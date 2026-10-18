"""Capped, cached Parquet-file acquisition for larger public corpus samples."""
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from modelchef.corpus import digest, file_digest, viewer_headers, viewer_page

RANGE_THRESHOLD_BYTES = 256 * 1024 * 1024
RANGE_CHUNK_BYTES = 32 * 1024 * 1024


class DownloadRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        if urlparse(newurl).scheme != 'https':
            raise ValueError('Parquet download refuses non-HTTPS redirects')
        redirected = super().redirect_request(request, response, code, message, headers, newurl)
        if redirected and urlparse(newurl).hostname != urlparse(request.full_url).hostname:
            redirected.remove_header('Authorization')
        return redirected


def download_file(url, path, size, timeout, headers, explain):
    import time
    from urllib.error import HTTPError, URLError
    for attempt in range(4):
        try:
            return _download_file(url, path, size, timeout, headers, explain)
        except (TimeoutError, ConnectionError, URLError) as error:
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 3:
                raise
            delay = min(5 * 2 ** attempt, 20)
            explain(f'Bulk transfer interrupted ({type(error).__name__}); retrying the same '
                    f'file in {delay}s, attempt {attempt + 2}/4. Verified files are retained.')
            time.sleep(delay)


def _download_file(url, path, size, timeout, headers, explain):
    stamp = path.with_suffix('.json')
    if path.exists() and stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get('url') == url and saved.get('size') == size:
            if path.stat().st_size != size or file_digest(path) != saved['sha256']:
                raise ValueError('Cached Parquet file checksum changed; refuse silent reuse')
            explain(f'Reusing verified Parquet file ({size:,} bytes).')
            return saved
    if size >= RANGE_THRESHOLD_BYTES:
        return _download_ranges(url, path, size, timeout, headers, explain)
    explain(f'Downloading one Parquet file ({size:,} bytes), rather than 100-row pages.')
    temporary = path.with_suffix('.part')
    try:
        with build_opener(DownloadRedirect()).open(Request(url, headers=headers), timeout=timeout) as response:
            received = 0
            milestone = 32 * 1024 * 1024
            with temporary.open('wb') as target:
                while chunk := response.read(min(1024 * 1024, size - received + 1)):
                    received += len(chunk)
                    if received > size:
                        raise ValueError('Parquet response exceeds its approved metadata size')
                    target.write(chunk)
                    if received >= milestone:
                        explain(f'Parquet transfer: {received:,} / {size:,} bytes.')
                        milestone += 32 * 1024 * 1024
            if received != size:
                raise ValueError('Parquet download was truncated; retry acquisition')
        temporary.replace(path)
        saved = {'url': url, 'size': size, 'sha256': file_digest(path)}
        stamp.write_text(json.dumps(saved), encoding='utf-8')
        return saved
    finally:
        temporary.unlink(missing_ok=True)


def _download_ranges(url, path, size, timeout, headers, explain):
    """Commit verified prefixes; never append an ignored or mismatched HTTP range."""
    import hashlib
    temporary = path.with_suffix('.range-part')
    state_path = path.with_suffix('.range-state.json')
    state = {'url': url, 'size': size, 'received': 0, 'etag': None, 'sha256': digest(b'')}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['url'] != url or state['size'] != size or not 0 <= state['received'] <= size:
            raise ValueError('Partial download identity changed')
        if not temporary.exists() or temporary.stat().st_size != state['received'] or file_digest(temporary) != state['sha256']:
            raise ValueError('Partial download checksum/size changed')
    elif temporary.exists():
        raise ValueError('Partial download has no verified state')
    hasher = hashlib.sha256()
    if temporary.exists():
        with temporary.open('rb') as previous:
            while chunk := previous.read(1024 * 1024):
                hasher.update(chunk)
    explain(f'Range transfer: resume at {state["received"]:,} / {size:,} bytes.')
    while state['received'] < size:
        start = state['received']
        end = min(size, start + RANGE_CHUNK_BYTES) - 1
        request_headers = {**headers, 'Range': f'bytes={start}-{end}'}
        if state['etag']:
            request_headers['If-Range'] = state['etag']
        with build_opener(DownloadRedirect()).open(Request(url, headers=request_headers), timeout=timeout) as response:
            expected = f'bytes {start}-{end}/{size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server did not honor the exact requested byte range')
            etag = response.headers.get('ETag')
            if state['etag'] and etag != state['etag']:
                raise ValueError('Source version changed during range transfer')
            payload = response.read(end - start + 2)
            if len(payload) != end - start + 1:
                raise ConnectionError('Incomplete range response; retry the same range')
        with temporary.open('ab') as target:
            target.write(payload)
        hasher.update(payload)
        state.update(received=end + 1, etag=etag, sha256=hasher.hexdigest())
        pending = state_path.with_suffix('.tmp')
        pending.write_text(json.dumps(state), encoding='utf-8')
        pending.replace(state_path)
        explain(f'Range transfer: verified {state["received"]:,} / {size:,} bytes.')
    temporary.replace(path)
    saved = {'url': url, 'size': size, 'sha256': state['sha256']}
    path.with_suffix('.json').write_text(json.dumps(saved), encoding='utf-8')
    state_path.unlink()
    return saved


def acquire_parquet(config, cache, explain, row_sink=None):
    import pyarrow.parquet as pq

    headers = viewer_headers()
    explain('Resolving bulk Parquet file metadata; authentication enabled.' if 'Authorization' in headers
            else 'Resolving bulk Parquet file metadata using anonymous access.')
    request = Request('https://datasets-server.huggingface.co/parquet?' +
                      urlencode({'dataset': config.dataset}), headers=headers)
    limit = min(2_000_000, config.max_download_bytes)
    payload = viewer_page(request, config.timeout_seconds, limit + 1, explain)
    if len(payload) > limit:
        raise ValueError('Parquet metadata exceeds byte cap')
    metadata = json.loads(payload)
    files = [item for item in metadata['parquet_files']
             if item['config'] == config.subset and item['split'] == config.split]
    if not files:
        raise ValueError('No Parquet files available for the selected subset/split; conversion may be pending')
    cache.mkdir(parents=True, exist_ok=True)
    records, manifests = [], []
    used, index, selected = len(payload), 0, 0
    # Only the next required file is prefetched. Parsing and transfer overlap;
    # row order, approved byte accounting and the existing verified cache remain unchanged.
    pending = None
    with ThreadPoolExecutor(max_workers=1) as downloads:
        for position, item in enumerate(files):
            path, saved, size = (pending.result() if pending is not None else
                _selected_parquet(item, config, cache, used, headers, explain))
            pending = None
            manifests.append(saved)
            used += size
            with pq.ParquetFile(path) as parquet:
                if config.text_field not in parquet.schema_arrow.names:
                    raise ValueError('Parquet file is missing the configured text_field')
                # Metadata proves that the row cap cannot be met in this file.
                # Do not download unneeded files merely because they are listed.
                if (position + 1 < len(files) and
                        index + parquet.metadata.num_rows < config.offset + config.max_documents):
                    pending = downloads.submit(_selected_parquet, files[position + 1], config,
                                               cache, used, headers, explain)
                for batch in parquet.iter_batches(batch_size=128, columns=[config.text_field]):
                    for text in batch.column(0).to_pylist():
                        if index >= config.offset:
                            if not isinstance(text, str):
                                raise ValueError('Parquet text_field must contain strings')
                            if len(text.encode()) > config.max_record_bytes:
                                raise ValueError('Parquet text record exceeds max_record_bytes')
                            record = {'text': text, 'source_index': index}
                            if row_sink is None:
                                records.append(record)
                            else:
                                row_sink(record)
                            selected += 1
                        index += 1
                        if selected >= config.max_documents:
                            break
                    if selected >= config.max_documents:
                        break
            explain(f'Selected {selected} rows from bulk files; {used:,} / {config.max_download_bytes:,} source bytes.')
            if selected >= config.max_documents:
                break
    return records if row_sink is None else selected, used, {'provider': 'hf_parquet', 'dataset': config.dataset,
                          'subset': config.subset, 'split': config.split, 'offset': config.offset,
                          'files': manifests, 'partial_conversion': metadata.get('partial', False),
                          'dataset_card': f'https://huggingface.co/datasets/{config.dataset}',
                          'revision': None, 'revision_note': 'Replay pins cached file and raw snapshot SHA256; conversion metadata is mutable.'}


def _selected_parquet(item, config, cache, used, headers, explain):
    url, size = item['url'], item['size']
    parsed = urlparse(url)
    prefix = '/datasets/' + quote(config.dataset, safe='/') + '/resolve/'
    if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co' or not parsed.path.startswith(prefix):
        raise ValueError('Unexpected Parquet download URL; only selected Hugging Face dataset files are allowed')
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError('Parquet metadata requires a positive file size')
    if used + size > config.max_download_bytes:
        raise ValueError('Next full Parquet file exceeds max_download_bytes; increase the approved cap or use a smaller local shard')
    path = cache / (digest(url.encode()) + '.parquet')
    saved = download_file(url, path, size, config.timeout_seconds, headers, explain)
    return path, saved, size
