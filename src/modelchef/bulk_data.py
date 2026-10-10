"""Capped, cached Parquet-file acquisition for larger public corpus samples."""
import json
from urllib.parse import quote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from modelchef.corpus import digest, file_digest, viewer_headers, viewer_page


class DownloadRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        if urlparse(newurl).scheme != 'https':
            raise ValueError('Parquet download refuses non-HTTPS redirects')
        redirected = super().redirect_request(request, response, code, message, headers, newurl)
        if redirected and urlparse(newurl).hostname != urlparse(request.full_url).hostname:
            redirected.remove_header('Authorization')
        return redirected


def download_file(url, path, size, timeout, headers, explain):
    stamp = path.with_suffix('.json')
    if path.exists() and stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get('url') == url and saved.get('size') == size:
            if path.stat().st_size != size or file_digest(path) != saved['sha256']:
                raise ValueError('Cached Parquet file checksum changed; refuse silent reuse')
            explain(f'Reusing verified Parquet file ({size:,} bytes).')
            return saved
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


def acquire_parquet(config, cache, explain):
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
    used, index = len(payload), 0
    for item in files:
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
        manifests.append(download_file(url, path, size, config.timeout_seconds, headers, explain))
        used += size
        with pq.ParquetFile(path) as parquet:
            if config.text_field not in parquet.schema_arrow.names:
                raise ValueError('Parquet file is missing the configured text_field')
            for batch in parquet.iter_batches(batch_size=128, columns=[config.text_field]):
                for text in batch.column(0).to_pylist():
                    if index >= config.offset:
                        if not isinstance(text, str):
                            raise ValueError('Parquet text_field must contain strings')
                        if len(text.encode()) > config.max_record_bytes:
                            raise ValueError('Parquet text record exceeds max_record_bytes')
                        records.append({'text': text, 'source_index': index})
                    index += 1
                    if len(records) >= config.max_documents:
                        break
                if len(records) >= config.max_documents:
                    break
        explain(f'Selected {len(records)} rows from bulk files; {used:,} / {config.max_download_bytes:,} source bytes.')
        if len(records) >= config.max_documents:
            break
    return records, used, {'provider': 'hf_parquet', 'dataset': config.dataset,
                          'subset': config.subset, 'split': config.split, 'offset': config.offset,
                          'files': manifests, 'partial_conversion': metadata.get('partial', False),
                          'dataset_card': f'https://huggingface.co/datasets/{config.dataset}',
                          'revision': None, 'revision_note': 'Replay pins cached file and raw snapshot SHA256; conversion metadata is mutable.'}
