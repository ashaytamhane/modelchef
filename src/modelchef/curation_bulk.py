"""Disk-backed ingredient mixing using complete public Parquet records."""
import json
import sqlite3
from collections import Counter
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request

import yaml

from modelchef.bulk_data import download_file
from modelchef.corpus import digest, file_digest, viewer_headers, viewer_page


def parquet_rows(source, folder, budget, timeout, progress):
    if budget <= 0:
        raise ValueError('Curation download budget is exhausted')
    headers = viewer_headers()
    payload = viewer_page(Request('https://datasets-server.huggingface.co/parquet?' +
        urlencode({'dataset': source.dataset}), headers=headers), timeout,
        min(2_000_000, budget) + 1, progress)
    if len(payload) > min(2_000_000, budget):
        raise ValueError('Bulk metadata exceeds 2 MB or the remaining download budget')
    files = [f for f in json.loads(payload)['parquet_files']
             if f['config'] == source.subset and f['split'] == source.split]
    if not files:
        raise ValueError(f'{source.name}: no Parquet files for the selected configuration')
    import pyarrow.parquet as pq
    used, index = len(payload), 0
    provenance = []
    for item in files:
        url, size = item['url'], item['size']
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co' or not parsed.path.startswith(
                '/datasets/' + quote(source.dataset, safe='/') + '/resolve/'):
            raise ValueError('Unexpected bulk dataset URL')
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or used + size > budget:
            raise ValueError('Next full shard exceeds the approved curation download budget')
        path = folder / (digest(url.encode()) + '.parquet')
        provenance.append(download_file(url, path, size, timeout, headers, progress))
        used += size
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=128):
                selected = []
                for row in batch.to_pylist():
                    current, index = index, index + 1
                    if current < source.offset:
                        continue
                    if current >= source.offset + source.max_rows:
                        break
                    selected.append({'row_idx': current, 'row': row})
                if selected:
                    yield selected, used, provenance
                if index >= source.offset + source.max_rows:
                    return


def curate_bulk(config, root, progress):
    from modelchef.curation import adapt_rows, curate
    folder = root / config.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    identity = digest(config.model_dump_json().encode())
    manifest_path = folder / 'attribution.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['recipe_sha256'] != identity:
            raise ValueError('Existing bulk curation uses a different recipe')
        for item in saved['snapshots'] + saved['collections']:
            if file_digest(folder / item['file']) != item['sha256']:
                raise ValueError('Bulk snapshot checksum changed')
        progress('Reusing verified complete bulk mixtures.')
        return saved
    database = folder / 'mixing.sqlite'
    connection = sqlite3.connect(database)
    downloaded, sources, snapshots = 0, [], []
    try:
        connection.executescript('DROP TABLE IF EXISTS examples; CREATE TABLE examples '
            '(collection TEXT, ordering TEXT, source TEXT, payload TEXT);')
        for source in config.sources:
            progress(f'Preparing ingredient {source.name}: {source.max_rows:,} source rows maximum.')
            accepted, skipped, count = 0, Counter(), 0
            raw_path = folder / (source.name + '-raw.jsonl')
            parquet_manifest = []
            if source.provider != 'hf_parquet':
                # Reuse bounded viewer/CSV adapters for the small social and rating sources.
                small = config.model_copy(deep=True)
                small.sources = [source]
                small.output_dir = config.output_dir + '/small-' + source.name
                small.max_download_bytes = config.max_download_bytes - downloaded
                path = folder / ('small-' + source.name + '.yaml')
                path.write_text(yaml.safe_dump(small.model_dump(mode='json')))
                result = curate(path, root, execute=True, progress=progress)
                downloaded += result['response_bytes']
                source_manifest = result['sources'][0]
                for collection in result['collections']:
                    for line in (root / small.output_dir / collection['file']).read_text().splitlines():
                        row = json.loads(line)
                        ordering = digest(f"{config.seed}:{row['split_group']}:{line}".encode())
                        connection.execute('INSERT INTO examples VALUES (?,?,?,?)',
                            (collection['file'][:-6], ordering, source.name, line))
                for item in result['snapshots']:
                    snapshots.append({**item, 'file': 'small-' + source.name + '/' + item['file']})
            else:
                cache = folder / 'parquet' / source.name
                cache.mkdir(parents=True, exist_ok=True)
                used = 0
                with raw_path.open('w', encoding='utf-8') as raw:
                    for rows, used, parquet_manifest in parquet_rows(source, cache,
                            config.max_download_bytes - downloaded, config.timeout_seconds, progress):
                        bounded_rows = []
                        for row in rows:
                            payload = json.dumps(row, ensure_ascii=False)
                            if len(payload.encode()) > 2_000_000:
                                skipped['oversized_source_record'] += 1
                                continue
                            raw.write(payload + '\n')
                            bounded_rows.append(row)
                        converted, rejected = adapt_rows(bounded_rows, source)
                        count += len(rows)
                        skipped.update(rejected)
                        accepted += len(converted)
                        for row in converted:
                            payload = json.dumps(row, ensure_ascii=False)
                            ordering = digest(f"{config.seed}:{row['split_group']}:{payload}".encode())
                            connection.execute('INSERT INTO examples VALUES (?,?,?,?)',
                                (source.collection, ordering, source.name, payload))
                            if source.pretraining_collection:
                                text_row = {key: value for key, value in row.items()
                                            if key not in {'prompt', 'response'}}
                                text_row['text'] = row['prompt'].rstrip() + '\nAssistant: ' + row['response']
                                connection.execute('INSERT INTO examples VALUES (?,?,?,?)',
                                    (source.pretraining_collection, ordering, source.name,
                                     json.dumps(text_row, ensure_ascii=False)))
                        connection.commit()
                        if count % 10000 < 128:
                            progress(f'{source.name}: read {count:,}, retained {accepted:,} complete examples.')
                downloaded += used
                if accepted < 3:
                    raise ValueError('Bulk ingredient retained fewer than three complete examples')
                sha = file_digest(raw_path)
                snapshots.append({'file': raw_path.name, 'sha256': sha})
                source_manifest = {**source.model_dump(), 'dataset_url':
                    'https://huggingface.co/datasets/' + source.dataset,
                    'snapshot_sha256': sha, 'source_rows': count, 'accepted_examples': accepted,
                    'skipped': dict(skipped), 'parquet_files': parquet_manifest,
                    'revision_note': 'Downloaded shards pinned by individual SHA256; mutable metadata.'}
            sources.append(source_manifest)
            connection.commit()
        connection.execute('CREATE INDEX IF NOT EXISTS ordered_collection ON examples(collection,ordering)')
        outputs = []
        names = [row[0] for row in connection.execute('SELECT DISTINCT collection FROM examples')]
        for name in sorted(names):
            path = folder / (name + '.jsonl')
            counts = Counter()
            with path.open('w', encoding='utf-8') as output:
                for source, payload in connection.execute(
                        'SELECT source,payload FROM examples WHERE collection=? ORDER BY ordering', (name,)):
                    output.write(payload + '\n')
                    counts[source] += 1
            outputs.append({'file': path.name, 'sha256': file_digest(path),
                            'examples': sum(counts.values()), 'source_counts': dict(counts)})
        result = {'schema_version': 1, 'name': config.name, 'recipe_sha256': identity,
                  'sources': sources, 'snapshots': snapshots, 'collections': outputs,
                  'response_bytes': downloaded, 'max_download_bytes': config.max_download_bytes,
                  'transformations': 'Complete role-preserving adapters; disk-backed deterministic hash ordering. '
                  'Declared source-row mixture, not exact token proportions. SHA-pinned source shards. '
                  'No clipped targets, invented reasoning or neural signals. Subsequent pipeline filtering recorded separately.'}
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
        temporary.replace(manifest_path)
        return result
    finally:
        connection.close()
