"""Compose attributed text collections with reproducible per-source example caps."""
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from modelchef.corpus import digest, file_digest


def compose(path, root):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    folder = (root / config['output_dir']).resolve()
    if not folder.is_relative_to((root / 'data').resolve()):
        raise ValueError('Output must be under the project data directory')
    identity = digest(json.dumps(config, sort_keys=True).encode())
    manifest_path = folder / 'attribution.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['recipe_sha256'] != identity or file_digest(folder / 'corpus.jsonl') != saved['corpus_sha256']:
            raise ValueError('Existing composition differs; choose a new output directory')
        return saved
    grouped, sources, inputs = {}, [], []
    for item in config['inputs']:
        manifest = (root / item['manifest']).resolve()
        source_manifest = json.loads(manifest.read_text(encoding='utf-8'))
        collection = next(c for c in source_manifest['collections'] if c['file'] == item['collection'])
        data = manifest.parent / collection['file']
        if file_digest(data) != collection['sha256']:
            raise ValueError('Input collection checksum changed')
        inputs.append({'manifest': str(manifest), 'manifest_sha256': file_digest(manifest),
                       'collection': str(data), 'collection_sha256': collection['sha256']})
        sources.extend(source_manifest['sources'])
        with data.open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if not isinstance(record.get('text'), str) and all(isinstance(record.get(k), str)
                                                                  for k in ['prompt', 'response']):
                    record['text'] = record['prompt'].rstrip() + '\nAssistant: ' + record['response']
                if not isinstance(record.get('text'), str):
                    raise TypeError('Composition requires text records')
                prefix = record['text'].split('\nAssistant:', 1)[0]
                record.setdefault('split_group', 'prompt:' + digest(' '.join(prefix.casefold().split()).encode()))
                ordering = digest(f"{config['seed']}:{record['text']}".encode())
                grouped.setdefault(record['source'], []).append((ordering, record))
    selected, seen = [], set()
    for name, records in sorted(grouped.items()):
        cap = config['max_examples_by_source'].get(name)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 3:
            raise ValueError(f'Configure a positive example cap for {name}')
        for ordering, record in sorted(records, key=lambda pair: pair[0])[:cap]:
            key = digest(record['text'].encode())
            if key not in seen:
                seen.add(key)
                selected.append((ordering, record))
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'corpus.jsonl'
    with output.open('w', encoding='utf-8', newline='\n') as stream:
        for _, record in sorted(selected, key=lambda pair: pair[0]):
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    manifest = {'name': config['name'], 'recipe_sha256': identity, 'sources': sources,
                'inputs': inputs, 'corpus_sha256': file_digest(output),
                'source_counts': dict(Counter(record['source'] for _, record in selected)),
                'source_bytes': {name: sum(len(record['text'].encode()) for _, record in selected
                                          if record['source'] == name) for name in grouped},
                'selection': 'Seeded SHA ordering, per-source example caps, exact-text deduplication; '
                             'normalized prompt split groups. These are not fixed token proportions.'}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    args = parser.parse_args()
    print(json.dumps(compose(args.recipe, Path.cwd()), indent=2))
