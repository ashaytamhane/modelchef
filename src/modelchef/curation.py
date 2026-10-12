"""Bounded public-data recipes, reusable dialogue adapters and attribution manifests."""
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml
from pydantic import Field, model_validator

from modelchef.corpus import digest, file_digest, viewer_headers, viewer_page
from modelchef.run_config import Section


class Citation(Section):
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    url: str = Field(pattern=r'^https://')


class Ingredient(Section):
    name: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    dataset: str = Field(pattern=r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
    provider: Literal['hf_rows', 'hf_parquet', 'https_csv'] = 'hf_rows'
    source_url: str | None = Field(default=None, pattern=r'^https://')
    subset: str = 'default'
    split: str = 'train'
    offset: int = Field(default=0, ge=0)
    max_rows: int = Field(default=64, ge=3, le=10000000)
    collection: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    pretraining_collection: str | None = Field(default=None, pattern=r'^[a-zA-Z0-9_-]+$')
    adapter: Literal['text', 'messages', 'persona_turn', 'empathy_turn', 'measurements', 'reasoning_fields']
    reasoning_fields: dict[str, str] = Field(default_factory=dict)
    answer_extraction: Literal['none', 'last_boxed'] = 'none'
    exclude_sources: list[str] = Field(default_factory=list)
    measurement_fields: dict[str, str] = Field(default_factory=dict)
    measurement_description: str = ''
    measurement_min: float = 0
    measurement_max: float = 5
    text_field: str = 'text'
    max_example_bytes: int = Field(default=1600, ge=64)
    system_prompt: str = ''
    system_prefix: str = ''
    reasoning: Literal['keep', 'require', 'exclude'] = 'keep'
    min_user_words: int = Field(default=0, ge=0)
    license: str = Field(min_length=1)
    license_url: str = Field(pattern=r'^https://')
    attribution: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)

    @model_validator(mode='after')
    def valid_source(self):
        if self.answer_extraction != 'none' and self.adapter != 'reasoning_fields':
            raise ValueError('Answer extraction requires a reasoning-fields adapter')
        if self.adapter == 'reasoning_fields' and set(self.reasoning_fields) != {'prompt', 'reasoning', 'answer'}:
            raise ValueError('Reasoning fields require prompt, reasoning and answer column mappings')
        if self.provider == 'hf_rows' and self.max_rows > 10000:
            raise ValueError('Large ingredients require hf_parquet bulk acquisition')
        if self.provider == 'https_csv' and not self.source_url:
            raise ValueError('https_csv requires source_url')
        if self.adapter == 'measurements' and (not self.measurement_fields or
                not self.measurement_description or self.measurement_min >= self.measurement_max):
            raise ValueError('Measurements require columns, a description and a valid range')
        return self


class CurationConfig(Section):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    output_dir: str
    seed: int = 42
    max_download_bytes: int = Field(default=16000000, gt=0)
    timeout_seconds: int = Field(default=30, gt=0, le=120)
    sources: list[Ingredient] = Field(min_length=1)

    @model_validator(mode='after')
    def unique(self):
        if len({s.name for s in self.sources}) != len(self.sources):
            raise ValueError('Ingredient names must be unique')
        return self


def transcript(messages, system='', system_prefix=''):
    """Keep role boundaries; never infer a persona from the reply being predicted."""
    instruction = '\n'.join(part for part in [system_prefix, system] if part)
    parts = ['System: ' + instruction] if instruction else []
    for message in messages:
        role, content = message.get('role'), message.get('content')
        if role not in {'system', 'user', 'assistant'} or not isinstance(content, str):
            raise ValueError('Only string system/user/assistant messages are supported')
        parts.append(role.capitalize() + ': ' + content.strip())
    return '\n'.join(parts)


def adapt_rows(rows, source):
    """Produce complete examples; reject oversized traces instead of clipping targets."""
    result, skipped, groups = [], Counter(), set()
    empathy = defaultdict(list)
    if source.adapter == 'empathy_turn':
        for entry in rows:
            empathy[entry['row']['conv_id']].append(entry)
        rows = [sorted(group, key=lambda x: x['row']['utterance_idx'])
                for group in empathy.values()]
    for entry in rows:
        if source.adapter == 'empathy_turn':
            turns = entry
            row, index = turns[0]['row'], turns[0]['row_idx']
            # The first speaker describes the situation; train the listener's first reply.
            if ([x['row']['utterance_idx'] for x in turns[:2]] != [1, 2]
                    or turns[0]['row']['speaker_idx'] == turns[1]['row']['speaker_idx']):
                skipped['incomplete_conversation'] += 1
                continue
            prompt = transcript([{'role': 'user', 'content':
                                 turns[0]['row']['utterance'].replace('_comma_', ',')}],
                                source.system_prompt, source.system_prefix)
            response = turns[1]['row']['utterance'].replace('_comma_', ',')
            group = str(row['conv_id'])
        else:
            row, index = entry['row'], entry['row_idx']
            if row.get('source') in source.exclude_sources:
                skipped['excluded_source'] += 1
                continue
            group = str(row.get('conv_id', index))
            if source.adapter == 'reasoning_fields':
                fields = {key: row.get(column) for key, column in source.reasoning_fields.items()}
                if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
                    skipped['incomplete_reasoning'] += 1
                    continue
                thought = fields['reasoning'].strip()
                answer = fields['answer'].strip()
                if source.answer_extraction == 'last_boxed':
                    marker = answer.rfind('\\boxed{')
                    depth, end = 1, marker + len('\\boxed{')
                    start = end
                    while marker >= 0 and end < len(answer) and depth:
                        depth += (answer[end] == '{') - (answer[end] == '}')
                        end += 1
                    if marker < 0 or depth or not answer[start:end-1].strip():
                        skipped['missing_boxed_answer'] += 1
                        continue
                    thought = (answer[:marker] + answer[end:]).strip()
                    answer = answer[start:end-1].strip()
                    if not thought:
                        skipped['empty_reasoning'] += 1
                        continue
                if any(tag in thought + answer for tag in ['<think>', '</think>']):
                    skipped['nested_reasoning_markers'] += 1
                    continue
                prompt = transcript([{'role': 'user', 'content': fields['prompt']}],
                                    source.system_prompt, source.system_prefix)
                response = '<think>\n' + thought + '\n</think>\n' + answer
                group = digest(' '.join(fields['prompt'].casefold().split()).encode())
                if len((prompt + response).encode()) > source.max_example_bytes:
                    skipped['oversized'] += 1
                    continue
                result.append({'prompt': prompt, 'response': response, 'source': source.name,
                               'source_row': index, 'split_group': 'reasoning-problem:' + group,
                               'upstream_source': row.get('source'), 'domain': row.get('domain')})
                continue
            if source.adapter in {'text', 'measurements'}:
                text = row[source.text_field]
                if not isinstance(text, str):
                    raise ValueError('Text adapter requires string text')
                if source.adapter == 'measurements':
                    values = []
                    for column, label in source.measurement_fields.items():
                        value = float(row[column])
                        if not source.measurement_min <= value <= source.measurement_max:
                            raise ValueError('Measurement outside the declared range')
                        values.append(f'{label}: {row[column]}')
                    text = (source.measurement_description + '\nConcept: ' + text + '\n' +
                            '; '.join(values) + '.')
                # Foundation documents can be long; skip rather than mutate them.
                if len(text.encode()) > source.max_example_bytes:
                    skipped['oversized'] += 1
                    continue
                result.append({'text': text, 'source': source.name, 'source_row': index,
                               'split_group': source.name + ':' + digest(text.encode())})
                continue
            if source.adapter == 'persona_turn':
                history = row['history']
                if not history or not row['candidates']:
                    skipped['empty'] += 1
                    continue
                persona = 'Role-play the following profile for this conversation:\n' + '\n'.join(row['personality'])
                # ConvAI/PersonaChat stores the gold response as the final candidate.
                response = row['candidates'][-1]
                messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': content}
                            for i, content in enumerate(history)]
                prompt = transcript(messages, persona, source.system_prefix)
                group = digest(json.dumps(row['personality'], sort_keys=True).encode())
            else:
                kwargs = row.get('chat_template_kwargs') or {}
                if kwargs.get('python_tools') or kwargs.get('xml_tools'):
                    skipped['tools'] += 1
                    continue
                system = kwargs.get('custom_instructions') or source.system_prompt
                messages = row['messages']
                candidates = []
                for i, message in enumerate(messages):
                    if message.get('role') != 'assistant' or i == 0:
                        continue
                    if (messages[i - 1].get('role') != 'user' or
                            len(messages[i - 1]['content'].split()) < source.min_user_words):
                        continue
                    response = message['content'].strip()
                    has_thought = bool(re.match(r'^<think>.*?</think>', response, re.DOTALL))
                    if (source.reasoning == 'require' and not has_thought or
                            source.reasoning == 'exclude' and '<think>' in response):
                        continue
                    if has_thought and not response.split('</think>', 1)[-1].strip():
                        continue
                    # Keep one turn per source conversation, avoiding adjacent-turn leakage.
                    prompt = transcript(messages[:i], system, source.system_prefix)
                    candidates.append((prompt, response))
                fitting = [(p, r) for p, r in candidates if
                           len((p + r).encode()) <= source.max_example_bytes]
                if not fitting:
                    skipped['no_complete_fitting_turn'] += 1
                    continue
                prompt, response = fitting[0]
        if not prompt.strip() or not response.strip():
            skipped['empty'] += 1
            continue
        if len((prompt + response).encode()) > source.max_example_bytes:
            skipped['oversized'] += 1
            continue
        example_key = digest((prompt + '\nAssistant: ' + response).encode())
        if example_key in groups:
            skipped['duplicate_example'] += 1
            continue
        groups.add(example_key)
        result.append({'prompt': prompt, 'response': response, 'source': source.name,
                       'source_row': index, 'split_group': source.name + ':' + group})
    return result, dict(skipped)


def curate(path, root, execute=False, progress=print):
    root = Path(root).resolve()
    config = CurationConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding='utf-8')))
    folder = (root / config.output_dir).resolve()
    if not folder.is_relative_to(root / 'data'):
        raise ValueError('Curation outputs must be under the project data directory')
    plan = {'name': config.name, 'output_dir': str(folder),
            'max_download_bytes': config.max_download_bytes,
            'sources': [s.model_dump() for s in config.sources],
            'note': ('Bounded complete Parquet shards with ordered source-row selection; '
                     'mixture proportions are measured after filtering.'
                     if any(s.provider == 'hf_parquet' for s in config.sources) else
                     'Bounded ordered viewer/CSV slices; not representative full-corpus sampling.')}
    if not execute:
        return plan
    if any(source.provider == 'hf_parquet' for source in config.sources):
        from modelchef.curation_bulk import curate_bulk
        return curate_bulk(config, root, progress)
    folder.mkdir(parents=True, exist_ok=True)
    identity = digest(config.model_dump_json().encode())
    manifest_path = folder / 'attribution.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved['recipe_sha256'] != identity:
            raise ValueError('Existing curation uses a different recipe; use a new output_dir')
        for item in saved['snapshots'] + saved['collections']:
            if file_digest(folder / item['file']) != item['sha256']:
                raise ValueError('Curated snapshot checksum changed; refuse silent reuse')
        progress('Reusing checksum-verified curated snapshots; no download.')
        return saved
    headers, downloaded, snapshots, sources = viewer_headers(), 0, [], []
    collections = defaultdict(list)
    for source in config.sources:
        raw_path = folder / (source.name + '-raw.json')
        # Failed preparation leaves verified raw responses for retry, not partially written mixtures.
        stamp = raw_path.with_suffix('.sha256')
        cache_key = digest(source.model_dump_json().encode())
        if raw_path.exists() and stamp.exists():
            cached = json.loads(stamp.read_text())
            if cached['key'] != cache_key or file_digest(raw_path) != cached['sha256']:
                raise ValueError('Cached ingredient snapshot changed')
            rows = json.loads(raw_path.read_text(encoding='utf-8'))
            downloaded += cached['response_bytes']
        else:
            rows, size = [], 0
            progress(f'Acquiring {source.name}: at most {source.max_rows} source rows; adapter={source.adapter}.')
            if source.provider == 'https_csv':
                remaining = config.max_download_bytes - downloaded
                if remaining <= 0:
                    raise ValueError('Curation exceeds approved download cap')
                # Stream only the requested rows; do not materialize the full CSV.
                with urlopen(Request(source.source_url, headers={'User-Agent': 'ModelChef/0.1'}),
                             timeout=config.timeout_seconds) as response:
                    def lines():
                        nonlocal size, downloaded
                        while True:
                            line = response.readline(min(65536, config.max_download_bytes - downloaded) + 1)
                            size += len(line)
                            downloaded += len(line)
                            if downloaded > config.max_download_bytes:
                                raise ValueError('Curation exceeds approved download cap')
                            if not line:
                                return
                            if len(line) > 65536:
                                raise ValueError('CSV line exceeds 64 KiB')
                            yield line.decode('utf-8-sig')
                    for index, row in enumerate(csv.DictReader(lines())):
                        if index < source.offset:
                            continue
                        rows.append({'row_idx': index, 'row': row})
                        if len(rows) >= source.max_rows:
                            break
            for offset in (range(source.offset, source.offset + source.max_rows, 100)
                           if source.provider == 'hf_rows' else []):
                query = urlencode({'dataset': source.dataset, 'config': source.subset, 'split': source.split,
                                   'offset': offset, 'length': min(100, source.offset + source.max_rows - offset)})
                remaining = config.max_download_bytes - downloaded
                if remaining <= 0:
                    raise ValueError('Curation exceeds approved download cap')
                payload = viewer_page(Request('https://datasets-server.huggingface.co/rows?' + query,
                                             headers=headers), config.timeout_seconds, remaining + 1, progress)
                size += len(payload)
                downloaded += len(payload)
                if downloaded > config.max_download_bytes:
                    raise ValueError('Curation exceeds approved download cap')
                page = json.loads(payload)
                if any(entry.get('truncated_cells') for entry in page['rows']):
                    raise ValueError('Viewer truncated a record; use a complete snapshot')
                rows.extend(page['rows'])
                if not page['rows']:
                    break
            raw_path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
            stamp.write_text(json.dumps({'key': cache_key, 'sha256': file_digest(raw_path),
                                         'response_bytes': size}), encoding='utf-8')
        if downloaded > config.max_download_bytes:
            raise ValueError('Cached sources exceed approved byte cap')
        converted, skipped = adapt_rows(rows, source)
        if len(converted) < 3:
            raise ValueError(f'{source.name}: only {len(converted)} complete examples fit; adjust sample/length limits')
        collections[source.collection].extend(converted)
        if source.pretraining_collection:
            if source.adapter in {'text', 'measurements'}:
                raise ValueError('pretraining_collection requires structured dialogue examples')
            # Full-token next-token training: retain the context, reasoning and final reply.
            collections[source.pretraining_collection].extend({
                'text': row['prompt'].rstrip() + '\nAssistant: ' + row['response'],
                'source': row['source'], 'source_row': row['source_row'],
                'split_group': row['split_group']} for row in converted)
        snapshots.append({'file': raw_path.name, 'sha256': file_digest(raw_path)})
        sources.append({**source.model_dump(), 'dataset_url': source.source_url or
                        'https://huggingface.co/datasets/' + source.dataset,
                        'snapshot_sha256': file_digest(raw_path), 'source_rows': len(rows),
                        'accepted_examples': len(converted), 'skipped': skipped,
                        'revision': None, 'revision_note': 'Mutable viewer; replay pins the saved response SHA256.'})
        progress(f'{source.name}: retained {len(converted)} complete examples; skipped={skipped}.')
    outputs = []
    for name, records in sorted(collections.items()):
        random.Random(config.seed).shuffle(records)
        output = folder / (name + '.jsonl')
        output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        outputs.append({'file': output.name, 'sha256': file_digest(output), 'examples': len(records),
                        'source_counts': dict(Counter(r['source'] for r in records))})
    manifest = {**plan, 'schema_version': 1, 'recipe_sha256': identity, 'sources': sources,
                'response_bytes': downloaded, 'snapshots': snapshots, 'collections': outputs,
                'transformations': 'Role-preserving transcripts; original persona/system instructions; '
                'one complete fitting turn per conversation/persona; gold PersonaChat candidate only; '
                'EmpatheticDialogues listener response; no clipped responses or invented reasoning. '
                'Collection proportions are observed example counts, not token weights. '
                'Optional pretraining collections serialize complete dialogues as text, retaining split groups. '
                'Measurement tables are serialized with declared labels, scale and interpretation; '
                'these are reported ratings, not neural recordings or fabricated experiences. '
                'Pipeline filtering, PII redaction and split manifests are separate artifacts.'}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
