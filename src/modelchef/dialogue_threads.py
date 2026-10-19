"""Reconstruct quality-filtered conversations from explicitly mapped flat trees."""
from collections import Counter

from modelchef.corpus import digest


def thread_pairs(rows, source, context=None):
    from modelchef.curation import transcript
    nodes = {} if context is None else context
    fields = source.thread_fields
    skipped, output = Counter(), []
    for entry in rows:
        row = entry['row']
        identity = row.get(fields['id'])
        if isinstance(identity, str) and identity:
            nodes[identity] = {key: row.get(column) for key, column in fields.items()}
    for entry in rows:
        node = nodes.get(entry['row'].get(fields['id']))
        if not node or source.thread_roles.get(node['role']) != 'assistant':
            continue
        rank = node['rank']
        if not isinstance(rank, int) or isinstance(rank, bool) or rank > source.thread_max_rank:
            skipped['unranked_or_lower_quality_reply'] += 1
            continue
        chain, seen, current, invalid = [], set(), node, None
        while current:
            identity = current['id']
            if identity in seen:
                invalid = 'cyclic_thread'
                break
            seen.add(identity)
            role = source.thread_roles.get(current['role'])
            if (current['language'] != source.thread_language or current['deleted'] is not False
                    or current['review'] is False or role not in {'user', 'assistant'}
                    or not isinstance(current['text'], str) or not current['text'].strip()):
                invalid = 'invalid_thread_message'
                break
            if source.thread_human_only and current['synthetic'] is not False:
                invalid = 'synthetic_or_unverified_human_message'
                break
            if source.reasoning == 'exclude' and any(tag in current['text'] for tag in ['<think>', '</think>']):
                invalid = 'reasoning_excluded'
                break
            chain.append({'role': role, 'content': current['text']})
            parent = current['parent']
            if parent is None:
                break
            current = nodes.get(parent)
            if current is None:
                invalid = 'missing_parent'
                break
        chain.reverse()
        if invalid:
            skipped[invalid] += 1
            continue
        if len(chain) < 2 or any(message['role'] != ('user' if i % 2 == 0 else 'assistant')
                                 for i, message in enumerate(chain)):
            skipped['invalid_role_sequence'] += 1
            continue
        prompt = transcript(chain[:-1], source.system_prompt, source.system_prefix)
        response = chain[-1]['content'].strip()
        if len((prompt + response).encode()) > source.max_example_bytes:
            skipped['oversized'] += 1
            continue
        output.append({'prompt': prompt, 'response': response, 'source': source.name,
                       'source_row': entry['row_idx'],
                       'split_group': source.name + ':thread:' + digest(chain[0]['content'].strip().casefold().encode())})
    return output, dict(skipped)
