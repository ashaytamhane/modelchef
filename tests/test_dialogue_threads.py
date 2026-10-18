from modelchef.curation import Ingredient, adapt_rows


def source():
    fields = {k: k for k in ['id', 'parent', 'role', 'text', 'language', 'rank',
                              'synthetic', 'deleted', 'review']}
    return Ingredient(name='threads', dataset='author/data', adapter='thread_turn',
        thread_fields=fields, thread_roles={'user': 'user', 'assistant': 'assistant'},
        collection='pairs', license='apache-2.0', license_url='https://example.org/license',
        attribution='Authors', citations=[{'title': 'Paper', 'authors': ['Author'],
                                         'url': 'https://example.org/paper'}])


def row(index, parent, role, text, **kwargs):
    return {'row_idx': index, 'row': {'id': str(index), 'parent': parent, 'role': role,
        'text': text, 'language': 'en', 'rank': 0, 'synthetic': False,
        'deleted': False, 'review': True, **kwargs}}


def test_thread_reconstruction_across_batches_and_shared_groups():
    context = {}
    assert adapt_rows([row(0, None, 'user', 'Hello')], source(), context)[0] == []
    first, _ = adapt_rows([row(1, '0', 'assistant', 'Hi!')], source(), context)
    second, _ = adapt_rows([row(2, '1', 'user', 'How are you?'),
        row(3, '2', 'assistant', 'Ready to help.')], source(), context)
    assert first[0]['prompt'] == 'User: Hello'
    assert second[0]['prompt'] == 'User: Hello\nAssistant: Hi!\nUser: How are you?'
    assert first[0]['split_group'] == second[0]['split_group']


def test_thread_rejects_synthetic_ancestors_missing_parents_and_lower_rank():
    rows = [row(0, None, 'user', 'Hello', synthetic=True),
            row(1, '0', 'assistant', 'Hi!'), row(2, 'missing', 'assistant', 'Incomplete'),
            row(3, None, 'user', 'A question'), row(4, '3', 'assistant', 'Lower rank', rank=1)]
    result, skipped = adapt_rows(rows, source())
    assert result == []
    assert skipped == {'synthetic_or_unverified_human_message': 1, 'missing_parent': 1,
                       'unranked_or_lower_quality_reply': 1}
