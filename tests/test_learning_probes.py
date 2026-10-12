import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('pydantic')
from modelchef.corpus import group_split
from modelchef.run_config import RunConfig


def test_overfit_selection_preserves_saved_holdouts_even_without_groups(tmp_path):
    select = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'scripts/overfit_probe.py'))['select_examples']
    config = RunConfig()
    for index, split in enumerate(['train', 'validation', 'test']):
        records = [{'prompt': f'{split} question {i}', 'response': f'{split} reply {i}',
                    'source_index': index*10+i} for i in range(5)]
        (tmp_path/(split+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in records))
    selected = select(tmp_path, 4, config)
    assert all(len(rows) == 4 for rows in selected.values())
    for split, rows in selected.items():
        assert all(group_split(row['split_group'], config) == split for row in rows)
        assert all(row['response'].startswith(split) for row in rows)
    with pytest.raises(ValueError, match='fewer'):
        select(tmp_path, 6, config)
