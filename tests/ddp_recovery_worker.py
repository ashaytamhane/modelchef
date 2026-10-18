"""Real DDP training with mocked S3 transport, including periodic recovery."""
import json
import os
import sys
from pathlib import Path

import yaml
from test_recovery import S3

from modelchef import recovery
from modelchef.distributed import run_distributed
from modelchef.run_config import RunConfig

root = Path.cwd()
uri = 's3://bucket/ddp/recovery'
os.environ['MODELCHEF_RECOVERY_S3_URI'] = uri
store = recovery.RecoveryStore(S3(), uri)
recovery._store = store
original = store.snapshot
published = []


def snapshot(workspace, folder, checkpoint):
    result = original(workspace, folder, checkpoint)
    published.append(checkpoint.name)
    if checkpoint.name == 'step-000001.pt':
        destination = root / 's3-restored' / str(len(published))
        restored = store.fetch(destination, 100_000_000)
        (root / 's3-checkpoint.txt').write_text(str(destination / restored['checkpoint']))
    return result


store.snapshot = snapshot
config = RunConfig.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text()))
resume = Path(sys.argv[2]) if len(sys.argv) > 2 else None
run_distributed(config, root, resume)
if int(os.environ['RANK']) == 0:
    assert 'step-000001.pt' in published
    if not resume:
        assert published[0] == 'step-000000.pt'
    (root / 's3-published.json').write_text(json.dumps(published))
