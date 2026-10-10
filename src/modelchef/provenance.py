"""Small, secret-free software identity records for reproducible experiments."""
import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


def environment_provenance(root):
    packages = {}
    for name in ('modelchef', 'torch', 'numpy', 'pydantic', 'tokenizers', 'safetensors',
                 'transformers', 'peft', 'bitsandbytes', 'transformer-engine', 'vllm'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        source.update(path.name.encode() + b'\0' + path.read_bytes())
    code = {'source_sha256': source.hexdigest(), 'git_revision': None, 'git_dirty': None}
    root = Path(root).resolve()
    if (root / '.git').exists():
        command = ['git', '--no-optional-locks', '-c', f'safe.directory={root}']
        try:
            code['git_revision'] = subprocess.check_output(
                command + ['rev-parse', 'HEAD'], cwd=root, text=True,
                stderr=subprocess.DEVNULL, timeout=5).strip()
            code['git_dirty'] = bool(subprocess.check_output(
                command + ['status', '--porcelain', '--untracked-files=no'], cwd=root,
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {'packages': packages, 'code': code}
