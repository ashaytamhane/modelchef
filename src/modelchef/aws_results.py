"""Small controller result archives; complete models remain in S3 recovery snapshots."""
import argparse
import tarfile
from pathlib import Path


def archive_metadata(root, output):
    root = Path(root).resolve()
    names = {'report.json', 'config.json', 'workflow.json', 'environment.json',
             'source-manifest.json', 'benchmark-manifest.json', 'data-attribution.json',
             'events.jsonl', 'metrics.jsonl', 'tensorboard.json', 'performance-gate.json'}
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted((root / 'artifacts').rglob('*')):
            if path.is_file() and not path.is_symlink() and path.name in names:
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    archive_metadata(args.root, args.output)


if __name__ == '__main__':
    main()
