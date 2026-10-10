"""Small EC2 readiness entry point sharing production diagnostics and smoke code."""
import argparse
import json
from pathlib import Path

import yaml

from modelchef.diagnostics import check_environment
from modelchef.progress import operation


def main():
    parser = argparse.ArgumentParser(prog='modelchef')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    check = commands.add_parser('check-config')
    check.add_argument('path', type=Path)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--accelerator', default='auto')
    smoke.add_argument('--precision', default='auto')
    parser.add_argument('--quiet', action='store_true')
    for subparser in commands.choices.values():
        subparser.add_argument('--quiet', action='store_true', default=argparse.SUPPRESS)
    args = parser.parse_args()
    with operation(args.command, args.quiet):
        if args.command == 'doctor':
            result = check_environment()
        elif args.command == 'smoke':
            from modelchef.smoke import run_smoke
            result = run_smoke(args.accelerator, args.precision)
        else:
            result = yaml.safe_load(args.path.read_text())
            if not isinstance(result, dict) or 'project' not in result or 'runtime' not in result:
                parser.error("Config must contain 'project' and 'runtime' sections")
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
