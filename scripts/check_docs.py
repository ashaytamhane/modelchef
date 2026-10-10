"""Check local documentation links, CLI examples, recipes and schema drift; no network."""
import ast
import json
import re
from pathlib import Path

import yaml

from modelchef.run_config import RunConfig
from modelchef.workflow import Workflow, cook


def check(root):
    root = Path(root).resolve()
    text = (root / 'README.md').read_text(encoding='utf-8')
    assert text.count('```') % 2 == 0, 'Unbalanced README code fences'
    headings = re.findall(r'^#{1,6}\s+(.+)$', re.sub(r'```.*?```', '', text, flags=re.DOTALL), re.MULTILINE)
    anchors = {re.sub(r'[^\w -]', '', h.lower()).replace(' ', '-') for h in headings}
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if link.startswith(('https://', 'http://', 'mailto:')):
            continue
        if link.startswith('#'):
            assert link[1:] in anchors, f'Missing README anchor: {link}'
        else:
            assert (root / link.split('#')[0]).exists(), f'Missing README file: {link}'
    tree = ast.parse((root / 'src/modelchef/cli.py').read_text(encoding='utf-8'))
    commands = {node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_parser' and node.args
                and isinstance(node.args[0], ast.Constant)}
    mentioned = set(re.findall(r'(?:python -m modelchef\.cli|modelchef) ([a-z][a-z-]*)', text))
    # Only command lines; prose uses the capitalized brand.
    assert not mentioned - commands, f'Unknown documented commands: {mentioned - commands}'
    reference = yaml.safe_load((root / 'configs/recipe-reference.yaml').read_text(encoding='utf-8'))
    RunConfig.model_validate(reference).validate_execution()
    defaults = RunConfig().model_dump(mode='json')

    def complete(expected, actual, path=''):
        for key, value in expected.items():
            assert key in actual, f'Annotated recipe omits {path}{key}'
            if isinstance(value, dict):
                complete(value, actual[key], path + key + '.')

    complete(defaults, reference)
    blocks = re.findall(r'```yaml\n(.*?)\n```', text, re.DOTALL)
    documented = next((yaml.safe_load(block) for block in blocks
                       if 'name: full-reference' in block), None)
    assert documented == reference, 'README reference recipe differs from checked-in recipe'
    for model, name in [(RunConfig, 'pretrain'), (Workflow, 'workflow')]:
        saved = json.loads((root / f'configs/{name}.schema.json').read_text(encoding='utf-8'))
        assert saved == model.model_json_schema(), f'{name} schema is stale'
    for path in (root / 'configs').glob('*.yaml'):
        if '.local.' in path.name:
            continue
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        if path.name.startswith('workflow-'):
            cook(path, root)
        elif isinstance(config, dict) and 'training' in config:
            RunConfig.model_validate(config).validate_execution()
    print('Documentation, commands, example recipes and schemas agree.')


if __name__ == '__main__':
    check(Path(__file__).resolve().parents[1])
