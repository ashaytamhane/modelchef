"""Bounded Neuron feasibility jobs, reusing the existing EC2 shutdown lifecycle."""
import io
import json
import tarfile
from pathlib import Path

import yaml

from modelchef.neuron import NeuronRecipe


def recipe(root, settings):
    config = NeuronRecipe.model_validate(yaml.safe_load(
        (Path(root) / settings['pipeline_config']).read_text(encoding='utf-8')))
    if (config.steps != 4 or config.checkpoint_every != 2 or config.batch_size != 1
            or config.sequence_length > 128 or config.vocab_size > 4096
            or config.model.layers > 2 or config.model.hidden_size > 128):
        raise ValueError('AWS Neuron probe requires four steps, checkpoints every two, '
                         'batch one, context <=128, vocabulary <=4096 and a tiny model')
    if settings.get('instance_type') != 'trn1.2xlarge':
        raise ValueError('Training probe requires trn1.2xlarge; Inferentia is an inference target')
    if settings['max_runtime_minutes'] > 60 or settings['storage_gb'] > 100:
        raise ValueError('Neuron feasibility jobs are limited to 60 minutes and 100 GB storage')
    return config


def build_bundle(root, settings):
    root = Path(root).resolve()
    config = recipe(root, settings)
    cap = min(int(settings.get('max_bundle_bytes', 64000000)), 64000000)
    token_file = (root / config.tokens_path).resolve()
    if token_file.stat().st_size > 1000000:
        raise ValueError('Neuron probe token input is limited to 1 MB')
    files = {'inputs/tokens.bin': token_file.read_bytes()}
    config.tokens_path = 'inputs/tokens.bin'
    config.output_dir = 'artifacts/neuron/probe'
    files['configs/job.yaml'] = yaml.safe_dump(config.model_dump(mode='json')).encode()
    for path in [root / 'pyproject.toml', root / 'README.md',
                 *sorted((root / 'src/modelchef').glob('*.py'))]:
        if path.stat().st_size + sum(map(len, files.values())) > cap:
            raise ValueError('Neuron source bundle exceeds configured cap')
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    if sum(map(len, files.values())) > cap:
        raise ValueError('Neuron source bundle exceeds configured cap')
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def preflight(root, settings):
    import numpy as np
    import torch

    from modelchef.neuron import create_decoder

    config = recipe(root, settings)
    values = np.fromfile(Path(root) / config.tokens_path, dtype='<u4')
    if len(values) <= config.sequence_length or values.max() >= config.vocab_size:
        raise ValueError('Invalid Neuron probe token input')
    model = create_decoder(config)
    ids = torch.from_numpy(values[:config.sequence_length + 1].astype('int64'))
    loss = torch.nn.functional.cross_entropy(model(ids[:-1][None]).reshape(-1, config.vocab_size),
                                            ids[1:])
    loss.backward()
    if not torch.isfinite(loss):
        raise ValueError('Neuron portable CPU preflight produced invalid loss')
    return {'passed': True, 'cpu_forward_backward': True, 'hardware_validation': False}


def run_probe(path):
    import torch

    from modelchef.neuron import compile_model, create_decoder, train
    from modelchef.progress import report

    config = NeuronRecipe.model_validate(yaml.safe_load(Path(path).read_text()))
    output = Path(config.output_dir)
    train(config)
    final_path = output / 'step-000004.pt'
    expected = torch.load(final_path, weights_only=True, map_location='cpu')
    # This disposable probe intentionally replays the final two optimizer steps.
    final_path.unlink()
    report('Restoring step two and replaying optimizer steps three and four on Trainium.')
    train(config, output / 'step-000002.pt')
    resumed = torch.load(final_path, weights_only=True, map_location='cpu')
    for key, value in expected['model'].items():
        torch.testing.assert_close(resumed['model'][key], value, rtol=1e-5, atol=1e-6)
    manifest = compile_model(config, final_path)
    report('Executing the compiled decoder on Neuron and comparing logits with CPU.')
    model = create_decoder(config).eval()
    model.load_state_dict(resumed['model'])
    compiled = torch.jit.load(str(output / 'compiled/model.pt'))
    ids = torch.arange(config.sequence_length)[None] % config.vocab_size
    with torch.no_grad():
        actual, reference = compiled(ids).cpu(), model(ids)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    result = {'status': 'completed', 'training_resume_verified': True,
              'compiled_logits_verified': True, 'compiled_on': 'trn1.2xlarge',
              'inf2_execution_verified': False, 'quality_evaluation': False,
              'max_logit_error': float((actual - reference).abs().max()), 'compilation': manifest}
    (output / 'probe-report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    import sys
    run_probe(sys.argv[1])
