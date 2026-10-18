"""Two-process CPU/Gloo FSDP2 numerical/recovery check; no downloaded models."""
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from modelchef.engines import create_model
from modelchef.fsdp_training import shard_native
from modelchef.run_config import Architecture
from modelchef.sharded_checkpoint import load_sharded, publish_sharded_to_s3, save_sharded


def main():
    rank, root = int(sys.argv[1]), Path(sys.argv[2]).resolve()
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root / 'rendezvous').as_uri(), rank=rank, world_size=2)
    try:
        architecture = Architecture(layers=2, hidden_size=16, attention_heads=2, kv_heads=1,
                                    intermediate_size=32, context_length=8)
        torch.manual_seed(7)
        reference = create_model(architecture, 300)
        with torch.device('meta'):
            model = create_model(architecture, 300)
        model = shard_native(model, 'fp32', 'cpu')
        assert model.head.weight is model.embedding.weight, 'Tied embeddings were broken'
        set_model_state_dict(model, reference.state_dict() if rank == 0 else {},
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, foreach=False)
        x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        y = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

        def update():
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[rank:rank+1])
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y[rank:rank+1].flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            return norm.full_tensor().item()

        def full_weights():
            return {k: v.full_tensor().detach().clone() if hasattr(v, 'full_tensor') else v.detach().clone()
                    for k, v in model.state_dict().items()}

        save_sharded(model, optimizer, root / 'initial-step0.dcp',
            {'step': 0, 'tokens_seen': 0, 'identities': {'synthetic': 'same-data'}},
            {'torch_rng': torch.get_rng_state()})
        assert all(state['step'].item() == 0 for state in optimizer.state.values())
        norm = update()
        reference_optimizer.zero_grad(set_to_none=True)
        logits = reference(x)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .5).item()
        reference_optimizer.step()
        assert abs(norm-reference_norm) < 1e-5, (norm, reference_norm)
        weights = full_weights()
        assert all(torch.allclose(weights[k], v, atol=2e-6, rtol=2e-5)
                   for k, v in reference.state_dict().items()), 'FSDP update differs from global-batch reference'
        workspace = root / 'guest'
        run = workspace / 'artifacts/training/run'
        checkpoint = run / 'checkpoints/step-000001.dcp'
        if rank == 0:
            checkpoint.parent.mkdir(parents=True)
            (run / 'config.json').write_text('{"stage": {}, "training": {"distributed": "fsdp2"}}')
            (run / 'tokenizer.json').write_text('frozen tokenizer')
            (run / 'tokens-train.bin').write_bytes(b'prepared tokens')
        dist.barrier()
        sampler = torch.Generator().manual_seed(17 + rank)
        state = {'sampler_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state()}
        identities = {'synthetic': 'same-data'}
        save_sharded(model, optimizer, checkpoint, {'step': 1, 'tokens_seen': 8,
                     'identities': identities}, state)
        from test_recovery import S3

        from modelchef import recovery
        client = S3()
        os.environ['MODELCHEF_RECOVERY_S3_URI'] = 's3://bucket/fsdp/recovery'
        if rank == 0:
            recovery._store = recovery.RecoveryStore(client, os.environ['MODELCHEF_RECOVERY_S3_URI'])
        publish_sharded_to_s3(workspace, run, checkpoint)
        if rank == 0:
            recovery._store.fetch(root / 'restored', 100_000_000)
            pointer = client.objects['fsdp/recovery/latest.json']
        dist.barrier()
        update()
        expected = full_weights()
        manifest, restored = load_sharded(model, optimizer, checkpoint, identities)
        assert manifest['step'] == 1
        assert torch.equal(restored['sampler_rng'], state['sampler_rng'])
        update()
        actual = full_weights()
        assert all(torch.equal(actual[k], v) for k, v in expected.items()), 'AdamW continuation differs'
        remote_copy = root / 'restored' / checkpoint.relative_to(workspace)
        _, remote_state = load_sharded(model, optimizer, remote_copy, identities)
        assert torch.equal(remote_state['sampler_rng'], state['sampler_rng'])
        assert torch.equal(remote_state['torch_rng'], state['torch_rng'])
        update()
        from_s3 = full_weights()
        assert all(torch.equal(from_s3[k], v) for k, v in expected.items()), 'S3 AdamW recovery differs'
        if rank == 0:
            (run / 'metrics.jsonl').write_text('new metrics force an upload')
            client.fail = True
        dist.barrier()
        try:
            publish_sharded_to_s3(workspace, run, checkpoint)
            raise AssertionError('S3 failure was not propagated to every rank')
        except RuntimeError as error:
            assert 'interrupted transfer' in str(error)
        if rank == 0:
            assert client.objects['fsdp/recovery/latest.json'] == pointer
        try:
            load_sharded(model, optimizer, checkpoint, {'synthetic': 'changed'})
            raise AssertionError('Changed data identity accepted')
        except RuntimeError as error:
            assert 'identities changed' in str(error)
        dist.barrier()
        if rank == 0:
            (root / 'result.json').write_text(json.dumps({'averaged_gradient_update': True,
                'tied_embeddings': True, 'adamw_resume_exact': True, 'rng_resume': True,
                'changed_identity_rejected': True, 's3_adamw_resume_exact': True,
                'upload_failure_propagated_to_all_ranks': True, 'step_zero_optimizer_unchanged': True}))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
