import json
import math
import struct
import wave
from pathlib import Path

import pytest

from modelchef.multimodal_config import MultimodalConfig
from modelchef.multimodal_data import read_manifest, split_records


def media_fixture(root, modality):
    rows = []
    for index in range(6):
        if modality == 'vision':
            Image = pytest.importorskip('PIL.Image')
            path = root / f'image-{index}.png'
            Image.new('RGB', (16, 16), (index * 30, 10, 80)).save(path)
            rows.append({'image': path.name, 'prompt': 'Describe the color.', 'response': f'Color {index}.'})
        else:
            path = root / f'audio-{index}.wav'
            with wave.open(str(path), 'wb') as target:
                target.setparams((1, 2, 16000, 1600, 'NONE', 'not compressed'))
                target.writeframes(b''.join(struct.pack('<h', int(200 * math.sin(i * (index + 1) / 20)))
                                           for i in range(1600)))
            rows.append({'audio': path.name, 'text': f'hello {index}'})
    manifest = root / 'media.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return manifest, rows


@pytest.mark.parametrize('modality', ['vision', 'speech'])
def test_media_bounds_grouped_splits_and_changed_content(tmp_path, modality):
    path, rows = media_fixture(tmp_path, modality)
    # Repeated assets must not leak across held-out splits.
    with path.open('a') as target:
        target.write(json.dumps(rows[0]) + '\n')
    config = MultimodalConfig(modality=modality, data={'manifest': str(path)}, model={'path': 'model'})
    records, manifest = read_manifest(config, tmp_path)
    assert manifest['unique_assets'] == 6 and manifest['records'] == 7
    splits = split_records(records, config)
    assets = [{row['media_sha256'] for row in values} for values in splits.values()]
    assert all(assets) and sum(map(len, assets)) == len(set.union(*assets))
    assert splits == split_records(records, config)
    config.data.max_media_bytes = 1
    with pytest.raises(ValueError, match='max_media_bytes'):
        read_manifest(config, tmp_path)


def test_invalid_audio_and_recipe_settings_fail_before_model_loading(tmp_path):
    path, _ = media_fixture(tmp_path, 'speech')
    with wave.open(str(tmp_path / 'audio-0.wav'), 'wb') as target:
        target.setparams((1, 2, 8000, 800, 'NONE', 'not compressed'))
        target.writeframes(b'\0' * 1600)
    config = MultimodalConfig(modality='speech', data={'manifest': str(path)}, model={'path': 'model'})
    with pytest.raises(ValueError, match='16000 Hz'):
        read_manifest(config, tmp_path)
    (tmp_path / 'audio-0.wav').write_bytes(b'not a WAV file')
    with pytest.raises(ValueError, match='Invalid or truncated'):
        read_manifest(config, tmp_path)
    with pytest.raises(ValueError, match='single process'):
        MultimodalConfig(modality='speech', data={'manifest': str(path)}, model={'path': 'model'},
                         training={'distributed': 'ddp', 'warmup_steps': 0})


def test_word_error_rate_distance():
    from modelchef.multimodal import edit_distance
    assert edit_distance(['a', 'b'], ['a', 'c', 'd']) == 2
    assert edit_distance([], ['a']) == 1
    assert edit_distance(['same'], ['same']) == 0


def test_chat_target_mask_and_context_bound(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('PIL')
    from modelchef.multimodal import encode
    manifest, _ = media_fixture(tmp_path, 'vision')
    config = MultimodalConfig(modality='vision', data={'manifest': str(manifest)}, model={'path': 'model'})
    row = read_manifest(config, tmp_path)[0][0]
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            prefix = 'user-image-prompt-assistant:'
            return prefix + (messages[-1]['content'][0]['text'] + '!' if len(messages) > 1 else '')
        def __call__(self, text, **kwargs):
            ids = torch.tensor([[ord(letter) for letter in text]])
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    batch = encode(row, Processor(), config)
    boundary = len('user-image-prompt-assistant:')
    assert (batch['labels'][0, :boundary] == -100).all()
    assert (batch['labels'][0, boundary:] != -100).all()
    config.max_text_tokens = 8
    with pytest.raises(ValueError, match='max_text_tokens'):
        encode(row, Processor(), config)


def tiny_model(root, modality):
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('torch')
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    folder = root / 'model'
    folder.mkdir()
    backend = Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    specials = (['<unk>', '<pad>', '<bos>', '<eos>', '<image>'] if modality == 'vision' else
                ['<unk>', '<|endoftext|>', '<|startoftranscript|>', '<|transcribe|>',
                 '<|translate|>', '<|notimestamps|>', '<|en|>'])
    backend.train_from_iterator(['hello 0 1 2 3 4 5 Describe the color Color.'] * 4,
        trainers.BpeTrainer(vocab_size=300, special_tokens=specials,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    backend.save(str(folder / 'tokenizer.json'))
    if modality == 'vision':
        tokenizer = transformers.GemmaTokenizerFast(tokenizer_file=str(folder / 'tokenizer.json'),
            unk_token='<unk>', pad_token='<pad>', bos_token='<bos>', eos_token='<eos>',
            additional_special_tokens=['<image>'], extra_special_tokens={'image_token': '<image>'})
        tokenizer.image_token = '<image>'
        image_processor = transformers.SiglipImageProcessor(size={'height': 16, 'width': 16})
        image_processor.image_seq_length = 4
        processor = transformers.PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        config = transformers.PaliGemmaConfig(vision_config={
            'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
            'num_attention_heads': 2, 'image_size': 16, 'patch_size': 8},
            text_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                         'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 8,
                         'vocab_size': len(tokenizer), 'pad_token_id': tokenizer.pad_token_id},
            image_token_index=tokenizer.convert_tokens_to_ids('<image>'), projection_dim=32,
            hidden_size=32, vocab_size=len(tokenizer))
        model = transformers.PaliGemmaForConditionalGeneration(config)
    else:
        tokenizer = transformers.WhisperTokenizerFast(tokenizer_file=str(folder / 'tokenizer.json'),
            unk_token='<unk>', bos_token='<|endoftext|>', eos_token='<|endoftext|>',
            pad_token='<|endoftext|>')
        processor = transformers.WhisperProcessor(
            feature_extractor=transformers.WhisperFeatureExtractor(chunk_length=1), tokenizer=tokenizer)
        config = transformers.WhisperConfig(vocab_size=len(tokenizer), d_model=16,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=50, max_target_positions=64,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.convert_tokens_to_ids('<|startoftranscript|>'),
            suppress_tokens=[], begin_suppress_tokens=[])
        model = transformers.WhisperForConditionalGeneration(config)
        model.generation_config.no_timestamps_token_id = tokenizer.convert_tokens_to_ids('<|notimestamps|>')
        model.generation_config.is_multilingual = False
        model.generation_config.task_to_id = {key: tokenizer.convert_tokens_to_ids(f'<|{key}|>')
                                            for key in ['transcribe', 'translate']}
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    return folder


@pytest.mark.parametrize('modality', ['vision', 'speech'])
@pytest.mark.parametrize('accelerator', ['cpu', 'cuda'])
def test_real_multimodal_training_resume_export_and_inference(tmp_path, modality, accelerator):
    torch = pytest.importorskip('torch')
    if accelerator == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip('BF16 CUDA hardware required')
    torch.set_num_threads(1)
    from modelchef.multimodal import encode, load_components, plan, predict, train
    model = tiny_model(tmp_path, modality)
    manifest, rows = media_fixture(tmp_path, modality)
    config = MultimodalConfig(modality=modality,
        data={'manifest': str(manifest)},
        model={'path': str(model), 'vision_format': 'suffix'},
        max_text_tokens=128,
        training={'accelerator': accelerator, 'precision': 'fp32' if accelerator == 'cpu' else 'bf16',
                  'steps': 2, 'warmup_steps': 0,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 1,
                  'log_every': 1, 'max_runtime_seconds': 120},
        evaluation={'max_examples': 1, 'max_new_tokens': 2})
    assert plan(config, tmp_path)['downloads'] is False
    if modality == 'vision':
        network, processor = load_components(config, tmp_path)
        encoder = next(name for name, _ in network.named_modules() if name.endswith('vision_tower'))
        config.model.freeze_encoders = True
        config.model.encoder_modules = [encoder]
        frozen = {key: value.clone() for key, value in network.state_dict().items()
                  if key.startswith(encoder + '.')}
        record = read_manifest(config, tmp_path)[0][0]
        batch = encode(record, processor, config)
        assert (batch['labels'] == -100).any() and (batch['labels'] != -100).any()
    result = train(config, tmp_path)
    assert result['status'] == 'completed' and math.isfinite(result['test']['loss'])
    final = torch.load(result['checkpoint'], weights_only=True)['model']
    if modality == 'vision':
        assert all(torch.equal(final[key].cpu(), value.cpu()) for key, value in frozen.items())
    middle = Path(result['run_dir']) / 'checkpoints/step-000001.pt'
    resumed = train(config, tmp_path, middle)
    restored = torch.load(resumed['checkpoint'], weights_only=True)['model']
    assert all(torch.equal(final[key], restored[key]) for key in final)
    media = tmp_path / rows[0]['image' if modality == 'vision' else 'audio']
    assert isinstance(predict(config, tmp_path, media, 'Describe the color.', result['export']), str)
    from modelchef.publishing import PublishConfig, publish
    published = publish(PublishConfig(export_dir=result['export'], repo_id='example/tiny',
        license='apache-2.0', summary='Random fixture', training_data='Synthetic local media',
        evaluation='Two-step protocol test', limitations='No model quality claim',
        pipeline_tag='image-text-to-text' if modality == 'vision' else 'automatic-speech-recognition'), tmp_path)
    assert not published['uploaded']
    assert {'preprocessor_config.json', 'processor_config.json'} & {item['name'] for item in published['files']}
    import io
    import tarfile

    import yaml

    from modelchef.aws_guest import guest_plan
    from modelchef.aws_training import build_bundle
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump(config.model_dump(mode='json')))
    bundle = build_bundle(tmp_path, {'pipeline_config': str(recipe), 'max_bundle_bytes': 10000000})
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        packed = MultimodalConfig.model_validate(yaml.safe_load(archive.extractfile('configs/job.yaml').read()))
        assert packed.model.path == 'inputs/model'
        assert packed.data.manifest == 'inputs/media.jsonl'
        assert 'inputs/model/model.safetensors' in archive.getnames()
        assert any(name.startswith('inputs/media/') for name in archive.getnames())
    assert guest_plan(packed)['extras'] == ['train', 'multimodal']
    assert 'multimodal' in guest_plan(packed)['command']
    with pytest.raises(ValueError, match='max_bundle_bytes'):
        build_bundle(tmp_path, {'pipeline_config': str(recipe), 'max_bundle_bytes': 1})
    config.data.max_media_bytes = 1
    with pytest.raises(ValueError, match='max_media_bytes'):
        train(config, tmp_path, middle)
