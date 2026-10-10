"""Bounded local media manifests, grouped held-out splits and PCM WAV decoding."""
import hashlib
import json
import random
import wave
from pathlib import Path

from modelchef.corpus import file_digest


def inspect_media(path, config):
    if not path.is_file() or path.stat().st_size > config.data.max_media_bytes:
        raise ValueError(f'Missing media or max_media_bytes exceeded: {path.name}')
    if config.modality == 'vision':
        from PIL import Image
        with Image.open(path) as image:
            if image.width * image.height > config.data.max_image_pixels:
                raise ValueError('Image exceeds max_image_pixels')
            image.verify()
    else:
        try:
            with wave.open(str(path), 'rb') as audio:
                if audio.getcomptype() != 'NONE' or audio.getsampwidth() != 2 or audio.getnchannels() != 1:
                    raise ValueError('Speech inputs require uncompressed mono PCM16 WAV')
                if audio.getframerate() != 16000:
                    raise ValueError('Speech inputs require 16000 Hz; resample before training')
                duration = audio.getnframes() / audio.getframerate()
                if duration <= 0 or duration > config.data.max_audio_seconds:
                    raise ValueError('Audio duration exceeds recipe bounds')
                if len(audio.readframes(audio.getnframes())) != audio.getnframes() * 2:
                    raise ValueError('Audio WAV payload is truncated')
        except (wave.Error, EOFError) as error:
            raise ValueError('Invalid or truncated PCM WAV input') from error
    return file_digest(path)


def read_manifest(config, root):
    path = (Path(root) / config.data.manifest).resolve()
    if path.stat().st_size > config.data.max_manifest_bytes:
        raise ValueError('Manifest exceeds max_manifest_bytes')
    records, seen, total = [], set(), 0
    field = 'image' if config.modality == 'vision' else 'audio'
    with path.open('rb') as source:
        while line := source.readline(config.data.max_record_bytes + 1):
            if len(line) > config.data.max_record_bytes:
                raise ValueError('Manifest record exceeds max_record_bytes')
            if not line.strip():
                continue
            row = json.loads(line)
            required = {field, 'prompt', 'response'} if field == 'image' else {field, 'text'}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f'Media record requires {sorted(required)}')
            if any(not isinstance(row[key], str) or not row[key].strip() for key in required):
                raise ValueError('Media paths and text must be nonempty strings')
            media = (path.parent / row[field]).resolve()
            checksum = inspect_media(media, config)
            if checksum not in seen:
                total += media.stat().st_size
                seen.add(checksum)
            if total > config.data.max_total_media_bytes:
                raise ValueError('Selected media exceeds max_total_media_bytes')
            records.append({**{key: row[key] for key in required}, field: str(media),
                            'media_sha256': checksum})
            if len(records) >= config.data.max_records:
                break
    if len(seen) < 3:
        raise ValueError('At least three distinct media assets are required for disjoint splits')
    identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return records, {'manifest_sha256': file_digest(path), 'selected_sha256': identity,
                     'records': len(records), 'unique_assets': len(seen), 'media_bytes': total}


def split_records(records, config):
    groups = sorted({row['media_sha256'] for row in records})
    random.Random(config.training.seed).shuffle(groups)
    validation = min(len(groups) - 2, max(1, int(len(groups) * config.data.validation_fraction)))
    test = min(len(groups) - validation - 1, max(1, int(len(groups) * config.data.test_fraction)))
    assignments = {key: name for name, keys in [
        ('validation', groups[:validation]), ('test', groups[validation:validation + test]),
        ('train', groups[validation + test:])] for key in keys}
    return {name: [row for row in records if assignments[row['media_sha256']] == name]
            for name in ['train', 'validation', 'test']}


def read_audio(path):
    import numpy as np
    with wave.open(str(path), 'rb') as source:
        if source.getframerate() != 16000 or source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError('Speech inputs require mono PCM16 WAV at 16000 Hz')
        return np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').astype('float32') / 32768
