"""Read physical safetensors payload sizes without loading model tensors."""

import json
from pathlib import Path
import struct


def tensor_category(name: str) -> str:
    """Explicit suffix rules; unknown tensors remain separate rather than guessed."""
    suffix = name.rsplit('.', 1)[-1].lower()
    if suffix in {'scales', 'scale', 'qzeros', 'zeros', 'zero_points', 'g_idx',
                  'weight_scale', 'weight_scale_inv', 'weight_zero_point',
                  'input_scale', 'input_zero_point', 'output_scale', 'output_zero_point'}:
        return 'quantization_metadata'
    if suffix in {'weight', 'weights', 'bias', 'qweight', 'q_weight'}:
        return 'weights'
    return 'other_tensors'


def inspect_checkpoint(model_path: Path):
    shards = sorted(model_path.glob('*.safetensors'))
    index = model_path / 'model.safetensors.index.json'
    if index.is_file():
        names = set(json.loads(index.read_text())['weight_map'].values())
        if any(Path(name).name != name for name in names):
            raise ValueError('Checkpoint shard names must be local filenames.')
        shards = [model_path / name for name in sorted(names)]
    if not shards:
        return None
    tensors = []
    names_seen = set()
    file_bytes = 0
    for shard in shards:
        file_size = shard.stat().st_size
        file_bytes += file_size
        with shard.open('rb') as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f'Invalid safetensors header: {shard.name}')
            size = struct.unpack('<Q', raw)[0]
            if not 0 < size <= min(file_size - 8, 100_000_000):
                raise ValueError(f'Invalid safetensors header size: {shard.name}')
            header = json.loads(handle.read(size))
        intervals = []
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            begin, end = tensor['data_offsets']
            if not (type(begin) is int and type(end) is int and 0 <= begin <= end <= file_size - 8 - size):
                raise ValueError(f'Invalid tensor offsets: {shard.name}:{name}')
            if name in names_seen:
                raise ValueError(f'Duplicate checkpoint tensor: {name}')
            names_seen.add(name)
            intervals.append((begin, end))
            tensors.append({'name': name, 'shard': shard.name, 'dtype': tensor['dtype'],
                            'shape': tensor['shape'], 'bytes': end - begin,
                            'category': tensor_category(name)})
        previous_end = 0
        for begin, end in sorted(intervals):
            if begin < previous_end:
                raise ValueError(f'Overlapping tensor offsets: {shard.name}')
            previous_end = end
    total = sum(tensor['bytes'] for tensor in tensors)
    if total <= 0:
        raise ValueError('Checkpoint contains no tensor data.')
    return {'tensor_bytes': total, 'file_bytes': file_bytes,
            'serialization_overhead_bytes': file_bytes - total, 'tensors': tensors}
