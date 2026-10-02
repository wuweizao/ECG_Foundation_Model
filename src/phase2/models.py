"""Phase-two adapters; never modify phase-one architecture sources."""
import hashlib
import types
from pathlib import Path
import torch
from src.models import build_model
from src.utils import sha256

BASELINE_SOURCE = Path('vendor/ptbxl_benchmark/code/models/xresnet1d.py')


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        value = value.detach().cpu().contiguous()
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def build(spec, local):
    family = spec['family']
    if family == 'xresnet1d101':
        expected = 'f1c789bfd0a50789ea9e9da6388fb049801ac58048bfa2955274ced803710928'
        if sha256(BASELINE_SOURCE) != expected:
            raise ValueError('Changed pinned xResNet source')
        module = types.ModuleType('registered_xresnet1d')
        source = BASELINE_SOURCE.read_text().replace(
            'from models.basic_conv1d import create_head1d, Flatten',
            'from src.phase2.baseline_head import create_head1d, Flatten')
        exec(compile(source, str(BASELINE_SOURCE), 'exec'), module.__dict__)
        return module.xresnet1d101(input_channels=12, num_classes=5)
    initialization = 'pretrained' if family in ['pretrained','linear_probe'] else 'scratch'
    model = build_model(initialization, local['checkpoint'])
    if family in ['linear_probe','random_probe']:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith('dense.')
    return model
