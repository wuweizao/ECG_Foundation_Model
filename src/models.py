import importlib.util
from pathlib import Path

import torch
import numpy as np


def build_model(initialization, checkpoint=None, linear_probe=False):
    source = Path(__file__).resolve().parents[1] / 'vendor/ECGFounder/net1d.py'
    if not source.exists():
        raise FileNotFoundError('Run scripts/setup.ps1 to fetch pinned official ECGFounder')
    spec = importlib.util.spec_from_file_location('ecgfounder_net1d', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Net1D(in_channels=12, base_filters=64, ratio=1,
                        filter_list=[64,160,160,400,400,1024,1024],
                        m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16,
                        stride=2, groups_width=16, verbose=False,
                        use_bn=False, use_do=False, n_classes=5)
    if initialization == 'pretrained':
        # Official checkpoint also stores a NumPy scalar validation score.
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype('float64'))]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        state = {k: v for k, v in state.items() if not k.startswith('dense.')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {'dense.weight', 'dense.bias'} or unexpected:
            raise ValueError(f'Incompatible official checkpoint: {missing}, {unexpected}')
    elif initialization != 'scratch':
        raise ValueError(initialization)
    if linear_probe:
        if initialization != 'pretrained':
            raise ValueError('Linear probe requires pretrained initialization')
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith('dense.')
    return model
