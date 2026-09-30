"""Cache frozen per-record features; no fitting and no cross-record statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dataset import ECGDataset
from .labels import CLASSES
from .utils import sha256


def frozen_dataset(model, frame, split, local, config, device):
    if split not in ['train','validation']:
        raise ValueError('Frozen training features cannot use test')
    all_frame = pd.read_csv(f'data/manifests/{split}.csv')
    root = Path('runs/frozen_features')
    root.mkdir(parents=True,exist_ok=True)
    output = root / f'{split}.npz'
    identity = dict(checkpoint=sha256(local['checkpoint']),manifest=sha256(f'data/manifests/{split}.csv'),
                    cache_identity=sha256(Path(local['cache'])/'identity.json'),
                    source=sha256('vendor/ECGFounder/net1d.py'),amp=config['amp'],
                    batch_size=config['batch_size'],torch=torch.__version__,device=device.type)
    if output.exists():
        data = np.load(output)
        if json.loads(str(data['identity'])) != identity:
            raise ValueError('Frozen feature cache identity mismatch')
        features,ids = data['features'],data['ecg_id']
    else:
        model.eval()
        model.return_features = True
        loader = DataLoader(ECGDataset(all_frame,local['cache']),batch_size=config['batch_size'])
        chunks = []
        with torch.inference_mode():
            for x,_ in loader:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                    enabled=config['amp'] and device.type=='cuda'):
                    _,features = model(x.to(device))
                chunks.append(features.float().cpu().numpy())
        model.return_features = False
        features,ids = np.concatenate(chunks),all_frame.ecg_id.to_numpy()
        tmp = output.with_suffix('.tmp')
        with open(tmp,'wb') as f:
            np.savez(f,features=features,ecg_id=ids,identity=json.dumps(identity,sort_keys=True))
        tmp.replace(output)
    positions = pd.Index(ids).get_indexer(frame.ecg_id)
    if (positions < 0).any():
        raise ValueError('Missing frozen feature record')
    return TensorDataset(torch.from_numpy(features[positions].copy()),
                         torch.from_numpy(frame[CLASSES].to_numpy(dtype=np.float32)))
