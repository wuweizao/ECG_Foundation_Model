"""Per-encoder frozen caches, supporting both random and pretrained controls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, TensorDataset
from src.dataset import ECGDataset
from src.labels import CLASSES
from src.utils import sha256
from .models import tensor_hash


def frozen_dataset(model, frame, split, local, spec, device):
    if split not in ['train','validation']:
        raise ValueError('Training feature cache cannot access test')
    manifest = Path(f'runs/phase2/manifests/{split}.csv')
    all_frame = pd.read_csv(manifest)
    encoder_hash = tensor_hash({n:v for n,v in model.state_dict().items() if not n.startswith('dense.')})
    root = Path('runs/phase2/features')/encoder_hash
    root.mkdir(parents=True,exist_ok=True)
    output = root/f'{split}.npz'
    identity = dict(encoder=encoder_hash,manifest=sha256(manifest),
                    preprocessing=sha256(Path(local['cache'])/'identity.json'),
                    amp=spec['amp'],batch_size=spec['batch_size'],torch=torch.__version__)
    with FileLock(str(output)+'.lock'):
        if output.exists():
            with np.load(output) as data:
                if json.loads(str(data['identity'])) != identity:
                    raise ValueError('Frozen cache identity mismatch')
                features, ids = data['features'].copy(), data['ecg_id'].copy()
        else:
            model.eval()
            model.return_features = True
            loader = DataLoader(ECGDataset(all_frame,local['cache']),batch_size=spec['batch_size'])
            chunks=[]
            try:
                with torch.inference_mode():
                    for x,_ in loader:
                        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                            enabled=spec['amp'] and device.type=='cuda'):
                            _, z = model(x.to(device))
                        chunks.append(z.float().cpu().numpy())
            finally:
                model.return_features=False
            features,ids=np.concatenate(chunks),all_frame.ecg_id.to_numpy()
            tmp=output.with_suffix('.tmp')
            with open(tmp,'wb') as f:
                np.savez(f,features=features,ecg_id=ids,identity=json.dumps(identity,sort_keys=True))
            tmp.replace(output)
    positions=pd.Index(ids).get_indexer(frame.ecg_id)
    if (positions<0).any():
        raise ValueError('Requested frozen record absent')
    return TensorDataset(torch.from_numpy(features[positions].copy()),
                         torch.from_numpy(frame[CLASSES].to_numpy(dtype=np.float32)))
