"""Fixed post-preprocessing corruptions with paired per-record random draws."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import ECGDataset
from .metrics import compute_metrics
from .models import build_model
from .train import predict
from .utils import seed_all

CONDITIONS = [('clean',0),('gaussian',20),('gaussian',10),('gaussian',5),
              ('wander',.2),('wander',.5),('dropout',1),('dropout',3),('dropout',6)]


def corrupt(x, kind, level, ecg_id):
    rng = np.random.default_rng(np.random.SeedSequence([2026,int(ecg_id)]))
    x = x.copy()
    if kind == 'gaussian':
        noise = rng.standard_normal(x.shape)
        noise *= np.sqrt(np.mean(x*x) / (np.mean(noise*noise) * 10**(level/10)))
        x += noise.astype(np.float32)
    elif kind == 'dropout':
        x[rng.permutation(12)[:int(level)]] = 0
    elif kind == 'wander':
        phase = rng.uniform(0,2*np.pi,size=(12,1))
        x += (0.5*np.sin(2*np.pi*level*np.arange(5000)[None,:]/500 + phase)).astype(np.float32)
    elif kind != 'clean':
        raise ValueError(kind)
    return x


class CorruptedDataset(Dataset):
    def __init__(self, frame, cache, kind, level):
        self.base = ECGDataset(frame,cache)
        self.kind, self.level = kind,level

    def __len__(self):
        return len(self.base)

    def __getitem__(self,i):
        x,y = self.base[i]
        ecg_id = self.base.frame.iloc[i].ecg_id
        return torch.from_numpy(corrupt(x.numpy(),self.kind,self.level,ecg_id)), y


def main():
    if not Path('experiments/test_lock.json').exists():
        raise RuntimeError('Freeze the main experiments before robustness evaluation')
    seed_all(42)
    local = json.loads(Path('configs/local.json').read_text())
    frame = pd.read_csv('data/manifests/test.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for name in ['scratch','pretrained']:
        run = Path('runs') / f'{name}_f0.1_s42'
        thresholds = json.loads((run / 'thresholds.json').read_text())
        model = build_model('scratch').to(device)
        model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
        clean_auc = None
        for kind,level in CONDITIONS:
            output = run / f'robustness_{kind}_{level}.npz'
            if kind == 'clean':
                data = np.load(run/'test_predictions.npz')
                y,prob = data['y'],data['p']
            elif output.exists():
                data = np.load(output)
                y,prob = data['y'],data['p']
            else:
                loader = DataLoader(CorruptedDataset(frame,local['cache'],kind,level),batch_size=16)
                y,prob = predict(model,loader,device)
                np.savez_compressed(output,y=y,p=prob,ecg_id=frame.ecg_id.to_numpy())
            metrics,_ = compute_metrics(y,prob,thresholds)
            if kind == 'clean':
                clean_auc = metrics['macro_auroc']
            bounded = np.clip(prob,1e-7,1-1e-7)
            entropy = -(bounded*np.log(bounded)+(1-bounded)*np.log(1-bounded))
            rows.append(dict(model=name,fraction=.1,seed=42,corruption=kind,level=level,
                             auroc_drop=clean_auc-metrics['macro_auroc'],
                             mean_binary_entropy=float(entropy.mean()),**metrics))
            print(name,kind,level,metrics['macro_auroc'],flush=True)
    pd.DataFrame(rows).to_csv('results/robustness.csv',index=False)


if __name__ == '__main__':
    main()
