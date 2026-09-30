"""One deterministic validation example; input saliency is not a causal explanation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .dataset import LEADS
from .labels import CLASSES
from .models import build_model
from .report import style,save
from .utils import seed_all,save_json


def main():
    seed_all(42)
    torch.set_num_threads(1)
    local = json.loads(Path('configs/local.json').read_text())
    frame = pd.read_csv('data/manifests/validation.csv')
    row = frame[frame.MI==1].sort_values('ecg_id').iloc[0]
    signal = np.load(Path(local['cache'])/f'{int(row.ecg_id)}.npy')
    style()
    fig,axes = plt.subplots(3,1,figsize=(11,7),sharex=True,
                           gridspec_kw={'height_ratios':[1,2,2]})
    time = np.arange(5000)/500
    axes[0].plot(time,signal[1],color='#333333',lw=.7)
    axes[0].set_ylabel('Lead II\nnormalized')
    metadata = dict(split='validation',ecg_id=int(row.ecg_id),target='MI',fraction=.1,seed=42,
                    selection='first MI-positive validation ECG by ecg_id, independent of predictions',
                    method='absolute input times gradient of MI logit',models={})
    for ax,name in zip(axes[1:],['scratch','pretrained']):
        model = build_model('scratch').eval()
        model.load_state_dict(torch.load(f'runs/{name}_f0.1_s42/best.pt',map_location='cpu',weights_only=True))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x = torch.from_numpy(signal.copy()).unsqueeze(0).requires_grad_(True)
        logit = model(x)[0,CLASSES.index('MI')]
        grad = torch.autograd.grad(logit,x)[0]
        importance = (grad*x).abs().detach().numpy()[0]
        scale = float(np.quantile(importance,.99))
        normalized = np.clip(importance/(scale+1e-12),0,1)
        threshold = json.loads(Path(f'runs/{name}_f0.1_s42/thresholds.json').read_text())[CLASSES.index('MI')]
        probability = float(torch.sigmoid(logit.detach()))
        ax.imshow(normalized,aspect='auto',extent=[0,10,11.5,-.5],cmap='Blues',vmin=0,vmax=1,
                  interpolation='nearest',rasterized=True)
        ax.set_yticks(np.arange(12),LEADS,fontsize=8)
        ax.set_ylabel(name.capitalize())
        ax.set_title(f'MI probability {probability:.3f} | validation threshold {threshold:.2f}',loc='left',fontsize=9,pad=5)
        metadata['models'][name] = dict(logit=float(logit.detach()),probability=probability,
                                        threshold=threshold,predicted_positive=probability>=threshold,
                                        normalization_99th_percentile=scale)
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'MI input saliency | validation ECG {int(row.ecg_id)} | 10% labels, seed 42',x=.08,ha='left')
    fig.text(.08,.01,'Absolute gradient × input; darker blue = larger relative sensitivity. Each model scaled to its own 99th percentile. Not causal evidence.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    save(fig,'saliency')
    save_json('results/saliency_metadata.json',metadata)


if __name__ == '__main__':
    main()
