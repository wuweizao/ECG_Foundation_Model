"""Qualitative frozen-initialization representations on validation, not test."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from .dataset import ECGDataset
from .labels import CLASSES
from .models import build_model
from .report import save, style
from .utils import seed_all, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--umap', action='store_true')
    parser.add_argument('--device', choices=['cpu','cuda'], default=None)
    args = parser.parse_args()
    local = json.loads(Path('configs/local.json').read_text())
    frame = pd.read_csv('data/manifests/validation.csv')
    output = Path('runs/representations')
    output.mkdir(parents=True,exist_ok=True)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    maps = {}
    info = {}
    for name in ['scratch','pretrained']:
        seed_all(42)
        target = output / f'{name}.npz'
        if not target.exists():
            model = build_model(name,local['checkpoint']).to(device).eval()
            model.return_features = True
            loader = DataLoader(ECGDataset(frame,local['cache']),batch_size=16)
            features = []
            with torch.inference_mode():
                for x,_ in loader:
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        _, f = model(x.to(device))
                    features.append(f.float().cpu().numpy())
            np.savez_compressed(target,features=np.concatenate(features),ecg_id=frame.ecg_id.to_numpy())
            del model
        data = np.load(target)
        if not np.array_equal(data['ecg_id'],frame.ecg_id.to_numpy()):
            raise ValueError('Representation order mismatch')
        if args.umap:
            from umap import UMAP
            reducer = UMAP(n_components=2,n_neighbors=30,min_dist=.1,metric='cosine',random_state=42,n_jobs=1)
            maps[name] = reducer.fit_transform(data['features'])
            info[name] = {'n_neighbors':30,'min_dist':.1,'metric':'cosine','seed':42}
        else:
            pca = PCA(n_components=2,random_state=42)
            maps[name] = pca.fit_transform(data['features'])
            info[name] = {'explained_variance_ratio':pca.explained_variance_ratio_.tolist()}
    style()
    fig,axes = plt.subplots(2,5,figsize=(15,6))
    for i,name in enumerate(['scratch','pretrained']):
        for j,label in enumerate(CLASSES):
            ax = axes[i,j]
            xy = maps[name]
            positive = frame[label].to_numpy().astype(bool)
            ax.scatter(xy[~positive,0],xy[~positive,1],s=3,c='#C9C9C9',alpha=.3,rasterized=True)
            ax.scatter(xy[positive,0],xy[positive,1],s=4,c='#3176B6',alpha=.4,rasterized=True)
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                ax.set_ylabel('Random initialization' if name=='scratch' else 'Pretrained initialization')
    method = 'UMAP' if args.umap else 'PCA'
    fig.suptitle(f'{method} of frozen encoder representations | PTB-XL validation fold 9',x=.08,ha='left')
    fig.text(.08,.015,f'Same {len(frame):,} ECGs; positive label shown in blue independently in each panel. Separate {method} fits; axes are not directly comparable. Qualitative only.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    save(fig,method.lower())
    save_json(f'results/{method.lower()}_metadata.json',info)


if __name__ == '__main__':
    main()
