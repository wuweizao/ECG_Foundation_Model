"""Per-label reliability; p=.8 means 80% positive event frequency, not accuracy."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .labels import CLASSES
from .report import COLORS, NAMES, save, style


def main():
    style()
    rows = []
    fig,axes = plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for model, marker, ls in [('scratch','s','--'),('pretrained','o','-')]:
        data = np.load(Path('runs') / f'{model}_f0.1_s42/test_predictions.npz')
        y,p = data['y'], data['p']
        for j,(label,ax) in enumerate(zip(CLASSES,axes.flat)):
            bins = np.minimum((p[:,j]*10).astype(int),9)
            xs,ys = [],[]
            for b in range(10):
                mask = bins==b
                if mask.any():
                    x, observed = float(p[mask,j].mean()),float(y[mask,j].mean())
                    xs.append(x)
                    ys.append(observed)
                    rows.append(dict(model=model,label=label,bin=b,n=int(mask.sum()),
                                     mean_probability=x,event_frequency=observed))
            ax.plot(xs,ys,marker=marker,ls=ls,label=NAMES[model],color=COLORS[model])
            ax.set_title(label,loc='left')
    for ax in list(axes.flat)[:5]:
        ax.plot([0,1],[0,1],':',color='#555555',lw=1)
        ax.set(xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.15)
    axes.flat[5].axis('off')
    axes.flat[4].legend(frameon=False,loc='lower right')
    fig.supxlabel('Mean predicted probability (10 equal-width bins)')
    fig.supylabel('Observed positive event frequency')
    fig.suptitle('Per-class calibration | 10% labels, seed 42, PTB-XL fold 10',x=.08,ha='left')
    fig.tight_layout()
    save(fig,'calibration')
    pd.DataFrame(rows).to_csv('results/reliability_bins.csv',index=False)


if __name__ == '__main__':
    main()
