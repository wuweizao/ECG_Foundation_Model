"""Display one training ECG and the exact model input without accessing test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from .dataset import LEADS
from .labels import CLASSES
from .report import save,style


def main():
    local = json.loads(Path('configs/local.json').read_text())
    frame = pd.read_csv('data/manifests/train.csv')
    row = frame.iloc[0]
    raw,metadata = wfdb.rdsamp(str(Path(local['data_root'])/row.filename_hr))
    x = np.load(Path(local['cache'])/f'{int(row.ecg_id)}.npy')
    names = [s.upper() for s in metadata['sig_name']]
    style()
    fig,axes = plt.subplots(6,2,figsize=(12,10),sharex=True)
    t = np.arange(5000)/500
    for j,ax in enumerate(axes.flat):
        ax.plot(t,raw[:,names.index(LEADS[j])],lw=.6,c='#3176B6')
        ax.set_title(LEADS[j],loc='left',fontsize=10)
        ax.grid(alpha=.15)
    labels = ', '.join(k for k in CLASSES if row[k])
    fig.suptitle(f'PTB-XL training ECG {int(row.ecg_id)} | {labels}',x=.08,ha='left')
    fig.supxlabel('Time (seconds)')
    fig.supylabel('Raw amplitude (mV)')
    fig.tight_layout(rect=(.03,.03,1,.96))
    save(fig,'ecg_example')
    fig,axes = plt.subplots(2,1,figsize=(11,5),sharex=True)
    axes[0].plot(t,raw[:,names.index('II')],lw=.7,c='#3176B6')
    axes[0].set_ylabel('Raw (mV)')
    axes[1].plot(t,x[1],lw=.7,c='#D08728')
    axes[1].set(ylabel='Normalized input',xlabel='Time (seconds)')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('Lead II preprocessing | training ECG example',x=.125,ha='left')
    fig.tight_layout()
    save(fig,'preprocessing')


if __name__ == '__main__':
    main()
