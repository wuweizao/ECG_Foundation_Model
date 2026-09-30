"""Reproducible static research figures from actual held-out results only."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .labels import CLASSES
from .plan import conditions
from .utils import save_json

COLORS = {'scratch': '#D08728', 'pretrained': '#3176B6', 'linear_probe': '#8B8740'}
NAMES = {'scratch': 'Scratch', 'pretrained': 'Pretrained', 'linear_probe': 'Linear probe'}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#242424', 'text.color': '#242424',
                         'figure.dpi': 130, 'savefig.dpi': 220})


def save(fig, name):
    Path('figures').mkdir(exist_ok=True)
    fig.savefig(f'figures/{name}.png', bbox_inches='tight')
    fig.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    style()
    df = pd.read_csv('results/metrics.csv')
    expected = {(r['model'],r['fraction'],r['seed']) for r in conditions()}
    actual = set(df[['model','fraction','seed']].itertuples(index=False,name=None))
    if actual != expected or len(df) != len(expected):
        raise ValueError('Cannot publish incomplete or duplicate experiment results')
    stats = df.groupby(['model', 'fraction']).agg(
        n_seeds=('seed','count'), auroc_mean=('macro_auroc','mean'), auroc_sd=('macro_auroc','std'),
        auprc_mean=('macro_auprc','mean'), auprc_sd=('macro_auprc','std'),
        f1_mean=('macro_f1','mean'), f1_sd=('macro_f1','std')).reset_index()
    stats.to_csv('results/summary.csv', index=False)
    for metric, filename in [('auroc','label_efficiency'), ('auprc','auprc')]:
        fig, ax = plt.subplots(figsize=(8,5))
        for model, marker, ls in [('scratch','s','--'), ('pretrained','o','-')]:
            part = stats[stats.model == model].sort_values('fraction')
            x, y = part.fraction.to_numpy()*100, part[f'{metric}_mean'].to_numpy()
            sd = part[f'{metric}_sd'].to_numpy()
            ax.plot(x,y,marker=marker,ls=ls,color=COLORS[model],label=NAMES[model])
            valid = np.isfinite(sd)
            ax.errorbar(x[valid],y[valid],yerr=sd[valid],fmt='none',capsize=4,color=COLORS[model])
        ax.set(xscale='log', xticks=[1,5,10,25,100], xticklabels=['1','5','10','25','100'],
               xlabel='Labeled training patients (%)', ylabel=f'Macro {metric.upper()}', ylim=(0,1.02))
        ax.minorticks_off()
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False, loc='lower right')
        ax.set_title(f'Label efficiency: macro {metric.upper()}',loc='left',pad=30)
        ax.text(0,1.035,'PTB-XL fold 10 | mean ± sample SD; 3 seeds at 1–25%, 1 seed at 100%',transform=ax.transAxes,fontsize=9)
        save(fig, filename)
    per_class = pd.read_csv('results/per_class.csv')
    part = per_class[(per_class.fraction == .1) & (per_class.model != 'linear_probe')]
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax, metric in zip(axes,['auroc','auprc']):
        for model, offset, marker in [('scratch',-.12,'s'), ('pretrained',.12,'o')]:
            sub = part[part.model == model].groupby('label')[metric].agg(['mean','std']).reindex(CLASSES)
            ax.errorbar(sub['mean'],np.arange(5)+offset,xerr=sub['std'],fmt=marker,
                        capsize=3,color=COLORS[model],label=NAMES[model])
        ax.set(xlim=(0,1.02),yticks=np.arange(5),yticklabels=CLASSES,xlabel=metric.upper())
        ax.grid(axis='x',alpha=.2)
        ax.legend(frameon=False,loc='lower left')
    axes[0].invert_yaxis()
    fig.suptitle('Per-class performance at 10% labeled patients',x=.125,ha='left')
    fig.text(.125,.9,'PTB-XL fold 10 | mean ± sample SD across 3 matched seeds',fontsize=10)
    fig.subplots_adjust(top=.82,wspace=.1)
    save(fig,'per_class')
    fig,ax = plt.subplots(figsize=(7,4.7))
    for model,offset,marker in [('pretrained',-.08,'o'),('linear_probe',.08,'D')]:
        part = stats[(stats.model==model)&stats.fraction.isin([.05,.1,1.])].sort_values('fraction')
        x = np.arange(3)+offset
        ax.scatter(x,part.auroc_mean,marker=marker,color=COLORS[model],label=NAMES[model],s=45)
        valid = part.auroc_sd.notna().to_numpy()
        ax.errorbar(x[valid],part.auroc_mean.to_numpy()[valid],yerr=part.auroc_sd.to_numpy()[valid],
                    fmt='none',capsize=4,color=COLORS[model])
    ax.set(xticks=np.arange(3),xticklabels=['5%','10%','100%'],ylim=(0,1.02),
           xlabel='Labeled training patients',ylabel='Macro AUROC')
    ax.set_title('Frozen encoder vs full fine-tuning',loc='left',pad=30)
    ax.text(0,1.04,'PTB-XL fold 10 | mean ± sample SD; n=3 at 5/10%, n=1 at 100%',transform=ax.transAxes,fontsize=9)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,loc='lower right')
    save(fig,'linear_probe')
    reference = stats[(stats.model=='scratch') & (stats.fraction==1)].auroc_mean.iloc[0]
    eligible = stats[(stats.model=='pretrained') & (stats.auroc_mean >= reference)].sort_values('fraction')
    gain = {'definition': 'Smallest tested patient fraction whose mean AUROC >= scratch 100% AUROC; descriptive, not statistical equivalence',
            'scratch_100_auroc': float(reference), 'pretrained_min_tested_fraction': None,
            'label_reduction': None, 'gain_factor': None}
    if len(eligible):
        fraction = float(eligible.iloc[0].fraction)
        gain.update(pretrained_min_tested_fraction=fraction,label_reduction=1-fraction,gain_factor=1/fraction)
    save_json('results/label_efficiency_gain.json',gain)
    lines = ['# Held-out results', '', 'All thresholds selected on fold 9. Test fold 10 was locked before evaluation.', '',
             '| Model | Patients | Seeds | Macro AUROC | Macro AUPRC | Macro F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in stats.itertuples():
        def fmt(mean,sd):
            return f'{mean:.4f} ± {sd:.4f}' if np.isfinite(sd) else f'{mean:.4f} (one seed)'
        lines.append(f'| {NAMES[r.model]} | {r.fraction:.0%} | {r.n_seeds} | {fmt(r.auroc_mean,r.auroc_sd)} | {fmt(r.auprc_mean,r.auprc_sd)} | {fmt(r.f1_mean,r.f1_sd)} |')
    lines += ['', 'Label efficiency gain is a discrete descriptive comparison, not an equivalence test.',
              'The fixed 30-epoch maximum does not establish performance after unlimited training or model-specific tuning.',
              'Seed variability combines patient subsampling and optimization. One full-data seed cannot estimate training variance.']
    Path('results/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
