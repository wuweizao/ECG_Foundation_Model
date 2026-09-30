"""Independent rank/confusion-count audit of the final exported results."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.labels import CLASSES
from src.plan import conditions,run_name
from src.utils import save_json


def rank_auc(y,p):
    npos,nneg = y.sum(),(1-y).sum()
    return (rankdata(p)[y.astype(bool)].sum()-npos*(npos+1)/2)/(npos*nneg)


def main():
    metrics = pd.read_csv(ROOT/'results/metrics.csv')
    convergence = pd.read_csv(ROOT/'results/convergence_audit.csv')
    per_class = pd.read_csv(ROOT/'results/per_class.csv')
    assert len(metrics)==33 and len(per_class)==165
    assert len(convergence)==2
    all_metrics = pd.concat([metrics.assign(root='runs'),convergence.assign(root='runs/convergence')])
    errors = []
    cases = [dict(**c,root='runs') for c in conditions()]
    cases += [dict(model=m,fraction=.01,seed=42,root='runs/convergence') for m in ['scratch','pretrained']]
    for condition in cases:
        run = ROOT/condition['root']/run_name(condition)
        data = np.load(run/'test_predictions.npz')
        y,p = data['y'],data['p']
        t = np.asarray(json.loads((run/'thresholds.json').read_text()))
        pred = p>=t
        mask = (all_metrics.model==condition['model']) & (all_metrics.fraction==condition['fraction']) & (all_metrics.seed==condition['seed']) & (all_metrics.root==condition['root'])
        assert mask.sum()==1
        row = all_metrics.loc[mask].iloc[0]
        auc = np.mean([rank_auc(y[:,j],p[:,j]) for j in range(5)])
        errors.append(abs(auc-row.macro_auroc))
        tp = ((y==1)&pred).sum(axis=0)
        fp = ((y==0)&pred).sum(axis=0)
        fn = ((y==1)&~pred).sum(axis=0)
        f1 = np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5,dtype=float),where=(2*tp+fp+fn)>0).mean()
        errors.append(abs(f1-row.macro_f1))
        sensitivity = (tp/y.sum(axis=0)).mean()
        specificity = (((y==0)&~pred).sum(axis=0)/(y==0).sum(axis=0)).mean()
        errors.extend([abs(sensitivity-row.macro_sensitivity),abs(specificity-row.macro_specificity)])
        # AUPRC is average precision (step-wise), not trapezoidal PR area.
        values = []
        for j in range(5):
            order = np.argsort(-p[:,j],kind='stable')
            positives = np.cumsum(y[order,j])
            ends = np.r_[np.flatnonzero(np.diff(p[order,j]) != 0),len(order)-1]
            cumulative_tp = positives[ends]
            increments = np.diff(np.r_[0,cumulative_tp])
            values.append(float(np.sum(increments*(cumulative_tp/(ends+1)))/y[:,j].sum()))
        errors.append(abs(np.mean(values)-row.macro_auprc))
    assert max(errors)<1e-7, max(errors)
    summary = pd.read_csv(ROOT/'results/summary.csv')
    for row in summary.itertuples():
        values = metrics[(metrics.model==row.model)&(metrics.fraction==row.fraction)].macro_auroc
        assert abs(values.mean()-row.auroc_mean)<1e-10
        if len(values)==1:
            assert np.isnan(row.auroc_sd)
        else:
            assert abs(values.std(ddof=1)-row.auroc_sd)<1e-10
    save_json(ROOT/'results/analysis_validation.json',dict(runs_checked=35,per_class_rows=165,
              independent_rank_auroc=True,independent_stepwise_average_precision=True,
              independent_confusion_metrics=True,sample_sd_verified=True,max_absolute_discrepancy=float(max(errors))))
    print('Independent metric and aggregation audit passed')


if __name__ == '__main__':
    main()
