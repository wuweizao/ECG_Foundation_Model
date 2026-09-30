import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .labels import CLASSES


def thresholds_from_validation(y, p):
    thresholds = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) != 2:
            raise ValueError('Validation class missing; cannot tune threshold')
        grid = np.linspace(0.01, 0.99, 99)
        scores = [f1_score(y[:, j], p[:, j] >= t, zero_division=0) for t in grid]
        thresholds.append(float(grid[np.argmax(scores)]))
    return thresholds


def compute_metrics(y, p, thresholds):
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError('Invalid predictions')
    if y.ndim != 2 or y.shape[1] != len(CLASSES) or len(thresholds) != len(CLASSES):
        raise ValueError('Expected five ordered independent labels')
    if not np.isin(y,[0,1]).all() or (p<0).any() or (p>1).any():
        raise ValueError('Expected binary targets and probabilities in [0,1]')
    pred = p >= np.asarray(thresholds)
    per_class = []
    for j, label in enumerate(CLASSES):
        truth, guess = y[:, j].astype(bool), pred[:, j]
        valid = len(np.unique(truth)) == 2
        tp, tn = (truth & guess).sum(), (~truth & ~guess).sum()
        per_class.append(dict(label=label, auroc=float(roc_auc_score(truth, p[:, j])) if valid else None,
                              auprc=float(average_precision_score(truth, p[:, j])) if valid else None,
                              f1=float(f1_score(truth, guess, zero_division=0)),
                              sensitivity=float(tp / truth.sum()) if truth.sum() else None,
                              specificity=float(tn / (~truth).sum()) if (~truth).sum() else None,
                              positives=int(truth.sum()), n=int(len(truth))))
    result = {}
    for key in ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity']:
        values = [r[key] for r in per_class]
        result['macro_' + key] = float(np.mean(values)) if all(v is not None for v in values) else None
    result['micro_f1'] = float(f1_score(y, pred, average='micro', zero_division=0))
    result['brier'] = float(np.mean((p-y)**2))
    eces = []
    for j in range(y.shape[1]):
        bins = np.minimum((p[:, j] * 10).astype(int), 9)
        eces.append(sum(float((bins == b).mean()) * abs(float(p[bins == b, j].mean() - y[bins == b, j].mean()))
                        for b in range(10) if (bins == b).any()))
    result['macro_ece'] = float(np.mean(eces))
    return result, per_class
