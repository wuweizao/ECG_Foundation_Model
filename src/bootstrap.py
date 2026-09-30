"""Paired patient-cluster bootstrap, conditional on each fitted seed pair."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def patient_indices(patient_ids, rng, groups=None):
    if groups is None:
        groups = {pid: np.flatnonzero(patient_ids == pid) for pid in np.unique(patient_ids)}
    unique = np.asarray(list(groups))
    selected = rng.choice(unique, len(unique), replace=True)
    return np.concatenate([groups[pid] for pid in selected])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    rows = []
    for fraction in [.01,.05,.1,.25,1.]:
        for seed in ([42] if fraction == 1 else [42,52,62]):
            a = np.load(Path('runs') / f'scratch_f{fraction:g}_s{seed}/test_predictions.npz')
            b = np.load(Path('runs') / f'pretrained_f{fraction:g}_s{seed}/test_predictions.npz')
            for field in ['y','ecg_id','patient_id']:
                if not np.array_equal(a[field], b[field]):
                    raise ValueError('Paired bootstrap requires identical ordered observations')
            rng = np.random.default_rng(seed)
            groups = {pid: np.flatnonzero(a['patient_id'] == pid) for pid in np.unique(a['patient_id'])}
            differences = {'macro_auroc': [], 'macro_auprc': []}
            metrics = {'macro_auroc': roc_auc_score, 'macro_auprc': average_precision_score}
            for _ in range(args.replicates):
                idx = patient_indices(a['patient_id'],rng,groups)
                y = a['y'][idx]
                if any(len(np.unique(y[:,j])) != 2 for j in range(y.shape[1])):
                    continue
                for name, fn in metrics.items():
                    differences[name].append(fn(y,b['p'][idx],average='macro') - fn(y,a['p'][idx],average='macro'))
            for name, fn in metrics.items():
                lower, upper = np.quantile(differences[name],[.025,.975])
                rows.append(dict(fraction=fraction,seed=seed,metric=name,
                                 delta=fn(a['y'],b['p'],average='macro')-fn(a['y'],a['p'],average='macro'),
                                 ci_low=lower,ci_high=upper,requested_replicates=args.replicates,
                                 valid_replicates=len(differences[name]),unit='patient',
                                 scope='conditional_on_fitted_seed_pair'))
            print(f'Bootstrap fraction={fraction} seed={seed}',flush=True)
    pd.DataFrame(rows).to_csv('results/bootstrap.csv',index=False)


if __name__ == '__main__':
    main()
