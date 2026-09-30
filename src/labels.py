"""SCP diagnostic code membership, including likelihood=0 per PTB-XL benchmark."""
import ast
import math

import numpy as np
import pandas as pd

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def map_labels(database, statements):
    mapping = statements.loc[statements.diagnostic == 1, 'diagnostic_class'].to_dict()
    out = database.copy()
    codes = out.scp_codes.map(ast.literal_eval)
    for label in CLASSES:
        out[label] = codes.map(lambda c: int(any(mapping.get(k) == label for k in c)))
    # Never reinterpret records without diagnostic labels as healthy negatives.
    return out.loc[out[CLASSES].sum(axis=1) > 0].copy()


def validate_folds(df):
    if df.patient_id.isna().any() or df.ecg_id.duplicated().any():
        raise ValueError('Missing patient ID or duplicate ECG ID')
    if not df.strat_fold.isin(range(1, 11)).all():
        raise ValueError('Invalid official fold')
    if df.groupby('patient_id').strat_fold.nunique().max() != 1:
        raise ValueError('Patient crosses official folds')


def patient_subset(train, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    patients = np.sort(train.patient_id.unique())
    # One permutation gives nested subsets across fractions; paired across models.
    selected = np.random.default_rng(seed).permutation(patients)[:max(1, math.ceil(len(patients) * fraction))]
    return train.loc[train.patient_id.isin(selected)].copy()
