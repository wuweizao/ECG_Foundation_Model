"""Audit official metadata, persist patient subsets, cache native 500 Hz signals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from .dataset import PREPROCESS_VERSION, preprocess
from .labels import CLASSES, map_labels, patient_subset, validate_folds
from .utils import save_json, sha256


def cache_record(args):
    root, cache, ecg_id, filename = args
    target = Path(cache) / f'{ecg_id}.npy'
    if target.exists():
        x = np.load(target)
        if x.shape == (12, 5000) and x.dtype == np.float32 and np.isfinite(x).all():
            return ecg_id
        raise ValueError(f'Invalid cached record: {ecg_id}')
    x, metadata = wfdb.rdsamp(str(Path(root) / filename))
    x = preprocess(x, metadata['sig_name'], metadata['fs'])
    tmp = target.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, x)
    tmp.replace(target)
    return ecg_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', default='configs/local.json')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    local = json.loads(Path(args.local).read_text())
    root, cache = Path(local['data_root']), Path(local['cache'])
    manifest = Path('data/manifests')
    manifest.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    database = pd.read_csv(root / 'ptbxl_database.csv')
    validate_folds(database)
    frame = map_labels(database, pd.read_csv(root / 'scp_statements.csv', index_col=0))
    frame = frame[['ecg_id', 'patient_id', 'strat_fold', 'filename_hr'] + CLASSES].sort_values('ecg_id')
    audit = {'version': '1.0.3', 'preprocessing': PREPROCESS_VERSION,
             'database_sha256': sha256(root / 'ptbxl_database.csv'),
             'statements_sha256': sha256(root / 'scp_statements.csv'),
             'raw_records': len(database), 'retained_records': len(frame),
             'excluded_no_diagnostic_label': len(database) - len(frame), 'splits': {}, 'subsets': {}}
    for split, mask in [('train', frame.strat_fold <= 8), ('validation', frame.strat_fold == 9), ('test', frame.strat_fold == 10)]:
        part = frame.loc[mask]
        part.to_csv(manifest / f'{split}.csv', index=False)
        audit['splits'][split] = {'records': len(part), 'patients': part.patient_id.nunique(),
                                  'positives': {k: int(part[k].sum()) for k in CLASSES}}
    train = frame.loc[frame.strat_fold <= 8]
    for seed in [42, 52, 62]:
        for fraction in [0.01, 0.05, 0.1, 0.25, 1.0]:
            if fraction == 1 and seed != 42:
                continue
            part = patient_subset(train, fraction, seed)
            key = f'train_f{fraction:g}_s{seed}'
            part.to_csv(manifest / f'{key}.csv', index=False)
            audit['subsets'][key] = {'records': len(part), 'patients': part.patient_id.nunique(),
                                      'positives': {k: int(part[k].sum()) for k in CLASSES},
                                      'sha256': sha256(manifest / f'{key}.csv')}
            if (part[CLASSES].sum() == 0).any():
                raise ValueError(f'Missing positive class in {key}; do not silently resample')
    # Cache identity prevents reuse after preprocessing or metadata changes.
    cache_id = cache / 'identity.json'
    identity = {k: audit[k] for k in ['version', 'preprocessing', 'database_sha256', 'statements_sha256']}
    if cache_id.exists() and json.loads(cache_id.read_text()) != identity:
        raise ValueError('Cache identity mismatch: choose a new cache directory')
    save_json(cache_id, identity)
    save_json('results/data_audit.json', audit)
    jobs = [(str(root), str(cache), int(r.ecg_id), r.filename_hr) for r in frame.itertuples()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in tqdm(pool.map(cache_record, jobs, chunksize=16), total=len(jobs), desc='Preprocessing'):
            pass
    save_json(cache / 'complete.json', {'records': len(jobs), 'identity': identity})
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
