"""Freeze all selected models before one coordinated held-out test evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ECGDataset
from .labels import CLASSES
from .metrics import compute_metrics
from .models import build_model
from .plan import conditions, run_name
from .train import predict
from .utils import save_json, seed_all, sha256


def freeze():
    entries = []
    for row in conditions():
        run = Path('runs') / run_name(row)
        if not (run / 'complete.json').exists():
            raise RuntimeError(f'Unfinished run: {run}')
        if json.loads((run/'complete.json').read_text())['best_sha256'] != sha256(run/'best.pt'):
            raise RuntimeError(f'Checkpoint differs from completed training: {run}')
        entries.append(dict(**row, weights=sha256(run / 'best.pt'),
                            config=sha256(run / 'config.json'), thresholds=sha256(run / 'thresholds.json')))
    code_files = ['src/dataset.py','src/labels.py','src/models.py','src/train.py','src/metrics.py',
                  'src/features.py','src/plan.py','src/utils.py','vendor/ECGFounder/net1d.py']
    lock = dict(runs=entries, test_manifest=sha256('data/manifests/test.csv'),
                code_sha256={name:sha256(name) for name in code_files})
    target = Path('experiments/test_lock.json')
    if target.exists() and json.loads(target.read_text()) != lock:
        raise RuntimeError('Frozen test experiment differs; do not retune after test access')
    save_json(target, lock)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-and-evaluate', action='store_true', required=True)
    args = parser.parse_args()
    entries = freeze()
    seed_all(42)
    local = json.loads(Path('configs/local.json').read_text())
    frame = pd.read_csv('data/manifests/test.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, per_class, provenance = [], [], []
    for entry in entries:
        run = Path('runs') / run_name(entry)
        config = json.loads((run / 'config.json').read_text())['config']
        thresholds = json.loads((run / 'thresholds.json').read_text())
        output = run / 'test_predictions.npz'
        if not output.exists():
            model = build_model('scratch').to(device)
            model.load_state_dict(torch.load(run / 'best.pt', weights_only=True, map_location=device))
            loader = DataLoader(ECGDataset(frame, local['cache']), batch_size=config['batch_size'], num_workers=0)
            y, prob = predict(model, loader, device, config['amp'])
            np.savez_compressed(output, y=y, p=prob, ecg_id=frame.ecg_id.to_numpy(), patient_id=frame.patient_id.to_numpy())
            del model
        predictions = np.load(output)
        if not np.array_equal(predictions['ecg_id'], frame.ecg_id.to_numpy()):
            raise ValueError('Prediction record order mismatch')
        if not np.array_equal(predictions['y'], frame[CLASSES].to_numpy()) or not np.array_equal(predictions['patient_id'],frame.patient_id.to_numpy()):
            raise ValueError('Prediction labels or patient IDs differ from locked test manifest')
        metrics, classes = compute_metrics(predictions['y'], predictions['p'], thresholds)
        key = {k: entry[k] for k in ['model', 'fraction', 'seed']}
        train_frame = pd.read_csv(f'data/manifests/train_f{entry["fraction"]:g}_s{entry["seed"]}.csv')
        rows.append(dict(**key, training_patients=train_frame.patient_id.nunique(),training_records=len(train_frame),
                         test_patients=frame.patient_id.nunique(),test_records=len(frame),**metrics))
        provenance.append(dict(**key,predictions_sha256=sha256(output),checkpoint_sha256=entry['weights']))
        per_class.extend(dict(**key, **c) for c in classes)
        print(run.name, metrics, flush=True)
    pd.DataFrame(rows).to_csv('results/metrics.csv', index=False)
    pd.DataFrame(per_class).to_csv('results/per_class.csv', index=False)
    save_json('results/prediction_provenance.json',provenance)


if __name__ == '__main__':
    main()
