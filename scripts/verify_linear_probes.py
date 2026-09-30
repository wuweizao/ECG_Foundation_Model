"""Before test access, verify cached-feature validation against full waveform inference."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import ECGDataset
from src.models import build_model
from src.plan import conditions,run_name
from src.train import predict
from src.utils import seed_all,save_json,sha256


def artifact_identity():
    return dict(validation_manifest=sha256(ROOT/'data/manifests/validation.csv'),
                checkpoints={run_name(c):sha256(ROOT/'runs'/run_name(c)/'best.pt')
                             for c in conditions() if c['model']=='linear_probe'},
                predictions={run_name(c):sha256(ROOT/'runs'/run_name(c)/'validation_predictions.npz')
                             for c in conditions() if c['model']=='linear_probe'})


def main():
    identity = artifact_identity()
    verification_path = ROOT/'results/linear_probe_verification.json'
    if verification_path.exists():
        existing = json.loads(verification_path.read_text())
        if existing.get('identity')==identity and existing.get('runs_verified')==7:
            print('Linear probe verification is current for these unchanged artifacts')
            return
    seed_all(42)
    local = json.loads((ROOT/'configs/local.json').read_text())
    frame = pd.read_csv(ROOT/'data/manifests/validation.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checks = []
    for condition in conditions():
        if condition['model']!='linear_probe':
            continue
        run = ROOT/'runs'/run_name(condition)
        config = json.loads((run/'config.json').read_text())['config']
        model = build_model('scratch').to(device)
        model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
        loader = DataLoader(ECGDataset(frame,local['cache']),batch_size=config['batch_size'])
        y,p = predict(model,loader,device,config['amp'])
        cached = np.load(run/'validation_predictions.npz')
        np.testing.assert_array_equal(y,cached['y'])
        np.testing.assert_allclose(p,cached['p'],atol=1e-6,rtol=1e-6)
        checks.append(dict(**condition,maximum_probability_difference=float(np.max(np.abs(p-cached['p'])))))
        del model
    save_json(verification_path,dict(runs_verified=len(checks),identity=identity,
              split='validation',full_waveform_matches_cached_feature_predictions=True,checks=checks))
    print('Verified all 7 linear probes against full-waveform validation inference')


if __name__ == '__main__':
    main()
