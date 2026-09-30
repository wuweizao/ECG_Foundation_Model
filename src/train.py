"""Training and validation only. This module never loads the test manifest."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ECGDataset
from .features import frozen_dataset
from .metrics import compute_metrics, thresholds_from_validation
from .models import build_model
from .utils import read_config, save_json, seed_all, sha256


@torch.inference_mode()
def predict(model, loader, device, amp=True):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == 'cuda'):
            logits = model(x.to(device))
        ys.append(y.numpy())
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train(config_path, fraction, seed, local_path='configs/local.json', output_root='runs'):
    config = read_config(config_path)
    local = json.loads(Path(local_path).read_text())
    label = 'linear_probe' if config['linear_probe'] else config['initialization']
    run = Path(output_root) / f'{label}_f{fraction:g}_s{seed}'
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'complete.json').exists():
        existing = json.loads((run / 'config.json').read_text())
        if existing['config'] != config:
            raise ValueError(f'Completed run has different config: {run}')
        for key, path in [('train_manifest_sha256', f'data/manifests/train_f{fraction:g}_s{seed}.csv'),
                          ('validation_manifest_sha256', 'data/manifests/validation.csv')]:
            if existing[key] != sha256(path):
                raise ValueError(f'Completed run has changed input manifest: {run}')
        complete = json.loads((run / 'complete.json').read_text())
        if complete['best_sha256'] != sha256(run / 'best.pt'):
            raise ValueError(f'Completed checkpoint changed: {run}')
        print(f'Skip completed {run}', flush=True)
        return run
    if Path('experiments/test_lock.json').exists():
        raise RuntimeError('Test has been unlocked: refusing additional main training')
    train_path = Path(f'data/manifests/train_f{fraction:g}_s{seed}.csv')
    val_path = Path('data/manifests/validation.csv')
    metadata = dict(config=config, fraction=fraction, seed=seed, model=label,
                    train_manifest_sha256=sha256(train_path), validation_manifest_sha256=sha256(val_path),
                    checkpoint_sha256=sha256(local['checkpoint']) if config['initialization'] == 'pretrained' else None,
                    torch=torch.__version__, cuda=torch.version.cuda)
    save_json(run / 'config.json', metadata)
    seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_frame, val_frame = pd.read_csv(train_path), pd.read_csv(val_path)
    if set(train_frame.patient_id) & set(val_frame.patient_id):
        raise ValueError('Patient leakage')
    train_loader = DataLoader(ECGDataset(train_frame, local['cache']), batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['workers'], pin_memory=device.type == 'cuda',
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(ECGDataset(val_frame, local['cache']), batch_size=config['batch_size'],
                            num_workers=config['workers'])
    model = build_model(config['initialization'], local['checkpoint'], config['linear_probe']).to(device)
    active_model = model
    if config['linear_probe']:
        train_loader = DataLoader(frozen_dataset(model,train_frame,'train',local,config,device),
                                  batch_size=config['batch_size'],shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(frozen_dataset(model,val_frame,'validation',local,config,device),
                                batch_size=config['batch_size'])
        active_model = model.dense
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss()
    best, bad_epochs, history = -float('inf'), 0, []
    start = time.time()
    for epoch in range(config['epochs']):
        model.train()
        if config['linear_probe']:
            model.eval()
            model.dense.train()
        loss_sum = 0.
        optimizer.zero_grad(set_to_none=True)
        accum = config['accumulation_steps']
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # Weight by sample count, including the final partial accumulation group.
            group_start = (step // accum) * accum * config['batch_size']
            group_samples = min(accum * config['batch_size'], len(train_frame) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config['amp'] and device.type == 'cuda'):
                loss = criterion(active_model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss in {run}')
            (loss * len(x) / group_samples).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * len(x)
        y, p = predict(active_model, val_loader, device, config['amp'])
        metrics, _ = compute_metrics(y, p, [0.5] * 5)
        score = metrics['macro_auroc']
        if score is None:
            raise ValueError('Validation AUROC undefined')
        history.append(dict(epoch=epoch+1, train_loss=loss_sum / len(train_frame),
                            validation_macro_auroc=score, elapsed_seconds=time.time()-start))
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        print(f'{run.name} epoch={epoch+1} loss={history[-1]["train_loss"]:.5f} val_auc={score:.5f} elapsed={time.time()-start:.0f}s', flush=True)
        if score > best:
            best, bad_epochs = score, 0
            tmp = run / 'best.tmp'
            torch.save(model.state_dict(), tmp)
            tmp.replace(run / 'best.pt')
            np.savez_compressed(run / 'validation_predictions.npz', y=y, p=p,
                                ecg_id=val_frame.ecg_id.to_numpy(), patient_id=val_frame.patient_id.to_numpy())
        else:
            bad_epochs += 1
        if bad_epochs >= config['patience']:
            break
    predictions = np.load(run / 'validation_predictions.npz')
    thresholds = thresholds_from_validation(predictions['y'], predictions['p'])
    save_json(run / 'thresholds.json', thresholds)
    save_json(run / 'complete.json', dict(best_validation_macro_auroc=best, epochs=len(history),
                                         duration_seconds=time.time()-start, best_sha256=sha256(run / 'best.pt')))
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local', default='configs/local.json')
    p.add_argument('--output-root', default='runs')
    a = p.parse_args()
    train(a.config, a.fraction, a.seed, a.local, a.output_root)


if __name__ == '__main__':
    main()
