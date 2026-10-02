"""Registered phase-two training; only train and validation data are accessible."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import DataLoader
from src.dataset import ECGDataset
from src.train import predict
from src.metrics import compute_metrics, thresholds_from_validation
from src.utils import save_json, seed_all, sha256
from .features import frozen_dataset
from .models import build, tensor_hash
from .register import verify_phase1


def guard():
    if Path('experiments/phase2/test_lock.json').exists():
        raise RuntimeError('Phase-two test is locked; additional training forbidden')
    registration=json.loads(Path('experiments/phase2/registration.json').read_text())
    for key,path in [('protocol_sha256','experiments/phase2/PROTOCOL.md'),
                     ('plan_sha256','experiments/phase2/plan.json')]:
        if sha256(path)!=registration[key]:
            raise RuntimeError(f'Registered artifact changed: {path}')
    for name,digest in {**registration['code_sha256'],**registration['manifest_sha256']}.items():
        if sha256(name)!=digest:
            raise RuntimeError(f'Registered input changed: {name}')
    verify_phase1()


def train(spec):
    plan=json.loads(Path('experiments/phase2/plan.json').read_text())
    if spec not in plan['initial_runs']:
        from .register import spec as make_spec
        selections=json.loads(Path('results/phase2/selection.json').read_text())
        key=f'{spec["family"]}_f{spec["fraction"]:g}'
        if spec['seed'] not in [52,62] or key not in selections:
            raise ValueError('Unregistered training specification')
        expected=make_spec('repeat',spec['family'],spec['fraction'],spec['seed'],selections[key]['lr'],60,12)
        if spec!=expected:
            raise ValueError('Repeat differs from validation-selected registered configuration')
    run=Path('runs/phase2')/spec['id']
    run.mkdir(parents=True,exist_ok=True)
    with FileLock(str(run/'.training.lock')):
        if (run/'complete.json').exists():
            metadata=json.loads((run/'config.json').read_text())
            complete=json.loads((run/'complete.json').read_text())
            if metadata['spec']!=spec or sha256(run/'best.pt')!=complete['best_sha256']:
                raise ValueError('Completed run identity mismatch')
            return run
        guard()
        return _train(spec,run)


def _train(spec,run):
    local=json.loads(Path('configs/local.json').read_text())
    manifest=Path('runs/phase2/manifests')
    train_path=manifest/f'train_f{spec["fraction"]:g}_s{spec["seed"]}.csv'
    val_path=manifest/'validation.csv'
    train_frame,val_frame=pd.read_csv(train_path),pd.read_csv(val_path)
    if set(train_frame.patient_id)&set(val_frame.patient_id):
        raise ValueError('Patient leakage')
    seed_all(spec['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build(spec,local).to(device)
    probe=spec['family'] in ['linear_probe','random_probe']
    metadata=dict(spec=spec,train_manifest_sha256=sha256(train_path),
                  validation_manifest_sha256=sha256(val_path),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  torch=torch.__version__,cuda=torch.version.cuda,
                  initial_state_sha256=tensor_hash(model.state_dict()))
    if spec['family']!='xresnet1d101':
        metadata['initial_head_sha256']=tensor_hash(model.dense.state_dict())
        metadata['initial_encoder_sha256']=tensor_hash({n:v for n,v in model.state_dict().items() if not n.startswith('dense.')})
    save_json(run/'config.json',metadata)
    train_data=ECGDataset(train_frame,local['cache'])
    val_data=ECGDataset(val_frame,local['cache'])
    active=model
    if probe:
        train_data=frozen_dataset(model,train_frame,'train',local,spec,device)
        val_data=frozen_dataset(model,val_frame,'validation',local,spec,device)
        active=model.dense
    train_loader=DataLoader(train_data,batch_size=spec['batch_size'],shuffle=True,
                            generator=torch.Generator().manual_seed(spec['seed']))
    val_loader=DataLoader(val_data,batch_size=spec['batch_size'])
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=spec['lr'],weight_decay=spec['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss()
    best,bad,history=-float('inf'),0,[]
    start=time.time()
    for epoch in range(spec['epochs']):
        model.train()
        if probe:
            model.eval()
            model.dense.train()
        loss_sum=0.
        optimizer.zero_grad(set_to_none=True)
        accum=spec['accumulation_steps']
        for step,(x,y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device)
            group_start=(step//accum)*accum*spec['batch_size']
            group_samples=min(accum*spec['batch_size'],len(train_frame)-group_start)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,
                                enabled=spec['amp'] and device.type=='cuda'):
                loss=criterion(active(x),y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, step {step}')
            (loss*len(x)/group_samples).backward()
            if (step+1)%accum==0 or step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum+=loss.item()*len(x)
        y,p=predict(active,val_loader,device,spec['amp'])
        metrics,_=compute_metrics(y,p,[.5]*5)
        score=metrics['macro_auroc']
        if score is None or not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation score')
        history.append(dict(epoch=epoch+1,train_loss=loss_sum/len(train_frame),
                            validation_macro_auroc=score,elapsed_seconds=time.time()-start))
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        print(f'{spec["id"]} epoch={epoch+1} val_auc={score:.5f} elapsed={time.time()-start:.0f}s',flush=True)
        if score>best:
            best,bad=score,0
            torch.save(model.state_dict(),run/'best.tmp')
            (run/'best.tmp').replace(run/'best.pt')
            np.savez_compressed(run/'validation_predictions.npz',y=y,p=p,
                                ecg_id=val_frame.ecg_id.to_numpy(),patient_id=val_frame.patient_id.to_numpy())
        else:
            bad+=1
        if bad>=spec['patience']:
            break
    with np.load(run/'validation_predictions.npz') as predictions:
        thresholds=thresholds_from_validation(predictions['y'],predictions['p'])
    save_json(run/'thresholds.json',thresholds)
    if probe:
        encoder=tensor_hash({n:v for n,v in model.state_dict().items() if not n.startswith('dense.')})
        if encoder!=metadata['initial_encoder_sha256']:
            raise AssertionError('Frozen encoder mutated')
    save_json(run/'complete.json',dict(best_validation_macro_auroc=best,epochs=len(history),
                                      duration_seconds=time.time()-start,best_sha256=sha256(run/'best.pt')))
    return run


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',required=True)
    a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text())
    try:
        train(spec)
    except FloatingPointError as exc:
        save_json(Path('runs/phase2')/spec['id']/'failed.json',dict(reason=str(exc),kind='numerical'))
        raise


if __name__=='__main__':
    main()
