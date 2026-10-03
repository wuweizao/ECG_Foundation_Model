"""Fixed-recipe training with budget-restricted validation BCE selection."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from filelock import FileLock
from src.dataset import ECGDataset
from src.models import build_model
from src.metrics import thresholds_from_validation
from src.utils import seed_all,save_json,sha256
from .register import verify_prior


def guard():
    if Path('experiments/phase3/test_lock.json').exists():
        raise RuntimeError('Phase-three test is locked; additional training forbidden')
    r=json.loads(Path('experiments/phase3/registration.json').read_text())
    for path,digest in {**r['files'],**r['manifests']}.items():
        if sha256(path)!=digest:
            raise ValueError(f'Registered phase-three input changed: {path}')
    verify_prior()


def thresholds(y,p):
    values,fallback=[],[]
    # The original helper requires five columns. For each supported column,
    # duplicate it temporarily; threshold search is class-independent.
    for j in range(5):
        missing=len(np.unique(y[:,j]))!=2
        fallback.append(missing)
        if missing:
            values.append(.5)
        else:
            values.append(thresholds_from_validation(np.repeat(y[:,j:j+1],5,axis=1),
                                                    np.repeat(p[:,j:j+1],5,axis=1))[0])
    return values,fallback


@torch.inference_mode()
def predict_bce(model,loader,device,amp):
    model.eval()
    ys,ps=[],[]
    numerator,elements=0.,0
    for x,y in loader:
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
            logits=model(x.to(device))
        logits=logits.float().cpu()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum')
        numerator+=float(loss)
        elements+=y.numel()
        ys.append(y.numpy())
        ps.append(torch.sigmoid(logits).numpy())
    if not np.isfinite(numerator):
        raise FloatingPointError('Nonfinite validation BCE')
    return np.concatenate(ys),np.concatenate(ps),numerator/elements


def train(spec):
    guard()
    plan=json.loads(Path('experiments/phase3/plan.json').read_text())
    if spec not in plan:
        raise ValueError('Unregistered experiment')
    run=Path('runs/phase3')/spec['id']
    run.mkdir(parents=True,exist_ok=True)
    with FileLock(str(run/'.training.lock')):
        if (run/'complete.json').exists():
            old=json.loads((run/'complete.json').read_text())
            if old['best_sha256']!=sha256(run/'best.pt') or json.loads((run/'config.json').read_text())!=spec:
                raise ValueError('Changed completed run')
            return
        _train(spec,run)


def _train(s,run):
    local=json.loads(Path('configs/local.json').read_text())
    tr,va=pd.read_csv(s['train_manifest']),pd.read_csv(s['validation_manifest'])
    if set(tr.patient_id)&set(va.patient_id):
        raise ValueError('Patient leakage')
    seed_all(s['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(s['family'],local['checkpoint']).to(device)
    train_loader=DataLoader(ECGDataset(tr,local['cache']),batch_size=s['batch_size'],shuffle=True,
                            generator=torch.Generator().manual_seed(s['seed']))
    val_loader=DataLoader(ECGDataset(va,local['cache']),batch_size=s['batch_size'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=s['lr'],weight_decay=s['weight_decay'])
    criterion=torch.nn.BCEWithLogitsLoss()
    save_json(run/'config.json',s)
    best,bad,history=float('inf'),0,[]
    start=time.time()
    for epoch in range(s['epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total=0.
        accum=s['accumulation_steps']
        for step,(x,y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device)
            group_start=(step//accum)*accum*s['batch_size']
            group_samples=min(accum*s['batch_size'],len(tr)-group_start)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=s['amp'] and device.type=='cuda'):
                loss=criterion(model(x),y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            (loss*len(x)/group_samples).backward()
            if (step+1)%accum==0 or step+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(x)
        y,p,score=predict_bce(model,val_loader,device,s['amp'])
        history.append(dict(epoch=epoch+1,train_loss=total/len(tr),validation_bce=score,
                            elapsed_seconds=time.time()-start))
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        print(f'{s["id"]} epoch={epoch+1} validation_bce={score:.6f} elapsed={time.time()-start:.0f}s',flush=True)
        if score<best:
            best,bad=score,0
            torch.save(model.state_dict(),run/'best.tmp')
            (run/'best.tmp').replace(run/'best.pt')
            np.savez_compressed(run/'validation_predictions.npz',y=y,p=p,
                                ecg_id=va.ecg_id.to_numpy(),patient_id=va.patient_id.to_numpy())
        else:
            bad+=1
        if bad>=s['patience']:
            break
    with np.load(run/'validation_predictions.npz') as d:
        values,fallback=thresholds(d['y'],d['p'])
    save_json(run/'thresholds.json',values)
    save_json(run/'threshold_fallback.json',fallback)
    save_json(run/'complete.json',dict(best_validation_bce=best,epochs=len(history),
              best_sha256=sha256(run/'best.pt'),duration_seconds=time.time()-start,
              fallback_classes=int(sum(fallback))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    a=parser.parse_args()
    s=json.loads(Path(a.spec).read_text())
    try:
        train(s)
    except Exception as exc:
        save_json(Path('runs/phase3')/s['id']/'failed.json',dict(type=type(exc).__name__,reason=str(exc)))
        raise


if __name__=='__main__':
    main()
