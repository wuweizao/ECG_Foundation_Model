"""Freeze all selected phase-three artifacts before any test prediction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import ECGDataset
from src.labels import CLASSES
from src.models import build_model
from src.metrics import compute_metrics
from src.utils import save_json,sha256,seed_all
from .train import predict_bce,guard
from .register import verify_prior

ROOT=Path('runs/phase3')
OUT=Path('results/phase3')


def freeze():
    target=Path('experiments/phase3/test_lock.json')
    if not target.exists():
        guard()
    verify_prior(full=True)
    plan=json.loads(Path('experiments/phase3/plan.json').read_text())
    entries=[]
    for s in plan:
        run=ROOT/s['id']
        if not (run/'complete.json').exists():
            raise RuntimeError(f'Incomplete required run: {s["id"]}')
        complete=json.loads((run/'complete.json').read_text())
        if complete['best_sha256']!=sha256(run/'best.pt') or json.loads((run/'config.json').read_text())!=s:
            raise ValueError('Changed completed experiment')
        entries.append(dict(spec=s,weights=sha256(run/'best.pt'),thresholds=sha256(run/'thresholds.json'),
                            config=sha256(run/'config.json'),history=sha256(run/'history.csv'),
                            validation_predictions=sha256(run/'validation_predictions.npz')))
    assert len(entries)==54
    lock=dict(test_previously_seen=True,entries=entries,test_manifest=sha256('data/manifests/test.csv'),
              registration=sha256('experiments/phase3/registration.json'),
              sources={str(p).replace('\\','/'):sha256(p) for p in Path('src/phase3').glob('*.py')})
    if target.exists() and json.loads(target.read_text())!=lock:
        raise RuntimeError('Phase-three test lock changed')
    save_json(target,lock)
    save_json(OUT/'test_lock.json',lock)
    return plan


def main():
    plan=freeze()
    local=json.loads(Path('configs/local.json').read_text())
    frame=pd.read_csv('data/manifests/test.csv')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows,classes,provenance=[],[],[]
    for s in plan:
        run=ROOT/s['id']
        output=run/'test_predictions.npz'
        if not output.exists():
            seed_all(s['seed'])
            model=build_model('scratch').to(device)
            model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
            loader=DataLoader(ECGDataset(frame,local['cache']),batch_size=s['batch_size'])
            y,p,bce=predict_bce(model,loader,device,s['amp'])
            np.savez_compressed(output,y=y,p=p,ecg_id=frame.ecg_id.to_numpy(),patient_id=frame.patient_id.to_numpy(),bce=bce)
            del model
        with np.load(output) as data:
            for k,expected in [('y',frame[CLASSES].to_numpy()),('ecg_id',frame.ecg_id.to_numpy()),('patient_id',frame.patient_id.to_numpy())]:
                if not np.array_equal(data[k],expected):
                    raise ValueError('Test prediction identity mismatch')
            metrics,per=compute_metrics(data['y'],data['p'],json.loads((run/'thresholds.json').read_text()))
            fixed,_=compute_metrics(data['y'],data['p'],[.5]*5)
            metrics.update(macro_f1_fixed05=fixed['macro_f1'],micro_f1_fixed05=fixed['micro_f1'],test_bce=float(data['bce']))
        tr,va=pd.read_csv(s['train_manifest']),pd.read_csv(s['validation_manifest'])
        for policy in (['budgeted','full_validation_control'] if s['policy']=='shared_full' else [s['policy']]):
            key=dict(policy=policy,fraction=s['fraction'],seed=s['seed'],model=s['family'],run=str(run).replace('\\','/'))
            rows.append(dict(**key,training_patients=tr.patient_id.nunique(),validation_patients=va.patient_id.nunique(),
                             total_labeled_patients=tr.patient_id.nunique()+va.patient_id.nunique(),
                             training_records=len(tr),validation_records=len(va),shared_full=s['policy']=='shared_full',**metrics))
            classes.extend(dict(**key,**c) for c in per)
        provenance.append(dict(id=s['id'],predictions_sha256=sha256(output)))
        print(s['id'],metrics['macro_auroc'],flush=True)
    assert len(rows)==60
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'per_class.csv',index=False)
    save_json(OUT/'prediction_provenance.json',provenance)


if __name__=='__main__':
    main()
