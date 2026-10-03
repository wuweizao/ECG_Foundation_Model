"""Immutable phase-three budget allocation and prior-stage seal."""
import json
import math
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.labels import CLASSES
from src.utils import save_json,sha256
from src.phase2.register import verify_phase1

ROOT=Path('experiments/phase3')
RUNS=Path('runs/phase3')
BASE='b12eb03'


def allocation(fraction,train_patients,val_patients):
    total=train_patients+val_patients
    budget=math.ceil(fraction*total)
    nv=(budget*val_patients)//total
    nt=budget-nv
    if not (1<=nt<=train_patients and 1<=nv<=val_patients):
        raise ValueError('Budget too small for the registered split ratio')
    return nt,nv


def subset(frame,n,seed):
    ids=np.sort(frame.patient_id.unique())
    chosen=np.random.default_rng(seed).permutation(ids)[:n]
    return frame.loc[frame.patient_id.isin(chosen)].copy()


def verify_prior(full=False):
    seal=json.loads((ROOT/'prior_seal.json').read_text())
    for path,digest in seal['files'].items():
        if sha256(path)!=digest:
            raise ValueError(f'Prior artifact changed: {path}')
    if full:
        verify_phase1(full=True)
        for path,digest in seal['weights'].items():
            if sha256(path)!=digest:
                raise ValueError(f'Prior weights changed: {path}')


def count(frame):
    return dict(patients=int(frame.patient_id.nunique()),records=len(frame),
                positives={k:int(frame[k].sum()) for k in CLASSES},
                negatives={k:int((frame[k]==0).sum()) for k in CLASSES})


def main():
    if (ROOT/'registration.json').exists():
        verify_prior(full=True)
        print('Existing registration preserved; prior integrity verified.')
        return
    verify_phase1(full=True)
    phase2=json.loads(Path('results/phase2/test_lock.json').read_text())
    weights={}
    for attempt in phase2['attempts']:
        if 'weights' in attempt:
            path=f'runs/phase2/{attempt["spec"]["id"]}/best.pt'
            if sha256(path)!=attempt['weights']:
                raise ValueError(f'Phase-two checkpoint mismatch: {path}')
            weights[path]=attempt['weights']
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    save_json(ROOT/'prior_seal.json',dict(commit=BASE,files={n:sha256(n) for n in names},weights=weights))
    train=pd.read_csv('data/manifests/train.csv')
    val=pd.read_csv('data/manifests/validation.csv')
    if set(train.patient_id)&set(val.patient_id):
        raise ValueError('Patient leakage')
    manifest=RUNS/'manifests'
    manifest.mkdir(parents=True,exist_ok=True)
    audit,rows,hashes=[],[],{}
    for fraction in [.01,.05,.1,.25,1.]:
        nt,nv=allocation(fraction,train.patient_id.nunique(),val.patient_id.nunique())
        for seed in [42,52,62]:
            tr=subset(train,nt,seed)
            for policy in (['shared_full'] if fraction==1 else ['budgeted','full_validation_control']):
                va=subset(val,nv,seed+10000) if policy=='budgeted' else val.copy()
                key=f'{policy}_f{fraction:g}_s{seed}'
                tp,vp=manifest/f'{key}_train.csv',manifest/f'{key}_validation.csv'
                tr.to_csv(tp,index=False)
                va.to_csv(vp,index=False)
                for p in [tp,vp]:
                    hashes[str(p).replace('\\','/')]=sha256(p)
                audit.append(dict(policy=policy,fraction=fraction,seed=seed,train=count(tr),validation=count(va),
                                  total_labeled_patients=nt+va.patient_id.nunique(),
                                  nominal_budget_patients=math.ceil(fraction*16740)))
                for family in ['scratch','pretrained']:
                    rows.append(dict(id=f'{key}_{family}',policy=policy,fraction=fraction,seed=seed,family=family,
                                     train_manifest=str(tp).replace('\\','/'),validation_manifest=str(vp).replace('\\','/'),
                                     epochs=60,patience=12,batch_size=16,accumulation_steps=2,
                                     lr=.0001,weight_decay=.01,amp=True,selection='minimum_validation_bce'))
    assert len(rows)==54
    save_json(ROOT/'plan.json',rows)
    save_json('results/phase3/data_audit.json',audit)
    sources=list(Path('src/phase3').glob('*.py'))
    registration=dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),prior_commit=BASE,
                      test_previously_seen=True,planned_runs=54,
                      files={str(p).replace('\\','/'):sha256(p) for p in [ROOT/'PROTOCOL.md',ROOT/'plan.json',*sources]},
                      manifests=hashes,test_manifest_sha256=sha256('data/manifests/test.csv'),
                      checkpoint_sha256=sha256(json.loads(Path('configs/local.json').read_text())['checkpoint']))
    save_json(ROOT/'registration.json',registration)
    print('Registered 54 runs with paired/nested patient manifests.',flush=True)


if __name__=='__main__':
    main()
