"""Second-stage freeze and held-out evaluation; no test-driven selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import ECGDataset
from src.labels import CLASSES
from src.train import predict
from src.metrics import compute_metrics
from src.utils import save_json, sha256, seed_all
from .models import build
from .register import verify_phase1, spec as make_spec
from .suite import choose

ROOT=Path('runs/phase2')
OUT=Path('results/phase2')


def completed_selection():
    plan=json.loads(Path('experiments/phase2/plan.json').read_text())
    all_specs=list(plan['initial_runs'])
    selected=[s for s in all_specs if s['group']!='search']
    choices={}
    for fraction in [.01,1.]:
        for family in ['scratch','pretrained','xresnet1d101']:
            trials=[s for s in all_specs if s['group']=='search' and s['family']==family and s['fraction']==fraction]
            candidates=[]
            for s in trials:
                complete=ROOT/s['id']/'complete.json'
                failed=ROOT/s['id']/'failed.json'
                if not complete.exists() and not failed.exists():
                    raise RuntimeError(f'Incomplete candidate: {s["id"]}')
                score=json.loads(complete.read_text())['best_validation_macro_auroc'] if complete.exists() else None
                candidates.append(dict(spec=s,score=score))
            winner=choose(candidates)['spec']
            selected.append(winner)
            choices[f'{family}_f{fraction:g}']=winner['id']
            for seed in [52,62]:
                s=make_spec('repeat',family,fraction,seed,winner['lr'],60,12)
                all_specs.append(s)
                selected.append(s)
    assert len(all_specs)==45 and len(selected)==33
    records=[]
    for s in all_specs:
        run=ROOT/s['id']
        if (run/'complete.json').exists():
            complete=json.loads((run/'complete.json').read_text())
            if sha256(run/'best.pt')!=complete['best_sha256']:
                raise ValueError(f'Changed checkpoint: {run}')
            if json.loads((run/'config.json').read_text())['spec']!=s:
                raise ValueError(f'Changed config: {run}')
            records.append(dict(spec=s,selected=s in selected,weights=complete['best_sha256'],
                                config=sha256(run/'config.json'),thresholds=sha256(run/'thresholds.json'),
                                validation_predictions=sha256(run/'validation_predictions.npz')))
        elif s['group']=='search' and (run/'failed.json').exists():
            records.append(dict(spec=s,selected=False,failed=sha256(run/'failed.json')))
        else:
            raise RuntimeError(f'Required run incomplete: {s["id"]}')
    return selected,records,choices


def verify_probes(selected,local,device):
    path=OUT/'probe_verification.json'
    existing=json.loads(path.read_text()) if path.exists() else {}
    frame=pd.read_csv(ROOT/'manifests/validation.csv')
    for s in selected:
        if s['family'] not in ['linear_probe','random_probe']:
            continue
        run=ROOT/s['id']
        identity=dict(weights=sha256(run/'best.pt'),predictions=sha256(run/'validation_predictions.npz'))
        if existing.get(s['id'],{}).get('identity')==identity:
            continue
        seed_all(s['seed'])
        model=build(s,local).to(device)
        model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
        loader=DataLoader(ECGDataset(frame,local['cache']),batch_size=s['batch_size'])
        y,p=predict(model,loader,device,s['amp'])
        with np.load(run/'validation_predictions.npz') as data:
            if not np.array_equal(y,data['y']) or not np.array_equal(p,data['p']):
                raise ValueError(f'Frozen-feature/full-waveform disagreement: {s["id"]}')
        existing[s['id']]=dict(identity=identity,bitwise_equal=True,max_probability_difference=0.)
        save_json(path,existing)
        del model


def freeze(selected,records,choices):
    from .train import guard
    lock_path=Path('experiments/phase2/test_lock.json')
    if not lock_path.exists():
        guard()
    verify_phase1(full=True)
    paths=['experiments/phase2/PROTOCOL.md','experiments/phase2/registration.json',
           'experiments/phase2/plan.json',*[str(p).replace('\\','/') for p in Path('src/phase2').glob('*.py')]]
    lock=dict(phase1_test_previously_seen=True,attempts=records,selection=choices,
              test_manifest=sha256('data/manifests/test.csv'),
              source_sha256={p:sha256(p) for p in paths},
              probe_verification_sha256=sha256(OUT/'probe_verification.json'))
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:
        raise RuntimeError('Second-stage lock differs; test cannot be reused to retune')
    save_json(lock_path,lock)
    save_json(OUT/'test_lock.json',lock)


def cases(selected):
    result=[]
    def add(group,family,fraction,seed,run,spec=None):
        result.append(dict(group=group,model=family,fraction=fraction,seed=seed,run=str(run).replace('\\','/'),spec=spec))
    for s in selected:
        group={'replication':'fixed_full','frozen_control':'frozen','search':'equal_search','repeat':'equal_search'}[s['group']]
        add(group,s['family'],s['fraction'],s['seed'],ROOT/s['id'],s)
        if s['family']=='linear_probe':
            add('frozen','linear_probe',1.,s['seed'],ROOT/s['id'],s)
    for family in ['scratch','pretrained','linear_probe']:
        add('fixed_full',family,1.,42,f'runs/{family}_f1_s42')
    for fraction in [.05,.1,1.]:
        for seed in ([42] if fraction==1 else [42,52,62]):
            add('frozen','linear_probe',fraction,seed,f'runs/linear_probe_f{fraction:g}_s{seed}')
    assert len(result)==45
    return result


def main():
    selected,records,choices=completed_selection()
    local=json.loads(Path('configs/local.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verify_probes(selected,local,device)
    freeze(selected,records,choices)
    frame=pd.read_csv('data/manifests/test.csv')
    rows,classes,provenance=[],[],[]
    old_provenance=json.loads(Path('results/prediction_provenance.json').read_text())
    for case in cases(selected):
        run=Path(case['run'])
        output=run/'test_predictions.npz'
        s=case['spec']
        if s is None:
            previous=[r for r in old_provenance if r['group']=='main' and r['model']==case['model'] and r['fraction']==case['fraction'] and r['seed']==case['seed']][0]
            if sha256(output)!=previous['predictions_sha256']:
                raise ValueError('Phase-one predictions changed')
        elif not output.exists():
            seed_all(s['seed'])
            model=build(s,local).to(device)
            model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
            loader=DataLoader(ECGDataset(frame,local['cache']),batch_size=s['batch_size'])
            y,p=predict(model,loader,device,s['amp'])
            np.savez_compressed(output,y=y,p=p,ecg_id=frame.ecg_id.to_numpy(),patient_id=frame.patient_id.to_numpy())
            del model
        with np.load(output) as data:
            for key,expected in [('y',frame[CLASSES].to_numpy()),('ecg_id',frame.ecg_id.to_numpy()),('patient_id',frame.patient_id.to_numpy())]:
                if not np.array_equal(data[key],expected):
                    raise ValueError('Test prediction identity mismatch')
            metrics,per_class=compute_metrics(data['y'],data['p'],json.loads((run/'thresholds.json').read_text()))
        key={k:case[k] for k in ['group','model','fraction','seed','run']}
        rows.append(dict(**key,**metrics,reused_phase1=s is None))
        classes.extend(dict(**key,**c) for c in per_class)
        provenance.append(dict(**key,predictions_sha256=sha256(output)))
        print(key,metrics['macro_auroc'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'per_class.csv',index=False)
    save_json(OUT/'prediction_provenance.json',provenance)


if __name__=='__main__':
    main()
