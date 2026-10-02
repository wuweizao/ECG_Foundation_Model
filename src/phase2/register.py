"""Seal phase one and materialize the predeclared phase-two candidate grid."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from src.utils import sha256, save_json

ROOT = Path('experiments/phase2')
RUNS = Path('runs/phase2')


def verify_phase1(full=False):
    seal = json.loads((ROOT/'phase1_seal.json').read_text())
    for name, digest in seal['tracked_files'].items():
        if sha256(name) != digest:
            raise ValueError(f'Phase-one tracked artifact changed: {name}')
    for name, digest in seal['untracked_inputs'].items():
        if sha256(name) != digest:
            raise ValueError(f'Phase-one input changed: {name}')
    if full:
        for name, digest in seal['selected_weights'].items():
            if sha256(name) != digest:
                raise ValueError(f'Phase-one selected weights changed: {name}')
    return seal


def spec(group, family, fraction, seed, lr, epochs=30, patience=7):
    return dict(id=f'{group}_{family}_f{fraction:g}_s{seed}_lr{lr:g}',
                group=group, family=family, fraction=fraction, seed=seed,
                lr=lr, epochs=epochs, patience=patience, batch_size=16,
                accumulation_steps=2, weight_decay=.01, amp=True, workers=0)


def main():
    ROOT.mkdir(exist_ok=True, parents=True)
    if (ROOT/'registration.json').exists():
        verify_phase1(full=True)
        print('Registration already exists; phase-one seal verified.')
        return
    names = subprocess.check_output(['git','ls-tree','-r','--name-only','a3b6bc4'],text=True).splitlines()
    lock = json.loads(Path('experiments/test_lock.json').read_text())
    inputs = {n:sha256(n) for n in ['experiments/test_lock.json',
              *[str(p).replace('\\','/') for p in Path('data/manifests').glob('*.csv')],
              'vendor/ECGFounder/net1d.py']}
    weights = {}
    for row in lock['runs']:
        path = f'{row["root"]}/{row["model"]}_f{row["fraction"]:g}_s{row["seed"]}/best.pt'
        if sha256(path) != row['weights']:
            raise ValueError(f'Existing lock invalid: {path}')
        weights[path] = row['weights']
    seal = dict(commit='a3b6bc4', tracked_files={n:sha256(n) for n in names},
                untracked_inputs=inputs, selected_weights=weights,
                archive_sha256=sha256(RUNS/'phase1-a3b6bc4.zip'))
    save_json(ROOT/'phase1_seal.json',seal)
    target = RUNS/'manifests'
    target.mkdir(exist_ok=True)
    for path in Path('data/manifests').glob('train*.csv'):
        shutil.copyfile(path,target/path.name)
    shutil.copyfile('data/manifests/validation.csv',target/'validation.csv')
    for seed in [52,62]:
        shutil.copyfile(target/'train.csv',target/f'train_f1_s{seed}.csv')
    rows = []
    for family in ['scratch','pretrained','linear_probe']:
        for seed in [52,62]:
            rows.append(spec('replication',family,1.,seed,.001 if family=='linear_probe' else .0001))
    for fraction in [.05,.1,1.]:
        for seed in [42,52,62]:
            rows.append(spec('frozen_control','random_probe',fraction,seed,.001))
    for fraction in [.01,1.]:
        for family in ['scratch','pretrained','xresnet1d101']:
            for lr in [.00001,.0001,.001]:
                rows.append(spec('search',family,fraction,42,lr,60,12))
    save_json(ROOT/'plan.json',dict(initial_runs=rows,repeat_rule=dict(
        group='repeat',seeds=[52,62],selection='best validation macro AUROC; smaller LR tie',
        model_fractions=6),expected_new_runs=45))
    code = list(Path('src/phase2').glob('*.py'))
    registration = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        phase1_commit='a3b6bc4',phase1_test_seen=True,
        protocol_sha256=sha256(ROOT/'PROTOCOL.md'),plan_sha256=sha256(ROOT/'plan.json'),
        code_sha256={str(p).replace('\\','/'):sha256(p) for p in code},
        manifest_sha256={str(p).replace('\\','/'):sha256(p) for p in target.glob('*.csv')},
        baseline_commit='cdbf4e66d7e57d9b6a2657b6024716212b8d0afa',
        baseline_sha256=sha256('vendor/ptbxl_benchmark/code/models/xresnet1d.py'))
    save_json(ROOT/'registration.json',registration)
    verify_phase1()
    print('Registered 33 initial runs + 12 selected-config repeats; phase one sealed.')


if __name__ == '__main__':
    main()
