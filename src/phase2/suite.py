"""Bounded local process queue, with validation-only LR selection."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from filelock import FileLock
from src.utils import save_json, sha256
from .register import spec as make_spec
from .train import guard

ROOT=Path('runs/phase2')
RESULTS=Path('results/phase2')


def choose(candidates):
    valid=[x for x in candidates if x.get('score') is not None]
    if not valid:
        raise RuntimeError('No successful candidate in a registered comparison')
    return sorted(valid,key=lambda x:(-x['score'],x['spec']['lr']))[0]


def run_queue():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parallel',type=int,default=2,choices=[1,2])
    args=parser.parse_args()
    guard()
    plan=json.loads(Path('experiments/phase2/plan.json').read_text())
    initial=plan['initial_runs']
    all_specs=list(initial)
    active={}
    selected={}
    spec_root=ROOT/'specs'
    spec_root.mkdir(exist_ok=True)
    log_root=ROOT/'logs'
    log_root.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True,parents=True)
    try:
        while True:
            for name,(process,stream,spec) in list(active.items()):
                code=process.poll()
                if code is None:
                    continue
                stream.close()
                del active[name]
                if code!=0 and not (ROOT/name/'failed.json').exists():
                    raise RuntimeError(f'Infrastructure failure in {name}; inspect its log. No silent retry.')
                if code!=0 and spec['group']!='search':
                    raise RuntimeError(f'Required non-search run failed: {name}')
            for fraction in [.01,1.]:
                for family in ['scratch','pretrained','xresnet1d101']:
                    key=f'{family}_f{fraction:g}'
                    if key in selected:
                        continue
                    trials=[s for s in initial if s['group']=='search' and s['family']==family and s['fraction']==fraction]
                    if not all((ROOT/s['id']/'complete.json').exists() or (ROOT/s['id']/'failed.json').exists() for s in trials):
                        continue
                    candidates=[]
                    for s in trials:
                        done=ROOT/s['id']/'complete.json'
                        candidates.append(dict(spec=s,score=json.loads(done.read_text())['best_validation_macro_auroc'] if done.exists() else None))
                    winner=choose(candidates)
                    selected[key]=dict(winner=winner['spec']['id'],lr=winner['spec']['lr'],candidates=candidates)
                    for seed in [52,62]:
                        all_specs.append(make_spec('repeat',family,fraction,seed,winner['spec']['lr'],60,12))
                    save_json(RESULTS/'selection.json',selected)
            finished=[s for s in all_specs if (ROOT/s['id']/'complete.json').exists()]
            failed=[s for s in all_specs if (ROOT/s['id']/'failed.json').exists()]
            pending=[s for s in all_specs if s['id'] not in active and s not in finished and s not in failed]
            while pending and len(active)<args.parallel:
                s=pending.pop(0)
                path=spec_root/f'{s["id"]}.json'
                save_json(path,s)
                stream=open(log_root/f'{s["id"]}.log','a',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-m','src.phase2.train','--spec',str(path)],stdout=stream,stderr=subprocess.STDOUT)
                active[s['id']]=(process,stream,s)
                print(f'Start {s["id"]} pid={process.pid}',flush=True)
            save_json(RESULTS/'status.json',dict(stage='training',completed=len(finished),
                      failed=len(failed),expected=45,active=list(active),selected_groups=len(selected)))
            if not active and not pending:
                if len(all_specs)!=45 or len(selected)!=6:
                    raise RuntimeError('Incomplete registered queue')
                save_json(RESULTS/'training_manifest.json',all_specs)
                save_json(RESULTS/'status.json',dict(stage='training_complete',completed=len(finished),failed=len(failed),expected=45))
                print('All 45 registered training attempts accounted for.',flush=True)
                break
            time.sleep(5)
    finally:
        for process,stream,spec in active.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            stream.close()


def main():
    with FileLock(str(ROOT/'.suite.lock'),timeout=0):
        run_queue()


if __name__=='__main__':
    main()
