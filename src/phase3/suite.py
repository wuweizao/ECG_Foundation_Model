"""Bounded queue and optional automatic locked evaluation after all runs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from filelock import FileLock
from src.utils import save_json,sha256
from .train import guard

ROOT=Path('runs/phase3')
OUT=Path('results/phase3')


def queue(parallel):
    guard()
    plan=json.loads(Path('experiments/phase3/plan.json').read_text())
    active={}
    for name in ['specs','logs']:
        (ROOT/name).mkdir(exist_ok=True)
    try:
        while True:
            for name,(process,stream) in list(active.items()):
                if process.poll() is None:
                    continue
                stream.close()
                del active[name]
                if process.returncode!=0:
                    raise RuntimeError(f'Training failed: {name}; inspect log; no automatic replacement')
            complete=[]
            for s in plan:
                run=ROOT/s['id']
                done=run/'complete.json'
                if done.exists():
                    complete.append(s['id'])
            pending=[s for s in plan if s['id'] not in complete and s['id'] not in active]
            while pending and len(active)<parallel:
                s=pending.pop(0)
                if (ROOT/s['id']/'failed.json').exists():
                    raise RuntimeError(f'Recorded failure requires explicit repair: {s["id"]}')
                path=ROOT/'specs'/f'{s["id"]}.json'
                save_json(path,s)
                stream=open(ROOT/'logs'/f'{s["id"]}.log','a',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-m','src.phase3.train','--spec',str(path)],stdout=stream,stderr=subprocess.STDOUT)
                active[s['id']]=(process,stream)
                print(f'Start {s["id"]} pid={process.pid}',flush=True)
            save_json(OUT/'status.json',dict(stage='training',completed=len(complete),expected=54,active=list(active)))
            if not active and len(complete)==54:
                save_json(OUT/'training_manifest.json',plan)
                save_json(OUT/'status.json',dict(stage='training_complete',completed=54,expected=54))
                return
            time.sleep(5)
    except Exception as exc:
        save_json(OUT/'status.json',dict(stage='failed',reason=str(exc),active=list(active)))
        raise
    finally:
        for process,stream in active.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
            stream.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parallel',type=int,choices=[1,2],default=2)
    p.add_argument('--finalize',action='store_true')
    a=p.parse_args()
    with FileLock(str(ROOT/'.suite.lock'),timeout=0):
        queue(a.parallel)
        if a.finalize:
            for stage,module in [('locked_test','src.phase3.evaluate'),('report','src.phase3.report'),('bootstrap','src.phase3.bootstrap')]:
                save_json(OUT/'status.json',dict(stage=stage,completed=54,expected=54))
                try:
                    subprocess.run([sys.executable,'-m',module],check=True)
                except Exception as exc:
                    save_json(OUT/'status.json',dict(stage='failed',failed_stage=stage,reason=str(exc)))
                    raise
            save_json(OUT/'status.json',dict(stage='complete',new_training_runs=54,comparison_rows=60,
                      test_previously_seen=True,independent_external_validation=False))


if __name__=='__main__':
    main()
