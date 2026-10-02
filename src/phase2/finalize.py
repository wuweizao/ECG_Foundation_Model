"""Run only after the registered training queue has completed."""
import subprocess
import sys
import argparse
from pathlib import Path
from filelock import FileLock
from src.utils import save_json
from .register import verify_phase1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for-training',action='store_true')
    args=parser.parse_args()
    if args.wait_for_training:
        print('Waiting for the active registered training queue to release its lock.',flush=True)
        with FileLock('runs/phase2/.suite.lock',timeout=-1,poll_interval=1):
            pass
    if not Path('results/phase2/training_manifest.json').exists():
        raise RuntimeError('Training queue has not completed')
    steps=[('probe_verification_and_locked_test','src.phase2.evaluate'),
           ('independent_audit_and_report','src.phase2.report'),
           ('paired_patient_bootstrap','src.phase2.bootstrap')]
    for stage,module in steps:
        save_json('results/phase2/status.json',dict(stage=stage))
        subprocess.run([sys.executable,'-m',module],check=True)
    verify_phase1(full=True)
    save_json('results/phase2/status.json',dict(stage='complete',training_attempts=45,
              selected_new_models=33,comparison_rows=45,phase1_unchanged=True,
              independent_external_validation=False))


if __name__=='__main__':
    main()
