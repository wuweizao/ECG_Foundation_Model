"""Finish the locked evaluation and all available result artifacts after training."""
import argparse
from pathlib import Path
import subprocess
import sys

from .utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-replicates',type=int,default=2000)
    args = parser.parse_args()
    steps = [
        ('training_audit',['-m','src.training_audit']),
        ('locked_test',['-m','src.evaluate','--freeze-and-evaluate']),
        ('main_figures',['-m','src.report']),
        ('robustness',['-m','src.robustness']),
        ('calibration',['-m','src.calibration']),
        ('extension_figures',['-m','src.extension_figures']),
        ('saliency',['-m','src.explain']),
        ('paired_patient_bootstrap',['-m','src.bootstrap','--replicates',str(args.bootstrap_replicates)]),
        ('independent_validation',['scripts/validate_results.py']),
        ('notebook',['scripts/build_notebook.py']),
        ('research_readout',['scripts/write_findings.py']),
    ]
    for stage,command in steps:
        save_json('results/status.json',dict(stage=stage))
        try:
            subprocess.run([sys.executable]+command,check=True)
        except Exception as error:
            save_json('results/status.json',dict(stage='failed',failed_step=stage,error=type(error).__name__))
            raise
    save_json('results/status.json',dict(stage='complete',main_runs=26,linear_probes=7,convergence_audits=2))


if __name__ == '__main__':
    main()
