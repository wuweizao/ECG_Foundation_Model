import argparse
import subprocess
import sys
from pathlib import Path

from .plan import conditions, run_name
from .utils import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-only', action='store_true')
    a = p.parse_args()
    planned = conditions()
    save_json('experiments/plan.json', planned)
    for i, row in enumerate(planned):
        save_json('results/status.json', {'stage': 'training', 'current': row, 'index': i+1,
                                         'total': len(planned)})
        subprocess.run([sys.executable, '-m', 'src.train', '--config', f'configs/{row["model"]}.yaml',
                        '--fraction', str(row['fraction']), '--seed', str(row['seed'])], check=True)
    if not a.train_only:
        subprocess.run([sys.executable, '-m', 'src.evaluate', '--freeze-and-evaluate'], check=True)
        subprocess.run([sys.executable, '-m', 'src.report'], check=True)
    save_json('results/status.json', {'stage': 'trained' if a.train_only else 'complete', 'runs': len(planned)})


if __name__ == '__main__':
    main()
