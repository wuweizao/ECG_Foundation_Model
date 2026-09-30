"""Report convergence limits without altering the predeclared test protocol."""
import json
from pathlib import Path

import pandas as pd

from .plan import conditions,run_name


def main():
    rows = []
    for condition in conditions():
        folder = Path('runs') / run_name(condition)
        history = pd.read_csv(folder / 'history.csv')
        config = json.loads((folder/'config.json').read_text())['config']
        best = history.loc[history.validation_macro_auroc.idxmax()]
        complete = json.loads((folder/'complete.json').read_text())
        rows.append(dict(**condition,epochs_completed=len(history),best_epoch=int(best.epoch),
                         best_validation_macro_auroc=best.validation_macro_auroc,
                         reached_epoch_limit=len(history)==config['epochs'],
                         best_in_last_three_epochs=best.epoch>len(history)-3,
                         training_seconds=complete['duration_seconds']))
    pd.DataFrame(rows).to_csv('results/training_audit.csv',index=False)


if __name__ == '__main__':
    main()
