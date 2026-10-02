"""Read-only progress summary for human supervision."""
import json
from pathlib import Path
import pandas as pd


def main():
    path=Path('results/phase2/status.json')
    if not path.exists():
        print('Phase two has not started.')
        return
    status=json.loads(path.read_text())
    print(json.dumps(status,ensure_ascii=False,indent=2))
    for name in status.get('active',[]):
        history=Path('runs/phase2')/name/'history.csv'
        if history.exists():
            data=pd.read_csv(history)
            if len(data):
                row=data.iloc[-1]
                print(f'{name}: epoch {int(row.epoch)}, validation AUROC {row.validation_macro_auroc:.4f}, '
                      f'best {data.validation_macro_auroc.max():.4f}, elapsed {row.elapsed_seconds/60:.1f} minutes')
        else:
            print(f'{name}: initializing model or extracting frozen features')


if __name__=='__main__':
    main()
