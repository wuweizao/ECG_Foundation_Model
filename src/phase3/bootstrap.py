"""Registered initialization and validation-budget contrasts."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from src.phase2.bootstrap import pair


def main():
    root=Path('results/phase3')
    df=pd.read_csv(root/'metrics.csv')
    jobs=[]
    for (policy,fraction,seed),sub in df.groupby(['policy','fraction','seed']):
        if fraction==1 and policy=='full_validation_control':
            continue
        jobs.append((f'initialization_{policy}',fraction,seed,'scratch','pretrained',
                     sub[sub.model=='scratch'].run.iloc[0],sub[sub.model=='pretrained'].run.iloc[0]))
    for (model,fraction,seed),sub in df[df.fraction<1].groupby(['model','fraction','seed']):
        jobs.append((f'validation_budget_{model}',fraction,seed,'full_validation_control','budgeted',
                     sub[sub.policy=='full_validation_control'].run.iloc[0],sub[sub.policy=='budgeted'].run.iloc[0]))
    assert len(jobs)==51
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(pair,jobs):
            rows.extend(result)
            print(result[0]['group'],result[0]['fraction'],result[0]['seed'],flush=True)
    pd.DataFrame(rows).to_csv(root/'bootstrap.csv',index=False)


if __name__=='__main__':
    main()
