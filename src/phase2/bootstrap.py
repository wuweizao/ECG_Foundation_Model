"""Patient-paired comparisons for registered phase-two contrasts."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from src.bootstrap import patient_indices

OUT=Path('results/phase2')


def pair(job):
    group,fraction,seed,a_name,b_name,a_path,b_path=job
    with np.load(Path(a_path)/'test_predictions.npz') as d:
        a={k:d[k].copy() for k in d.files}
    with np.load(Path(b_path)/'test_predictions.npz') as d:
        b={k:d[k].copy() for k in d.files}
    for k in ['y','ecg_id','patient_id']:
        if not np.array_equal(a[k],b[k]):
            raise ValueError('Nonpaired observations')
    groups={pid:np.flatnonzero(a['patient_id']==pid) for pid in np.unique(a['patient_id'])}
    rng=np.random.default_rng(seed)
    funcs={'macro_auroc':roc_auc_score,'macro_auprc':average_precision_score}
    samples={k:[] for k in funcs}
    for _ in range(2000):
        idx=patient_indices(a['patient_id'],rng,groups)
        y=a['y'][idx]
        if any(len(np.unique(y[:,j]))!=2 for j in range(5)):
            continue
        for key,fn in funcs.items():
            samples[key].append(fn(y,b['p'][idx],average='macro')-fn(y,a['p'][idx],average='macro'))
    rows=[]
    for key,fn in funcs.items():
        low,high=np.quantile(samples[key],[.025,.975])
        rows.append(dict(group=group,fraction=fraction,seed=seed,reference=a_name,comparison=b_name,
                         metric=key,delta=fn(a['y'],b['p'],average='macro')-fn(a['y'],a['p'],average='macro'),
                         ci_low=low,ci_high=high,replicates=2000,valid_replicates=len(samples[key]),
                         scope='conditional_on_fitted_pair',unit='patient'))
    return rows


def main():
    data=pd.read_csv(OUT/'metrics.csv')
    jobs=[]
    for (group,fraction,seed),sub in data.groupby(['group','fraction','seed']):
        contrasts={'fixed_full':[('scratch','pretrained')],
                   'frozen':[('random_probe','linear_probe')],
                   'equal_search':[('scratch','pretrained'),('scratch','xresnet1d101'),('xresnet1d101','pretrained')]}[group]
        for a,b in contrasts:
            jobs.append((group,fraction,seed,a,b,sub[sub.model==a].run.iloc[0],sub[sub.model==b].run.iloc[0]))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(pair,jobs):
            rows.extend(result)
            print(f'Bootstrap {result[0]["group"]} {result[0]["fraction"]} seed {result[0]["seed"]}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'bootstrap.csv',index=False)


if __name__=='__main__':
    main()
