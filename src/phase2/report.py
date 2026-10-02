"""Aggregate-only phase-two tables, audit and figures; never change selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from src.utils import save_json

OUT=Path('results/phase2')
FIG=Path('figures/phase2')
COLORS={'scratch':'#D08728','pretrained':'#3176B6','linear_probe':'#8B8740',
        'random_probe':'#8E6EAA','xresnet1d101':'#397E70'}
NAMES={'scratch':'Scratch Net1D','pretrained':'Pretrained Net1D',
       'linear_probe':'Pretrained frozen','random_probe':'Random frozen','xresnet1d101':'xResNet1D-101'}


def independent_metrics(y,p,t):
    auc,ap=[],[]
    for j in range(y.shape[1]):
        positive=y[:,j].astype(bool)
        npos=positive.sum()
        auc.append((rankdata(p[:,j])[positive].sum()-npos*(npos+1)/2)/(npos*(len(y)-npos)))
        order=np.argsort(-p[:,j],kind='stable')
        ends=np.r_[np.flatnonzero(np.diff(p[order,j])!=0),len(y)-1]
        tp=np.cumsum(y[order,j])[ends]
        ap.append((np.diff(np.r_[0,tp])*tp/(ends+1)).sum()/npos)
    pred=p>=np.asarray(t)
    tp=((y==1)&pred).sum(0)
    fp=((y==0)&pred).sum(0)
    fn=((y==1)&~pred).sum(0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(y.shape[1]),where=2*tp+fp+fn>0)
    return dict(macro_auroc=float(np.mean(auc)),macro_auprc=float(np.mean(ap)),
                macro_f1=float(f1.mean()),micro_f1=float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum())),
                macro_sensitivity=float((tp/y.sum(0)).mean()),
                macro_specificity=float((((y==0)&~pred).sum(0)/(y==0).sum(0)).mean()))


def audit(df):
    if len(df)!=45 or df[['group','model','fraction','seed']].duplicated().any():
        raise ValueError('Missing/duplicated selected result rows')
    errors=[]
    for row in df.itertuples():
        run=Path(row.run)
        with np.load(run/'test_predictions.npz') as d:
            independent=independent_metrics(d['y'],d['p'],json.loads((run/'thresholds.json').read_text()))
        errors.extend(abs(value-getattr(row,k)) for k,value in independent.items())
    if max(errors)>1e-7:
        raise AssertionError(f'Independent metrics mismatch: {max(errors)}')
    save_json(OUT/'analysis_validation.json',dict(rows_checked=len(df),unique_prediction_files=df.run.nunique(),
              max_absolute_discrepancy=float(max(errors)),rank_auroc=True,stepwise_ap=True,confusion_metrics=True))


def training_audit():
    specs=json.loads((OUT/'training_manifest.json').read_text())
    rows=[]
    for s in specs:
        run=Path('runs/phase2')/s['id']
        if (run/'failed.json').exists() and not (run/'complete.json').exists():
            rows.append(dict(**s,status='numerical_failure'))
            continue
        history=pd.read_csv(run/'history.csv')
        config=json.loads((run/'config.json').read_text())
        frame=pd.read_csv(f'runs/phase2/manifests/train_f{s["fraction"]:g}_s{s["seed"]}.csv')
        best=int(history.loc[history.validation_macro_auroc.idxmax(),'epoch'])
        rows.append(dict(**s,status='complete',epochs_completed=len(history),best_epoch=best,
                         best_validation_macro_auroc=float(history.validation_macro_auroc.max()),
                         reached_epoch_limit=len(history)==s['epochs'],
                         best_in_last_three_epochs=best>=len(history)-2,
                         training_patients=frame.patient_id.nunique(),training_records=len(frame),
                         examples_seen=len(frame)*len(history),
                         optimizer_steps=int(np.ceil(len(frame)/(s['batch_size']*s['accumulation_steps'])))*len(history),
                         parameters=config['parameters'],duration_seconds=float(history.elapsed_seconds.iloc[-1])))
    pd.DataFrame(rows).to_csv(OUT/'training_audit.csv',index=False)


def figures(summary):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for group,title in [('fixed_full','Full-label replication: original fixed recipe'),
                        ('frozen','Frozen representations: matched head-training budget'),
                        ('equal_search','Equal-search comparison: 3 learning rates per model')]:
        part=summary[summary.group==group]
        fractions=sorted(part.fraction.unique())
        models=list(part.model.unique())
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
        for ax,metric in zip(axes,['auroc','auprc']):
            for i,model in enumerate(models):
                sub=part[part.model==model].sort_values('fraction')
                x=np.arange(len(fractions))+(i-(len(models)-1)/2)*.15
                ax.errorbar(x,sub[f'{metric}_mean'],yerr=sub[f'{metric}_sd'],
                            fmt=['s','o','D'][i],capsize=4,color=COLORS[model],label=NAMES[model])
            ax.set(xticks=np.arange(len(fractions)),xticklabels=[f'{f:.0%}' for f in fractions],
                   ylim=(0,1.02),xlabel='Labeled training patients',ylabel=f'Macro {metric.upper()}')
            ax.grid(axis='y',alpha=.2)
        axes[1].legend(frameon=False,loc='lower right')
        fig.suptitle(title,x=.08,ha='left')
        fig.text(.08,.89,'PTB-XL fold 10 (previously examined) | mean ± sample SD, 3 seeds; full validation labels',fontsize=9)
        fig.subplots_adjust(top=.8,bottom=.15,wspace=.16)
        for ext in ['png','pdf']:
            fig.savefig(FIG/f'{group}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)


def main():
    df=pd.read_csv(OUT/'metrics.csv')
    audit(df)
    training_audit()
    summary=df.groupby(['group','model','fraction']).agg(n_seeds=('seed','count'),
        auroc_mean=('macro_auroc','mean'),auroc_sd=('macro_auroc','std'),
        auprc_mean=('macro_auprc','mean'),auprc_sd=('macro_auprc','std'),
        f1_mean=('macro_f1','mean'),f1_sd=('macro_f1','std')).reset_index()
    if not (summary.n_seeds==3).all():
        raise ValueError('Every selected comparison requires 3 seeds')
    summary.to_csv(OUT/'summary.csv',index=False)
    figures(summary)
    lines=['# 第二阶段结果', '',
           '本阶段在看过第一阶段测试结果后登记，属于扩展研究，不是独立确认或外部验证。',
           '原始第一阶段结果保持封存；下面每项均为三个 seed 的均值 ± 样本标准差。', '',
           '| 比较 | 模型 | 训练标注比例 | Macro AUROC | Macro AUPRC |',
           '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.group} | {r.model} | {r.fraction:.0%} | {r.auroc_mean:.4f} ± {r.auroc_sd:.4f} | {r.auprc_mean:.4f} ± {r.auprc_sd:.4f} |')
    lines+=['', '## 学习率选择', '',
            '每个模型在 1% 和 100% 下分别以 seed 42 搜索三个学习率，仅按验证 AUROC 选择；seed 52/62 固定使用所选学习率。', '',
            '| 模型/比例 | 所选 LR |', '|---|---:|']
    selection=json.loads((OUT/'selection.json').read_text())
    for k,v in selection.items():
        lines.append(f'| {k} | {v["lr"]:g} |')
    lines+=['', '## 解读边界', '',
            '- 固定配置与等搜索预算回答不同问题，不能把两个表混为一个控制变量实验。',
            '- 搜索机会、训练上限相同，不代表 FLOPs/耗时相同，也不代表全局最优。',
            '- 冻结模型头部未做独立优化器搜索，不能从其结果推断表征的最优线性可分性。',
            '- 完整验证标签仍参与选择，标注比例仅指训练患者。',
            '- 样本 SD 混合了低标注子集与初始化变异；不能分离二者。',
            '- 患者配对 bootstrap 条件于已训练模型，不能当作训练不确定性的替代。',
            '- 原 PTB-XL 测试集已被第一阶段查看；本阶段没有新增外部泛化证据。']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Independent phase-two metric checks and aggregate figures completed.',flush=True)


if __name__=='__main__':
    main()
