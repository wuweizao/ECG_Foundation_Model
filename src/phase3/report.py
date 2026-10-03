"""Independent metrics audit and explicit annotation-budget plots."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import save_json
from src.phase2.report import independent_metrics

OUT=Path('results/phase3')
FIG=Path('figures/phase3')


def main():
    df=pd.read_csv(OUT/'metrics.csv')
    if len(df)!=60 or df.run.nunique()!=54 or df[['policy','model','fraction','seed']].duplicated().any():
        raise ValueError('Incomplete/duplicate phase-three results')
    errors=[]
    for r in df.itertuples():
        run=Path(r.run)
        with np.load(run/'test_predictions.npz') as d:
            checked=independent_metrics(d['y'],d['p'],json.loads((run/'thresholds.json').read_text()))
            fixed=independent_metrics(d['y'],d['p'],[.5]*5)
        errors.extend(abs(v-getattr(r,k)) for k,v in checked.items())
        errors.append(abs(fixed['macro_f1']-r.macro_f1_fixed05))
    if max(errors)>1e-7:
        raise AssertionError('Independent metric discrepancy')
    save_json(OUT/'analysis_validation.json',dict(comparison_rows=60,unique_models=54,
              independent_rank_auc_ap_confusion=True,max_absolute_discrepancy=float(max(errors))))
    summary=df.groupby(['policy','model','fraction']).agg(n_seeds=('seed','count'),
        training_patients=('training_patients','first'),validation_patients=('validation_patients','first'),
        total_labeled_patients=('total_labeled_patients','first'),
        auroc_mean=('macro_auroc','mean'),auroc_sd=('macro_auroc','std'),
        auprc_mean=('macro_auprc','mean'),auprc_sd=('macro_auprc','std'),
        f1_mean=('macro_f1','mean'),f1_sd=('macro_f1','std'),f1_fixed05_mean=('macro_f1_fixed05','mean')).reset_index()
    if not (summary.n_seeds==3).all():
        raise ValueError('Three seeds required for every summary')
    summary.to_csv(OUT/'summary.csv',index=False)
    audit=[]
    for s in json.loads(Path('experiments/phase3/plan.json').read_text()):
        run=Path('runs/phase3')/s['id']
        history=pd.read_csv(run/'history.csv')
        complete=json.loads((run/'complete.json').read_text())
        audit.append(dict(id=s['id'],policy=s['policy'],model=s['family'],fraction=s['fraction'],seed=s['seed'],
                          best_epoch=int(history.loc[history.validation_bce.idxmin(),'epoch']),
                          epochs_completed=len(history),reached_epoch_limit=len(history)==s['epochs'],
                          best_validation_bce=complete['best_validation_bce'],fallback_classes=complete['fallback_classes'],
                          examples_seen=len(pd.read_csv(s['train_manifest']))*len(history),
                          duration_seconds=complete['duration_seconds']))
    pd.DataFrame(audit).to_csv(OUT/'training_audit.csv',index=False)
    FIG.mkdir(parents=True,exist_ok=True)
    colors={'scratch':'#D08728','pretrained':'#3176B6'}
    for actual,name in [(True,'actual_annotation_budget'),(False,'matched_training_patients')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,metric in zip(axes,['auroc','auprc']):
            for model in ['scratch','pretrained']:
                for policy,marker,ls in [('budgeted','o','-'),('full_validation_control','s','--')]:
                    sub=summary[(summary.model==model)&(summary.policy==policy)].sort_values('fraction')
                    x=sub.total_labeled_patients if actual else sub.fraction*100
                    ax.errorbar(x,sub[f'{metric}_mean'],yerr=sub[f'{metric}_sd'],fmt=marker,linestyle=ls,
                                capsize=3,color=colors[model],label=f'{model}; '+('budgeted val' if policy=='budgeted' else 'full val'))
            ax.set(xscale='log',ylim=(0,1.02),ylabel=f'Macro {metric.upper()}',
                   xlabel='Actual labeled train + validation patients' if actual else 'Nominal budget (%) / matched training subsets')
            if not actual:
                ax.set_xticks([1,5,10,25,100],labels=['1','5','10','25','100'])
            ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        axes[1].legend(frameon=False,loc='lower right',fontsize=9)
        fig.suptitle('Validation annotation budget: actual label use' if actual else 'Validation restriction with the same training patients',x=.08,ha='left')
        fig.text(.08,.9,'Previously examined PTB-XL fold 10 | mean ± SD, 3 seeds | 100% endpoint shared | fixed recipe, BCE selection',fontsize=9)
        fig.subplots_adjust(top=.8,bottom=.16,wspace=.2)
        for ext in ['png','pdf']:
            fig.savefig(FIG/f'{name}.{ext}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    lines=['# 第三阶段：训练与验证合计标注预算', '',
           '这是已查看前两阶段测试结果后的回顾性预算模拟，不是独立确认、前瞻性标注成本或外部验证。',
           '两种策略使用相同训练患者、同一固定学习率与验证 BCE 选择规则；完整验证对照使用更多标签。', '',
           '| 验证策略 | 模型 | 名义预算 | 训练患者 | 验证患者 | 合计 | AUROC | AUPRC |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.policy} | {r.model} | {r.fraction:.0%} | {r.training_patients} | {r.validation_patients} | {r.total_labeled_patients} | {r.auroc_mean:.4f} ± {r.auroc_sd:.4f} | {r.auprc_mean:.4f} ± {r.auprc_sd:.4f} |')
    lines+=['', '100% 两种策略共享同一训练，统计表复用终点，不能作为两组独立重复。',
            '标签预算指本阶段算法使用的患者标签数，不包含测试标注、预训练成本或历史调研/既往实验的标签信息。',
            '固定配置不是模型性能上限。19 人等小验证子集可能有类别缺失或较大选择噪声；未重抽样本，缺类阈值固定0.5。',
            'bootstrap 为已拟合模型条件下的患者配对描述性区间，未作多重比较显著性或等效性声明。']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Phase-three independent metric audit and aggregate report passed.',flush=True)


if __name__=='__main__':
    main()
