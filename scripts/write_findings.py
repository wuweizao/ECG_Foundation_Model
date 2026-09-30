"""Compose the final Chinese research readout entirely from validated artifacts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd


def main():
    summary = pd.read_csv(ROOT/'results/summary.csv')
    metrics = pd.read_csv(ROOT/'results/metrics.csv')
    robustness = pd.read_csv(ROOT/'results/robustness.csv')
    budget = pd.read_csv(ROOT/'results/convergence_audit.csv')
    training = pd.read_csv(ROOT/'results/training_audit.csv')
    gain = json.loads((ROOT/'results/label_efficiency_gain.json').read_text())
    check = json.loads((ROOT/'results/analysis_validation.json').read_text())
    assert check['runs_checked']==35
    def select(model,fraction):
        return summary[(summary.model==model)&(summary.fraction==fraction)].iloc[0]
    def value(row,metric):
        mean,sd = row[f'{metric}_mean'],row[f'{metric}_sd']
        return f'{mean:.4f} ± {sd:.4f}' if pd.notna(sd) else f'{mean:.4f}（单 seed）'
    wins = sum(select('pretrained',f).auroc_mean>select('scratch',f).auroc_mean for f in [.01,.05,.1,.25,1.])
    lines = ['# 实验结果与解释', '',
             f'完成 26 次主实验、7 次线性探测及 2 次训练预算敏感性对照。固定 30 轮上限下，预训练模型在 5 个已测标注比例中的 {wins} 个比例取得更高的平均 Macro AUROC。该比较使用同架构、同患者子集和同一验证/测试集。', '',
             '| 标注患者比例 | Scratch AUROC | Pretrained AUROC | Scratch AUPRC | Pretrained AUPRC |',
             '|---|---:|---:|---:|---:|']
    for fraction in [.01,.05,.1,.25,1.]:
        a,b = select('scratch',fraction),select('pretrained',fraction)
        lines.append(f'| {fraction:.0%} | {value(a,"auroc")} | {value(b,"auroc")} | {value(a,"auprc")} | {value(b,"auprc")} |')
    lines += ['', '1–25% 为三个 seed 的均值 ± 样本标准差；100% 为一个 seed。全部结果来自 PTB-XL 官方 fold 10，模型与分类阈值由 fold 9 决定。', '', '## 标注效率', '']
    fraction = gain['pretrained_min_tested_fraction']
    if fraction is None:
        lines.append('在本次已测比例中，预训练模型未达到 Scratch 100% 的 Macro AUROC，不能据此宣称标注节省。')
    elif fraction==1:
        lines.append('预训练模型仅在 100% 比例达到 Scratch 100% 的 Macro AUROC；当前离散网格没有显示标注节省。')
    else:
        lines.append(f'预训练模型在已测网格中最早于 {fraction:.0%} 标注患者比例达到 Scratch 100% 的 AUROC（{gain["scratch_100_auroc"]:.4f}），对应 PTB-XL 下游标注患者量减少 {1-fraction:.0%}、描述性效率倍数 {1/fraction:g}×。这不是统计等效检验，也不是在未测试比例上的插值结论；不计入基础模型原先的大规模临床标注成本。')
    a = budget[budget.model=='scratch'].iloc[0]
    b = budget[budget.model=='pretrained'].iloc[0]
    main_a = metrics[(metrics.model=='scratch')&(metrics.fraction==.01)&(metrics.seed==42)].iloc[0]
    main_b = metrics[(metrics.model=='pretrained')&(metrics.fraction==.01)&(metrics.seed==42)].iloc[0]
    lines += ['', '## 训练预算敏感性', '',
              f'在测试前登记的 1%、seed 42 对照中，30 轮上限时 Scratch / Pretrained AUROC 为 {main_a.macro_auroc:.4f} / {main_b.macro_auroc:.4f}；100 轮上限、patience 15 时为 {a.macro_auroc:.4f} / {b.macro_auroc:.4f}，实际分别训练 {int(a.epochs_completed)} / {int(b.epochs_completed)} 轮。',
              f'主实验/探测的 33 次训练中，{int(training.reached_epoch_limit.sum())} 次达到轮数上限，{int(training.best_in_last_three_epochs.sum())} 次的最佳验证轮次位于最后三轮。该审计必须与主曲线一起解释；不能把固定预算结果当作各模型充分调优后的性能上限。', '',
              '## 冻结编码器与完整微调', '',
              '| 标注患者比例 | Full fine-tuning AUROC | Linear probe AUROC |',
              '|---|---:|---:|']
    for fraction in [.05,.1,1.]:
        lines.append(f'| {fraction:.0%} | {value(select("pretrained",fraction),"auroc")} | {value(select("linear_probe",fraction),"auroc")} |')
    lines += ['', '## 噪声与校准', '',
              '以下增强比较固定为 10%、seed 42。噪声添加在预处理之后；两模型使用相同的逐记录扰动，不重新选择阈值。', '',
              '| 条件 | Scratch AUROC | Pretrained AUROC |', '|---|---:|---:|']
    for kind,level,label in [('clean',0,'Clean'),('gaussian',5,'Gaussian 5 dB'),('dropout',6,'Drop 6 leads')]:
        subset = robustness[(robustness.corruption==kind)&(robustness.level==level)]
        av = subset[subset.model=='scratch'].iloc[0].macro_auroc
        bv = subset[subset.model=='pretrained'].iloc[0].macro_auroc
        lines.append(f'| {label} | {av:.4f} | {bv:.4f} |')
    subset = metrics[(metrics.fraction==.1)&(metrics.seed==42)]
    a,b = subset[subset.model=='scratch'].iloc[0],subset[subset.model=='pretrained'].iloc[0]
    lines += ['', f'干净测试集 Brier Score：Scratch {a.brier:.4f}，Pretrained {b.brier:.4f}；Macro ECE：{a.macro_ece:.4f} / {b.macro_ece:.4f}。两者均越低越好。这些是当前标签与队列上的概率校准描述，不能据此推断临床部署可靠性。', '',
              '## 不确定性与边界', '',
              '- bootstrap 以患者为单位、在同 seed 的模型间配对，95% CI 条件于已拟合模型，不包含完整训练不确定性；结果见 `bootstrap.csv`。未作多重比较显著性声明。',
              '- PCA/UMAP 使用同一验证集、分别降维，仅说明可视化结构，不能代替分类性能证据。',
              '- Binary entropy 可能在噪声下下降；低熵不保证输入在分布内或预测正确。',
              '- 未进行严格外部医院/设备验证、第二基础模型或 ECG-text 零样本实验，不宣称这些能力已被证实。',
              '- 数据完整性、独立指标复算、一次同环境逐位复现及 notebook 执行记录均保存在 results 目录。']
    (ROOT/'results/FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote results/FINDINGS.md from validated result tables')


if __name__ == '__main__':
    main()
