# Label-Efficient Multi-Disease ECG Classification with a Pretrained Foundation Model

基于 ECG Foundation Model 的低标注多疾病分类与泛化能力评估。

本项目比较 **ECGFounder 预训练初始化与同架构随机初始化**在 PTB-XL 五类诊断超类多标签任务中的标注效率。模型输出为 5 个独立 sigmoid 概率，不是 softmax 五分类。

## 实验设计

- PTB-XL 1.0.3，原生 500 Hz、12 导联、10 秒。
- 官方 folds 1–8 / 9 / 10 分别用于训练 / 验证 / 测试；患者不重叠。
- 按训练患者抽取 1%、5%、10%、25%、100%；同 seed 的子集嵌套，两模型配对使用同一子集。
- 1–25% 使用 seeds 42/52/62；100% 使用 seed 42，共 **26 次主实验**。
- 5%、10%、100% 增加预训练编码器冻结的 linear probe，共 **7 次策略对照**。
- 主指标 Macro AUROC、Macro AUPRC；同时报告 Macro/Micro F1、Sensitivity、Specificity、逐类结果。
- 指标以 ECG 记录为单位计算；划分、标注比例和 bootstrap 的独立单位为患者，不把二者混用。
- checkpoint、早停和逐类 F1 阈值仅由验证集决定。33 次主实验/探测加 2 次预算敏感性对照全部完成后锁定配置、权重和阈值，再统一打开测试集。
- 预算敏感性对照：1%、seed 42，两模型同为最多 100 轮、patience 15。由验证日志触发并在测试前登记，单独报告，不替换 30 轮主曲线。

完整方法和边界见 [实验协议](experiments/PROTOCOL.md)。数据计数、类别分布和各子集哈希见 [数据审计](results/data_audit.json)。本项目没有使用模拟性能填充结果；正式指标由锁定后的实际预测生成。

## 快速复现

推荐 Python 3.10 和支持本机 GPU 的 PyTorch CUDA 版本；本次实际环境是 RTX 5070 Ti 16 GB，PyTorch 2.11.0+cu128。CPU 可用于检查代码，不推荐运行完整训练。

```powershell
conda create -n ecg_foundation python=3.10
conda activate ecg_foundation
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
./scripts/setup.ps1
Copy-Item configs/local.example.json configs/local.json
# 编辑 local.json 指向 PTB-XL 和官方权重。
python -m src.prepare --workers 8
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
python -m src.run_suite --train-only
python -m src.convergence
python -m src.evaluate --freeze-and-evaluate
python -m src.report
```

Linux/macOS 可手动 `git clone https://github.com/PKUDigitalHealth/ECGFounder.git vendor/ECGFounder`，并 checkout `04edac702b61c91face519774ddcc0cd712fef23`，然后 `pip install -r requirements.txt`。训练入口相同。

数据获取见 [data/README.md](data/README.md)。从 [官方 Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder) 下载 `12_lead_ECGFounder.pth`。应核对 SHA-256：`ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`。

单条件训练示例：

```powershell
python -m src.train --config configs/pretrained.yaml --fraction 0.1 --seed 42
```

同配置已完成的实验会核对清单和权重后跳过。中断但未完成的实验从固定 seed 重新运行，**不是从中途 optimizer 状态恢复**。测试集锁定后拒绝新增主实验训练。本机活动锁 `experiments/test_lock.json` 不进 Git；发布的审计副本是 `results/test_lock.json`。因此全新 clone 可直接重新训练，原工作目录则保持锁定。复现实验应在新副本中运行，保留原始发布结果，不能将新结果伪装为本次实验。

## 增强分析

```powershell
python -m src.representations
pip install -r requirements-analysis.txt
python -m src.representations --umap
python -m src.robustness
python -m src.calibration
python -m src.explain
python -m src.bootstrap --replicates 2000
```

- PCA：同一验证集 ECG 的随机初始化/预训练编码器表征；五个标签分别着色，保留多标签关系。仅作为定性展示。
- UMAP：相同记录、固定 seed、cosine 距离、30 邻居、min_dist=0.1；两模型分别降维，坐标不可直接对应，不按图形分离度推断分类性能。
- 噪声：在 **预处理后的输入**加入 20/10/5 dB 高斯噪声、0.2/0.5 Hz 漂移、1/3/6 导联缺失；两模型使用完全相同的逐记录扰动。不是采集设备原始噪声仿真。
- 校准：逐标签 Brier、等宽 10-bin ECE、reliability diagrams。多标签概率 0.8 表示约 80% 阳性事件率，不是“80% 整体预测正确”。
- bootstrap：按测试患者成簇有放回抽样，保留同一患者全部记录，计算每个匹配 seed 下 Pretrained − Scratch 的 95% 百分位 CI；不将训练 seed 的不确定性与测试抽样不确定性混为一谈。
- 输入敏感性：选择第一条 MI 阳性的验证记录，显示绝对值 gradient × input、预测概率和验证阈值；不挑选成功案例，也不将梯度视为因果解释。

## 输出

```text
configs/       # 两组主实验及 linear probe 配置
src/           # 数据、训练、锁定测试、统计和可视化
experiments/   # 预定协议、33-run 清单、测试锁
results/       # metrics / per_class / summary / bootstrap / audit
figures/       # PNG + PDF，均由真实结果生成
notebooks/     # 可从头执行的结果展示 notebook
tests/         # 患者隔离、标签、指标、扰动与抽样检查
runs/          # 本地权重、预测、逐 epoch 训练日志（不进 Git）
```

## 解释边界

1. 采用固定最大训练轮数、相同优化器和学习率的受控比较，不等于对每个模型单独调优后的最高可达性能。达到训练上限的情况单独记录。
2. 低标注 mean ± SD 为三个 seed 的样本标准差；100% 只有一个 seed，不画虚构误差条。
3. Label Efficiency Gain 使用离散已测比例：预训练平均 AUROC 达到 Scratch 100% 结果的最小比例。它是描述性指标，不是统计等效检验，也不插值伪造中间结果。
4. ECGFounder 使用大规模临床标注预训练，因此不能把本项目自动标为纯 self-supervised learning。官方论文将 PTB-XL 用作外部评估；本项目则是 PTB-XL 内部患者隔离的下游适配实验，不额外宣称未知医院泛化。
5. 第二基础模型、严格外部数据泛化和 ECG-text/zero-shot 未纳入本次核心协议。不得将同五类文本标签的监督对齐称作未见疾病零样本泛化。
6. 部分训练并行执行，耗时受调度和设备负载影响；不将日志 wall time 当作公平速度对比。训练样本暴露量和 optimizer step 数另行审计。

## 来源

- [ECGFounder 官方代码与预处理说明](https://github.com/PKUDigitalHealth/ECGFounder)
- [ECGFounder 论文](https://arxiv.org/abs/2410.04133)
- [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

第三方代码归属和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

逐记录波形、PCA/UMAP 散点和 saliency 示例保留在本地，不随仓库发布；相应代码可按以上命令重新生成。GitHub 图表仅发布汇总性能统计，不包含单条 ECG 或逐记录预测。
