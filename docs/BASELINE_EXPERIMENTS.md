# 基线对比实验方案

更新：2026-10-15。

实时应用流程见 [工作流](WORKFLOW.md)。本文总结本地静态研究设计；基线训练尚未完成，没有可报告的精度提升。

## 1. 当前状态与适用范围

本地已完成 FER2013 像素重复审计，并保留 `ferlab` 实验代码，但尚无已完成的训练对比结果。当前 GitHub 仓库仅保存实时核心代码及文档，**不包含 `ferlab`**；下文命令只能在保留完整实验框架的本地项目运行。

## 2. 研究问题与统一设置

研究假设：使用全局置信度降低噪声样本权重时，少数类的正确难样本也可能被压低；混合全局排序与类内排序，可能在抑制错误标签的同时保留少数类学习信号。

这是假设，不是已验证创新。教师预测置信度不能直接当作标签正确概率；类内排序也可能保留系统性错标。

### 数据与公平设置

| 项目 | 设置 |
| --- | --- |
| 数据 | FER2013 图片版，48×48 灰度，七分类 |
| 标准划分 | train 28,709 / val 3,589 / test 3,589 |
| 排重划分 | val 3,309 / test 3,275；排除先前划分中像素完全一致的样本 |
| 骨干 | SmallInputResNet18，基础通道 16，随机初始化 |
| 首轮预算 | 12 epochs，warmup 3，batch size 256 |
| 优化 | AdamW，学习率 0.001，weight decay 0.0001，余弦衰减 |
| 随机种子 | 42、2026、3407，各方法使用配对种子 |
| 增强 | 双视图，随机水平翻转、轻微平移、对比度扰动 |
| 选择模型 | 排重验证集 macro-F1 最高的 checkpoint |

这是小输入、窄通道残差网络先导实验，不是标准 ImageNet 预训练 ResNet-18，也不是 SCN 等论文的完整复现。所有方法共享初始化规则、样本顺序、增强和优化器；仍需报告时间成本差异，不能将同 epoch 数理解为同 FLOPs。

像素审计发现 557 个跨划分重复组，57 个标签冲突重复组；这些数量来自当前本地数据版本。排重不能消除近重复、划分内部重复或人物身份泄漏。

## 3. 对比组与消融

| 方法 | 类别加权 | 双视图 JS 一致性 | EMA 教师可靠性 | 用途 |
| --- | --- | --- | --- | --- |
| `ce` | 无 | 无 | 无 | 普通交叉熵基线；仍使用同样双视图训练 |
| `balanced` | 有 | 无 | 无 | 检验类别不平衡处理的收益 |
| `consistent` | 有 | 有 | 无 | 检验一致性正则的增量收益 |
| `global` | 有 | 有 | 全局排序，α=0 | 可靠性加权基线 |
| `classwise` | 有 | 有 | 全局／类内混合，α=0.5 | 候选方法 |
| `no_floor` | 有 | 有 | 同混合排序，权重下限=0 | 去除权重下限的消融 |

首轮运行前五种方法，共 **5×3=15 次训练**；加入 `no_floor` 后共 18 次。`no_floor` 不在默认 suite 列表中，需显式指定。

类别权重由训练标签计数的平方根倒数得到，按类别均值归一化后裁剪到 `[0.5, 3]`。教师参数 EMA 衰减为 0.99；每轮结束用教师对训练集推理，计算：

```text
margin_i = p_teacher(observed_label_i) - max(p_teacher(other_labels))
r_i = (1 - α) × global_percentile(margin_i)
      + α × within_observed_class_percentile(margin_i)
w_i = floor + (1 - floor) × r_i
```

默认 floor=0.2。并列分数取平均秩，避免文件顺序决定权重。权重用于下一轮，预热后渐进启用；可靠性计算仅使用训练集，不引入验证或测试标签。

两视图交叉熵取平均后，按类别权重和渐进样本权重加权并归一化；一致性方法额外加 `0.1 × ramp × JS`。`global` 与 `classwise` 的主要差异应只保留 α，以便解释类内排序的作用。

## 4. 执行顺序与命令

以下命令仅供**本地完整项目**使用；当前 GitHub 核心版本不包含 `ferlab`。训练另需 NumPy、Pillow 和合适版本的 PyTorch。

```powershell
# 1. 准备指定格式的 FER2013 JPG ZIP，生成数据和重复审计
python -m ferlab.prepare --zip archive.zip --output data/fer2013

# 2. 可选：CPU 单轮检查，不作为精度结果
python -m ferlab.train --data data/fer2013 --output runs/check --method ce --epochs 1 --batch-size 32 --device cpu

# 3. 首轮 15 次对照；suite 默认 CUDA
python -m ferlab.suite --output runs/pilot --epochs 12 --warmup 3 --seeds 42 2026 3407

# 4. 预先另定方案：包含 no_floor 的 18 次实验
python -m ferlab.suite --output runs/ablation --epochs 12 --warmup 3 --methods ce balanced consistent global classwise no_floor --seeds 42 2026 3407
```

准备器要求七类父目录、Training/PublicTest/PrivateTest 文件名前缀及指定数量的 48×48 灰度 JPG；不直接接受任意 FER2013 CSV。

`suite` 冻结 `plan.json`，顺序训练，全部训练结束后评估测试 checkpoint，再生成报告。单次 `train` 可用 `--device cpu`，但 `suite` 当前没有 CPU 参数。修改方案使用新输出目录；不要用测试结果反复选择超参数。

## 5. 评估与结果记录

主要比较排重测试集 **macro-F1**，同时报告标准测试 accuracy / macro-F1、balanced accuracy、各类 recall/F1、混淆矩阵以及 NLL、Brier、15-bin ECE。

同一种子计算 `classwise − global` 的配对差值，报告各次方向及均值；跨种子结果用均值 ± 样本标准差。三个种子不足以单独支持强显著性结论，必要时增加种子及独立数据集验证。

| 方法 | 标准测试 accuracy | 排重测试 macro-F1 | 少数类 recall | 训练时间/峰值显存 |
| --- | --- | --- | --- | --- |
| ce | 未运行 | 未运行 | 未运行 | 未运行 |
| balanced | 未运行 | 未运行 | 未运行 | 未运行 |
| consistent | 未运行 | 未运行 | 未运行 | 未运行 |
| global | 未运行 | 未运行 | 未运行 | 未运行 |
| classwise | 未运行 | 未运行 | 未运行 | 未运行 |
| no_floor | 未运行 | 未运行 | 未运行 | 未运行 |

应保留 `config.json`、数据/标签哈希、`history.jsonl`、`best.pt`、`last.pt`、`result.json`、预测分数与噪声审计。完整 suite 才生成 `results.csv` 和实验报告。当前没有这些已完成训练结果，表格不填入预计值或模拟精度。

后续可预注册 20% / 40% 训练标签随机错误实验；0% 额外噪声不代表原标签全正确。需要人工复核难样本、检查少数类保留率并扩展跨数据集评估，才能支撑“保护正确难样本”的机制结论。

## 6. 两条路线如何衔接

实时八分类含轻蔑，FER2013 七分类不含轻蔑，标签顺序也不同；不能直接按索引比较，不能把 FER2013 实验分数当成当前 EmotiEffNet 的准确率。

若候选方法在静态评测中稳定改善，下一步才是定义类别映射、导出模型、验证预处理一致性，再进行真实摄像头测试。部署侧另测端到端延迟、P95 延迟、漏检、遮挡、多人交叉、标签抖动和阈值覆盖率。

此前 12 个静态样本的检测与分类耗时中位数约 24.07 ms、P95 25.68 ms，只是本机小样本功能记录，不包含完整 GUI 显示链路，也不能证明稳定 FPS 或识别准确率。

## 7. 本次更新范围

本次仅整理实验方案，不启动训练，不上传训练代码、数据和权重。所有“未运行”项保留为空缺状态，不填入预计值或模拟精度。
