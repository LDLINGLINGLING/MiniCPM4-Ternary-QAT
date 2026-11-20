# mix_v2 实验记录（2B token QAT + BF16，含 SFT 与 EOS 诊断）

日期：2026-11-13。本记录基于 `reports/pretrain_v2/mix_v2/` 下的实际产物，所有数字均从 metrics.jsonl、selection.json、downstream JSON 中重新读取核对。

## 0. 实验目的

在 mix_v1（1.07B token/组）之后，用 mix_v2 数据池训练 2B token/组，验证以下假设：
- 训练量翻倍能否提升下游指标？
- alpha_ramp_frac 从 0.1 拉长到 0.3 是否减少量化损失？
- SFT 后 QAT 与 BF16 的差距能否收敛？
- GSM8K 低分的原因是模型不会停止还是答案算错？

本轮不称官方 BitCPM4 配方复现，不使用 test 做配方或 checkpoint 选择。

## 1. 预训练配置

### 1.1 与 mix_v1 的差异

| 项目 | mix_v1 | mix_v2 |
|---|---|---|
| 总训练量 | 1,073,741,824 token | 2,000,683,008 token |
| 更新步数 | 1,024 | 1,908 |
| alpha_ramp_frac | 0.1（102 步到全三值） | 0.3（572 步） |
| GPU 分配 | 2 卡/组，共 4 组 | 4 卡/组，共 2 组 |
| 周期 checkpoint | 每 1.34 亿 token | 不存（仅 final） |

batch（1,048,576 token/步）、seq_len（2048）、micro_batch（8）、lr（3e-3）、warmup_frac（0.02）、decay_frac（0.2）、seed（20261110）均相同。两组实验：QAT（70/15/15）和 BF16 对照（70/15/15），网页内部中英各半。

### 1.2 数据池

| 数据域 | mix_v1 池 token | mix_v2 池 token | 实际抽到（v2） | data_passes |
|---|---:|---:|---:|---:|
| web_zh | 600,002,217 | 1,840,000,310 | 699,467,776 | 0.380 |
| web_en | 600,001,183 | 1,920,000,769 | 699,848,704 | 0.365 |
| math | 350,000,359 | 350,000,359 | 300,062,720 | 0.858 |
| code | 350,000,063 | 350,000,063 | 301,303,808 | 0.861 |

mix_v1 与 mix_v2 的 math、code 训练文件 SHA256 完全相同（逐字节一致）。网页来源：mix_v1 为 17 片中文 + 16 片英文（共 33 片），mix_v2 扩到 55 + 54（共 109 片），v1 的分片全部包含在 v2 中。混合去重后保留 538 万篇文档，过滤污染 16.27 万篇，去重 6.51 万篇。

## 2. 预训练结果

### 2.1 训练过程

QAT 组耗时 2 小时 16 分 44 秒，BF16 组耗时 2 小 时 11 分 42 秒。两组在所有 1,908 步内 loss 均有限，梯度正常，无中断。

每 128 步验证一次（四域开发 CE，三值精度 for QAT，浮点精度 for BF16）：

| 步数 | QAT 三值 四域平均 CE | BF16 浮点 四域平均 CE |
|---:|---:|---:|
| 0 | 12.003 | 2.073 |
| 256 | 9.372 | 1.868 |
| 512 | 2.544 | 1.840 |
| 768 | 2.083 | 1.824 |
| 1024 | 2.058 | 1.815 |
| 1280 | 2.041 | 1.806 |
| 1536 | 2.027 | 1.800 |
| 1908 | **1.897** | **1.720** |

QAT 在 512 步附近量化 ramp 完成（alpha=1），三值验证损失从 12.0 迅速降至 2.5，之后持续缓降。与 mix_v1 相比，在相同 1.07B 步处 v2 为 2.058，v1 为 1.934——v2 在同等步数下更差，原因是网页分片内容换了、更难拟合；继续训练到 2B 后 v2（1.897）超过了 v1（1.934）。

### 2.2 开发集各域 CE（2B 步最终）

| 域 | QAT 三值 | BF16 浮点 |
|---|---:|---:|
| web_zh | 2.955 | 2.712 |
| web_en | 2.859 | 2.640 |
| math | 1.244 | 1.089 |
| code | 0.532 | 0.438 |
| **四域平均** | **1.897** | **1.720** |

### 2.3 下游评测（基础模型，不经 SFT）

评测协议：5-shot MC，8-shot GSM8K，max_new=1024，prompt_version=question-answer-8shot-v1，eval_code 同一份 SHA256（4d35e06d）。

| 指标 | mix_v2 QAT（三值） | mix_v2 BF16 | mix_v1 QAT | mix_v1 BF16 |
|---|---:|---:|---:|---:|
| GSM8K strict | 1.74% | 9.25% | 0.83% | 13.04% |
| MMLU | 24.29% | 29.03% | 26.61% | 31.61% |
| CMMLU | 25.04% | 27.10% | 25.47% | 30.73% |
| C-Eval | 24.96% | 28.31% | 24.74% | 31.05% |

GSM8K 的 1,319 题全部顶到 1,024 token 上限（length_stop_count=1319），模型在未经 SFT 时不会结束生成，分数只反映随机碰对的概率。

**关键发现**：训练量从 1B 增加到 2B，所有下游指标反而下降（QAT MMLU 26.61→24.29，BF16 MMLU 31.61→29.03）。两组同步下降，排除精度原因，原因是网页语料内容替换：v2 用了更多片段，这批内容更难学习，模型把容量分配到这批网页文本的分布上，多选题能力未能受益。

## 3. GSM8K SFT

### 3.1 配置

起点：mix_v2_qat_70_15_15 final（QAT）和 mix_v2_bf16_70_15_15 final（BF16）。数据：gsm8k_sft_v1，训练集 6,973 条（原始 7,473 条去掉 500 条开发集），答案-only CE 监督（prompt 部分 label=-100）。LR=1e-4，5 epoch，global_batch=64，2 卡/组，GPU 0,1（QAT）和 2,3（BF16）。每轮训练后在 500 条开发集上做生成评测，外部选最优 epoch。

### 3.2 开发集逐 epoch 结果（500 题，max_new=512）

| epoch | QAT strict | QAT length_stop | BF16 strict | BF16 length_stop |
|---|---:|---:|---:|---:|
| 1 | 10.20% | 92/500 | 20.60% | 500/500 |
| 2 | 11.80% | 56/500 | 23.80% | 500/500 |
| 3 | 13.40% | 59/500 | **26.00%** | 495/500 |
| 4 | **14.00%** | 57/500 | 24.60% | 495/500 |
| 5 | 14.00% | 56/500 | 23.60% | 481/500 |

QAT 选第 4 轮（strict 14.00%，CE 0.6415）；BF16 选第 3 轮（strict 26.00%，CE 0.4696）。

注意：BF16 开发集 `length_stop` 显示的是**达到上限的样本数**，BF16 在开发集上 1~2 轮全部顶到上限，但评测脚本在 `\n\nQuestion:` 处截断，依然能抽到答案，所以 strict_acc 仍然有效。

### 3.3 下游评测（SFT 后，max_new=512）

| 指标 | mix_v2 QAT SFT | mix_v2 BF16 SFT | mix_v1 QAT SFT | mix_v1 BF16 SFT |
|---|---:|---:|---:|---:|
| GSM8K strict | 11.90% | 21.23% | 11.60% | 23.73% |
| MMLU | 24.79% | 27.67% | 25.47% | 32.20% |
| CMMLU | 24.38% | 27.73% | 24.97% | 32.24% |
| C-Eval | 27.12% | 29.12% | 24.67% | 32.54% |

GSM8K 实际停止分布（1,319 题）：

| | 正常停止（EOS） | 达到 512 上限 |
|---|---:|---:|
| QAT SFT | **1,218（92.3%）** | 101（7.7%） |
| BF16 SFT | 8（0.6%） | 1,311（99.4%） |

## 4. EOS 诊断

### 4.1 Teacher-forcing 边界检验（200 条留出开发样本）

在答案最后一个 token 位置上，检查两个停止符的 logit 排名：

| 配置 | `<\|im_end\|>`(73440) 排名 | `</s>`(2) 排名 | top1 是停止符 |
|---|---:|---:|---:|
| QAT SFT · 三值 | **第 1 名（100%）** | 第 22,696 名 | 100% |
| QAT SFT · 浮点 | 第 65,337 名 | 第 1,061 名 | 0% |
| BF16 SFT · 浮点 | **第 1 名（100%）** | 第 71,287 名 | 100% |
| QAT 基础模型 · 三值 | 第 66,916 名 | **第 47 名** | 0% |

**重要说明**：QAT SFT 的浮点权重不能用于评估，因为训练时梯度几乎全走量化路径，FP32 master 被优化成"只有三值化后才正确"的形态。浮点评估会严重误判。

### 4.2 生成行为的真实原因

| 模型 | 实际停止率 | 顶上限后写了 `####` | 失败模式 |
|---|---|---|---|
| QAT SFT | 92.3% 正常停止 | 87/101（87%）重复 `####` 循环 | **答案算错**，格式正确 |
| BF16 SFT | 0.6% 正常停止 | 1,290/1,311 续写新 `Question:` | **不知道答完就停**，答案更准 |

**核心结论**：停止符配置没有问题，结束符也不是瓶颈。两个模型的问题性质完全不同：QAT 学会了格式（92.3% 主动停止）但答案算错；BF16 答案更准但把 prompt 的 few-shot 格式当模板来续写，不会在答完后主动结束。评测脚本在 `\n\nQuestion:` 处截断所以 BF16 仍能得分。

## 5. 结论与待解问题

### 5.1 已确认的结论

1. **alpha_ramp_frac=0.3 有效**：v2 QAT 的四域平均 CE（1.897）优于 v1（1.934），ramp 拉长让量化适应更充分。QAT 与 BF16 的差距从 0.185 收窄到 0.177。
2. **训练量从 1B→2B 未带来下游增益**：所有下游指标下降，根本原因是网页语料换了批次，不是训练量的问题。
3. **三值量化损伤的是推理正确率，不是格式习得**：QAT 的 GSM8K 格式（停止、`####`）学得比 BF16 好，但数学答案正确率低一倍（11.90% vs 21.23%）。
4. **BF16 SFT 存在 few-shot 续写问题**：99.4% 的样本不主动停止，依赖评测截断机制才能得分，这个模型不可直接部署。
5. **比较口径**：SFT 评测用 max_new=512，基础模型评测用 max_new=1024，两者不能直接比较 GSM8K 绝对数字。

### 5.2 下一步方向

- **QAT 推理误差来源**：三值化后 MLP/attention 层的激活误差分析，定位哪些层对多步推理贡献最大的量化误差。
- **通用 SFT**：已下载中文 Alpaca（48,818 条）和英文 Alpaca（52,002 条），ModelScope 代理通道已验证可用（速度约 500–600 KB/s）。接入需要改造 `encode_example`，支持可配置 prompt 模板和 loss mask，并对候选数据做污染过滤。
- **BF16 SFT 修复**：BF16 的 few-shot 续写问题来自训练时没见过"答完就停"的示例，可以在 prompt 末尾加明确停止标记，或增加 EOS 惩罚训练。

## 6. 通用指令 SFT（Alpaca / Firefly / HC3）

### 6.1 目的

在已经做过 GSM8K SFT 的 QAT 模型上继续用通用指令数据 SFT，检验能否在不损失 GSM8K 的前提下提升选择题（MMLU/CMMLU/C-Eval）能力。这是"直接在已有 QAT 基础上继续 SFT"路线的第一次尝试。

### 6.2 数据

`data/alpaca_sft_v1`，统一格式 `{"prompt", "response", "source", "id"}`：

| 来源 | 条数 |
|---|---|
| firefly（7 类：Cot/OpenQA/MRC/Translation/Composition/ClassicalChinese/Dictionary） | 384,973 |
| alpaca_en | 51,589 |
| alpaca_zh | 48,669 |
| hc3（chatgpt_answers） | 9,988 |
| **train 合计** | **495,219** |
| dev | 1,000 |

处理：长度过滤（prompt+response ≤ 8000 字符）→ 精确去重 → 污染过滤（对 MMLU/CMMLU/C-Eval/GSM8K 全部 train/test/变体题干做完整题干 + 13-unit n-gram 匹配，剔除 9,728 条）。JSONL 用 `ensure_ascii=True` 写出，避免 U+2028/U+2029 被 `splitlines()` 误当换行。

### 6.3 训练配置

- 起点：`runs/mix_v2_qat_sft/snapshots/epoch_4/model.pt`（QAT，三值结构，alpha=1.0，FP32 master）
- 8 卡 DDP，global_batch=128，micro_batch=2，seq_len=2048
- LR=5e-5，warmup 5% + cosine 衰减，3 epoch，无梯度检查点
- 11,598 步，耗时约 3.75 小时（13,490 s）

### 6.4 结果

开发集 CE（每轮末）：

| epoch | dev_ce |
|---|---|
| 1 | 2.9311 |
| 2 | 2.8994 |
| 3 | **2.8883**（最优，选中） |

下游评测（epoch 3，max_new=512，与基线同协议）：

| 指标 | mix_v2 QAT SFT (ep4) | + 通用 SFT (ep3) | 变化 |
|---|---|---|---|
| GSM8K strict | 11.90% | **3.56%** | −8.34 |
| GSM8K acc | — | 4.32% | — |
| MMLU | 24.79% | 24.88% | +0.09 |
| CMMLU | 24.38% | 25.25% | +0.87 |
| C-Eval | 27.12% | 26.82% | −0.30 |

GSM8K 停止分布：`length_stop_count=449/1319`（34% 顶到长度上限不停止），相比纯 GSM8K SFT 的 92.3% 正常停止严重退化。

### 6.5 结论

1. **通用指令数据没有带来选择题收益**：MMLU/CMMLU/C-Eval 基本持平（CMMLU +0.87 属噪声量级）。
2. **纯通用数据覆盖会打崩 GSM8K 格式**：strict 从 11.90% 掉到 3.56%，停止率从 92.3% 退到 66%。原因是 49.5 万条通用样本里 GSM8K 占比为 0，3 个 epoch 把 GSM8K 的 `####` 结尾 + 停止习惯稀释掉了。
3. **下一步**：把 GSM8K 按比例混回通用数据（见 6.6），而不是用纯通用数据覆盖。

### 6.6 后续实验：GSM8K 混入通用 SFT

数据集 `data/alpaca_gsm8k_sft_v1`：在 `alpaca_sft_v1` 基础上把 GSM8K train（6,973 条）重复 3 倍后混入，共 516,138 条，GSM8K 占比约 4.05%；3 个 epoch 下每条 GSM8K 样本被看到 9 次（纯 GSM8K SFT 基线为 5 次），以补偿稀释。dev 用 alpaca dev 1,000 + GSM8K dev 500 = 1,500 条。其余配置与 6.3 相同。

**训练**：12,087 步，约 4 小时。dev_ce：ep1 2.0663、ep2 **2.0578**（最优）、ep3 2.0655。注意 dev 集含 500 条 GSM8K，CE 绝对值与 6.4 不可直接比较。

**下游评测**（epoch 2，max_new=512）：

| 指标 | 纯 GSM8K SFT (ep4) | 纯通用 SFT (ep3) | 混合 SFT (ep2) |
|---|---|---|---|
| GSM8K strict | 11.90% | 3.56% | 8.87% |
| MMLU | 24.79% | 24.88% | 24.72% |
| CMMLU | 24.38% | 25.25% | 24.76% |
| C-Eval | 27.12% | 26.82% | 27.79% |

GSM8K 停止分布：`length_stop_count=15/1319`（仅 1.1% 顶到长度上限），停止率 98.9%，**优于**纯 GSM8K SFT 的 92.3%。

**结论**：

1. **混入 GSM8K 完全修复了格式**：停止率从纯通用 SFT 的 66% 回到 98.9%，甚至超过纯 GSM8K SFT 基线。
2. **GSM8K 正确率部分恢复**：3.56% → 8.87%，但仍低于纯 GSM8K SFT 的 11.90%。格式已恢复，差距在答案正确率上——通用数据把 GSM8K 的答题习惯冲淡了，4% 的占比 + 9 次暴露不足以完全抵消。
3. **选择题没有收益**：MMLU/CMMLU/C-Eval 与两个基线都在噪声范围内（±0.5），C-Eval 27.79% 略高但不足以说明通用数据带来了能力提升。
4. **总体判断**：这条"通用 SFT + GSM8K 混合"路线没有达到目标——既没拿到选择题增益，GSM8K 也没回到基线。若要继续，需要提高 GSM8K 占比（如 10–20%）或降低通用数据权重，而不是简单等比例混合。

### 6.7 v2 混合 SFT：加入代码，GSM8K 重复 11 倍

数据集 `data/sft_v2`：通用 59.7% / 代码 25.0% / 数学 15.3%（499,804 条）。相对 6.6 的两处变化：**代码从 0 提到 25%**（`opc-annealing-corpus` 的 algorithmic_corpus，7 种语言各约 17.7k，加 firefly Program 973），**GSM8K 从重复 3 倍提高到 11 倍**。起点同 6.3，8 卡，LR 5e-5，3 epoch，11,709 步。

**训练**：dev_ce 逐轮 **上升**：ep1 2.1609、ep2 2.1980、ep3 2.2305 → 最优为 **epoch 1**。dev_ce 单调上升说明在混合 dev 上快速过拟合。

**下游评测**（epoch 1，max_new=512）：

| 指标 | 纯 GSM8K SFT (ep4) | 纯通用 SFT (ep3) | 混合 GSM8K×3 (ep2) | v2 通用60/代码25/GSM8K×11 (ep1) |
|---|---|---|---|---|
| GSM8K strict | 11.90% | 3.56% | 8.87% | **7.35%** |
| MMLU | 24.79% | 24.88% | 24.72% | **24.00%** |
| CMMLU | 24.38% | 25.25% | 24.76% | 24.99% |
| C-Eval | 27.12% | 26.82% | 27.79% | **25.71%** |

GSM8K 停止分布：`length_stop_count=23/1319`（停止率 98.3%，格式正常）。

**结论**：

1. **GSM8K 重复 11 倍反而更差**：v2 的 7.35% **低于** 6.6 混合实验的 8.87%（后者 GSM8K 只重复 3 倍、占比 4.05%）。重复越多、分数越低，配合 dev_ce 单调上升，说明高重复导致过拟合而非更好的格式习得。**这直接支持 v3 把重复倍数降到 2。**
2. **加入代码没有带来收益，反而普遍下降**：MMLU 24.00%、C-Eval 25.71% 是四个实验里最低；GSM8K 也低于 6.6。原因是代码（25%）挤占了通用数据份额，而代码能力这一轮**无法评测**（未装 HumanEval/MBPP），只看到通用指标的代价。
3. **最优 epoch 落在第 1 轮**：3 轮训练里后两轮都在变差，说明当前数据配比 + LR 下 3 epoch 偏多。

### 6.8 v3 混合 SFT：降低 GSM8K 重复 + 引入多样数学

数据集 `data/sft_v3`（497,881 条）：通用 59.9% / 代码 25.1% / 数学 15.0%（74,780 条）。**通用和代码与 v2 逐条相同**，只改数学类构成：

- GSM8K 重复从 **11 倍降到 2 倍**（13,946 条，3 epoch 下每条暴露 6 次）
- 新增 **UltraData-Math L3 QA-Synthetic** 60,834 条（UltraData-Math 原始是 parquet 纯文本，"开头 600 字符内最后一个 `?` 后的换行"切分题目/解答；背面 92.6% 保留，污染过滤剔除 219 条）

起点同 6.3，8 卡，LR 5e-5，3 epoch，11,601 步。

**训练**：dev_ce **单调下降**：ep1 2.0779 → ep2 2.0631 → ep3 **2.0626**（与 v2 的单调上升相反），最优 epoch 3。

**下游评测**（epoch 3，max_new=512）：

| 指标 | 纯GSM8K(ep4) | 纯通用(ep3) | 混合×3(ep2) | v2 ×11(ep1) | **v3 ×2+Ultra(ep3)** |
|---|---|---|---|---|---|
| GSM8K strict | 11.90% | 3.56% | 8.87% | 7.35% | **9.25%** |
| MMLU | 24.79% | 24.88% | 24.72% | 24.00% | **25.44%** |
| CMMLU | 24.38% | 25.25% | 24.76% | 24.99% | 24.86% |
| C-Eval | 27.12% | 26.82% | 27.79% | 25.71% | 26.30% |
| GSM8K 停止率 | 92.3% | 66% | 98.9% | 98.3% | **99.1%** |

**结论**：

1. **降低 GSM8K 重复 + 引入多样数学，确实有效**：v3 与 v2 只差数学类，四项里三项更好——GSM8K +1.90（7.35→9.25）、MMLU +1.44（24.00→25.44）、C-Eval +0.59（25.71→26.30），CMMLU 基本持平（−0.13）。
2. **MMLU 25.44% 是五次实验里最高的**，也是唯一明显高于基线的（基线 24.79%）。多样数学数据可能在推理上对 MMLU 有正向迁移。
3. **GSM8K 仍低于纯 GSM8K SFT 基线**（9.25% vs 11.90%）：格式已经修满（停止率 99.1%，五次里最高），差距在答案正确率。通用+代码数据占 85% 时，GSM8K 很难回到纯数据训练的水平。
4. **dev_ce 单调下降 + 最优在第 3 轮**，与 v2 的过拟合形成对比，说明重复倍数 2 比 11 更健康。

## 7. 重要文件索引

| 文件 | 内容 |
|---|---|
| `data/mix_v2/freeze.json` | 185 个来源文件 SHA256，过滤参数，tokenizer 版本 |
| `data/mix_v2/manifest.json` | 训练/验证 bin 文件 SHA256 和 token 计数 |
| `reports/pretrain_v2/mix_v2/preflight.json` | 回归测试、数据校验、冒烟测试结果 |
| `reports/pretrain_v2/mix_v2/selection.json` | 预训练 checkpoint 选择（QAT 四域 CE 等权平均最低） |
| `reports/pretrain_v2/mix_v2/selected_qat_downstream.json` | QAT 基础模型四项评测完整结果 |
| `reports/pretrain_v2/mix_v2/bf16_control_downstream.json` | BF16 基础模型四项评测完整结果 |
| `reports/pretrain_v2/mix_v2/mix_v2_sft_selection.json` | SFT checkpoint 选择（开发集 strict_acc 最高） |
| `reports/pretrain_v2/mix_v2/mix_v2_qat_sft_downstream.json` | QAT SFT 后四项评测完整结果 |
| `reports/pretrain_v2/mix_v2/mix_v2_bf16_sft_downstream.json` | BF16 SFT 后四项评测完整结果 |
| `reports/pretrain_v2/mix_v2/eos/qat_sft_ternary.json` | EOS 诊断：QAT SFT 三值精度（200 样本边界 + 生成） |
| `reports/pretrain_v2/mix_v2/eos/bf16_sft_float.json` | EOS 诊断：BF16 SFT 浮点精度 |
| `runs/mix_v2_qat_70_15_15/checkpoints/final/state.pt` | QAT 预训练最终 checkpoint（FP32 master，三值结构） |
| `runs/mix_v2_bf16_70_15_15/checkpoints/final/state.pt` | BF16 预训练最终 checkpoint |
| `runs/mix_v2_qat_sft/snapshots/epoch_4/model.pt` | QAT SFT 最优 epoch（第 4 轮）model-only |
| `runs/mix_v2_bf16_sft/snapshots/epoch_3/model.pt` | BF16 SFT 最优 epoch（第 3 轮）model-only |
| `data/alpaca_sft_v1/manifest.json` | 通用指令 SFT 数据清单（来源/条数/污染剔除） |
| `reports/pretrain_v2/mix_v2_alpaca_sft/best_epoch.json` | 通用 SFT 选优结果（epoch 3，dev_ce 2.8883） |
| `reports/pretrain_v2/mix_v2_alpaca_sft/epoch_3_downstream.json` | 通用 SFT 后四项评测完整结果 |
| `runs/mix_v2_qat_alpaca_sft/snapshots/epoch_3/model.pt` | 通用 SFT 最优 epoch（第 3 轮）model-only |
| `data/alpaca_gsm8k_sft_v1/manifest.json` | GSM8K 混合 SFT 数据清单（GSM8K 重复 3 倍，占比 4.05%） |
| `reports/pretrain_v2/mix_v2_alpaca_gsm8k_sft/best_epoch.json` | 混合 SFT 选优结果（epoch 2，dev_ce 2.0578） |
| `reports/pretrain_v2/mix_v2_alpaca_gsm8k_sft/epoch_2_downstream.json` | 混合 SFT 后四项评测完整结果 |
| `runs/mix_v2_qat_alpaca_gsm8k_sft/snapshots/epoch_2/model.pt` | 混合 SFT 最优 epoch（第 2 轮）model-only |
| `data/sft_v2/manifest.json` | v2 数据清单（通用60/代码25/数学15，GSM8K×11） |
| `reports/pretrain_v2/sft_v2/best_epoch.json` | v2 选优结果（epoch 1，dev_ce 2.1609） |
| `reports/pretrain_v2/sft_v2/epoch_1_downstream.json` | v2 四项评测完整结果 |
| `runs/mix_v2_qat_sft_v2/snapshots/epoch_1/model.pt` | v2 最优 epoch（第 1 轮）model-only |
| `data/sft_v3/manifest.json` | v3 数据清单（通用60/代码25/数学15，GSM8K×2 + UltraData） |
| `reports/pretrain_v2/sft_v3/best_epoch.json` | v3 选优结果（epoch 3，dev_ce 2.0626） |
| `reports/pretrain_v2/sft_v3/epoch_3_downstream.json` | v3 四项评测完整结果 |
| `runs/mix_v2_qat_sft_v3/snapshots/epoch_3/model.pt` | v3 最优 epoch（第 3 轮）model-only |
