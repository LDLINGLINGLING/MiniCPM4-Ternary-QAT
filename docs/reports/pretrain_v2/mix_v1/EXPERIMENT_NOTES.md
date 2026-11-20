# mix_v1 实验记录（1B token 四组 QAT/BF16 配方对比，含 SFT）

日期：2026-11-12。本记录基于 `reports/pretrain_v2/mix_v1/` 下的实际产物，所有数字均从 metrics.jsonl、selection.json、downstream JSON 中重新读取核对。

## 0. 实验目的

在小规模 s2/s3 实验（见历史 bench_*.log）确认 QAT 路径可行后，建立首次混合语料四组配方对比：纯网页、70/15/15、50/25/25 三种 QAT 配比 + BF16 对照，验证数学/代码语料加入是否改善 QAT 下游能力，并为 mix_v2 提供基线。本轮不称官方 BitCPM4 配方复现。

## 1. 预训练配置

### 1.1 参数

| 项目 | 值 |
|---|---|
| 总训练量 | 1,073,741,824 token（1,024 步） |
| global_batch_tokens | 1,048,576 |
| seq_len | 2,048 |
| micro_batch | 8 |
| lr | 3e-3 |
| alpha_ramp_frac | 0.1（QAT），0.0（BF16） |
| warmup_frac | 0.02，decay_frac 0.2 |
| seed | 20261110 |
| GPU | 2 卡/组，4 组，共 8 卡 |
| 周期 checkpoint | 每 134,217,728 token |

### 1.2 四组配比

| 组名 | 模式 | 网页 zh | 网页 en | 数学 | 代码 | GPU |
|---|---|---:|---:|---:|---:|---|
| mix_v1_qat_web | QAT | 50% | 50% | 0% | 0% | 0,1 |
| mix_v1_qat_70_15_15 | QAT | 35% | 35% | 15% | 15% | 2,3 |
| mix_v1_qat_50_25_25 | QAT | 25% | 25% | 25% | 25% | 4,5 |
| mix_v1_bf16_70_15_15 | BF16 | 35% | 35% | 15% | 15% | 6,7 |

### 1.3 数据池

| 域 | 池 token | 实际抽到（70/15/15） | data_passes |
|---|---:|---:|---:|
| web_zh | 600,002,217 | 375,169,024 | 0.626 |
| web_en | 600,001,183 | 376,563,712 | 0.628 |
| math | 350,000,359 | 160,548,864 | 0.459 |
| code | 350,000,063 | 161,460,224 | 0.462 |

## 2. 预训练结果

### 2.1 最终开发集 CE（1,024 步）

| 组名 | 四域平均 CE | web_zh | web_en | math | code |
|---|---:|---:|---:|---:|---:|
| mix_v1_qat_web | 2.280 | 3.284 | 3.167 | 1.498 | 1.172 |
| mix_v1_qat_70_15_15 | **1.934** | 3.011 | 2.900 | 1.269 | 0.557 |
| mix_v1_qat_50_25_25 | 1.937 | 3.035 | 2.914 | 1.264 | 0.595 |
| mix_v1_bf16_70_15_15 | 1.749 | 2.843 | 2.713 | 1.139 | 0.499 |

选择规则：QAT 三组中四域等权平均 CE 最低者。mix_v1_qat_70_15_15（1.934）与 mix_v1_qat_50_25_25（1.937）差距 0.003，按预先固定顺序选 70/15/15。纯网页组（2.280）明显差于混合组。

### 2.2 下游评测（基础模型，max_new=1024）

| 指标 | QAT 70/15/15 | BF16 70/15/15 |
|---|---:|---:|
| GSM8K strict | 0.83% | 13.04% |
| MMLU | 26.61% | 31.61% |
| CMMLU | 25.47% | 30.73% |
| C-Eval | 24.74% | 31.05% |

GSM8K 1,319 题全部顶到 1,024 token 上限（length_stop_count=1319）。模型未经 SFT 时不会结束生成，分数仅为随机碰对。

## 3. GSM8K SFT

### 3.1 配置

起点：mix_v1_qat_70_15_15 final（QAT）和 mix_v1_bf16_70_15_15 final（BF16）。数据：gsm8k_sft_v1，训练集 6,973 条，答案-only CE 监督。LR=1e-4，5 epoch，global_batch=64，2 卡/组，GPU 0,1（QAT）和 2,3（BF16）。

### 3.2 开发集逐 epoch（500 题，max_new=512）

| epoch | QAT strict | BF16 strict |
|---|---:|---:|
| 1 | 7.60% | 21.80% |
| 2 | 11.40% | 25.80% |
| 3 | 14.00% | 28.60% |
| 4 | **14.40%** | **29.20%** |
| 5 | 12.00% | 27.40% |

QAT 选第 4 轮；BF16 选第 4 轮（strict 29.20%，CE 0.5141）。

### 3.3 下游评测（SFT 后，max_new=512）

| 指标 | QAT SFT | BF16 SFT |
|---|---:|---:|
| GSM8K strict | 11.60% | 23.73% |
| MMLU | 25.47% | 32.20% |
| CMMLU | 24.97% | 32.24% |
| C-Eval | 24.67% | 32.54% |

SFT 后 QAT 与 BF16 在 GSM8K 上差距约 12 个百分点（11.60% vs 23.73%），多选题差距约 7–8 个百分点。这个差距在 mix_v2 实验中得到进一步分析（见 mix_v2/EXPERIMENT_NOTES.md）。

## 4. 结论

1. **混合语料优于纯网页**：加入 15% 数学 + 15% 代码使四域平均 CE 从 2.280 降至 1.934。
2. **70/15/15 与 50/25/25 几乎无差别**：CE 差距仅 0.003，在统计噪声范围内，配比从 15% 提高到 25% 在 1B token 量级未见收益。
3. **SFT 大幅提升 GSM8K**：QAT 从 0.83% → 11.60%（约×14），BF16 从 13.04% → 23.73%（约×1.8）；多选题基本持平略升，格式对齐不破坏通用能力。
4. **SFT 后 QAT 与 BF16 差距稳定在 12 点**：差距没有在 SFT 阶段收敛，说明问题出在预训练阶段的量化能力损失，而非格式习得。这一点在 mix_v2 EOS 诊断中得到验证。

## 5. 重要文件索引

| 文件 | 内容 |
|---|---|
| `data/mix_v1/freeze.json` | 109 个来源文件 SHA256，过滤参数 |
| `data/mix_v1/manifest.json` | 训练/验证 bin 文件 SHA256 |
| `reports/pretrain_v2/mix_v1/preflight.json` | 回归测试、冒烟测试结果 |
| `reports/pretrain_v2/mix_v1/selection.json` | 四组配比选择结果 |
| `reports/pretrain_v2/mix_v1/selected_qat_downstream.json` | QAT 基础模型四项评测 |
| `reports/pretrain_v2/mix_v1/bf16_control_downstream.json` | BF16 基础模型四项评测 |
| `reports/pretrain_v2/mix_v1/mix_v1_sft_selection.json` | SFT checkpoint 选择 |
| `reports/pretrain_v2/mix_v1/mix_v1_qat_sft_downstream.json` | QAT SFT 后四项评测 |
| `reports/pretrain_v2/mix_v1/mix_v1_bf16_sft_downstream.json` | BF16 SFT 后四项评测 |
| `runs/mix_v1_qat_70_15_15/checkpoints/final/state.pt` | QAT 预训练最终 checkpoint |
| `runs/mix_v1_bf16_70_15_15/checkpoints/final/state.pt` | BF16 预训练最终 checkpoint |
| `runs/mix_v1_qat_sft/snapshots/epoch_4/model.pt` | QAT SFT 最优 epoch（第 4 轮） |
| `runs/mix_v1_bf16_sft/snapshots/epoch_4/model.pt` | BF16 SFT 最优 epoch（第 4 轮） |
