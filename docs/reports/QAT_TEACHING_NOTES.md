# 三值 QAT 全流程教学文档

**读者**：刚接触量化感知训练（QAT）的工程师 / 学生；有 PyTorch 与 Transformer 基础，没做过低位量化训练。
**目的**：用这个仓库真实跑过的实验，把三件事讲透 ——（1）三值 QAT 的**原理**，（2）**为什么一步一步这样做实验**、每一步学到什么，（3）代码怎么写、踩了哪些坑。所有数字、曲线都取自 `reports/` 与 `runs/*/metrics.jsonl` 的真实产物，代码取自 `qat/` 的真实实现。
**范围**：MiniCPM4-0.5B 的三值 QAT 预训练 + 多轮 SFT，仅此一条线。

---

# 第一部分 原理

## 1. 从浮点到三值：量化到底损失了什么

普通权重是 FP32/BF16。三值量化把每个权重压成三种取值 `{-s, 0, +s}`，`s` 是该输出通道的缩放系数。乘加几乎退化成加减和符号运算，显存与算力都省。

代价是**信息损失**。对一行权重 `W_i = (w₁…w_d)`，我们只能用 `{-s, 0, +s}` 三个刻度去逼近 d 个连续值。用均方误差衡量，给定这一行，量化误差下界大致是 `s/2` 每元素 —— 所以 `s` 的选取直接决定误差。

**为什么是 absmean（平均绝对值）？** 使量化误差最小的 `s` 没有闭式解（依赖哪些元素被舍成 0）。`s_i = mean|W_ij|` 是一个廉价且稳健的近似：它把这一行的「典型幅度」作为刻度，使大多数元素落在 ±1 附近而不是全部饱和或全部归零。**为什么按行（per-output-channel）而不是全局一个 `s`？** 不同输出通道的权重动态范围差很多，各自一个 `s` 才能让每一行都用到完整的三值范围。

```python
def ternary_quantize(w: torch.Tensor):
    """Return (codes in {-1,0,1}, per-row scale) for a 2-D weight."""
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=EPS)   # 每行一个 s_i
    codes = (w / scale).round().clamp(-1, 1)                   # round-to-nearest + 截断
    return codes, scale
```

`dim=1`（对 in 维）配 `keepdim=True` 得到每行一个标量；`clamp(-1,1)` 把越界的舍入结果拉回三值；`clamp(min=EPS)` 防止整行全零时除零。

## 2. 可导性问题与 STE：整个 QAT 的枢纽

`round()` 的导数几乎处处为 0，直接反传梯度会消失。QAT 的解法是 **STE（straight-through estimator）**：反向时假装量化是恒等映射，把梯度原样透传。

```
class TernaryLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        if self.quantize:
            w_q = ternary_dequant(w.float()).to(w.dtype)
            w = w + self.alpha.to(w.dtype) * (w_q - w).detach()   # ← 枢纽
        return F.linear(x, w, self.bias)
```

这一行同时完成三件事，值得逐字拆：

| 片段 | 作用 |
|---|---|
| `w_q - w` | 量化误差 `Δ = Q(W) − W` |
| `.detach()` | **切断 `Δ` 的梯度** —— 于是 `∂L/∂W` 只经由第一个 `w` 流动，量化被当成恒等（这就是 STE） |
| `alpha *` | 用一个 0→1 的标量**渐进地**注入 `Δ` |

前向实际用的是 `W_fwd = (1−α)·W + α·Q(W)`：`α=0` 时是纯浮点，`α=1` 时是纯三值，中间是两者的线性插值。因为 `Δ` 被 detach，反向梯度对 `W` 是满的（`∂W_fwd/∂W = 1`），所以 **FP32 master 权重照常被优化**，量化只是前向的「假动作」。

**关键约定：参数始终保留 FP32 master。** 量化只在 forward 发生。这样优化器累积的微小更新不会被三值网格吃掉。

## 3. alpha ramp：为什么不能一步到位

如果一上来就 `α=1`，权重从浮点瞬间跳到三值，前向分布突变，损失会爆、梯度会炸。`alpha_ramp` 让 `α` 从 0 线性升到 1，模型有时间「适应」量化约束：

```python
def alpha_at(step, total, cfg):
    """alpha_ramp_tokens (absolute) takes precedence over alpha_ramp_frac."""
    if cfg['alpha_ramp_tokens']:
        ramp = cfg['alpha_ramp_tokens'] // cfg['global_batch_tokens']
    else:
        ramp = int(total * cfg['alpha_ramp_frac'])
    return 1.0 if ramp <= 0 else min(1.0, step / ramp)
```

**这一步的代价在曲线里看得见**（见 §8、图 `figs/pretrain_gradnorm.png`）：训练损失会在 `α` 快速上升段出现一个**尖峰**，尖峰大小取决于 ramp 有多快。

- `mix_v1` 用 `ramp=0.1`（102 步到全三值）：损失在 step 91–121 冲到 **2.96**，梯度范数冲到 **3.88**。
- `mix_v2` 把 ramp 拉到 `0.3`（572 步）：损失只升到 **~2.75**，梯度范数峰值 **~1.5**。

**ramp 越慢，尖峰越矮、适应越平滑，最终三值 CE 越低**（§8 有完整对照）。这是本项目最重要的超参结论之一。

## 4. 一个反直觉现象：QAT 模型的「浮点前向」是坏的

看 §8 的验证曲线（图 `figs/pretrain_val.png` 左）：QAT 训练结束时，**三值前向 CE = 1.897，浮点前向 CE = 10.19**。浮点前向竟然比三值前向差 5 倍以上。BF16 组正好相反（浮点 1.720，三值 22.4）。

**原因**：训练时梯度几乎全走量化路径，FP32 master 被优化成「**只有三值化之后才正确**」的形态 —— 它本身是一组没有独立意义的潜变量，`Q(W)` 才是那个「模型」。BF16 组的权重是三值前向时毫无意义的浮点值，所以三值前向爆炸。

**工程后果（务必记住）**：
- **QAT 模型必须用 `α=1` 的三值前向评测**，用浮点前向评测会得到完全错误的结论。
- checkpoint 里要保存的是 FP32 master（`assert_fp32_master` 会强制校验），因为三值化是确定性的、可从 master 重建。

## 5. 训练日程：WSD 与 μP

**WSD 学习率日程**（Warmup-Stable-Decay，不是 cosine）：

```python
def lr_at(step, total, cfg):
    """WSD: linear warmup, constant, linear decay to min_lr_ratio over the final decay_frac."""
    warm = max(1, int(total * cfg['warmup_frac']))
    decay_start = int(total * (1 - cfg['decay_frac']))
    if step < warm:
        return (step + 1) / warm
    if step < decay_start:
        return 1.0                     # 平台期
    frac = (step - decay_start) / max(1, total - decay_start)
    return 1.0 - frac * (1.0 - cfg['min_lr_ratio'])
```

线性升温 → **平台期恒定** → 末端线性衰减。平台期的好处是「训练到任意 checkpoint 都能代表该阶段」，做消融和比较时不用重跑完整日程。本仓库 `warmup_frac=0.02`、`decay_frac=0.2`。

**μP 参数分组**（maximal update parametrization）：矩阵层乘 `dim_model_base / hidden_size`，嵌入与 norm 层不衰减：

```python
def param_groups(model, cfg):
    scale = conf.dim_model_base / conf.hidden_size if cfg['mup_lr_scale'] else 1.0
    ...
    return [
        dict(params=matrices, lr_mult=scale, weight_decay=cfg['weight_decay'], name='matrices'),
        dict(params=embed,    lr_mult=1.0,   weight_decay=0.0, name='embed'),
        dict(params=other,    lr_mult=1.0,   weight_decay=0.0, name='norms'),
    ]
```

意义：同一套超参跨模型宽度可迁移 —— 换更大模型不必重调 LR。

## 6. 量化哪些层

```python
TARGET_SUFFIXES = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')

def convert_to_ternary(model, suffixes=None):
    ...
    if (type(child) is nn.Linear and child_name in targets and '.layers.' in f'.{full}'):
        ...  # 原地替换成 TernaryLinear
```

两层过滤缺一不可：

- **按名字**只碰 7 类投影矩阵（QAT 惯例：量化 attention 与 MLP 的线性层）。
- **按路径**必须含 `.layers.`，显式排除 `lm_head` 和 `embed_tokens`（词表相关层保持浮点）。

MiniCPM4-0.5B 24 层 × 7 投影 = **168 个矩阵**被三值化。

## 7. 远程代码的两个硬性约束

```python
def lm_logits(model, input_ids):
    """Windows are packed with no padding; the remote flash-attention path insists on a
    mask, so an all-ones mask is passed."""
    mask = torch.ones_like(input_ids)
    return model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
```

1. **必须显式传 `attention_mask`** —— MiniCPM4 走 flash-attention 路径，不带 mask 报错。窗口是密排无 padding 的，传全 1 mask。
2. **必须 `use_cache=False`** —— 训练/验证关掉 KV cache，否则与梯度检查点冲突。

---

# 第二部分 预训练实验：实验设计的推理链

这一部分回答「为什么一步步做这些实验」。整条链是**逐步排除变量**的过程：每一轮只改一个因素，用上一轮的结论决定下一轮。

```
s2  超参筛选      ──► 定下 LR=3e-3, ramp=0.1
s3  受控对照      ──► 量化代价多大？CE 恢复≠下游恢复 → 需要更丰富的数据配比
mix_v1  引入混合  ──► 网页/数学/代码 配比怎么定？(3 组 QAT + BF16 对照)
mix_v2  放大+调ramp ──► 训练量翻倍有用吗？ramp 拉长有用吗？
```

## 8. 阶段 s2：先定超参，再谈别的

**动机**：在动数据配方之前，必须先确定「三值 QAT 用什么 LR、什么 ramp」，否则后面所有对比都混了两个变量。

**设计**：8 组网格，约 5 亿 token，纯网页中英，LR ∈ {3e-4, 1e-3, 3e-3, 1e-2} × ramp ∈ {0, 0.1}。

| run | LR | ramp | 三值均值 CE |
|---|---:|---:|---:|
| **s2_ramp0.1_lr3e-3** | **3e-3** | **0.1** | **3.041** |
| s2_ramp0_lr3e-3 | 3e-3 | 0.0 | 3.054 |
| s2_ramp0.1_lr1e-3 | 1e-3 | 0.1 | 3.073 |
| ⋮ | ⋮ | ⋮ | ⋮ |
| s2_ramp0_lr1e-2 | 1e-2 | 0.0 | 3.656 |

**结论**：`LR=3e-3, ramp=0.1` 最优；有 ramp 普遍优于无 ramp（最多差 0.013）；LR 过大/过小都明显变差。进入后续阶段的主线配置由此固定。

## 9. 阶段 s3：量化到底损失了多少能力？

**动机（假设）**：三值 QAT 续训，量化代价有多大？能不能用知识蒸馏补回来？

**设计**：约 9.4 亿 token，纯 Ultra-FineWeb 中英，4 组对照 —— BF16 续训（两个 LR）、QAT（ramp 0.1, LR 3e-3）、QAT+原模型蒸馏。

| 组 | 中文 CE | 英文 CE | 均值 CE |
|---|---:|---:|---:|
| 原模型（起点） | 3.362 | 2.992 | 3.177 |
| A2 BF16 | 2.812 | 2.617 | **2.715** |
| **B QAT** | 3.076 | 2.865 | **2.970** |
| D QAT+蒸馏（原模型为师） | 3.109 | 2.875 | 2.992 |

**下游评测**（这是本轮最重要的发现）：

| 模型 | MMLU | CMMLU | C-Eval | GSM8K |
|---|---:|---:|---:|---:|
| 原始 MiniCPM4-0.5B | 50.5% | 66.4% | 68.0% | 30.3% |
| A2 BF16 续训 | 44.0% | 46.7% | 49.6% | 18.9% |
| **B QAT 续训** | 23.8% | 25.1% | 23.9% | 1.8% |

**结论（两条，都改变了后续方向）**：

1. **CE 恢复 ≠ 下游恢复**：QAT 的验证 CE 从 13.33 降到 2.97（算术上恢复了「语言建模」），但 MMLU 只剩 23.8%，**≈ 四选一随机水平**。9.4 亿 token 的三值 QAT 恢复了 PPL，知识和推理能力基本丢失。
2. **纯网页数据本身就不够**：BF16 续训在**同样纯网页数据**上，MMLU 也从 50.5% 掉到 44.0%。也就是说**数据配比不足（缺数学/代码/指令类）是重要因素，不是量化独有的问题**。

→ **直接推出 mix_v1：引入网页/数学/代码四域混合。**

**蒸馏的插曲**：用更好的 A2 当老师，三值 CE 2.970→2.937（改善约 0.03）；用比学生更差的原模型当老师（D）反而变差。验证了「老师必须优于学生」的直觉，也说明蒸馏不是本项目的关键杠杆，不再继续。

## 10. 阶段 mix_v1：四域配比怎么定？

**动机**：s3 证明纯网页不够。那么网页/数学/代码该按什么比例？

**设计**：约 1.07B token/组，4 组各 2 卡 —— QAT 三档配比 + BF16 对照。网页内部中英各半，共享同一数据池，**所有 data_passes ≤ 1**（不重复数据）。

| GPU | 模式 | 网页/数学/代码 |
|---|---|---|
| 0–1 | QAT | 100/0/0 |
| 2–3 | QAT | 70/15/15 |
| 4–5 | QAT | 50/25/25 |
| 6–7 | BF16 对照 | 70/15/15 |

**结论**：三组 QAT 里四域等权平均 CE 最低者胜出，作为后续起点；BF16 只作精度对照，不参与 QAT 选优。这一轮建立了「用四域开发 CE 选配比」的评测纪律。

## 11. 阶段 mix_v2：训练量翻倍 + 拉长 ramp

**动机（两个独立假设）**：训练量 1B→2B 能否提升下游？ramp 0.1→0.3 能否减少量化损失？

**设计**：只改这两个变量，其余全部与 mix_v1 相同。

| 项目 | mix_v1 | mix_v2 |
|---|---|---|
| 总训练量 | 1.074B token | 2.001B token |
| 更新步数 | 1,024 | 1,908 |
| **alpha_ramp_frac** | **0.1（102 步）** | **0.3（572 步）** |
| GPU 分配 | 2 卡/组 ×4 | 4 卡/组 ×2 |

数据池：web_zh 1.84B / web_en 1.92B / math 350M / code 350M。math、code 两轮**逐字节相同**（SHA256 一致），所以**网页语料是唯一变量**（v1 33 片 → v2 109 片，v1 全含于 v2）。实际抽到：web_zh 699M、web_en 700M、math 300M、code 301M。

### 11.1 训练曲线（真实数据）

**图 `figs/pretrain_train_loss.png`（左）**：mix_v2 QAT vs BF16 的训练 CE。注意 `α` 从 0 升到 1 的过程中，两条曲线都上扬一下再回落 —— 这是量化注入造成的。

**图 `figs/pretrain_gradnorm.png`**：梯度范数在 ramp 完成处的尖峰。`mix_v1`（ramp 0.1）冲到 3.88，`mix_v2`（ramp 0.3）只到 ~1.5。

采样后的训练 CE（每 ~128 步取一点）：

| step | mix_v2 QAT train CE | mix_v2 BF16 train CE |
|---:|---:|---:|
| 1 | 2.53 | 2.53 |
| 61 (α≈0.10) | 2.36 | 2.35 |
| 361 (α≈0.63) | 2.34 | 2.25 |
| 571 (α=1.00) | 2.62 | 2.25 |
| 1021 | 2.61 | 2.33 |
| 1908 | 2.34 | 2.15 |

**解释**：step 570 附近 QAT 的 train CE 从 ~2.4 抬到 ~2.6 —— 这不是「训练变差」，而是 `α=1` 后模型必须用三值权重去拟合同一批数据，损失天花板本来就更高。BF16 没有这个台阶。

### 11.2 验证曲线：量化适应期与「浮点前向崩溃」

**图 `figs/pretrain_val.png`** 是本轮最有教学价值的一张。左图：验证 CE，三值前向（实线）vs 浮点前向（虚线）。

| step | QAT 三值 | QAT 浮点 | BF16 三值 | BF16 浮点 |
|---:|---:|---:|---:|---:|
| 0 | 12.003 | 2.073 | 12.003 | 2.073 |
| 256 | 9.372 | 2.380 | 13.524 | 1.868 |
| 384 | 6.373 | 6.777 | 14.741 | 1.853 |
| 512 | 2.544 | 10.057 | 15.559 | 1.840 |
| 1024 | 2.058 | 9.853 | 19.144 | 1.815 |
| 1908 | **1.897** | 10.193 | 22.444 | **1.720** |

**三条读法**：

1. **12.0 → 2.5 的悬崖就是量化适应期**：`α` 升到 1 之前，三值前向用的是一半浮点一半三值的权重、量化误差巨大（step 0 CE 12.0）；`α=1` 之后模型已经学会在三值约束下工作，CE 骤降。
2. **QAT 的浮点前向随训练上升**（2.07→10.19）：master 越来越「只为三值服务」，与 §4 的原理完全一致。**所以 QAT 评测必须用三值列，不能用浮点列。**
3. **BF16 的三值前向爆炸**（12→22.4）：把一个浮点权重直接三值化就是垃圾，这是「不训练直接量化」为什么失败的直接证据。

**图 `figs/pretrain_val.png`（右）**：`mix_v1`（ramp 0.1，紫）在 step 128 就降到 2.36（ramp 早结束），`mix_v2`（ramp 0.3，红）此时还在 11.03，到 step 640 才降到 2.13。两条曲线最终：v1 在 1024 步 1.934，v2 在 1908 步 **1.897**。

> **注意对照口径**：v1 只有 1024 步，v2 有 1908 步。在**相同**的 1024 步处，v2 是 2.058，v1 是 1.934 —— v2 在同等步数下**更差**，因为网页语料换了一批更难拟合的。继续训练到 2B 后 v2（1.897）才超过 v1 的最终值（1.934）。

### 11.3 各域最终 CE

| 域 | mix_v2 QAT（三值） | mix_v2 BF16（浮点） |
|---|---:|---:|
| web_zh | 2.955 | 2.712 |
| web_en | 2.859 | 2.640 |
| math | 1.244 | 1.089 |
| code | 0.532 | 0.438 |
| **四域平均** | **1.897** | **1.720** |

code 域 CE 最低（0.53）是因为代码 token 可预测性高 —— 这也解释了后面 SFT 里「带代码的数据集 train CE 天然更低」（§13）。

### 11.4 下游评测与关键反直觉发现

| 指标 | mix_v2 QAT | mix_v2 BF16 | mix_v1 QAT | mix_v1 BF16 |
|---|---:|---:|---:|---:|
| GSM8K strict | 1.74% | 9.25% | 0.83% | 13.04% |
| MMLU | 24.29% | 29.03% | 26.61% | 31.61% |
| CMMLU | 25.04% | 27.10% | 25.47% | 30.73% |
| C-Eval | 24.96% | 28.31% | 24.74% | 31.05% |

**发现**：

1. **ramp 拉长有效**：三值 CE 1.934→1.897，QAT 与 BF16 的差距从 0.185 收窄到 0.177。
2. **训练量翻倍，下游反而全线下降**（QAT MMLU 26.61→24.29，BF16 MMLU 31.61→29.03）。**QAT 与 BF16 同步下降 → 不是量化问题**，而是网页语料换了一批、更难学，模型把容量分配到新分布上，多选题没受益。这是「CE 更低 ≠ 下游更好」的又一个实例。
3. **GSM8K 全部 1,319 题顶到长度上限**（`length_stop_count=1319`）：没 SFT 的模型**不会停止生成**，分数只反映碰对概率 —— 此时讨论 GSM8K 高低没有意义。**直接推出 SFT。**

---

# 第三部分 SFT 实验：四轮配比探索

## 12. SFT 基础：只对答案算 loss

统一数据格式 `{"prompt", "response", "source", "id"}`。编码时 prompt 部分屏蔽（label = −100），只监督 response：

```python
def encode_record(tok, record, max_length):
    """BOS + prompt tokens (no supervision) + response tokens + EOS (supervised)."""
    prompt_ids = tok.encode(record['prompt'], add_special_tokens=True, truncation=False)
    if prompt_ids and prompt_ids[-1] == tok.eos_token_id:      # 去掉 prompt 末尾误加的 EOS
        prompt_ids = prompt_ids[:-1]
    resp_ids = tok.encode(' ' + record['response'], add_special_tokens=False, truncation=False)
    if not resp_ids or resp_ids[-1] != tok.eos_token_id:
        resp_ids.append(tok.eos_token_id)
    ids = prompt_ids + resp_ids
    if len(ids) > max_length:
        return None                                            # 超长直接丢，不截断
    labels = [-100] * len(prompt_ids) + resp_ids               # prompt 部分屏蔽
    return {'input_ids': ids, 'labels': labels, ...}
```

三个要点：

- **loss mask**：只对 response 计算 loss，教「怎么回答」而不是「复述问题」。
- **超长直接丢弃**而不是截断：截断会砍掉答案尾部，教出不完整答案。
- response 前加空格，稳定 tokenization 边界。

**loss 归约按 response token 数**（分母只数非 −100 的位置）：

```python
active = lbls[:, 1:].ne(-100)
n = int(active.sum().item())
ce = F.cross_entropy(..., reduction='sum')
(ce / max(n, 1)).backward()
```

梯度累积用 `no_sync` 减少通信（只在最后一个 micro-batch 同步）。所有 SFT 起点相同：`runs/mix_v2_qat_sft/snapshots/epoch_4/model.pt`（已做过 GSM8K SFT 的 QAT 模型），加载后强制 `set_alpha(1.0)`，并断言初始 checkpoint 是 FP32 master。

### 12.1 先做一轮 GSM8K-only SFT：修复格式

**动机**：§11.4 发现基础模型不会停止。先用纯 GSM8K 教它格式。

**曲线（图 `figs/gsm8k_sft_loss.png`）**：train CE 从 1.48 一路崩到 **0.11** —— 典型的**记忆化**（只有 6,973 条数据，5 epoch）。

**结果**：开发集 strict 从 ~1% 升到 **14.00%**，停止率 92.3%。**格式学会了（主动输出 EOS + `####`），但这不是「变聪明」，是把格式刻进权重。** 4 轮之后继续训练不再涨（ep4=ep5=14%）。

### 12.2 主线目标与四轮推理链

从这一轮起的目标变成：**在不用纯 GSM8K 覆盖的前提下，保住 GSM8K 格式/能力，同时用通用数据提升选择题（MMLU/CMMLU/C-Eval）。** 四轮实验，每轮的结论决定下一轮：

```
轮1 纯通用            ──► 选择题没涨，GSM8K 崩了 → 必须把 GSM8K 混回去
轮2 通用 + GSM8K×3     ──► 格式修好了(98.9%)，正确率只回到 8.87% → 想靠加代码补
轮3 v2: +代码 +GSM8K×11 ──► 反而全线更差，dev_ce 单调上升=过拟合 → 重复倍数是毒
轮4 v3: GSM8K×2 +多样数学 ──► 三项更好，MMLU 最优，dev_ce 单调下降
```

## 13. 四轮结果与曲线

**图 `figs/sft_curves.png`** 是四轮的核心图。左：每轮末的 dev_ce；右：训练 CE（平滑）。

### 轮 1：纯通用数据（GSM8K 占比 0）

数据 `data/alpaca_sft_v1`，495,219 条（firefly 38.5万 + alpaca 英 5.2万 + 中 4.9万 + hc3 1万）。

| 指标 | 纯 GSM8K SFT (基线) | + 通用 SFT |
|---|---:|---:|
| GSM8K strict | 11.90% | **3.56%** |
| MMLU | 24.79% | 24.88% |
| CMMLU | 24.38% | 25.25% |
| C-Eval | 27.12% | 26.82% |
| GSM8K 停止率 | 92.3% | **66%**（449/1319 顶上限） |

dev_ce：2.9311 → 2.8994 → 2.8883（在降，但那是通用数据自己的 loss）。

**结论**：通用数据**没带来选择题收益**（基本持平），却把 GSM8K 打崩（格式退化到 66% 停止）。原因：49.5 万条里 GSM8K 占比 0，3 个 epoch 把 `####` 结尾 + 停止习惯稀释掉了。**教训：不能用纯通用数据覆盖专项能力。**

### 轮 2：GSM8K 混回通用数据（重复 3 倍）

数据 `data/alpaca_gsm8k_sft_v1`：GSM8K train（6,973 条）×3 混入，共 516,138 条，**GSM8K 占比 4.05%**，3 epoch 下每条被看到 9 次。

| 指标 | 纯 GSM8K | 纯通用 | 混合 GSM8K×3 |
|---|---:|---:|---:|
| GSM8K strict | 11.90% | 3.56% | **8.87%** |
| MMLU / CMMLU / C-Eval | — | — | 24.72 / 24.76 / 27.79 |
| GSM8K 停止率 | 92.3% | 66% | **98.9%** |

dev_ce：2.0663 → 2.0578 → 2.0655（最优 ep2）。

**结论**：混入 GSM8K **完全修复格式**（停止率 66%→98.9%，甚至超过纯基线 92.3%），正确率部分恢复（3.56%→8.87%）但仍低于 11.90%。选择题依然无收益。→ 假设：是不是数据不够多元（缺代码）？于是加代码。

### 轮 3（v2）：加代码 + GSM8K 重复 11 倍

数据 `data/sft_v2`：通用 59.7% / 代码 25.0% / 数学 15.3%，499,804 条。代码来自 `opc-annealing-corpus` 的 algorithmic_corpus（7 语言均衡）+ firefly Program。**相对轮 2 改了两处：代码 0→25%、GSM8K 重复 3→11 倍。**

**训练信号（图 `figs/sft_curves.png` 左，红线）**：dev_ce **逐轮上升** 2.1609 → 2.1980 → 2.2305 —— **过拟合**，最优落在 **epoch 1**。

| 指标 | 混合×3 (ep2) | v2 ×11 (ep1) |
|---|---:|---:|
| GSM8K strict | 8.87% | **7.35%** ↓ |
| MMLU | 24.72% | **24.00%** ↓ |
| CMMLU | 24.76% | 24.99% |
| C-Eval | 27.79% | **25.71%** ↓ |

**结论（两条，直接决定 v3）**：

1. **GSM8K 重复 11 倍反而更差**（7.35% < 8.87%）。重复越多分数越低，配合 dev_ce 单调上升 → 高重复是**过拟合**，不是更好的格式习得。
2. **加代码没带来收益，反而普遍下降**：MMLU 24.00%、C-Eval 25.71% 是四轮里最低。代码占 25% 挤占了通用份额，而**代码能力这一轮无法评测**（没装 HumanEval/MBPP），只看到通用指标的代价。

### 轮 4（v3）：降重复 + 引入多样数学

数据 `data/sft_v3`（497,881 条）：**通用和代码与 v2 逐条相同**，只改数学类构成：

- GSM8K 重复 **11 → 2**（13,946 条，3 epoch 下每条暴露 **6 次**，v2 是 33 次）。
- 新增 **UltraData-Math L3 QA-Synthetic 60,834 条**（UltraData-Math 原始是 parquet 纯文本，按「开头 600 字符内最后一个 `?` 后换行」切分题目/解答；污染过滤剔除 219 条）。

**训练信号（图 `figs/sft_curves.png` 左，橙线）**：dev_ce **单调下降** 2.0779 → 2.0631 → 2.0626，最优 **epoch 3** —— 与 v2 的过拟合**方向相反**，说明重复倍数 2 更健康。

| 指标 | 纯GSM8K | 纯通用 | 混合×3 | v2 ×11 | **v3 ×2+Ultra** |
|---|---:|---:|---:|---:|---:|
| GSM8K strict | 11.90% | 3.56% | 8.87% | 7.35% | **9.25%** |
| MMLU | 24.79% | 24.88% | 24.72% | 24.00% | **25.44%** |
| CMMLU | 24.38% | 25.25% | 24.76% | 24.99% | 24.86% |
| C-Eval | 27.12% | 26.82% | 27.79% | 25.71% | 26.30% |
| GSM8K 停止率 | 92.3% | 66% | 98.9% | 98.3% | **99.1%** |

**结论**：

1. 与 v2 只差数学类，**四项里三项更好**（GSM8K +1.90、MMLU +1.44、C-Eval +0.59）。
2. **MMLU 25.44% 是五次实验最高，也是唯一明显高于基线（24.79%）的** —— 多样数学数据对选择题可能有正向迁移。
3. GSM8K 仍低于纯基线 11.90%：格式已修满（停止率 99.1% 最高），差距在**答案正确率**。通用+代码占 85% 时，GSM8K 很难回到纯数据水平。
4. dev_ce 单调下降 + 最优在第 3 轮，与 v2 过拟合形成对照 —— **重复倍数是关键变量**。

### 13.1 一个读曲线的陷阱

看 **图 `figs/sft_curves.png` 右**：v2/v3 的训练 CE（~1.2）**远低于**轮 1/轮 2（~2.6）。这不是「v2/v3 学得更好」——而是它们的**数据构成不同**：代码数据可预测性高（§11.3 里 code 域 CE 只有 0.53），把整体 CE 拉低了。**不同数据配比的 CE 绝对值不可直接比较**；dev_ce 同理（轮 2 的 dev 集含 500 条 GSM8K，轮 1 不含）。**跨实验只能比同一份评测（同一个下游 test set）上的指标。**

---

# 第四部分 数据工程

## 14. 可复现的窗口采样

数据被预处理成扁平 `uint32` token 流。`TokenStream` 切成不重叠的 `seq_len+1` 窗口：

```python
self.window = seq_len + 1
counts = np.asarray([len(f) // self.window for f in self.files], dtype=np.int64)
self.offsets = np.concatenate([[0], np.cumsum(counts)])
```

`MixedSampler` 按域权重采样，关键是**可重放**：

```python
def _perm(self, lang_i, epoch):
    key = (lang_i, epoch)
    if key not in self._perm_cache:
        rng = np.random.default_rng([self.seed, lang_i, epoch])
        ...
        self._perm_cache[key] = rng.permutation(self.streams[lang_i].num_windows)
```

每个 (域, epoch) 用固定 seed 派生排列 → 中断恢复、续训对照、不同 run 之间都能**逐样本对齐**。

`plan()` 会**预先算完整日程**并在启动前检查 `data_passes ≤ 1`（不足就报错，而不是静默重复数据）—— 避免「训练后期偷偷重复」这种无声的实验污染。

## 15. 污染过滤：只索引题干

核心设计：**绝不索引选项/答案**（A/B/C/D 高频，会误杀正常文本），只索引题干。

```python
NGRAM = 13
UNIT_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[㐀-䶿一-鿿豈-﫿\U00020000-\U000323af]")
```

两级匹配：

1. **完整题干嵌入**（归一化 NFKC + casefold + 空白折叠后整题作为子串出现）。
2. **13-unit n-gram**：文本切成英文词/数字/单个 CJK 字，任何连续 13 个 unit 命中 → 剔除整篇（抓「改动几个字」的近似重复）。

用 `pyahocorasick` 自动机加速。

**局限（必须诚实说明）**：精确子串 + 13-gram 会**过度剔除**模板化 boilerplate，也会**漏掉**语义改写；而且它只清训练语料里的基准题，**无法消除基座模型本身已有的污染**。

---

# 第五部分 踩过的坑

真实项目里一半时间花在数据格式和静默错误上。以下每个都实际发生过。

### 5.1 `splitlines()` 会吃掉 U+2028/U+2029

`data/gsm8k_sft_v1/train.jsonl` 用 `ensure_ascii=False` 写出，JSON 字符串里含原始行分隔符 U+2028/U+2029，`str.splitlines()` **把这些也当换行**，每行被切碎，`json.loads` 报 `Unterminated string`。

```python
# 正确做法：直接迭代文件句柄（Python 只按 \n/\r/\r\n 切）
with Path(path).open(encoding='utf-8') as fh:
    for line in fh:
        ...
```

后续所有 JSONL 一律 `ensure_ascii=True` 写出。

### 5.2 浮点权重不能用来评测 QAT 模型

见 §4 与 §11.2。EOS 诊断里，QAT SFT 的 FP32 master 在**浮点前向**下把停止符排到第 65,337 名，**三值前向**下排第 1 名（100%）。**结论：QAT 必须用 `α=1` 三值前向评测。**

### 5.3 停止符诊断揭示两种失败模式

| 模型 | 停止率 | 失败模式 |
|---|---:|---|
| QAT SFT | 92.3% 正常停止 | 顶上限后 87% 重复 `####` 循环 → **答案算错，格式正确** |
| BF16 SFT | 0.6% 正常停止 | 99% 续写新 `Question:` → **不知道答完就停，答案更准** |

**教训**：GSM8K 低分不能只看总分。停止符配置本身没问题，两个模型问题性质完全不同，诊断必须拆到「生成行为」层。

### 5.4 从数据泄漏到「验证脚本自己出错」

v3 数学切分最初用空行切，但 UltraData 的题目和解答**经常只隔一个换行**，导致解答开头（「To find the modulus…」）漏进 prompt —— **等于把答案的一半放进题目**。改成「开头 600 字符内最后一个 `?` 后换行」切分。

随后写脚本检查泄漏时，正则 `\n\s*(...)` 匹配到了 prompt 自己的结尾换行，报「100% 泄漏」。对 prompt 先 `.rstrip()` 后，真实泄漏率 9.3%，且都是合法的多问子题。**教训：验证脚本本身也要被怀疑；数字太极端先查自己。**

### 5.5 Launcher 的 PID 竞态

`mkdir -p dir && nohup ... & echo $! > dir/launcher.pid`：目录没建好 `nohup` 会失败，且 `$!` 捕获的是子 shell 而非真实训练进程。**修法**：`mkdir -p` 单独先跑，再用 `pgrep -f <script>` 定位真实 PID。

---

# 第六部分 结论

- **三值 QAT 的核心**：一行 `W + α·(Q(W)−W).detach()` 同时完成假量化、STE、渐进注入。
- **最重要超参**：`alpha_ramp_frac`。0.1 → 0.3 使三值 CE 1.934 → 1.897，并把 ramp 完成处的梯度尖峰从 3.88 压到 ~1.5。
- **CE 与下游不可互推**：s3、mix_v2 两次证明「CE 更低 / 训练更多」不保证下游更好；必须用同一份下游 test set 比较。
- **数据配比**：通用 60 / 代码 25 / 数学 15 这条线上，**降低专项重复倍数 + 引入多样数学**（v3）是四轮最好；GSM8K 格式靠混入修复（停止率 99.1%），但正确率在通用数据占 85% 时难以回到纯数据水平。
- **评测纪律**：QAT 必须用三值（α=1）前向；污染过滤只索引题干；不同配比的 CE 不可直接比；验证数字太极端先查脚本。

---

## 附录 A：图目录（`reports/figs/`）

| 图 | 内容 |
|---|---|
| `pretrain_train_loss.png` | mix_v2 训练 CE（QAT vs BF16）+ ramp 速度对 loss 尖峰的影响 |
| `pretrain_gradnorm.png` | ramp 完成处的梯度范数尖峰（mix_v1 vs mix_v2） |
| `pretrain_val.png` | 验证 CE 三值 vs 浮点前向；mix_v1 vs mix_v2 三值曲线 |
| `sft_curves.png` | 四轮 SFT 的 dev_ce / train CE |
| `gsm8k_sft_loss.png` | GSM8K-only SFT 的记忆化（CE 1.48→0.11） |

## 附录 B：关键文件索引

| 文件 | 作用 |
|---|---|
| [qat/quant.py](qat/quant.py) | 三值量化核心：`ternary_quantize` / `TernaryLinear` / `convert_to_ternary` |
| [qat/train.py](qat/train.py) | 预训练 QAT/BF16：WSD、alpha ramp、μP、双精度验证 |
| [qat/train_alpaca_sft.py](qat/train_alpaca_sft.py) | SFT 训练器：loss mask、DDP no_sync、dev CE 选优 |
| [qat/data.py](qat/data.py) | `TokenStream` / `MixedSampler`：可重放窗口采样 |
| [qat/decontam.py](qat/decontam.py) | 基准污染过滤：完整题干 + 13-unit n-gram |
| [qat/prepare_sft_v2.py](qat/prepare_sft_v2.py) / [prepare_sft_v3.py](qat/prepare_sft_v3.py) | v2/v3 配比构建 |
| [qat/mix_gsm8k_sft.py](qat/mix_gsm8k_sft.py) | GSM8K 重复混入通用数据 |
| [reports/pretrain_v2/mix_v2/EXPERIMENT_NOTES.md](reports/pretrain_v2/mix_v2/EXPERIMENT_NOTES.md) | 全部实验的原始结果记录 |
| [reports/OVERVIEW.md](reports/OVERVIEW.md) | 各阶段实验总览（含 s2/s3） |
| [reports/figs/](reports/figs/) | 本文所有 loss / CE 曲线图 |
