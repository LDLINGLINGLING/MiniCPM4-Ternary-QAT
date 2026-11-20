# MiniCPM4-0.5B 三值模型 bitnet.cpp 部署与推理验证

日期：2026-11-19。本文只记录本地私有 fork 的工程改动、真实运行结果与结论；训练过程见 `EXPERIMENT_NOTES.md`。

## 0. 结果概览

已在 Microsoft bitnet.cpp 的 vendored llama.cpp/GGML 中实现两种新格式，并完成真实 x86 CPU 推理：

| checkpoint / 部署格式 | 原生 HF 四域 macro CE | llama.cpp 四域 macro CE | 差值 | GGUF 文件 |
|---|---:|---:|---:|---:|
| `mix_v2_qat_70_15_15` / `I2_S_ROW` | 1.897251 | **1.899053** | +0.001801 | 694,336,128 B（662.171 MiB） |
| `mix_v3_bitcpm_pg32` / `I2_S_G32` | 1.884792 | **1.886447** | +0.001655 | 738,056,832 B（703.866 MiB） |

两种模型均完成：168 个投影的量化与 GGUF 回读、prefill、cached decode、真实 greedy 生成，以及四域各 256 个固定 2048-token 窗口的 CE。每域有效 next-token 预测数均为 **524,288**，八组任务均 exit=0。

**结论：两种部署都保持了原 checkpoint 的量化粒度和精度，没有重训，也没有把一种粒度转换成另一种。** G32 与 ROW 来自不同训练 run，二者的模型质量差异不能解释成“仅更换存储格式”的因果效果。

## 1. 为什么不能直接复用旧 I2_S

旧版活跃的 `I2_S` 是**整张量一个 scale**，而两个 checkpoint 分别是：

- ROW：每个输出通道、即每个权重输出行一个 absmean scale；
- G32：每行每连续 32 个输入权重一个 absmean scale。

此前实测把 G32 checkpoint 强行放进旧 per-tensor 路径后，四域 macro CE 从 **1.8848** 变为 **3.4265**。因此这里不是简单修改打包代码，而是需要新的静态 GGML 类型、GGUF 物理布局及对应计算 kernel。

两个 checkpoint 保存的是 FP32 master，而不是已经三值化的部署权重。转换必须走：

```text
FP32 master state.pt
  → 逐位不变的 FP32 HF safetensors 中间目录
  → F32 GGUF
  → 按 checkpoint 原生粒度量化成 I2_S_ROW 或 I2_S_G32
```

不能先保存反量化三值 `0, ±s` 再重算 absmean；那样会把 scale 错算成 `s × nonzero_density`。导出器同时校验 run 配置和全部 168 个完整投影名称，禁止 ROW/G32 交叉导出。

## 2. 新格式与计算语义

### 2.1 权重量化

对一个量化区域 `g`：

```text
s_g = max(mean_{j in g}(|W_j|), 1e-8)
q_j = clamp(round(W_j / s_g), -1, 1)
stored_code_j = q_j + 1             # 0/1/2
W_deploy_j = q_j × s_g
```

每个 2-bit code 按字节内高位到低位打包。物理布局如下：

| 格式 | GGML type | scale 数量 | 每行字节数 | 约束 |
|---|---:|---:|---:|---|
| `I2_S_ROW` | 43 | 每行 1 个 FP32 | `K/4 + 4` | K 为 32 的倍数 |
| `I2_S_G32` | 44 | 每行 `K/32` 个 FP32 | `K/4 + 4×K/32` | K 为 32 的倍数 |

两种格式均不使用旧 `I2_S` 的 32-byte trailer。GGML sizing、stride、GGUF offset、loader 和 kernel 使用同一物理行大小。旧 `I2_S` 枚举、布局和执行路径未改。

### 2.2 激活与点积

activation 仍量化为 int8。对 32 维块，整数计算为：

```text
integer_dot_g = Σ(code_i × q8_i) − Σ(q8_i)
```

- ROW：整行累加整数贡献，最后乘一个行 scale，再除以 activation scale；
- G32：每组分别计算 activation correction，乘该组 scale，按原组序累加，最后除以 activation scale。

因此新增的是**权重量化粒度**，不是把训练改成 activation 三值，也不是声称端侧结果只含 weight-roundtrip 误差。

### 2.3 x86 kernel

实现了：GEMV、GEMM、行/列尾部、真实多线程二维路径、量化 `get_rows`、标量回退、AVX2 和 VNNI。

- AVX-VNNI/AVX512-VNNI 使用 `_mm256_dpbusd_epi32`；
- AVX2 回退使用 `_mm256_maddubs_epi16` + `_mm256_madd_epi16`；
- code 作为 unsigned 0/1/2，activation 作为 signed int8；
- SIMD 只改变指令实现，不改变整数点积语义。

主要改动文件：

| 文件 | 内容 |
|---|---|
| `ggml/include/ggml.h`、`ggml/src/ggml.c` | 类型枚举、traits、物理大小、量化/反量化分发 |
| `ggml/src/gguf.cpp`、`gguf-py/gguf/*` | GGUF 类型、文件大小和 Python reader/writer |
| `ggml/src/ggml-cpu/quants.c` | 打包、反量化、ROW/G32 dot、AVX2/VNNI |
| `ggml/src/ggml-cpu/ggml-cpu.c` | int8 workspace、多线程 mul_mat 分发 |
| `ggml/src/ggml-cpu/ggml-cpu-i2s.c` | 二维 GEMM/GEMV 与尾部 |
| `ggml/src/ggml-cpu/ops.cpp` | 量化权重 get_rows |
| `src/llama-quant.cpp`、`tools/quantize/quantize.cpp` | 量化范围、回退和 CLI 类型 |
| `tests/test-i2s-scaled.cpp` | 新格式已知答案及 legacy 回归 |
| `tools/bitnet-eval/bitnet-eval.cpp` | 原始 token 的 prefill/decode/logits/CE 工具 |

MiniCPM 图继续复用 fork 现有 Granite 路径，保留 Q/K permutation、embedding/residual/logit scaling 和 long/short RoPE factors。HF 的 `hidden_states / 4` 与 GGUF `logit_scale=4`、图中 `1/f_logit_scale` 已核对，不另造一套模型架构。

## 3. 模型产物

| 产物 | 字节数 | 说明 |
|---|---:|---|
| 任一 F32 GGUF | 2,038,028,928 | 1943.616 MiB，转换中间产物 |
| `mix_v2_qat_70_15_15-I2_S_ROW.gguf` | **694,336,128** | 662.171 MiB；llama-quantize payload 660.56 MiB，10.88 BPW |
| `mix_v3_bitcpm_pg32-I2_S_G32.gguf` | **738,056,832** | 703.866 MiB；llama-quantize payload 702.25 MiB，11.57 BPW |

每个量化 GGUF 均有 **221 tensors = 168 个新类型投影 + 53 个 F32 张量**。219 个 HF 权重之外的两个张量是转换生成的 long/short RoPE factors。embedding、lm_head/output、norm 和 RoPE factors 保留 F32，因此整模型有效 BPW 明显高于投影本身的 2-bit 编码，不能直接把 10.88/11.57 BPW 当成三值投影的存储位宽。

ROW/G32 文件分别为 F32 文件的 **34.069% / 36.214%**。G32 多出的 43,720,704 B 主要来自每 32 权重一个 FP32 scale。

## 4. 分层验证结果

### 4.1 Kernel 已知答案与 legacy 回归

最终 SIMD 二进制重新运行 `test-i2s-scaled`，覆盖 K={32,64,128,256}、rows={1,3,5}、5 个 activation 列、零激活、scale、打包字节序、反量化、GEMV/GEMM 尾部、物理 stride、`ggml_nbytes`，以及真实三线程 GGML mul_mat。

实际结尾：

```text
ok: GGML CPU mul_mat i2_s K=128 rows=5 cols=4 threads=3
ok: GGML CPU mul_mat i2_s_row K=128 rows=5 cols=5 threads=3
ok: GGML CPU mul_mat i2_s_g32 K=128 rows=5 cols=5 threads=3
all scaled I2_S known-answer tests passed
```

旧 `I2_S` 的独立打包、反量化和真实 GGML 图回归也通过。完整输出保存为 `bitnet_cpp/test_i2s_scaled.log`。

### 4.2 GGUF 逐权重回读

验证器对全部 221 个张量检查 shape、类型、物理字节数、Q/K permutation、三值 code、scale 和 F32 精确值：

| 指标 | ROW | G32 |
|---|---:|---:|
| 量化 / 精确 F32 张量 | 168 / 53 | 168 / 53 |
| 非边界 code mismatch | **0** | **0** |
| FP32 half-threshold code 差异 | 205 | 29 |
| 最大边界归一化偏差 | 9.5367e-7 | 8.9407e-8 |
| 最大 scale 绝对误差 | 2.0117e-7 | 8.9407e-8 |
| 最大 scale 相对误差 | 4.6271e-6 | 5.3194e-7 |
| code 相同时最大权重误差 | 2.0117e-7 | 8.9407e-8 |

这些结果**不能描述为逐位完全相同的三值 code**。准确结论是：非半阈值位置完全一致；所有边界翻转均已计数，最大归一化边界距离不超过 9.54e-7，来源于 C++ 与 PyTorch FP32 reduction-order 差异。

包含边界翻转时，ROW/G32 最大单权重差分别为 0.0713753 / 0.0668058；这是一档三值 code 翻转造成的幅值，不应与“同 code 的 scale 误差”混为一谈。

### 4.3 HF 与 llama.cpp logits：prefill 和 cached decode

协议：web_zh 原始 token，32-token prefill + 8 个 cached decode 位置，vocab=73,448；HF 使用 FP32 master、168 个 TernaryLinear、alpha=1、CUDA BF16 autocast。native 使用 int8 activation。

| 指标（40 位置总体） | ROW | G32 |
|---|---:|---:|
| max absolute logit error | 1.295019 | 1.681273 |
| mean absolute logit error | 0.101823 | 0.087666 |
| RMSE | 0.135220 | 0.124521 |
| argmax agreement | 95.0% | 97.5% |
| native CE | 4.573949 | 4.564454 |
| HF CE | 4.566381 | 4.531025 |
| prefill argmax agreement | 93.75% | 100% |
| cached-decode argmax agreement | 100% | 87.5% |

短序列 cached-decode native/HF CE：ROW **2.228561 / 2.204185**，G32 **2.649003 / 2.680007**。

最大的单个 logit 误差必须结合 int8 activation、HF BF16 autocast 和浮点累加顺序理解，不能当成纯权重回读误差；权重回读误差已在 §4.2 单独报告。40 个位置只用于执行路径对齐，不用于评价整体模型质量。

### 4.4 一个完整 2048-token 窗口的 HF 对照

域为 web_zh、window=0，CE 使用 2048 个 shifted targets：

| 模型 | HF CE | llama.cpp CE | 差值 |
|---|---:|---:|---:|
| ROW | 3.570723 | 3.573723 | +0.003000 |
| G32 | 3.541007 | 3.540195 | −0.000812 |

这一步排除了“只有 32-token 短序列能对齐”的假象，也为后面的完整四域评测提供了直接依据。

## 5. 完整固定窗口 CE

### 5.1 协议

- 每个模型分别在 web_zh、web_en、math、code 上运行；
- 每域取原始 uint32 token stream 的前 256 个互不重叠窗口；
- 每窗读取 2049 token：2048 inputs + 1 个 shifted target；
- 每窗重置 KV memory，所有 prefill 位置请求 logits；
- 不重新 tokenize，不混入 cached decode；
- 每域 524,288 predictions，与原生 HF evaluation 使用同一批窗口。

### 5.2 逐域真实结果

| 模型 | 域 | 原生 HF CE | llama.cpp CE | llama − HF | llama top-1 |
|---|---|---:|---:|---:|---:|
| ROW | web_zh | 2.954505 | **2.956743** | +0.002238 | 46.0934% |
| ROW | web_en | 2.858753 | **2.860963** | +0.002210 | 44.0302% |
| ROW | math | 1.243507 | **1.245192** | +0.001685 | 71.3264% |
| ROW | code | 0.532241 | **0.533313** | +0.001072 | 86.0779% |
| **ROW** | **四域 macro** | **1.897251** | **1.899053** | **+0.001801** | — |
| G32 | web_zh | 2.937009 | **2.939107** | +0.002098 | 46.2677% |
| G32 | web_en | 2.845672 | **2.847640** | +0.001968 | 44.1811% |
| G32 | math | 1.232789 | **1.234289** | +0.001500 | 71.4926% |
| G32 | code | 0.523697 | **0.524752** | +0.001055 | 86.2673% |
| **G32** | **四域 macro** | **1.884792** | **1.886447** | **+0.001655** | — |

逐任务 JSON 还校验了 `windows=256`、`sequence_length=2048`、`decode_steps=0`、`token_predictions=524288` 和有限 CE/top-1。汇总见 `bitnet_cpp/ce_summary.json`。

### 5.3 结论

1. 两种格式在四域上都只有 **0.001055–0.002238** 的 CE 偏差，macro 偏差分别为 **+0.001801 / +0.001655**，没有出现旧 per-tensor 路径那种粒度错误。
2. 八组差值均为小幅正值，符合 native int8 activation、HF BF16 autocast 和不同浮点累加实现共同造成的微小系统差异；不能把该差值全部归因于三值权重。
3. llama.cpp 下 G32 macro 比 ROW 低 **0.012606**，原生 HF 下低 **0.012460**；部署基本保留了两个 checkpoint 原有的相对差异。
4. 八组 CE 是 8×24 线程并发运行的，耗时包含 CPU/内存带宽争用，**不作为性能结果**；性能只看隔离基准。

## 6. 隔离性能测试

### 6.1 标准 llama-bench

环境：Intel Xeon Platinum 8480+，CPU backend，`n_gpu_layers=0`。两个模型顺序执行，不与 CE 或其他推理并发。参数为 `p=512`、`n=128`、`threads=16,64`、`repetitions=3`、`n_batch=2048`、`n_ubatch=512`，保留工具默认 warmup 和 mmap。

| 模型 | 线程 | 512-token prefill，tok/s | 128-token decode，tok/s |
|---|---:|---:|---:|
| ROW | 16 | 169.316 ± 25.492 | 101.230 ± 9.267 |
| ROW | 64 | **525.705 ± 43.981** | **133.395 ± 5.854** |
| G32 | 16 | 190.826 ± 33.683 | 91.686 ± 2.863 |
| G32 | 64 | 483.244 ± 16.051 | 128.685 ± 2.036 |

表中为 3 次均值 ± llama-bench 报告的 stddev，原始每次样本保存在 `bench_row.json` / `bench_g32.json`。prefill 波动较大，尤其第一次样本偏慢，因此不能只挑最快一次作为结果。

整个 benchmark 进程峰值 RSS：ROW **751,992 KiB（734.4 MiB）**，G32 **794,640 KiB（776.0 MiB）**。这是覆盖两种线程配置的套件级峰值；真实 CLI、固定 `c=512` 的内存见 §7。

结论：

1. 16→64 线程，ROW/G32 prefill 分别提升 **3.10× / 2.53×**，decode 只提升 **1.32× / 1.40×**。decode 不随线程线性扩展，已受到内存带宽和线程同步限制。
2. 16 线程时 G32 prefill 较快，64 线程时 ROW 较快；decode 则 ROW 在 16/64 线程分别高 **10.4% / 3.7%**。因此不能笼统宣称“G32 一定更快”，应按线程数和工作负载选择。
3. G32 文件比 ROW 大约 **6.30%**，并增加每组 scale 读取与修正；其优势是保持对应 checkpoint 的精度粒度，不是无条件吞吐优势。
4. 本结果只代表该 Xeon 服务器，不能外推到手机、ARM、Metal 或 CUDA。

### 6.2 完整 2048-token prefill 的 SIMD 前后对照

使用同一个 web_zh window=0、`n_batch=n_ubatch=2048`，此前还保留了一组开发和优化过程中的直接对照：

| 模型 / 实现 | 线程 | 总耗时 | prefill 耗时 | CE |
|---|---:|---:|---:|---:|
| ROW scalar | 16 | 46.756 s | 46.689 s | 3.573722934 |
| ROW scalar | 64 | 19.185 s | 19.165 s | 3.573722934 |
| ROW SIMD | 64 | **8.152 s** | **8.122 s** | 3.573722934 |
| G32 SIMD | 64 | **6.734 s** | **6.709 s** | 3.540194581 |

同样 64 线程下，ROW 从 scalar 到 SIMD 加速 **2.35×**，CE 完全不变。该表是单次长窗口测量，和 §6.1 的 512-token、3 次标准基准互补；注意力长度和测量协议不同，不能直接比较两组 tok/s。

此外，优化前后两份 40×73,448 FP32 logits 文件均逐字节一致：SHA256 分别为 `dd13f809…9589`（ROW）和 `f36b3805…dd7a`（G32）。这证明 SIMD 没有改变这些固定输入的数值输出。

## 7. 真实 greedy 生成与内存

统一命令：prompt=`The capital of France is`，`n=24`、`t=tb=16`、`c=512`、`temp=0`、`top-k=1`、`--no-warmup --single-turn`。下表直接对比优化前后相同 CLI 路径：

| 模型 | 指标 | scalar | 最终 SIMD | 变化 |
|---|---|---:|---:|---:|
| ROW | prompt | 31.3 t/s | 123.2 t/s | 3.94× |
| ROW | generation | 24.7 t/s | **62.7 t/s** | **2.54×** |
| ROW | wall，包括加载 | 7.805 s | 6.567 s | 1.19× |
| ROW | peak RSS | 717,124 KiB | **716,224 KiB（699.4 MiB）** | 基本不变 |
| G32 | prompt | 27.7 t/s | 108.3 t/s | 3.91× |
| G32 | generation | 23.5 t/s | **51.8 t/s** | **2.20×** |
| G32 | wall，包括加载 | 8.151 s | 6.602 s | 1.23× |
| G32 | peak RSS | 759,680 KiB | **759,096 KiB（741.3 MiB）** | 基本不变 |

两次 greedy 文本保持一致：

- ROW：`The capital of France is Brunswick, which is located in the province of Brunswick. Brunswick is a small town in the province`
- G32：输出以 `The capital of Germany is(["Auflauter"] ...` 开始，完整内容见日志。

这些是**未 SFT 的 base model**，G32 的短样例不是有用回答；生成质量不作为 kernel 正确性的唯一判据，精度判断使用 §4–5 的权重、logits 和固定窗口 CE。

24-token CLI 的 generation 数字低于 128-token llama-bench 稳态结果，主要因为短测量的首尾开销占比更高。wall time 还包含模型加载，所以 wall 加速比不能代替 decode kernel 加速比。完整输出、命令和 wrapper 统计保存在 `generate_*_scalar.log` / `generate_*_simd.log`。

## 8. 复现命令

以下变量对应本次实际路径：

```bash
ROOT=/chubao/tj-train-ssd-21/liudan190
QAT=$ROOT/QAT_minicpm
BN=$ROOT/bitnet.cpp
LL=$BN/3rdparty/llama.cpp
OUT=$QAT/reports/pretrain_v2/mix_v3_bitcpm/bitnet_cpp
```

构建和已知答案：

```bash
cmake --build "$BN/build" --target llama-cli llama-quantize \
  llama-bitnet-eval llama-bench test-i2s-scaled -j 8
"$BN/build/bin/test-i2s-scaled"
```

FP32 master 导出（输出目录必须不存在，工具拒绝覆盖）：

```bash
python3 "$QAT/tools/export_bitnet_hf.py" \
  --checkpoint "$QAT/runs/mix_v2_qat_70_15_15/checkpoints/final/state.pt" \
  --run-config "$QAT/runs/mix_v2_qat_70_15_15/resolved_config.json" \
  --base-model "$QAT/MiniCPM4-0.5B-fjzcuwlf" \
  --output-dir "$QAT/exports/mix_v2_qat_70_15_15_fp32_master" \
  --granularity row

python3 "$QAT/tools/export_bitnet_hf.py" \
  --checkpoint "$QAT/runs/mix_v3_bitcpm_pg32/checkpoints/final/state.pt" \
  --run-config "$QAT/runs/mix_v3_bitcpm_pg32/resolved_config.json" \
  --base-model "$QAT/MiniCPM4-0.5B-fjzcuwlf" \
  --output-dir "$QAT/exports/mix_v3_bitcpm_pg32_fp32_master" \
  --granularity g32
```

F32 GGUF 与目标格式（另一个模型按同样参数替换路径和类型）：

```bash
python3 "$LL/convert_hf_to_gguf.py" \
  "$QAT/exports/mix_v2_qat_70_15_15_fp32_master" \
  --outfile "$QAT/gguf/mix_v2_qat_70_15_15-F32.gguf" --outtype f32
"$BN/build/bin/llama-quantize" \
  "$QAT/gguf/mix_v2_qat_70_15_15-F32.gguf" \
  "$QAT/gguf/mix_v2_qat_70_15_15-I2_S_ROW.gguf" I2_S_ROW 16

python3 "$LL/convert_hf_to_gguf.py" \
  "$QAT/exports/mix_v3_bitcpm_pg32_fp32_master" \
  --outfile "$QAT/gguf/mix_v3_bitcpm_pg32-F32.gguf" --outtype f32
"$BN/build/bin/llama-quantize" \
  "$QAT/gguf/mix_v3_bitcpm_pg32-F32.gguf" \
  "$QAT/gguf/mix_v3_bitcpm_pg32-I2_S_G32.gguf" I2_S_G32 16
```

GGUF 回读验证（G32 替换三个对应路径和 `--granularity g32`）：

```bash
python3 "$QAT/tools/verify_bitnet_gguf.py" \
  --gguf "$QAT/gguf/mix_v2_qat_70_15_15-I2_S_ROW.gguf" \
  --hf-dir "$QAT/exports/mix_v2_qat_70_15_15_fp32_master" \
  --granularity row --llama-dir "$LL" --qat-root "$QAT" \
  --output "$OUT/mix_v2_row_roundtrip.json"
```

完整 CE 单域命令；其余七组只替换模型、token 域和输出前缀，协议不变：

```bash
OMP_NUM_THREADS=24 "$BN/build/bin/llama-bitnet-eval" \
  --model "$QAT/gguf/mix_v2_qat_70_15_15-I2_S_ROW.gguf" \
  --tokens "$QAT/data/mix_v2/web_zh_val.bin" \
  --seq-len 2048 --windows 256 --threads 24 --threads-batch 24 \
  --metrics-out "$OUT/row_web_zh_ce.json"
```

统一性能基准：

```bash
"$BN/build/bin/llama-bench" \
  -m "$QAT/gguf/mix_v2_qat_70_15_15-I2_S_ROW.gguf" \
  -p 512 -n 128 -t 16,64 -r 3 -ngl 0 -o json
```

真实生成：

```bash
"$BN/build/bin/llama-cli" \
  -m "$QAT/gguf/mix_v2_qat_70_15_15-I2_S_ROW.gguf" \
  -p 'The capital of France is' -n 24 -t 16 -tb 16 -c 512 \
  --temp 0 --top-k 1 --no-warmup --no-display-prompt --single-turn
```

## 9. 限制与最终结论

- **仅支持并验证 x86 CPU**；未实现或验证 Metal/CUDA/ARM。
- activation 是 int8；HF 对照使用 BF16 autocast，因此端到端差异不是纯 weight-roundtrip 误差。
- 三值 code 在非半阈值位置完全一致，半阈值翻转已如实统计，不宣称全部 code bitwise-identical。
- legacy `I2_S` 已由已知答案和真实 GGML 回归覆盖，但本轮没有重新评测官方 BitNet 模型的完整 CE/性能。
- 结果仅适用于当前 MiniCPM4-0.5B dense GQA 转换路径，不外推到其他架构或官方模型。
- 两种格式都以 FP32 embedding/lm_head 为代价保留现有数值语义；进一步压缩这些 F32 张量不在本轮范围。
- 所有修改只保留在本地私有 fork；未提交、未 push、未创建上游 PR。

**最终结论：两种 checkpoint 均已完成可运行、可复现、数值对齐的 bitnet.cpp x86 部署。ROW/G32 的完整 macro CE 分别只偏移 +0.001801/+0.001655；最终 SIMD 使真实 16 线程短生成达到 62.7/51.8 token/s，稳态 64 线程 decode 达到 133.4/128.7 token/s，峰值内存没有因 SIMD 增加。**

## 10. 结果文件索引

| 文件（均在本报告目录下） | 内容 |
|---|---|
| `bitnet_cpp/ce_summary.json` | 八组最终 CE、top-1、native 对照及校验后的 macro |
| `bitnet_cpp/{row,g32}_{domain}_ce.json/.log` | 每域完整原始结果与 256 窗日志 |
| `bitnet_cpp/mix_v2_row_roundtrip.json` | ROW 全张量回读结果 |
| `bitnet_cpp/mix_v3_g32_roundtrip.json` | G32 全张量回读结果 |
| `bitnet_cpp/mix_v2_row_logits_comparison.json` | ROW prefill/decode HF logits 对照 |
| `bitnet_cpp/mix_v3_g32_logits_comparison.json` | G32 prefill/decode HF logits 对照 |
| `bitnet_cpp/full_window_hf_check.json` | 完整 2048-token HF 对照 |
| `bitnet_cpp/test_i2s_scaled.log` | 最终 kernel 与 legacy 回归输出 |
| `bitnet_cpp/simd_logits_regression.log` | SIMD 前后 logits SHA256 与 cmp 结果 |
| `bitnet_cpp/benchmark_summary.json` | 吞吐、波动、内存和真实生成加速汇总 |
| `bitnet_cpp/bench_{row,g32}.json/.log` | llama-bench 原始结果 |
| `bitnet_cpp/generate_{row,g32}_{scalar,simd}.log` | 真实生成文本、速度、命令和峰值 RSS |
| `bitnet_cpp/{export,convert,quantize}_*.log` | 导出、F32 转换和量化原始日志 |
