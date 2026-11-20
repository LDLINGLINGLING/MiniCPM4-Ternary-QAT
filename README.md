# MiniCPM4-0.5B 三值 QAT 与原生 x86 推理

本仓库是一套可运行的 MiniCPM4-0.5B 三值量化感知训练（QAT）、数据构建、SFT、模型导出和 CPU 推理工程。部署部分包含修改后的 Microsoft `bitnet.cpp` 及其 vendored `llama.cpp/GGML`，同时支持两种已经训练好的原生量化粒度：

- `I2_S_ROW`：每个输出权重行一个 scale，对应 `mix_v2_qat_70_15_15`。
- `I2_S_G32`：每行每连续 32 个输入权重一个 scale，对应 `mix_v3_bitcpm_pg32`。

两种粒度不能交叉转换。仓库不包含训练好的 checkpoint、原始语料、完整模型 GGUF 或构建产物；这些文件需要按下文下载或生成。

## 1. 先看结论与验证边界

### 1.1 已保留的历史正式实验结果

以下数字来自原实验保存的 JSON、日志和报告，不是本次整理目录后重新训练或重新跑完整评测得到的：

| checkpoint / 部署格式 | 原生 HF 四域 macro CE | llama.cpp 四域 macro CE | 差值 | GGUF 大小 |
|---|---:|---:|---:|---:|
| `mix_v2_qat_70_15_15` / `I2_S_ROW` | 1.897251 | **1.899053** | +0.001801 | 694,336,128 B，662.171 MiB |
| `mix_v3_bitcpm_pg32` / `I2_S_G32` | 1.884792 | **1.886447** | +0.001655 | 738,056,832 B，703.866 MiB |

四域评测使用相同原始 uint32 token stream：每域前 256 个互不重叠窗口，每窗 2049 token，形成 2048 次 next-token prediction，即每域 **524,288** 次预测。完整逐域结果见[部署报告](docs/reports/pretrain_v2/mix_v3_bitcpm/BITNET_CPP_DEPLOYMENT.md)和 [CE 汇总](docs/reports/pretrain_v2/mix_v3_bitcpm/bitnet_cpp/ce_summary.json)。

### 1.2 本次功能整理后的实际检查

| 检查 | 最新真实结果 | 覆盖范围 |
|---|---|---|
| Python 离线测试 | **105 passed in 28.20s** | 下载格式校验、tokenize、数据恢复、混合采样、SFT、导出、严格恢复辅助逻辑 |
| Python 包构建与安装 | 成功 | wheel 构建、隔离解释器安装及 `qat/tools/data_pipeline` 导入 |
| 原生构建包装脚本 | 成功 | `llama-cli`、`llama-quantize`、`llama-bitnet-eval`、`llama-bench`、`test-i2s-scaled` |
| 原生已知答案测试 | **27 项通过** | legacy `I2_S`、ROW/G32 打包、反量化、GEMV/GEMM、尾部和真实 GGML 多线程路径 |

本次没有在新目录重跑正式训练、全量网络下载、完整模型导出、四域 CE 或性能实验。下载脚本使用合成数据做无网络测试；历史正式结果只作为保留证据引用，不冒充新执行结果。包安装检查使用 `--no-deps --no-build-isolation` 和带系统 site-packages 的隔离解释器，只验证项目自身 wheel 与导入，不声称已在无预装依赖的全新环境解析并安装全部第三方依赖。

### 1.3 适用边界

- 训练器需要支持 BF16 的 CUDA GPU；分布式训练使用 NCCL。它不是 CPU 训练器。
- 新部署格式只实现并验证了 **x86 CPU**，不声称支持 ARM、Metal 或 CUDA。
- `GGML_NATIVE=ON` 会按构建机器生成优化代码；换 CPU 后应重新构建。
- 基础模型、数据集和第三方源码各有自己的许可条件。下载或清洗不代表获得再分发授权。
- 不向上游 `llama.cpp` 直接提交本仓库的 vendored 修改；这里是部署适配所需的完整本地源码树。

## 2. 仓库结构与 DOCX 章节代码地图

代码按功能唯一归属，不按章节复制多份相同实现。技术分享文档的章节通过下表映射到可运行代码。

| 路径 | 功能 |
|---|---|
| `data_pipeline/download/` | 模型、预训练语料、SFT 语料和评测集下载、校验与 manifest |
| `data_pipeline/pretrain/` | parquet/JSONL 预处理、去污、去重、tokenize、冻结、恢复和 READY 发布 |
| `data_pipeline/sampling.py` | 训练时按权重在线选择域与固定窗口 |
| `data_pipeline/sft/` | GSM8K、通用指令、sft_v2、sft_v3 数据构建 |
| `qat/quant.py` | 三值量化器、`TernaryLinear`、STE、ROW/G32 语义 |
| `qat/train.py` | 预训练 QAT、BF16 对照、WSD、μP、验证与严格恢复 |
| `qat/train_gsm8k.py` | answer-only GSM8K SFT |
| `qat/train_alpaca_sft.py` | 通用混合 JSONL SFT |
| `qat/eval_downstream.py` | MMLU、CMMLU、C-Eval、GSM8K 下游评测 |
| `qat/run_*.py` | 固定卡数、固定 checkpoint 的历史实验复现启动器 |
| `tools/` | FP32 master 导出、GGUF 回读、HF/native logits 比较 |
| `bitnet.cpp/` | 修改后的完整推理源码树，含 vendored llama.cpp/GGML |
| `configs/` | 历史训练配置、数据使用计划和评测文件映射 |
| `scripts/` | 构建、测试、导出量化包装脚本 |
| `tests/` | 无网络、小规模、可重复的离线测试 |
| `docs/` | 技术分享、逐轮实验报告和真实结果证据 |

| 技术分享章节 | 对应代码和材料 |
|---|---|
| 1. 项目背景与最终成果 | 本 README、`docs/reports/` |
| 2. 三值量化基础 | `qat/quant.py`、`bitnet.cpp/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c` |
| 3. 三值 QAT 如何训练 | `qat/train.py`、`qat/checkpoint_utils.py`、`qat/quant.py` |
| 4. 数据配比与逐轮训练改进 | `data_pipeline/`、`configs/`、`qat/run_mix*.py`、`qat/run_sft*.py`、实验报告 |
| 5. 训练到部署总体架构 | `tools/export_bitnet_hf.py`、`scripts/export_and_quantize.sh`、`bitnet.cpp/` |
| 6. 模型导出与 GGUF 格式 | `tools/`、vendored `conversion/`、`gguf-py/`、`ggml/src/gguf.cpp` |
| 7. x86 原生推理 kernel | `ggml/src/ggml-cpu/quants.c`、`ggml-cpu-i2s.c`、`ggml-cpu.c`、`ops.cpp` |
| 8. 部署正确性证明 | `tests/`、`test-i2s-scaled.cpp`、`tools/verify_bitnet_gguf.py`、`tools/compare_bitnet_logits.py`、`tools/bitnet-eval/` |
| 9. 性能、内存与真实生成 | `llama-bench`、`llama-cli`、`docs/reports/pretrain_v2/mix_v3_bitcpm/bitnet_cpp/` |
| 10. 工程经验与边界 | 技术分享文档、原始实验报告和本 README |

阅读材料：

- [技术分享 Markdown](docs/TECHNICAL_SHARING.md)
- [技术分享 Word，含嵌图和原生公式](docs/TECHNICAL_SHARING.docx)
- [QAT 教学笔记](docs/reports/QAT_TEACHING_NOTES.md)
- [完整部署报告](docs/reports/pretrain_v2/mix_v3_bitcpm/BITNET_CPP_DEPLOYMENT.md)
- [mix_v3 训练与部署原报告](docs/reports/pretrain_v2/mix_v3_bitcpm/EXPERIMENT_NOTES.md)

## 3. 环境、安装与快速自检

### 3.1 依赖

基础环境：

- Linux，Python 3.10 或更高版本。
- 数据构建：足够的内存和磁盘；完整原始语料及 token stream 需要数十 GB 以上空间，具体由下载 plan 决定。
- 训练：支持 BF16 的 CUDA GPU、匹配 CUDA 的 PyTorch、NCCL。
- 原生构建：CMake、C/C++ 编译器、OpenMP；AVX2/VNNI 由 CPU 和编译器能力决定。

建议先安装与本机 CUDA 匹配的 PyTorch，再安装项目，避免通用依赖解析替换成不合适的 PyTorch wheel：

```bash
git clone YOUR_REPOSITORY_URL
cd MiniCPM4-Ternary-QAT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`requirements.txt` 等价于安装项目和测试依赖：

```bash
python -m pip install -r requirements.txt
```

若机器的 Python 没有 `ensurepip`，标准 venv 创建可能失败。可以复用现有 pip，并明确接受系统 site-packages 可见这一限制：

```bash
python -m venv --without-pip --system-site-packages .venv
python -m pip --python "$PWD/.venv/bin/python" install -e ".[dev]"
```

不要在训练环境中直接安装 vendored `llama.cpp/requirements.txt` 的全集合：其中的转换器依赖可能固定另一个 PyTorch 版本。本项目的 `pyproject.toml` 已覆盖当前 MiniCPM 转换、训练和验证所需 Python 包；GGUF 转换脚本设置 `gguf-py` 的 `PYTHONPATH`，回读工具则自行把该目录加入模块搜索路径。

### 3.2 无网络测试

所有命令均从仓库根目录执行：

```bash
bash scripts/test_offline.sh
```

测试不需要下载模型或数据，不访问网络。它会在临时目录构造小型 parquet、JSONL、CSV、tokenizer 和 checkpoint fixture；真实本地 tokenizer 的 spawn/CLI 链路也会执行，但不进行 CUDA 训练。

### 3.3 构建与原生测试

```bash
JOBS=16 bash scripts/build_bitnet.sh
bash scripts/test_native.sh
```

自定义构建目录：

```bash
BITNET_BUILD_DIR=/path/to/build JOBS=16 bash scripts/build_bitnet.sh
BITNET_BUILD_DIR=/path/to/build bash scripts/test_native.sh
```

必须保留脚本中的 `-DLLAMA_BUILD_COMMON=ON`，否则作为子项目嵌入时不会生成 `llama-cli` target。没有 Git 元数据时 CMake 可能报告 `ggml commit: unknown`；这不影响本地源码构建和数值测试。

## 4. 下载模型与全部数据

### 4.1 下载纪律

- 下载目录应是独立的新目录，不要指向源码目录，也不要在下载目录中运行解释器或构建工具。
- 本文统一从仓库根目录用 `python -I` 读取外部数据。
- `.partial` 不是已发布数据。下载器完成大小、SHA256 和格式检查后才将文件改成正式名称。
- 正式文件已存在时不会被静默覆盖；更换来源或参数时使用新的输出目录。
- HF token 只从 `HF_TOKEN` 环境变量读取，不写入配置或 manifest。公开模型通常不需要 token。
- 当前整理未执行全量网络下载。上游文件是否仍可用、实际大小和哈希以运行时生成的 manifest 为准。

默认目录布局如下；`data/`、`eval_data/` 和 `MiniCPM4-0.5B*/` 已被 `.gitignore` 排除：

```text
MiniCPM4-0.5B-fjzcuwlf/
data/
  ufw_raw_v2/
  math_raw_v1/ultradata_math/
  math_raw_v1/ultradata_math_l3/
  code_raw_v1/opc_annealing/
  sft_raw/alpaca/
  sft_raw/firefly/
  sft_raw/hc3/
  eval_downloads/
eval_data/
```

`MiniCPM4-0.5B-fjzcuwlf` 只是兼容历史配置的本地目录名，不是另一个模型版本。

### 4.2 MiniCPM4-0.5B 基座

来源为 Hugging Face `openbmb/MiniCPM4-0.5B`，下载器默认使用 `hf-mirror.com`：

```bash
python -I data_pipeline/download/download_minicpm.py \
  --repo openbmb/MiniCPM4-0.5B \
  --output MiniCPM4-0.5B-fjzcuwlf
```

需要固定上游版本时显式传入 commit、tag 或 branch：

```bash
python -I data_pipeline/download/download_minicpm.py \
  --repo openbmb/MiniCPM4-0.5B \
  --revision RESOLVED_REVISION \
  --output MiniCPM4-0.5B-fjzcuwlf
```

输出包含模型 config、tokenizer、模型权重、远程建模代码和 `download-manifest.json`。训练会使用 `trust_remote_code=True`，因此只能加载自己审阅并信任的模型目录。

### 4.3 预训练四个域

| 域 | 上游数据集 | 历史选择 |
|---|---|---|
| `web_zh` / `web_en` | ModelScope `OpenBMB/Ultra-FineWeb` | 每语言 100 个 shard，seed 20261109 |
| `math` L2 | ModelScope `OpenBMB/UltraData-Math` | L2-preview 选 12 个 shard |
| `math` L3 | 同上 | QA-Synthetic 与 Textbook-Exercise-Synthetic 各选 2 个 shard |
| `code` | ModelScope `AI-ModelScope/opc-annealing-corpus` | 三个根目录各选 20 个 JSONL shard |

Ultra-FineWeb：

```bash
python -I data_pipeline/download/download_modelscope_ufw.py \
  --output data/ufw_raw_v2 \
  --per-language 100 \
  --seed 20261109 \
  --workers 4
```

首次只验证两个文件时，可在同一命令末尾加 `--limit 2`。plan 仍保存完整选择，之后去掉 `--limit` 并复用同一目录可继续下载，不会重新抽样。

UltraData-Math L2：

```bash
python -I data_pipeline/download/download_modelscope_files.py \
  --repo OpenBMB/UltraData-Math \
  --root data/UltraData-Math-L2-preview \
  --suffix .parquet \
  --per-root 12 \
  --seed 20261109 \
  --output data/math_raw_v1/ultradata_math \
  --workers 2
```

UltraData-Math L3：

```bash
python -I data_pipeline/download/download_modelscope_files.py \
  --repo OpenBMB/UltraData-Math \
  --root data/UltraData-Math-L3/QA-Synthetic \
  --root data/UltraData-Math-L3/Textbook-Exercise-Synthetic \
  --suffix .parquet \
  --per-root 2 \
  --seed 20261109 \
  --output data/math_raw_v1/ultradata_math_l3 \
  --workers 2
```

OPC：

```bash
python -I data_pipeline/download/download_modelscope_files.py \
  --repo AI-ModelScope/opc-annealing-corpus \
  --root algorithmic_corpus \
  --root synthetic_code_snippet \
  --root synthetic_qa \
  --suffix .jsonl \
  --per-root 20 \
  --seed 20261109 \
  --output data/code_raw_v1/opc_annealing \
  --workers 2
```

每个目录中的 `plan.json` 是冻结选择。通用 ModelScope 下载器在 repo、roots、seed、per-root 或 suffix 与已有 plan 不同时会拒绝混合两次抽样；Ultra-FineWeb 下载器复用同 seed 的既有完整 plan，`--per-language` 不会重抽已有选择。做不同规模的小实验时请使用不同输出目录。

### 4.4 下游评测数据

一个命令获取 GSM8K、MMLU、CMMLU 和 C-Eval：

```bash
python -I data_pipeline/download/download_eval_data.py \
  --tasks gsm8k mmlu cmmlu ceval \
  --work-dir data/eval_downloads \
  --output eval_data
```

来源：

- GSM8K：`openai/grade-school-math` 的 train/test 与 socratic JSONL。
- MMLU：原始 Berkeley `data.tar`。
- CMMLU：`haonan-li/CMMLU` 的 `data.zip`。
- C-Eval：`hkust-nlp/ceval` 的 `data.zip`。

压缩包保留在独立 work 目录。脚本只在内存中读取 `configs/eval/*/mapping.json` 指定的 CSV，不执行压缩包内容，也不把成员路径直接解压到磁盘。生成布局为：

```text
eval_data/
  gsm8k/{train,test,socratic_train,socratic_test}.jsonl
  mmlu/mapping.json
  mmlu/data/{dev,test}/<subject>_<split>.csv
  cmmlu/mapping.json
  cmmlu/{dev,test}/<subject>.csv
  ceval/mapping.json
  ceval/{dev,val}/<subject>_<split>.csv
```

C-Eval 的 test label 不公开，因此评测器使用 `val`。MMLU 原始压缩包中的 `val` 不是当前正式评测输入，故脚本只发布 `dev` 和 `test`。

### 4.5 SFT 原始数据

Alpaca 英文和中文：

```bash
python -I data_pipeline/download/download_alpaca.py \
  --languages en zh \
  --output data/sft_raw/alpaca
```

来源分别为 `tatsu-lab/stanford_alpaca` 和 `ymcui/Chinese-LLaMA-Alpaca`。脚本保留上游 JSON，并生成训练器需要的 `instruction,input,output` CSV 和实际 SHA256 manifest。

**历史复现边界：**原实验的 Alpaca CSV 没保留来源 commit 和原始字节哈希。因此新下载流程可以重新获取上游数据，但不能在没有哈希对照的情况下声称与历史 CSV 逐字节相同。要严格复现历史数值，应使用原实验冻结的数据和 manifest。

Firefly：

```bash
python -I data_pipeline/download/download_modelscope_files.py \
  --repo AI-ModelScope/firefly-train-1.1M \
  --root '' \
  --suffix .jsonl \
  --per-root 0 \
  --seed 20261109 \
  --output data/sft_raw/firefly \
  --workers 1
```

历史文件为 `firefly-train-1.1M.jsonl`，1,171,119,212 bytes，SHA256 为 `31866264531f849b3eddd5a8bc51e6e611226425654188ac05c94fd283a658c7`。运行时仍以下载 plan 和实际校验为准。

HC3-Chinese：

```bash
python -I data_pipeline/download/download_modelscope_files.py \
  --repo AI-ModelScope/HC3-Chinese \
  --root '' \
  --suffix .jsonl \
  --per-root 0 \
  --seed 20261109 \
  --output data/sft_raw/hc3 \
  --workers 2
```

`--per-root 0` 表示选择该根目录下全部匹配文件。通用下载器支持预训练 `text`、Firefly `kind/input/target`、HC3 `question + answers`、instruction JSONL 和 Alpaca CSV 的格式检查。

辅助工具：

- `download_ultrafineweb.py`：Hugging Face 版 Ultra-FineWeb 下载入口。
- `download_ranges.py`：根据已有 HF manifest 做断点/范围下载。
- 所有下载器都可先运行 `python -I <脚本> --help` 查看参数；help 不访问网络。

## 5. 预训练数据处理

这里有两个不同阶段，不能混为一谈：

1. **离线数据构建**：原始文件经过冻结、去重、去污、分域、tokenize，发布为带 `READY` 的 token stream。
2. **在线训练混合**：`MixedSampler` 在训练时按配置权重选择域和固定窗口，不重新读取原始 parquet，也不重新清洗文本。

### 5.1 单步 parquet → uint32：适合小数据与自定义实验

```bash
python -I data_pipeline/pretrain/prepare_data.py \
  --tokenizer MiniCPM4-0.5B-fjzcuwlf \
  --input zh=data/example/zh.parquet \
  --input en=data/example/en.parquet \
  --output data/ufw_small \
  --val-docs 2000 \
  --min-chars 50 \
  --workers 8 \
  --batch-docs 500 \
  --seed 1234
```

输入 parquet 必须有 `content` 列。每篇文档编码为：

```text
[BOS] + tokenizer(text, add_special_tokens=False) + [EOS=2]
```

对 MiniCPM4 当前 tokenizer，BOS 为 1。输出为 little-endian uint32：

```text
<lang>_train.bin
<lang>_val.bin
<lang>_train_doclens.npy
<lang>_val_doclens.npy
meta.json
```

输出目录必须不存在。该简单入口不生成生产 `manifest.json/READY`，也不做完整的跨源冻结和去污；正式四域数据使用下一节。

### 5.2 正式四域数据：freeze → run/resume → READY

先完成第 4 节全部计划下载和评测数据准备，再冻结。freeze 只接受 plan 中已经完整发布的文件；未完成文件会被排除，而且之后下载完成也不会自动加入这次快照。因此不要在下载仍继续时冻结正式数据。

```bash
python -I data_pipeline/pretrain/prepare_mix.py freeze \
  --root "$PWD" \
  --output "$PWD/data/mix_v2" \
  --tokenizer "$PWD/MiniCPM4-0.5B-fjzcuwlf" \
  --workers 16
```

默认读取四个生产 plan：

```text
data/ufw_raw_v2/plan.json
data/math_raw_v1/ultradata_math/plan.json
data/math_raw_v1/ultradata_math_l3/plan.json
data/code_raw_v1/opc_annealing/plan.json
```

自定义位置时重复传 `--plan`：

```bash
python -I data_pipeline/pretrain/prepare_mix.py freeze \
  --root "$PWD" \
  --output "$PWD/data/custom_mix" \
  --tokenizer "$PWD/MiniCPM4-0.5B-fjzcuwlf" \
  --plan /path/to/ufw/plan.json \
  --plan /path/to/math-l2/plan.json \
  --plan /path/to/math-l3/plan.json \
  --plan /path/to/opc/plan.json
```

先提交两个 batch 验证恢复，再继续：

```bash
python -I data_pipeline/pretrain/prepare_mix.py run \
  --root "$PWD" --output "$PWD/data/mix_v2" --max-batches 2

python -I data_pipeline/pretrain/prepare_mix.py resume \
  --root "$PWD" --output "$PWD/data/mix_v2"
```

第二条命令会跑到全部目标完成。也可以在长时间任务中再次使用 `run --max-batches N` 主动暂停。不要对同一输出目录并发启动两个构建进程；文件锁会拒绝第二个 owner。

训练 token 上限：

| 域 | 训练 token cap |
|---|---:|
| `web_zh` | 1,840,000,000 |
| `web_en` | 1,920,000,000 |
| `math` | 350,000,000 |
| `code` | 350,000,000 |

构建过程包括：

- 固定 plan、原始文件大小和 SHA256。
- tokenizer、评测集、软件版本和数据代码身份冻结。
- 规范化文本后的全局去重。
- 对 MMLU/CMMLU/C-Eval/GSM8K 及明确覆盖变体的题干去污。
- 按稳定文本 hash 划分验证集，而非每次重新随机切分。
- SQLite 提交点、输出 fsync、未提交尾部恢复。
- 所有域同时满足训练和验证要求后才发布 `READY`。

最终输出：

```text
data/mix_v2/
  freeze.json
  manifest.json
  READY
  web_zh_train.bin
  web_zh_val.bin
  web_en_train.bin
  web_en_val.bin
  math_train.bin
  math_val.bin
  code_train.bin
  code_val.bin
  staging/state.sqlite
```

`manifest.json` 记录每个 stream 的相对路径、token 数和 SHA256。训练加载器要求 `READY` 中的 manifest hash 匹配，并验证每个文件大小和哈希。不要只复制 `.bin` 而漏掉 manifest/READY。

### 5.3 在线混合、窗口和恢复

`data_pipeline/sampling.py` 的关键语义：

- 每条 stream 按 `seq_len + 1` 切窗口；`seq_len=2048` 时每窗 2049 token。
- 输入是前 2048 个 token，target 是向右移动一位的 2048 个 token。
- 训练前生成完整的 domain/window schedule；域选择服从配置权重，域内窗口按固定 seed 排列。
- `max_data_passes` 在完整计划层面检查，超出会报错，而不是训练中途静默重复。
- 验证使用每域固定前 N 个窗口。
- 严格恢复会检查模型、优化器、数据身份、采样游标和各 rank RNG。

mix_v2 的训练权重为 `[0.35, 0.35, 0.15, 0.15]`，对应 `web_zh/web_en/math/code`，总量 **2,000,683,008 tokens，1908 steps**。实际使用量见[配置计划](configs/data/mix_v2_schedule.json)：

| 域 | 实际 tokens | 数据 passes |
|---|---:|---:|
| web_zh | 699,467,776 | 0.380331 |
| web_en | 699,848,704 | 0.364683 |
| math | 300,062,720 | 0.857741 |
| code | 301,303,808 | 0.861289 |

mix_v3 G32 配置总量 **2,248,146,944 tokens，2144 steps**，主配比与 mix_v2 对齐。不要将“训练权重”误当成各域最终实际 token 数；精确值以 schedule 和 metrics 为准。

## 6. 预训练 QAT 与 BF16 对照

### 6.1 量化语义

每个量化区域的规则为：

```text
s = max(mean(abs(W)), 1e-8)
q = clamp(round(W / s), -1, 1)
Q(W) = q × s
```

`TernaryLinear` 用 STE 更新 FP32 master：

```text
w_forward = w + alpha × (Q(w) - w).detach()
```

24 层 × 7 个投影，共 168 个三值投影。embedding、lm_head、norm 和 RoPE factors 保持浮点。checkpoint 保存 FP32 master，而不是已经反量化的 `0,±s`。

### 6.2 运行 ROW 与 G32 配置

历史正式 run 使用 4 卡。先确保 `data/mix_v2/READY` 和基座模型存在，再从仓库根目录运行：

```bash
torchrun --standalone --nproc_per_node=4 \
  qat/train.py \
  --config configs/pretrain/mix_v2_qat_70_15_15.json \
  --run-dir runs/mix_v2_qat_70_15_15
```

G32：

```bash
torchrun --standalone --nproc_per_node=4 \
  qat/train.py \
  --config configs/pretrain/mix_v3_bitcpm_pg32.json \
  --run-dir runs/mix_v3_bitcpm_pg32
```

BF16 对照示例：

```bash
torchrun --standalone --nproc_per_node=4 \
  qat/train.py \
  --config configs/pretrain/mix_v2_bf16_70_15_15.json \
  --run-dir runs/mix_v2_bf16_70_15_15
```

注意：

- 必须显式给 `configs/pretrain/*.json` 搭配 `--run-dir runs/<name>`；否则默认输出到配置文件所在目录。
- JSON 中的相对模型、数据和 manifest 路径相对**进程当前工作目录**解析，因此上述命令都从仓库根目录运行。
- 旧配置没有 `quant_group` 字段时明确表示 per-row；G32 配置必须显式写 `"quant_group": 32`。
- 这些配置是正式实验规模，不代表单卡显存足够。减少 micro-batch、world size 或 token 数属于新实验，应另存配置，不能仍声称复现原训练。
- 验证同时记录浮点前向和 alpha=1 三值前向；模型选择不能只看浮点 master 的 CE。

主要输出：

```text
runs/<name>/
  resolved_config.json
  metrics.jsonl
  checkpoints/latest/state.pt
  checkpoints/final/state.pt
  snapshots/                 # 配置请求时产生
  TensorBoard event files
```

### 6.3 暂停与严格恢复

`--max-steps` 是绝对 optimizer update 上限，只暂停，不缩短 LR/alpha 日程：

```bash
torchrun --standalone --nproc_per_node=4 \
  qat/train.py \
  --config configs/pretrain/mix_v2_qat_70_15_15.json \
  --run-dir runs/mix_v2_qat_70_15_15 \
  --max-steps 10
```

从同一 run 恢复，并提高或移除暂停上限：

```bash
torchrun --standalone --nproc_per_node=4 \
  qat/train.py \
  --config configs/pretrain/mix_v2_qat_70_15_15.json \
  --run-dir runs/mix_v2_qat_70_15_15 \
  --resume runs/mix_v2_qat_70_15_15/checkpoints/latest/state.pt
```

严格 run 不会用另一个 checkpoint 隐式恢复。换数据、模型、world size、代码或日程可能使身份检查失败；应修复不一致或新建 run，而不是绕过检查。

### 6.4 历史启动器不是通用 CLI

`qat/run_mix.py`、`run_mix_v2.py`、`run_mix_v3.py` 和 `run_mix_v3_ablation.py` 保留固定多卡拓扑、固定输出目录、完整 preflight 和真实 GPU smoke：

- mix_v1/mix_v2 包含多个固定 arm。
- mix_v3 两个 4 卡 arm 总启动需要 8 卡。
- 普通训练优先使用上面的 `qat/train.py` 直接 CLI。

迁移后 preflight 的代码身份已经包含 `qat/**/*.py` 和 `data_pipeline/**/*.py`。因此旧 preflight 记录不能冒充新代码已验证；需要重新 preflight。历史报告中的旧身份和执行路径被原样保留，没有回填伪造。

## 7. SFT 数据构建

### 7.1 GSM8K answer-only 数据

```bash
python -I data_pipeline/sft/gsm8k_data.py \
  --train-file eval_data/gsm8k/train.jsonl \
  --test-file eval_data/gsm8k/test.jsonl \
  --output-dir data/gsm8k_sft_v1 \
  --dev-size 500 \
  --seed 1234
```

规则：

- 对原始 train 去重、划分 train/dev，保留前 8 条原始 train 作为 few-shot。
- test 只用于只读重叠审计，不参与划分、过滤或训练决策。
- 输出 `train.jsonl`、`dev.jsonl`、`fewshot.jsonl` 和带逐文件 SHA256 的 `manifest.json`。
- 输出目录必须不存在；manifest 是完成标记。

### 7.2 通用指令数据 alpaca_sft_v1

```bash
python -I data_pipeline/sft/prepare_alpaca_sft.py \
  --alpaca-zh data/sft_raw/alpaca/alpaca_zh.csv \
  --alpaca-en data/sft_raw/alpaca/alpaca_en.csv \
  --hc3-dir data/sft_raw/hc3 \
  --firefly data/sft_raw/firefly/firefly-train-1.1M.jsonl \
  --root "$PWD" \
  --output data/alpaca_sft_v1 \
  --dev-size 1000 \
  --seed 20261110
```

处理包括长度过滤、去重、对四套 benchmark 去污和 train/dev 切分。Firefly 只保留 `Cot/OpenQA/MRC/Translation/Composition/ClassicalChinese/Dictionary`。统一记录为：

```json
{"prompt": "instruction\n", "response": "answer", "source": "source", "id": "unique-id"}
```

历史 `alpaca_sft_v1` 实测为 train 495,219、dev 1,000、去污移除 9,728；train 来源为 Firefly 384,973、Alpaca EN 51,589、Alpaca ZH 48,669、HC3 9,988。新获取的数据若上游字节不同，数量可以不同，以新 manifest 为准。

### 7.3 通用数据 + GSM8K×N

```bash
python -I data_pipeline/sft/mix_gsm8k_sft.py \
  --alpaca data/alpaca_sft_v1 \
  --gsm data/gsm8k_sft_v1 \
  --output data/alpaca_gsm8k_sft_v1 \
  --repeat 3 \
  --seed 20261110
```

重复副本有不同 ID，输出 manifest 会记录真实重复数和 GSM8K share。

### 7.4 sft_v2：general/code/math

```bash
python -I data_pipeline/sft/prepare_sft_v2.py \
  --general data/alpaca_sft_v1 \
  --gsm data/gsm8k_sft_v1 \
  --code-dir data/code_raw_v1/opc_annealing/algorithmic_corpus \
  --firefly data/sft_raw/firefly/firefly-train-1.1M.jsonl \
  --benchmark-root "$PWD" \
  --output data/sft_v2 \
  --total 500000 \
  --code-frac 0.25 \
  --math-frac 0.15 \
  --math-repeat 11 \
  --seed 20261110
```

实际构造顺序不是简单对三个源独立截断：

1. 先按 `math_repeat` 产生 math。
2. 收集 Firefly Program，再按 code 目标补充 OPC。
3. 最后用 general 补足 `--total`。
4. code 去污后移除的样本不会自动从其他源补回。

因此 math 超过名义比例时 general 会自动减少；源数据不足、清洗移除或小 quota 下，最终总量和比例以 manifest 为准。负 quota、空代码目录和空最终数据集会明确报错，不会写出误导性的空数据。

历史 sft_v2：train 499,804、dev 1,500，其中 general 298,297、code 124,804、GSM8K 76,703（repeat 11）。v2 manifest 的 `categories` 使用 `source.split('/')[0]`，可能出现 `firefly`、`alpaca_zh`、`gsm8k` 等键；这里的 general/code/math 是按 source 规则汇总后的解释，不改写历史统计语义。

### 7.5 sft_v3：复用 v2 general/code，只改 math

```bash
python -I data_pipeline/sft/prepare_sft_v3.py \
  --v2 data/sft_v2 \
  --gsm data/gsm8k_sft_v1 \
  --ultra data/math_raw_v1/ultradata_math_l3/data/UltraData-Math-L3/QA-Synthetic \
  --benchmark-root "$PWD" \
  --output data/sft_v3 \
  --math-size 75000 \
  --gsm-repeat 2 \
  --seed 20261110
```

v3 原样复用 v2 的 general/code，仅将 math 改为 GSM8K×2 + UltraData-Math L3 QA-Synthetic。GSM8K 已超过 `math-size` 时 UltraData quota 为零，不读取 UltraData；不会去读取不必要的 parquet。

历史 sft_v3：train 497,881、dev 1,500；general 298,297、code 124,804、math 74,780。math 中 UltraData 60,834、GSM8K 13,946，math 去污移除 220，移除后没有自动补足到 75,000。

## 8. SFT 训练

### 8.1 answer-only GSM8K

标准协议为 2 张 BF16 CUDA GPU、global batch 64、最多 5 epochs：

```bash
torchrun --standalone --nproc_per_node=2 \
  qat/train_gsm8k.py \
  --data-dir data/gsm8k_sft_v1 \
  --run-dir runs/gsm8k_sft_lr1e-5 \
  --model MiniCPM4-0.5B-fjzcuwlf \
  --init-ckpt runs/mix_v2_qat_70_15_15/checkpoints/final/state.pt \
  --init-mode qat \
  --lr 1e-5 \
  --epochs 5 \
  --global-batch 64 \
  --micro-batch 2
```

- 只监督 answer + EOS，prompt label 为 `-100`。
- 超过 `max-length` 的样本直接报错，不静默截断。
- 初始化 checkpoint 的邻近 `resolved_config.json/config.json` 决定 ROW 或 G32，防止 ROW 被静默按 G32 继续训练。
- `--allow-other-world-size` 只用于显式诊断，不改变“标准协议为两卡”的事实。
- `--max-steps` 只暂停；恢复使用同一训练参数和 `--resume <run>/checkpoints/latest/state.pt`。
- epoch model-only snapshot 用于评测；完整 resume 必须使用 checkpoint 下的 `state.pt`。

### 8.2 通用混合 SFT

适用于 `alpaca_sft_v1`、`alpaca_gsm8k_sft_v1`、`sft_v2`、`sft_v3`：

```bash
torchrun --standalone --nproc_per_node=8 \
  qat/train_alpaca_sft.py \
  --data-dir data/sft_v3 \
  --run-dir runs/mix_v2_qat_sft_v3 \
  --model MiniCPM4-0.5B-fjzcuwlf \
  --init-ckpt runs/mix_v2_qat_sft/snapshots/epoch_4/model.pt \
  --init-mode qat \
  --lr 5e-5 \
  --epochs 3 \
  --global-batch 128 \
  --micro-batch 2
```

通用 trainer 同样只监督 response + EOS，并从 init checkpoint 推导量化粒度。过长样本跳过。它支持单/多 CUDA rank，但 8 卡、batch 128 是上述历史 run 的配置，不是最低硬件要求。

`configs/sft/*.json` 是训练完成后的 resolved config 记录，**不是所有 SFT trainer 都能直接读取的统一输入配置**。复现时把记录中的参数映射到上述 CLI，并显式给出当前路径。

`qat/run_sft_v2.py`、`run_sft_v3.py`、`run_alpaca_sft.py`、`run_alpaca_gsm8k_sft.py` 是固定 8 卡、固定 init checkpoint 和固定输出目录的历史启动器，不是通用数据/训练 CLI。`run_alpaca_sft.py` 只接受已通过 size + SHA256 校验的正式 Firefly 文件，不会把 `.partial` 自行改名发布。

## 9. 下游评测

先按第 4.4 节准备 `eval_data/`，再运行：

```bash
python -I qat/eval_downstream.py \
  --model MiniCPM4-0.5B-fjzcuwlf \
  --ckpt runs/mix_v2_qat_70_15_15/checkpoints/final/state.pt \
  --precision ternary \
  --tasks mmlu cmmlu ceval gsm8k \
  --gsm-test-file eval_data/gsm8k/test.jsonl \
  --gsm-fewshot-file data/gsm8k_sft_v1/fewshot.jsonl \
  --gsm-max-new 512 \
  --out reports/mix_v2/downstream.json
```

协议：

- MMLU test、CMMLU test、C-Eval val 为 5-shot。
- 选择题比较 `" A"` 到 `" D"` 的下一 token logits。
- GSM8K test 为 8-shot greedy generation，解析数值答案。
- `--precision ternary` 按 checkpoint 的原生 ROW/G32 粒度做一次三值前向替换；`float` 是浮点对照。
- `--ckpt none` 评测发布的基座权重。
- 评测器使用 CUDA，不提供 CPU 正式评测路径。
- `--limit` 只用于小样本冒烟，其结果不能与完整 benchmark 数字比较。

输出 JSON 记录权重、评测代码、GSM8K test/few-shot 文件的 SHA256 和逐项样本。相同设置的已完成 task 可以复用；设置或输入身份改变时应使用新的输出文件。

## 10. 导出 FP32 master 与原生 GGUF

### 10.1 为什么必须经过 FP32 master

正确链路：

```text
FP32 master state.pt
  → 逐位不变的 FP32 HF safetensors 中间目录
  → F32 GGUF
  → 按 checkpoint 原生粒度量化成 I2_S_ROW 或 I2_S_G32
```

不能先导出反量化 `0,±s` 再求 absmean；那会得到 `s × nonzero_density`，改变 scale。导出器检查 run granularity 和全部 168 个投影，不允许 ROW/G32 交叉导出。读取 `state.pt` 使用 `weights_only=False`，因此只能导出自己训练或明确信任的 checkpoint。

### 10.2 一条命令完成导出、转换、量化和回读

先完成原生构建。ROW：

```bash
bash scripts/export_and_quantize.sh \
  runs/mix_v2_qat_70_15_15/checkpoints/final/state.pt \
  runs/mix_v2_qat_70_15_15/resolved_config.json \
  row \
  exports/mix_v2_row \
  MiniCPM4-0.5B-fjzcuwlf
```

G32：

```bash
bash scripts/export_and_quantize.sh \
  runs/mix_v3_bitcpm_pg32/checkpoints/final/state.pt \
  runs/mix_v3_bitcpm_pg32/resolved_config.json \
  g32 \
  exports/mix_v3_g32 \
  MiniCPM4-0.5B-fjzcuwlf
```

输出目录必须不存在。生成：

```text
exports/<name>/
  hf_fp32_master/
  model-F32.gguf
  model-I2_S_ROW.gguf       # 或 model-I2_S_G32.gguf
  roundtrip.json
```

可使用 `BITNET_BUILD_DIR` 和 `QUANTIZE_THREADS` 指定构建目录与量化线程。完整 F32 GGUF 历史大小约 1.90 GiB，需为中间文件和原生 GGUF 预留空间。

本次只验证了脚本语法、组成它的工具和新位置的原生构建；没有在新目录对完整 checkpoint 再执行整条导出脚本。历史完整导出日志和回读结果保留在 `docs/reports/pretrain_v2/mix_v3_bitcpm/bitnet_cpp/`。

### 10.3 单独执行 GGUF 回读

```bash
python -I tools/verify_bitnet_gguf.py \
  --gguf exports/mix_v2_row/model-I2_S_ROW.gguf \
  --hf-dir exports/mix_v2_row/hf_fp32_master \
  --granularity row \
  --output exports/mix_v2_row/roundtrip.json
```

G32 时同时替换模型路径和 `--granularity g32`。验证全部 221 个 tensor：168 个量化投影和 53 个 F32 tensor，包括 Q/K permutation、scale、物理大小和浮点张量。

历史结果中，两种模型非 half-threshold code mismatch 都为 0；ROW/G32 分别有 205/29 个边界 code 差异，全部位于 FP32 half threshold。不能把这些结果简化成“所有 code 逐位相同”，完整误差表见部署报告。

## 11. 原生 x86 推理与分层验证

### 11.1 物理格式

每个 2-bit code 按字节高位到低位打包，`stored_code=q+1`，即 0/1/2：

| 格式 | GGML type | 每行 codes | 每行 scales | 每行字节数 |
|---|---:|---:|---:|---:|
| `I2_S_ROW` | 43 | `K/4` bytes | 1 个 FP32 | `K/4 + 4` |
| `I2_S_G32` | 44 | `K/4` bytes | `K/32` 个 FP32 | `K/4 + 4×K/32` |

两者不使用 legacy `I2_S` 的 32-byte trailer。activation 量化为 int8，整数点积利用：

```text
sum(q × q8) = sum(code × q8) - sum(q8)
```

ROW 在整行 correction 后乘一个 scale；G32 每 32 维分别 correction、乘组 scale 再按组累加。AVX2 和 AVX-VNNI/AVX512-VNNI 只改变整数实现，不改变这一语义。

### 11.2 HF/native prefill 与 cached decode logits

先让 native 对一个窗口输出 32-token prefill + 8 个 cached decode logits：

```bash
mkdir -p reports/native_check
bitnet.cpp/build/bin/llama-bitnet-eval \
  --model exports/mix_v2_row/model-I2_S_ROW.gguf \
  --tokens data/mix_v2/web_zh_val.bin \
  --seq-len 32 \
  --windows 1 \
  --decode-steps 8 \
  --threads 16 \
  --threads-batch 16 \
  --logits-out reports/native_check/row_logits.bin \
  --metrics-out reports/native_check/row_native.json
```

再计算 HF 参考并比较：

```bash
python -I tools/compare_bitnet_logits.py \
  --hf-dir exports/mix_v2_row/hf_fp32_master \
  --tokens data/mix_v2/web_zh_val.bin \
  --native-logits reports/native_check/row_logits.bin \
  --granularity row \
  --device cuda:0 \
  --autocast bf16 \
  --output reports/native_check/row_comparison.json
```

`--logits-out` 或 `--decode-steps > 0` 要求 `--windows 1`。该短序列用于验证执行路径，不用于评价整体模型质量。`llama-bitnet-eval` **没有标准 `--help` 选项**，参数以其源码和本节为准。

### 11.3 完整固定窗口 CE

对四个域分别运行：

```bash
for domain in web_zh web_en math code; do
  bitnet.cpp/build/bin/llama-bitnet-eval \
    --model exports/mix_v2_row/model-I2_S_ROW.gguf \
    --tokens "data/mix_v2/${domain}_val.bin" \
    --seq-len 2048 \
    --windows 256 \
    --decode-steps 0 \
    --threads 24 \
    --threads-batch 24 \
    --metrics-out "reports/native_check/row_${domain}_ce.json"
done
```

协议要求：

- 输入是原始 uint32 stream，不重新 tokenize。
- 每窗读取 2049 token，计算 2048 个 shifted targets。
- 每窗清空 KV memory。
- 每域必须是 256 个互不重叠的前导窗口、524,288 个 predictions。
- 不加 `--logits-out`，避免写出巨量 logits。
- 多任务并发时的耗时不作为性能数据；四域 macro 是四个域 CE 的算术平均。

G32 使用自己的 GGUF 和输出文件名。正式对照必须沿用同一数据 manifest 和窗口，不可换成重新 tokenize 的文本。

### 11.4 真实 greedy 生成

```bash
bitnet.cpp/build/bin/llama-cli \
  -m exports/mix_v2_row/model-I2_S_ROW.gguf \
  -p "The capital of France is" \
  -n 24 \
  -t 16 \
  -tb 16 \
  -c 512 \
  --temp 0 \
  --top-k 1 \
  --no-warmup \
  --no-display-prompt \
  --single-turn \
  < /dev/null
```

`--single-turn` 和关闭 stdin 防止 CLI 进入交互。基础模型或未完成指令对齐的 checkpoint 不能以一句话质量判断 kernel 是否正确；正确性应看前面的分层数值证据。

可用 Python 测量 wall time 和子进程峰值 RSS，无需依赖 `/usr/bin/time`：

```bash
python -I - <<'PY'
import json
import resource
import subprocess
import time

cmd = [
    "bitnet.cpp/build/bin/llama-cli",
    "-m", "exports/mix_v2_row/model-I2_S_ROW.gguf",
    "-p", "The capital of France is",
    "-n", "24", "-t", "16", "-tb", "16", "-c", "512",
    "--temp", "0", "--top-k", "1", "--no-warmup",
    "--no-display-prompt", "--single-turn",
]
started = time.perf_counter()
result = subprocess.run(cmd, stdin=subprocess.DEVNULL, check=False)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
print(json.dumps({
    "returncode": result.returncode,
    "wall_seconds": time.perf_counter() - started,
    "peak_rss_kib": usage.ru_maxrss,
}, indent=2))
raise SystemExit(result.returncode)
PY
```

### 11.5 隔离性能测试

不要在 CE 或其他推理任务并发时测性能。两个模型应顺序执行：

```bash
bitnet.cpp/build/bin/llama-bench \
  -m exports/mix_v2_row/model-I2_S_ROW.gguf \
  -p 512 \
  -n 128 \
  -t 16,64 \
  -r 3 \
  -ngl 0 \
  -o json \
  > reports/native_check/bench_row.json
```

历史协议使用 `n_batch=2048`、`n_ubatch=512`、工具默认 warmup 和 mmap。若当前构建默认值不同，应显式设置并记录实际命令。保留每次样本，不要只挑最快一次。

## 12. 历史性能、内存与解释

环境为 Intel Xeon Platinum 8480+，CPU backend，`n_gpu_layers=0`，512-token prompt、128-token generation、3 次重复。结果不适用于手机、ARM、Metal 或 CUDA。

| 模型 | 线程 | prefill tok/s | decode tok/s |
|---|---:|---:|---:|
| ROW | 16 | 169.316 ± 25.492 | 101.230 ± 9.267 |
| ROW | 64 | **525.705 ± 43.981** | **133.395 ± 5.854** |
| G32 | 16 | 190.826 ± 33.683 | 91.686 ± 2.863 |
| G32 | 64 | 483.244 ± 16.051 | 128.685 ± 2.036 |

16 线程真实短生成与峰值 RSS：

| 模型 | SIMD CLI 生成 | 峰值 RSS |
|---|---:|---:|
| ROW | 62.7 tok/s | 716,224 KiB |
| G32 | 51.8 tok/s | 759,096 KiB |

结论：

1. 两种部署都保持了对应 checkpoint 的量化粒度和精度，没有为重训或部署互相转换。
2. G32 与 ROW 来自不同训练 run，CE 差异不能解释为“只更换存储格式”的因果效果。
3. decode 不随线程线性增长，受到内存带宽和同步限制。
4. G32 文件更大且增加组 scale 工作，不代表任何负载下都比 ROW 快。
5. GGUF 还保留 53 个 F32 tensor，整模型有效 BPW 明显高于三值投影的 2-bit 编码。

原始每次 benchmark、生成日志、CE JSON 和汇总均在[结果目录](docs/reports/pretrain_v2/mix_v3_bitcpm/bitnet_cpp/)。

## 13. 实验配置与逐轮报告

`configs/pretrain/` 保留 s2、s3、mix_v1、mix_v2、mix_v3 的 23 份主实验配置；`configs/sft/` 保留 10 份 SFT resolved config。不要只运行最终配置而忽略此前失败尝试对设计结论的影响。

| 阶段 | 主要变量 | 记录 |
|---|---|---|
| s2 | 8 组 LR × alpha ramp | [s2 screening](docs/reports/s2_screening.md) |
| s3 | BF16、QAT、两类蒸馏 | [s3 comparison](docs/reports/s3_comparison.md) |
| mix_v1 | web-only、50/25/25、70/15/15 与 BF16 对照 | [mix_v1 notes](docs/reports/pretrain_v2/mix_v1/EXPERIMENT_NOTES.md) |
| mix_v2 | 扩网页、2.0007B tokens、拉长 ramp、逐轮 SFT | [mix_v2 notes](docs/reports/pretrain_v2/mix_v2/EXPERIMENT_NOTES.md) |
| mix_v3 | G32 与 LR 两臂、部署对齐 | [mix_v3 notes](docs/reports/pretrain_v2/mix_v3_bitcpm/EXPERIMENT_NOTES.md) |

技术分享第 4 章按轮次记录了真实数据配比、失败尝试、下游结果和下一步决策。历史报告中的绝对路径是当时的执行证据，未为了“看起来可移植”而篡改；新运行应使用本 README 的仓库相对命令。

## 14. 测试矩阵与故障定位

| 层级 | 命令/工具 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| Python 离线 | `bash scripts/test_offline.sh` | 数据格式、采样、恢复辅助、导出规则、CLI 小数据链路 | 完整 CUDA 训练、真实网络、完整模型质量 |
| Kernel | `bash scripts/test_native.sh` | 已知答案、尾部、legacy、真实 GGML 多线程 | GGUF 模型配置和端到端 logits |
| GGUF roundtrip | `tools/verify_bitnet_gguf.py` | tensor 类型、shape、scale、code、F32 保留 | activation 量化和模型图 |
| Logits | `llama-bitnet-eval` + `compare_bitnet_logits.py` | prefill、cached decode、HF/native 执行路径 | 整体数据域质量 |
| 固定窗口 CE | `llama-bitnet-eval --windows 256` | 四域相同 token 协议的端到端 CE | 隔离性能 |
| 真实生成 | `llama-cli --single-turn` | 实际加载、生成、无 crash、速度和 RSS | 单独证明数值正确 |
| Benchmark | `llama-bench` | 指定机器上的 prefill/decode 吞吐 | 其他硬件或后端性能 |

推荐排错顺序：先修 kernel，再修 GGUF，再修模型图/logits，最后才看 CE 和生成。不要用端到端“能输出文字”掩盖底层数值错误。

## 15. 上传 GitHub 前检查

仓库已用 `.gitignore` 排除：

- `data/`、`eval_data/`、`runs/`、`exports/`、`gguf/`、`models/`。
- `MiniCPM4-0.5B*/`。
- `*.pt`、`*.pth`、`*.safetensors`、生成的 `*.gguf`、`*.partial`。
- `build/`、虚拟环境、缓存和普通运行日志。

`bitnet.cpp/3rdparty/llama.cpp/models/*.gguf` 是上游 Git 索引中的 19 个 tokenizer 回归 fixture，约 75 MB，已显式保留；它们不含本项目的训练权重。`docs/reports/**/*.log` 也被显式保留，因为它们是小型真实结果证据。技术分享 DOCX 和机制 PNG 是教学材料，不是模型二进制。

初始化自己的 Git 仓库后，至少检查：

```bash
git status --short --ignored
git diff --check
find . -type f -size +20M -not -path './.git/*' -print
```

还应确认：

1. 没有 HF token、内部 URL、账号密码或本机私密路径进入新文件。
2. 没有把 checkpoint、原始数据、F32 GGUF、构建目录或 `.partial` 加入暂存区。
3. 历史报告中的路径仅作为历史证据保留，新 README 不依赖原项目目录。
4. `bitnet.cpp` 的第三方许可和 NOTICE 文件随源码保留。
5. 上传前重新运行 `bash scripts/test_offline.sh` 和 `bash scripts/test_native.sh`。

本仓库不通过 Git LFS 伪装提供“已包含模型”的体验；模型、数据和结果产物按本文命令显式获取或生成。
