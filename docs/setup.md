# 环境搭建（窗口 A 任务 A.2）

## 1. 安装 Python 3.11

- 官网下载安装（勾选 Add to PATH）：https://www.python.org/downloads/
- 或命令行：`winget install Python.Python.3.11`

## 2. 创建虚拟环境并安装依赖

在仓库根目录（`C:\Users\zyb\source\repos\QuantCliff`）执行：

```powershell
# 本机实际使用 uv 管理（也可用 py -3.11 -m venv）
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

说明：`llama-cpp-python` 在 Windows 有官方预编译 wheel（CPU AVX2 版），**无需安装 C++ 编译器**。若 pip 报错装不上，用 `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`。
说明：`torch` 从 PyPI 默认装 CPU 版（约 2GB）；`transformers`/`scikit-learn` 供实验 3（探针）使用，窗口 A 装好只做排雷。建议设 HF 缓存到仓库根目录：`$env:HF_HOME = "C:\Users\zyb\source\repos\QuantCliff\hf_cache"`。

## 3. 配置 HF 镜像

`scripts/download.py` 已内置 `HF_ENDPOINT=https://hf-mirror.com`，无需手动配置。
若手工使用 `hf` 命令或 `datasets` 库，先执行：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_XET_HIGH_PERFORMANCE = "1"
```

## 4. git 代理（push 失败时）

本仓库已配置本地代理（`git config http.proxy http://127.0.0.1:7897`）。
若代理端口变化：`git config http.proxy http://127.0.0.1:<新端口>`（同理 https.proxy）。
取消代理：`git config --unset http.proxy`。

## 5. 下载模型与数据

模型统一存**仓库根目录** `C:\Users\zyb\source\repos\QuantCliff\models`（已 git 排除）。

### 5.1 窗口 A 已有模型（2 档）

```powershell
python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant Q4_K_M --out C:\Users\zyb\source\repos\QuantCliff\models
python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant F16 --out C:\Users\zyb\source\repos\QuantCliff\models
```

实际文件名是 `Qwen3-1.7B-f16.gguf`（小写 f16）。

### 5.2 窗口 B 全量矩阵（3 模型 × 7 档 + 三元）

2026-10-17 远程核对结论：`ggml-org` 的 Qwen3/Gemma 只有 3 档，Qwen2.5 仓库名不存在。
正式实验改用 `unsloth/Qwen3-1.7B-GGUF`、`Qwen/Qwen2.5-1.5B-Instruct-GGUF`、
`unsloth/gemma-3-1b-it-GGUF`（均为 7 档），三元模型用
`prism-ml/Ternary-Bonsai-1.7B-gguf`（Q2_0 441.8 MiB）。

```powershell
# 1) 只预览计划，不下载
python scripts/download_matrix.py

# 2) 联网核对远端文件是否都存在
python scripts/download_matrix.py --verify-remote

# 3) 正式下载全部 22 个文件（约 17GB，建议寒假 B1 执行）
python scripts/download_matrix.py --execute

# 4) 只下载单个模型 / 单个档位
python scripts/download_matrix.py --model qwen3-1.7b --quant Q2_K --execute
```

默认保存到 `models/<模型名>/` 子目录。**同一个模型只使用同一仓库的量化文件**：
实测 ggml-org 与 unsloth 的同名 Q4_K_M 大小/哈希不同，混用会引入额外变量。
脚本会跳过已存在且非空的文件，可随时中断重跑。

说明：Qwen3 / Gemma 的全档位仓库只提供 BF16 作为高精度基线（与 F16 精度等价），
`analyze.py` 会把 BF16/F16/fp16 统一记作 F16 档，报告中注明。

### 5.3 基准数据

MMLU-Pro、TruthfulQA、C-Eval 由 `eval_loop.py` 首次运行时自动经镜像下载，
统一缓存在仓库根目录 `hf_cache/`（已 git 排除）。首次运行需要联网，之后离线可跑。

## 6. 评测循环与六个基准

统一接口见 `scripts/benchmarks/__init__.py`。六个基准的题量与建议命令：

```powershell
# MMLU-Pro 150（4 选 1 裁剪）
python scripts/eval_loop.py --model models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf --bench mmlu_pro --n 150 --out results\q4_mmlu_pro.csv

# TruthfulQA 100（817 题中转 4 选 1）
python scripts/eval_loop.py --model models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf --bench truthfulqa --n 100 --out results\q4_truthfulqa.csv

# IFEval 100（自写 10 类可判指令）
python scripts/eval_loop.py --model models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf --bench ifeval --n 100 --out results\q4_ifeval.csv

# C-Eval 100（高中语文/数学/物理/计算机网络各 25）
python scripts/eval_loop.py --model models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf --bench ceval --n 100 --out results\q4_ceval.csv

# needle 20（约 4K tokens，必须给足 n_ctx）
python scripts/eval_loop.py --model models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf --bench needle --n 20 --n_ctx 8192 --out results\q4_needle.csv

# 中文事实 30
python scripts/eval_loop.py --model models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf --bench chinese_facts --n 30 --out results\q4_chinese_facts.csv
```

断点续跑与结果保护：

```powershell
# 中断后继续跑同一文件（跳过已完成题目，重跑报错题目）
python scripts/eval_loop.py ... --out results\q4_mmlu_pro.csv --resume

# 不加 --resume 且文件已存在时，旧文件会自动改名归档（results/ 原始 CSV 永不删除）
```

每个 CSV 旁边会生成同名 `.meta.json`，记录模型路径、题量、seed、n_ctx、正确率等运行元数据。

## 7. 最小闭环验证（任务 A.4）

```powershell
python scripts/eval_loop.py --model C:\Users\zyb\source\repos\QuantCliff\models\Qwen3-1.7B-Q4_K_M.gguf --bench mmlu_pro --n 50 --out results\pilot_q4_mmlupro.csv
python scripts/eval_loop.py --model C:\Users\zyb\source\repos\QuantCliff\models\Qwen3-1.7B-f16.gguf --bench mmlu_pro --n 50 --out results\pilot_f16_mmlupro.csv
python -m unittest discover -s tests
```

验收标准：两条 CSV 各 50 行；F16 正确率明显高于 Q4；单元测试全绿。

## 8. 结果汇总与分析

```powershell
python scripts/analyze.py --results-dir results --out-prefix results\summary
```

输出：

- `results/summary_overall.csv`：按基座 × 档位汇总（正确率、解析率、Wilson 95% CI、相对退化率）
- `results/summary_by_bench.csv`：按模型 × 基准汇总（含乱码率、3-gram 重复率、重复行率）
- `results/summary.md`：可直接贴进报告/答辩材料的 Markdown 表

默认对同一 `(model, bench, item_id)` 保留最后写入的一行，避免 pilot 与全量结果重复计数；
需要保留全部原始行时加 `--no-dedupe`。

## 9. 探针路线排雷（任务 A.5）

实验 3 走 transformers 路线（llama-cpp 取不到隐藏层激活），窗口 A 先做 5 分钟 smoke test：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "C:\Users\zyb\source\repos\QuantCliff\hf_cache"
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B'); print('probe route OK, layers =', m.config.num_hidden_layers)"
```

验收：打印层数且无报错（首次下载约 1.2GB 到仓库根目录 hf_cache/），即探针路线可行。
