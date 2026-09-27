# 环境搭建（窗口 A 任务 A.2）

## 1. 安装 Python 3.11

- 官网下载安装（勾选 Add to PATH）：https://www.python.org/downloads/
- 或命令行：`winget install Python.Python.3.11`

## 2. 创建虚拟环境并安装依赖

在仓库根目录（`C:\Users\zyb\source\repos\QuantCliff`）执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

说明：`llama-cpp-python` 在 Windows 有官方预编译 wheel（CPU AVX2 版），**无需安装 C++ 编译器**。若 pip 报错装不上，用 `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`。
说明：`torch` 从 PyPI 默认装 CPU 版（约 2GB）；`transformers`/`scikit-learn` 供实验 3（探针）使用，窗口 A 装好只做排雷。建议设 HF 缓存到仓库根目录：`$env:HF_HOME = "C:\Users\zyb\source\repos\QuantCliff\hf_cache"`。

## 3. 配置 HF 镜像

`scripts/download.py` 已内置 `HF_ENDPOINT=https://hf-mirror.com`，无需手动配置。
若手工使用 `hf` 命令或 `datasets` 库，先执行：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
```

## 4. git 代理（push 失败时）

本仓库已配置本地代理（`git config http.proxy http://127.0.0.1:7897`）。
若代理端口变化：`git config http.proxy http://127.0.0.1:<新端口>`（同理 https.proxy）。
取消代理：`git config --unset http.proxy`。

## 5. 下载模型与数据

模型统一存**仓库根目录** `C:\Users\zyb\source\repos\QuantCliff\models`（已 git 排除），窗口 A 只需两个档位：

```powershell
python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant Q4_K_M --out C:\Users\zyb\source\repos\QuantCliff\models
python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant F16 --out C:\Users\zyb\source\repos\QuantCliff\models
```

MMLU-Pro 数据由 `scripts/eval_loop.py` 运行时自动经镜像下载。

## 6. 最小闭环验证（任务 A.4）

```powershell
python scripts/eval_loop.py --model C:\Users\zyb\source\repos\QuantCliff\models\Qwen3-1.7B-Q4_K_M.gguf --bench mmlu_pro --n 50 --out results\pilot_q4_mmlupro.csv
python scripts/eval_loop.py --model C:\Users\zyb\source\repos\QuantCliff\models\Qwen3-1.7B-F16.gguf --bench mmlu_pro --n 50 --out results\pilot_f16_mmlupro.csv
python -m unittest discover -s tests
```

验收标准：两条 CSV 各 50 行；F16 正确率明显高于 Q4；单元测试全绿。

## 7. 探针路线排雷（任务 A.5）

实验 3 走 transformers 路线（llama-cpp 取不到隐藏层激活），窗口 A 先做 5 分钟 smoke test：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "C:\Users\zyb\source\repos\QuantCliff\hf_cache"
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B'); print('probe route OK, layers =', m.config.num_hidden_layers)"
```

验收：打印层数且无报错（首次下载约 1.2GB 到仓库根目录 hf_cache/），即探针路线可行。
