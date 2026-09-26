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

模型存 **D 盘**（例如 `D:\QuantCliff\models`），窗口 A 只需两个档位：

```powershell
python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant Q4_K_M --out D:\QuantCliff\models
python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant F16 --out D:\QuantCliff\models
```

MMLU-Pro 数据由 `scripts/eval_loop.py` 运行时自动经镜像下载。

## 6. 最小闭环验证（任务 A.4）

```powershell
python scripts/eval_loop.py --model D:\QuantCliff\models\Qwen3-1.7B-Q4_K_M.gguf --bench mmlu_pro --n 50 --out results\pilot_q4_mmlupro.csv
python scripts/eval_loop.py --model D:\QuantCliff\models\Qwen3-1.7B-F16.gguf --bench mmlu_pro --n 50 --out results\pilot_f16_mmlupro.csv
python -m unittest discover -s tests
```

验收标准：两条 CSV 各 50 行；F16 正确率明显高于 Q4；单元测试全绿。
