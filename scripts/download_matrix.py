"""窗口 B 任务 B1：按实验矩阵批量下载 GGUF 模型。

2026-10-17 远程核对结果（重要）：
    - `ggml-org/Qwen3-1.7B-GGUF` 只有 Q4_K_M / Q8_0 / F16 三档，**不是**开题方案假设的全套；
    - `ggml-org/gemma-3-1b-it-GGUF` 同样只有三档；
    - `ggml-org/Qwen2.5-1.5B-Instruct-GGUF` 仓库不存在；
    - 因此改用下面这些已确认含全部 7 档的仓库。
    - Qwen3 / Gemma 的全档位仓库只提供 BF16 作为高精度基线；BF16 与 F16 精度等价，
      分析脚本统一记作 F16 档（报告里会注明）。

用法：
    # 只打印计划，不下载（默认）
    python scripts/download_matrix.py

    # 先核对远端文件是否都存在（联网，几秒）
    python scripts/download_matrix.py --verify-remote

    # 正式下载全部模型（约 20~30GB，寒假 B1 执行）
    python scripts/download_matrix.py --execute

    # 只下载某一模型 / 某一档位
    python scripts/download_matrix.py --model qwen3-1.7b --quant Q4_K_M --execute

默认保存到 `models/<模型名>/` 子目录，保证同一个模型只使用同一来源的量化文件
（2026-10-17 核对：ggml-org 与 unsloth 的同名 Q4_K_M 大小/哈希不同，不能混用）。
本地文件已存在且非空时自动跳过，可安全重复执行。
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

# 已验证的下载矩阵：label -> 精确文件名
MATRIX = {
    "qwen3-1.7b": {
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "note": "主模型；F16 档实际为 BF16（精度等价基线）",
        "files": {
            "Q2_K": "Qwen3-1.7B-Q2_K.gguf",
            "Q3_K_M": "Qwen3-1.7B-Q3_K_M.gguf",
            "Q4_K_M": "Qwen3-1.7B-Q4_K_M.gguf",
            "Q5_K_M": "Qwen3-1.7B-Q5_K_M.gguf",
            "Q6_K": "Qwen3-1.7B-Q6_K.gguf",
            "Q8_0": "Qwen3-1.7B-Q8_0.gguf",
            "F16": "Qwen3-1.7B-BF16.gguf",
        },
    },
    "qwen2.5-1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "note": "验证模型 1；F16 档为官方 fp16",
        "files": {
            "Q2_K": "qwen2.5-1.5b-instruct-q2_k.gguf",
            "Q3_K_M": "qwen2.5-1.5b-instruct-q3_k_m.gguf",
            "Q4_K_M": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "Q5_K_M": "qwen2.5-1.5b-instruct-q5_k_m.gguf",
            "Q6_K": "qwen2.5-1.5b-instruct-q6_k.gguf",
            "Q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
            "F16": "qwen2.5-1.5b-instruct-fp16.gguf",
        },
    },
    "gemma-3-1b": {
        "repo": "unsloth/gemma-3-1b-it-GGUF",
        "note": "验证模型 2；F16 档实际为 BF16（精度等价基线）",
        "files": {
            "Q2_K": "gemma-3-1b-it-Q2_K.gguf",
            "Q3_K_M": "gemma-3-1b-it-Q3_K_M.gguf",
            "Q4_K_M": "gemma-3-1b-it-Q4_K_M.gguf",
            "Q5_K_M": "gemma-3-1b-it-Q5_K_M.gguf",
            "Q6_K": "gemma-3-1b-it-Q6_K.gguf",
            "Q8_0": "gemma-3-1b-it-Q8_0.gguf",
            "F16": "gemma-3-1b-it-BF16.gguf",
        },
    },
    "ternary-bonsai-1.7b": {
        "repo": "prism-ml/Ternary-Bonsai-1.7B-gguf",
        "note": "实验 2 三元模型；只需 Q2_0（441.8 MiB）",
        "files": {
            "Q2_0": "Ternary-Bonsai-1.7B-Q2_0.gguf",
        },
    },
}

ALL_LABELS = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "Q2_0"]


def _selected(args):
    models = list(MATRIX) if args.model == "all" else [args.model]
    for model in models:
        if model not in MATRIX:
            raise SystemExit(f"未知模型 {model}；可选：{', '.join(MATRIX)}")
        entry = MATRIX[model]
        for label, filename in entry["files"].items():
            if args.quant in ("all", label):
                yield model, entry["repo"], label, filename


def verify_remote():
    api = HfApi()
    ok = True
    for model, entry in MATRIX.items():
        try:
            remote = set(api.list_repo_files(entry["repo"]))
        except Exception as exc:
            print(f"[失败] {model}: {entry['repo']} -> {exc}")
            ok = False
            continue
        missing = [f for f in entry["files"].values() if f not in remote]
        if missing:
            ok = False
            print(f"[缺文件] {model}: {', '.join(missing)}")
        else:
            print(f"[通过] {model}: {entry['repo']}（{len(entry['files'])} 档）")
    return ok


def main():
    ap = argparse.ArgumentParser(description="按矩阵下载 GGUF 模型")
    ap.add_argument("--out", default="models", help="本地保存根目录")
    ap.add_argument("--flat", action="store_true",
                    help="直接存到 --out 下（默认按模型建子目录，避免不同来源混用）")
    ap.add_argument("--model", default="all",
                    help="all 或 " + " / ".join(MATRIX))
    ap.add_argument("--quant", default="all",
                    help="all 或 " + " / ".join(ALL_LABELS))
    ap.add_argument("--execute", action="store_true",
                    help="真正下载；不加此参数只打印计划")
    ap.add_argument("--verify-remote", action="store_true",
                    help="联网核对远端文件是否存在后退出")
    args = ap.parse_args()

    if args.verify_remote:
        raise SystemExit(0 if verify_remote() else 1)

    root = Path(args.out)
    plan = list(_selected(args))
    print(f"共 {len(plan)} 个文件，保存根目录：{root.resolve()}")
    print(f"模式：{'下载' if args.execute else '仅预览（加 --execute 才会下载）'}\n")

    failures = []
    for model, repo, label, filename in plan:
        model_dir = root if args.flat else root / model
        local = model_dir / filename
        if local.exists() and local.stat().st_size > 0:
            size = local.stat().st_size / 1024 / 1024
            print(f"[跳过] {model:20s} {label:5s} {local}（已有 {size:.0f} MiB）")
            continue
        print(f"[计划] {model:20s} {label:5s} {repo} :: {filename}")
        print(f"       本地 -> {local}")
        if not args.execute:
            continue
        try:
            path = hf_hub_download(
                repo_id=repo, filename=filename, local_dir=str(model_dir))
            print(f"[完成] {path}")
        except Exception as exc:
            failures.append((filename, str(exc)))
            print(f"[失败] {filename} -> {exc}")

    if failures:
        print("\n以下文件下载失败，可重跑本脚本断点续传：")
        for name, err in failures:
            print(f"  - {name}: {err}")
        raise SystemExit(1)
    print("\n全部完成。")


if __name__ == "__main__":
    main()
