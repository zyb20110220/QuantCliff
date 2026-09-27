"""窗口 A 任务 A.3：从 hf-mirror 镜像下载 GGUF 模型。

用法示例：
    python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant Q4_K_M --out C:\\Users\\zyb\\source\\repos\\QuantCliff\\models
    python scripts/download.py --repo ggml-org/Qwen3-1.7B-GGUF --quant F16 --out C:\\Users\\zyb\\source\\repos\\QuantCliff\\models

原理：列出远端仓库全部文件，按关键字匹配 .gguf 文件名后下载（无需硬编码文件名）。
"""
import argparse
import os
from pathlib import Path

# 国内镜像：必须先于 huggingface_hub 导入设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def find_file(repo_id: str, quant: str) -> str:
    """列出仓库所有文件，返回文件名中包含 quant 关键字的 .gguf 文件。"""
    files = HfApi().list_repo_files(repo_id)
    ggufs = [f for f in files if f.endswith(".gguf")]
    matches = [f for f in ggufs if quant.lower() in Path(f).stem.lower()]
    if not matches:
        raise SystemExit(
            f"未找到含 '{quant}' 的 GGUF。仓库内现有 GGUF：\n  " + "\n  ".join(ggufs)
        )
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description="下载 GGUF 模型（走 hf-mirror）")
    ap.add_argument("--repo", required=True,
                    help="HF 仓库 id，如 ggml-org/Qwen3-1.7B-GGUF")
    ap.add_argument("--quant", required=True,
                    help="量化关键字，如 Q4_K_M / F16 / Q2_K")
    ap.add_argument("--out", default="models", help="本地保存目录")
    args = ap.parse_args()

    fname = find_file(args.repo, args.quant)
    print(f"[1/2] 匹配到文件：{fname}")
    path = hf_hub_download(
        repo_id=args.repo, filename=fname, local_dir=args.out)
    print(f"[2/2] 已保存：{path}")


if __name__ == "__main__":
    main()
