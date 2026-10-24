"""窗口 B 任务 B1：模型兼容性冒烟测试。

目的：在跑全量 10,500 题之前，先用每个模型家族的代表档位做一次
"加载 + 生成几个 token" 的检查，提前发现：
    - GGUF 架构是否被当前 llama-cpp-python 支持；
    - 三元 Q2_0 需要 Prism 的 llama.cpp（group-64）；
    - Gemma-3 是否需要额外的 chat template 处理。

只测代表档位（低比特 + 高精度），不重复测全部 7 档；全量实验会自然覆盖。
结果写入 results/model_smoke.csv。
"""
import csv
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_cpp import Llama  # noqa: E402

from download_matrix import MATRIX  # noqa: E402
from eval_loop import _generate_prism_cli  # noqa: E402

# (模型目录名, 档位标签, 后端)
CASES = [
    ("qwen3-1.7b", "Q2_K", "llama_cpp"),
    ("qwen3-1.7b", "Q4_K_M", "llama_cpp"),
    ("qwen3-1.7b", "F16", "llama_cpp"),
    ("qwen2.5-1.5b", "Q2_K", "llama_cpp"),
    ("gemma-3-1b", "Q2_K", "llama_cpp"),
    ("ternary-bonsai-1.7b", "Q2_0", "prism_cli"),
]

PROMPT = "用一句话介绍太阳。"
MAX_TOKENS = 8
PRISM_CLI = "models/_runtime/prism-llama/llama-completion.exe"


def main():
    rows = []
    for model, quant, backend in CASES:
        entry = MATRIX[model]
        filename = entry["files"][quant]
        path = Path("models") / model / filename
        print(f"\n=== {model} / {quant}（{backend}）===")
        print(f"文件：{path}")

        if not path.exists():
            rows.append({
                "model": model, "quant": quant, "backend": backend,
                "file": filename, "status": "MISSING", "seconds": 0,
                "output": "", "error": "file not found",
            })
            print("[缺失] 文件不存在")
            continue

        t0 = time.time()
        try:
            if backend == "prism_cli":
                output, _tokens = _generate_prism_cli(
                    PRISM_CLI, str(path), PROMPT, MAX_TOKENS, 512, 42)
            else:
                llm = Llama(model_path=str(path), n_ctx=512, seed=42,
                            verbose=False)
                resp = llm(PROMPT, temperature=0.0, top_k=1,
                           max_tokens=MAX_TOKENS)
                output = resp["choices"][0]["text"].strip()
                del llm
                gc.collect()
            seconds = round(time.time() - t0, 1)
            rows.append({
                "model": model, "quant": quant, "backend": backend,
                "file": filename, "status": "OK", "seconds": seconds,
                "output": output, "error": "",
            })
            print(f"[通过] {seconds}s  输出：{output!r}")
        except Exception as exc:
            seconds = round(time.time() - t0, 1)
            error = f"{type(exc).__name__}: {exc}"
            rows.append({
                "model": model, "quant": quant, "backend": backend,
                "file": filename, "status": "FAIL", "seconds": seconds,
                "output": "", "error": error,
            })
            print(f"[失败] {seconds}s  {error}")

    out = Path("results") / "model_smoke.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "quant", "backend", "file", "status", "seconds",
            "output", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已写入：{out}")

    failed = [r for r in rows if r["status"] != "OK"]
    if failed:
        print(f"有 {len(failed)} 个代表档位未通过，需要处理后再跑全量。")
        raise SystemExit(1)
    print("全部代表档位通过 ✅")


if __name__ == "__main__":
    main()
