"""窗口 A 任务 A.4：核心评测循环。

对单个（模型档位, 基准）组合跑题，逐题调用贪婪解码，判分后写 CSV。
所有基准模块遵循统一接口（见 benchmarks/README 约定，即本文件下方的说明）：

    load(n, seed)          -> list[item]      # 加载 n 道题（seed 固定可复现）
    build_prompt(item)     -> str             # 构造完整提示词
    judge(output, item)    -> dict            # {"correct": bool, "parse_ok": bool}
    MAX_TOKENS             -> int             # 输出长度上限

用法示例：
    python scripts/eval_loop.py --model D:\\QuantCliff\\models\\Qwen3-1.7B-Q4_K_M.gguf \
        --bench mmlu_pro --n 50 --out results\\pilot_q4_mmlupro.csv --tag pilot
"""
import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

# 保证以 python scripts/eval_loop.py 方式运行时能 import benchmarks 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_cpp import Llama  # noqa: E402


def run(model_path: str, bench_name: str, n: int, out_csv: str,
        tag: str = "", seed: int = 42, n_ctx: int = 4096):
    bench = importlib.import_module(f"benchmarks.{bench_name}")
    items = bench.load(n, seed)
    print(f"基准 {bench_name}：加载 {len(items)} 题")

    llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
    print(f"模型加载完成：{Path(model_path).name}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "bench", "item_id", "output", "parse_ok",
              "correct", "answer", "tokens", "seconds", "tag"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx, item in enumerate(items, 1):
            prompt = bench.build_prompt(item)
            t0 = time.time()
            # 贪婪解码：temperature=0 且 top_k=1，保证结果可复现
            resp = llm(prompt, temperature=0.0, top_k=1, max_tokens=bench.MAX_TOKENS)
            seconds = time.time() - t0
            output = resp["choices"][0]["text"].strip()
            tokens = resp["usage"]["completion_tokens"]
            res = bench.judge(output, item)
            w.writerow([Path(model_path).stem, bench_name, item.get("id", idx),
                        output, res["parse_ok"], res["correct"],
                        item.get("answer", ""), tokens, round(seconds, 2), tag])
            f.flush()
            if idx % 10 == 0 or idx == len(items):
                print(f"  进度 {idx}/{len(items)}  用时 {seconds:.1f}s/题")
    print(f"结果已写入：{out_path}")


def main():
    ap = argparse.ArgumentParser(description="QuantCliff 核心评测循环")
    ap.add_argument("--model", required=True, help="GGUF 模型文件路径")
    ap.add_argument("--bench", required=True, help="基准模块名，如 mmlu_pro")
    ap.add_argument("--n", type=int, default=50, help="题目数量")
    ap.add_argument("--out", required=True, help="结果 CSV 路径")
    ap.add_argument("--tag", default="", help="实验标签（写入 CSV，便于溯源）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_ctx", type=int, default=4096)
    args = ap.parse_args()
    run(args.model, args.bench, args.n, args.out, args.tag, args.seed, args.n_ctx)


if __name__ == "__main__":
    main()
