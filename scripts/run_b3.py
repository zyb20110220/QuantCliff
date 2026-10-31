"""窗口 B 任务 B3：两个验证模型的七档全量实验。

模型：
    - qwen2.5-1.5b  （Qwen/Qwen2.5-1.5B-Instruct-GGUF）
    - gemma-3-1b    （unsloth/gemma-3-1b-it-GGUF）

矩阵与 B2 相同：7 档 × 6 基准 × 500 题/档 = 3,500 题/模型。
两个模型合计 7,000 题。

用法：
    # 先跑两个模型的 pilot（60 题，约 10 分钟）
    python scripts/run_b3.py --pilot

    # 只跑 Qwen2.5 全量（推荐先跑这个）
    python scripts/run_b3.py --model qwen2.5-1.5b

    # 只跑 Gemma 全量
    python scripts/run_b3.py --model gemma-3-1b

    # 两个模型一起跑
    python scripts/run_b3.py

    # 只看计划
    python scripts/run_b3.py --dry-run

输出：results/b3_<模型>_<档位>_<基准>.csv
pilot：results/b3pilot_<模型>_<档位>_<基准>.csv
全部带 --resume，可随时中断重跑。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results"

sys.path.insert(0, str(SCRIPTS))
from download_matrix import MATRIX  # noqa: E402
from run_b2 import BENCHES, PILOT_N, QUANTS, _is_complete  # noqa: E402

MODELS = ["qwen2.5-1.5b", "gemma-3-1b"]


def _model_path(model: str, quant: str) -> Path:
    filename = MATRIX[model]["files"][quant]
    return REPO_ROOT / "models" / model / filename


def main():
    ap = argparse.ArgumentParser(description="B3 验证模型七档全量实验")
    ap.add_argument("--model", default="all",
                    help="all 或 " + " / ".join(MODELS))
    ap.add_argument("--quant", default="all",
                    help="all 或逗号分隔，如 Q2_K,Q4_K_M")
    ap.add_argument("--bench", default="all",
                    help="all 或逗号分隔，如 mmlu_pro,needle")
    ap.add_argument("--pilot", action="store_true",
                    help="每项只跑约 10% 题目，用于链路验证")
    ap.add_argument("--tag", default="b3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise SystemExit(f"未知模型：{bad}；可选：{MODELS}")

    quants = QUANTS if args.quant == "all" else [
        q.strip() for q in args.quant.split(",") if q.strip()]
    benches = BENCHES if args.bench == "all" else [
        b for b in BENCHES if b[0] in {x.strip() for x in args.bench.split(",")}]

    unknown = [q for q in quants if q not in QUANTS]
    if unknown:
        raise SystemExit(f"未知档位：{unknown}；可选：{QUANTS}")

    prefix = "b3pilot" if args.pilot else "b3"
    plan = []
    for model in models:
        for quant in quants:
            for bench, full_n, n_ctx in benches:
                n = PILOT_N[bench] if args.pilot else full_n
                out = RESULTS / f"{prefix}_{model}_{quant}_{bench}.csv"
                plan.append((model, quant, bench, n, n_ctx, out))

    print(f"B3 计划：{len(plan)} 个（模型 × 基准）组合")
    print(f"模型：{models}")
    print(f"档位：{quants}")
    print(f"基准：{[b[0] for b in benches]}")
    print(f"模式：{'pilot' if args.pilot else '全量'}；输出前缀：{prefix}_\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = 0
    started = time.time()

    for i, (model, quant, bench, n, n_ctx, out) in enumerate(plan, 1):
        model_path = _model_path(model, quant)
        meta_path = out.with_name(out.stem + ".meta.json")
        if _is_complete(meta_path, n):
            print(f"[{i}/{len(plan)}] 跳过（已完成）：{model} / {quant} / {bench}")
            skipped += 1
            continue

        cmd = [
            sys.executable, str(SCRIPTS / "eval_loop.py"),
            "--model", str(model_path),
            "--bench", bench,
            "--n", str(n),
            "--n_ctx", str(n_ctx),
            "--out", str(out),
            "--tag", args.tag,
            "--resume",
        ]
        print(f"\n[{i}/{len(plan)}] {model} / {quant} / {bench}  "
              f"n={n}  n_ctx={n_ctx}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            seconds = time.time() - t0
            if proc.returncode != 0:
                failures.append(
                    (model, quant, bench, f"exit {proc.returncode}"))
                print(f"  [失败] exit {proc.returncode}，继续下一个组合")
            else:
                print(f"  [完成] {seconds/60:.1f} 分钟")
        except KeyboardInterrupt:
            print("\n已中断；重跑本脚本会从断点继续。")
            raise SystemExit(130)
        except Exception as exc:
            failures.append((model, quant, bench, str(exc)))
            print(f"  [异常] {exc}")

    elapsed = time.time() - started
    print(f"\nB3 结束：{len(plan) - skipped - len(failures)} 成功，"
          f"{skipped} 跳过，{len(failures)} 失败，总耗时 {elapsed/60:.1f} 分钟")
    if failures:
        print("失败列表：")
        for model, quant, bench, err in failures:
            print(f"  - {model} / {quant} / {bench}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
