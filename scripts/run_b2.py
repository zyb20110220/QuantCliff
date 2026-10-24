"""窗口 B 任务 B2：主模型 Qwen3-1.7B 七档全量实验。

实验矩阵（每档 500 题）：
    MMLU-Pro 150 + TruthfulQA 100 + IFEval 100 + C-Eval 100
    + needle 20（n_ctx=8192）+ 中文事实 30

用法：
    # 先跑 pilot（每项约 10%，约 210 题，用于验证全链路）
    python scripts/run_b2.py --pilot

    # 正式全量（7 档 × 500 = 3500 题）
    python scripts/run_b2.py

    # 只跑某一档 / 某一基准
    python scripts/run_b2.py --quant Q2_K,Q4_K_M --bench mmlu_pro

    # 只看计划，不执行
    python scripts/run_b2.py --dry-run

每个组合调用一次 eval_loop.py，输出到 results/b2_<quant>_<bench>.csv
（pilot 用 b2pilot_ 前缀）。全部带 --resume，可随时中断重跑。
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

MODEL_KEY = "qwen3-1.7b"
QUANTS = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

# (基准名, 全量题数, n_ctx)
BENCHES = [
    ("mmlu_pro", 150, 4096),
    ("truthfulqa", 100, 4096),
    ("ifeval", 100, 4096),
    ("ceval", 100, 4096),
    ("needle", 20, 8192),
    ("chinese_facts", 30, 4096),
]

PILOT_N = {
    "mmlu_pro": 10,
    "truthfulqa": 5,
    "ifeval": 5,
    "ceval": 5,
    "needle": 2,
    "chinese_facts": 3,
}


def _model_path(quant: str) -> Path:
    filename = MATRIX[MODEL_KEY]["files"][quant]
    return REPO_ROOT / "models" / MODEL_KEY / filename


def _is_complete(meta_path: Path, n: int) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (meta.get("rows_total", 0) >= n
            and meta.get("errors", 1) == 0)


def main():
    ap = argparse.ArgumentParser(description="B2 主模型七档全量实验")
    ap.add_argument("--quant", default="all",
                    help="all 或逗号分隔，如 Q2_K,Q4_K_M")
    ap.add_argument("--bench", default="all",
                    help="all 或逗号分隔，如 mmlu_pro,needle")
    ap.add_argument("--pilot", action="store_true",
                    help="每项只跑约 10% 题目，用于链路验证")
    ap.add_argument("--tag", default="b2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    quants = QUANTS if args.quant == "all" else [
        q.strip() for q in args.quant.split(",") if q.strip()]
    benches = BENCHES if args.bench == "all" else [
        b for b in BENCHES if b[0] in {x.strip() for x in args.bench.split(",")}]

    unknown = [q for q in quants if q not in QUANTS]
    if unknown:
        raise SystemExit(f"未知档位：{unknown}；可选：{QUANTS}")

    prefix = "b2pilot" if args.pilot else "b2"
    plan = []
    for quant in quants:
        for bench, full_n, n_ctx in benches:
            n = PILOT_N[bench] if args.pilot else full_n
            out = RESULTS / f"{prefix}_{quant}_{bench}.csv"
            plan.append((quant, bench, n, n_ctx, out))

    print(f"B2 计划：{len(plan)} 个（模型 × 基准）组合")
    print(f"档位：{quants}")
    print(f"基准：{[b[0] for b in benches]}")
    print(f"模式：{'pilot' if args.pilot else '全量'}；输出前缀：{prefix}_\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = 0
    started = time.time()

    for i, (quant, bench, n, n_ctx, out) in enumerate(plan, 1):
        model_path = _model_path(quant)
        meta_path = out.with_name(out.stem + ".meta.json")
        if _is_complete(meta_path, n):
            print(f"[{i}/{len(plan)}] 跳过（已完成）：{quant} / {bench}")
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
        print(f"\n[{i}/{len(plan)}] {quant} / {bench}  n={n}  n_ctx={n_ctx}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            seconds = time.time() - t0
            if proc.returncode != 0:
                failures.append((quant, bench, f"exit {proc.returncode}"))
                print(f"  [失败] exit {proc.returncode}，继续下一个组合")
            else:
                print(f"  [完成] {seconds/60:.1f} 分钟")
        except KeyboardInterrupt:
            print("\n已中断；重跑本脚本会从断点继续。")
            raise SystemExit(130)
        except Exception as exc:
            failures.append((quant, bench, str(exc)))
            print(f"  [异常] {exc}")

    elapsed = time.time() - started
    print(f"\nB2 结束：{len(plan) - skipped - len(failures)} 成功，"
          f"{skipped} 跳过，{len(failures)} 失败，总耗时 {elapsed/60:.1f} 分钟")
    if failures:
        print("失败列表：")
        for quant, bench, err in failures:
            print(f"  - {quant} / {bench}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
