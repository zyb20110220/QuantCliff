"""窗口 C 任务 C2：实验 2 —— 三元（1.58-bit）能力退化图谱。

模型：prism-ml/Ternary-Bonsai-1.7B（Q2_0 g64，训练式三值量化）
底模与主实验同为 Qwen3-1.7B，因此可以把三元点挂到 PTQ 位宽曲线上做 RQ5 对照。

运行时：Prism 预编译 llama.cpp（主线 llama.cpp 不认 Q2_0 g64 的量化格式），
所以固定使用 --backend prism_cli。该后端每道题启动一次外部进程，
单题耗时会明显高于 llama-cpp-python，属预期行为。

用法：
    # 先跑 pilot 验证链路（约 30 题）
    python scripts/run_c2.py --pilot

    # 正式全量（500 题）
    python scripts/run_c2.py

    # 只跑某个基准
    python scripts/run_c2.py --bench mmlu_pro,chinese_facts

    # 只看计划
    python scripts/run_c2.py --dry-run

输出：results/c2_Q2_0_<bench>.csv（pilot 用 c2pilot_ 前缀），
      每个组合带 .meta.json，全部带 --resume，可随时中断重跑。
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results"

sys.path.insert(0, str(SCRIPTS))
from download_matrix import MATRIX  # noqa: E402
from run_b2 import BENCHES, PILOT_N, _is_complete  # noqa: E402

MODEL_KEY = "ternary-bonsai-1.7b"
QUANT = "Q2_0"

# Prism 预编译 llama.cpp 的入口（Windows CPU 版）
PRISM_CLI = REPO_ROOT / "models" / "_runtime" / "prism-llama" / "llama-completion.exe"


def _model_path() -> Path:
    filename = MATRIX[MODEL_KEY]["files"][QUANT]
    return REPO_ROOT / "models" / MODEL_KEY / filename


def main():
    ap = argparse.ArgumentParser(description="C2 三元模型六基准实验")
    ap.add_argument("--bench", default="all",
                    help="all 或逗号分隔，如 mmlu_pro,needle")
    ap.add_argument("--pilot", action="store_true",
                    help="每项只跑约 10% 题目，用于链路验证与测速")
    ap.add_argument("--tag", default="c2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    benches = BENCHES if args.bench == "all" else [
        b for b in BENCHES if b[0] in {x.strip() for x in args.bench.split(",")}]
    if not benches:
        raise SystemExit(f"未知基准：{args.bench}")

    model_path = _model_path()
    prefix = "c2pilot" if args.pilot else "c2"
    plan = []
    for bench, full_n, n_ctx in benches:
        n = PILOT_N[bench] if args.pilot else full_n
        out = RESULTS / f"{prefix}_{QUANT}_{bench}.csv"
        plan.append((bench, n, n_ctx, out))

    total_n = sum(p[1] for p in plan)
    print(f"C2 计划：{len(plan)} 个基准组合，共 {total_n} 题")
    print(f"模型：{model_path.name}")
    print(f"后端：prism_cli（{PRISM_CLI.name}）")
    print(f"模式：{'pilot' if args.pilot else '全量'}；输出前缀：{prefix}_\n")

    # 前置检查：文件缺失要尽早报错，而不是跑到第 3 个基准才失败
    if not model_path.exists():
        raise SystemExit(f"找不到模型文件：{model_path}\n请先运行 python scripts/download_matrix.py --model {MODEL_KEY}")
    if not PRISM_CLI.exists():
        raise SystemExit(f"找不到 Prism 运行时：{PRISM_CLI}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = 0
    started = time.time()

    for i, (bench, n, n_ctx, out) in enumerate(plan, 1):
        meta_path = out.with_name(out.stem + ".meta.json")
        if _is_complete(meta_path, n):
            print(f"[{i}/{len(plan)}] 跳过（已完成）：{QUANT} / {bench}")
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
            "--backend", "prism_cli",
            "--cli_path", str(PRISM_CLI),
            "--resume",
        ]
        print(f"\n[{i}/{len(plan)}] {QUANT} / {bench}  n={n}  n_ctx={n_ctx}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        t0 = time.time()
        try:
            import subprocess
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            seconds = time.time() - t0
            if proc.returncode != 0:
                failures.append((bench, f"exit {proc.returncode}"))
                print(f"  [失败] exit {proc.returncode}，继续下一个基准")
            else:
                per_q = seconds / max(n, 1)
                print(f"  [完成] {seconds/60:.1f} 分钟（{per_q:.1f} 秒/题）")
        except KeyboardInterrupt:
            print("\n已中断；重跑本脚本会从断点继续。")
            raise SystemExit(130)
        except Exception as exc:  # noqa: BLE001
            failures.append((bench, str(exc)))
            print(f"  [异常] {exc}")

    elapsed = time.time() - started
    print(f"\nC2 结束：{len(plan) - skipped - len(failures)} 成功，"
          f"{skipped} 跳过，{len(failures)} 失败，总耗时 {elapsed/60:.1f} 分钟")
    if failures:
        print("失败列表：")
        for bench, err in failures:
            print(f"  - {bench}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
