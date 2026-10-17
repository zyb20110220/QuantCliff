"""QuantCliff 核心评测循环（窗口 B 增强版）。

对单个（模型档位, 基准）组合跑题，逐题调用贪婪解码，判分后写 CSV。
所有基准模块遵循统一接口（见 benchmarks/__init__.py）：

    load(n, seed)          -> list[item]      # 加载 n 道题（seed 固定可复现）
    build_prompt(item)     -> str             # 构造完整提示词
    judge(output, item)    -> dict            # {"correct": bool, "parse_ok": bool}
    MAX_TOKENS             -> int             # 输出长度上限

窗口 B 前新增的三项能力：
    1. 断点续跑：--resume 时跳过已完成的题，只补跑中断/报错的题；
    2. 逐题容错：单题推理异常不再中断整轮，错误写入 error 列后可续跑；
    3. 结果保护：不加 --resume 而目标文件已存在时，先自动改名归档，绝不覆盖旧结果。

用法示例：
    python scripts/eval_loop.py --model models/Qwen3-1.7B-Q4_K_M.gguf \
        --bench mmlu_pro --n 50 --out results/pilot_q4_mmlupro.csv --tag pilot
    # 长上下文基准必须给足 n_ctx
    python scripts/eval_loop.py --model models/Qwen3-1.7B-Q4_K_M.gguf \
        --bench needle --n 20 --n_ctx 8192 --out results/q4_needle.csv
"""
import argparse
import csv
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 保证以 python scripts/eval_loop.py 方式运行时能 import benchmarks 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_cpp import Llama  # noqa: E402

HEADER = ["model", "bench", "item_id", "output", "parse_ok",
          "correct", "answer", "tokens", "seconds", "tag", "error"]


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _answer_text(item) -> str:
    """把标准答案转成 CSV 友好的字符串（chinese_facts 的答案可能是列表）。"""
    answer = item.get("answer", "")
    if isinstance(answer, (list, tuple, set)):
        return " | ".join(str(x) for x in answer)
    return str(answer)


def _read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in HEADER})


def _append_row(path: Path, row):
    # 首次写入用 utf-8-sig（Excel 友好）；追加用 utf-8，避免文件中间再出现 BOM
    has_content = path.exists() and path.stat().st_size > 0
    encoding = "utf-8" if has_content else "utf-8-sig"
    with open(path, "a", newline="", encoding=encoding) as f:
        csv.writer(f).writerow(row)


def _archive(path: Path):
    """旧结果改名备份，落实'results/ 原始 CSV 永不删除'。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.stem}_bak_{ts}{path.suffix}")
    path.replace(dest)
    print(f"旧结果已归档：{dest.name}")
    return dest


def run(model_path: str, bench_name: str, n: int, out_csv: str,
        tag: str = "", seed: int = 42, n_ctx: int = 4096,
        resume: bool = False):
    bench = importlib.import_module(f"benchmarks.{bench_name}")
    items = bench.load(n, seed)
    print(f"基准 {bench_name}：加载 {len(items)} 题")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not resume:
        _archive(out_path)

    rows = _read_rows(out_path) if resume else []

    # 兼容窗口 A 的旧表头（没有 error 列）
    if rows and set(HEADER) - set(rows[0].keys()):
        print("检测到旧表头，正在补齐 error 列……")
        _write_rows(out_path, rows)
        rows = _read_rows(out_path)

    # 续跑时丢弃上次报错的行，让它们重新跑（避免重复计数）
    if resume and rows:
        good = [r for r in rows if not str(r.get("error", "")).strip()]
        if len(good) != len(rows):
            print(f"将重跑 {len(rows) - len(good)} 条上次报错的题")
            _write_rows(out_path, good)
            rows = good

    if not out_path.exists():
        _write_rows(out_path, [])

    done = {str(r.get("item_id")) for r in rows
            if not str(r.get("error", "")).strip()}
    if done:
        print(f"断点续跑：已完成 {len(done)} 题，本次只补跑剩余题目")

    llm = Llama(model_path=model_path, n_ctx=n_ctx, seed=seed, verbose=False)
    print(f"模型加载完成：{Path(model_path).name}")

    started = datetime.now().isoformat(timespec="seconds")
    ran = 0
    for idx, item in enumerate(items, 1):
        item_id = str(item.get("id", idx))
        if item_id in done:
            continue

        prompt = bench.build_prompt(item)
        t0 = time.time()
        output, tokens, error = "", 0, ""
        correct = parse_ok = False
        try:
            # 贪婪解码：temperature=0 且 top_k=1，保证结果可复现
            resp = llm(prompt, temperature=0.0, top_k=1,
                       max_tokens=bench.MAX_TOKENS)
            output = resp["choices"][0]["text"].strip()
            tokens = resp["usage"]["completion_tokens"]
            res = bench.judge(output, item)
            correct, parse_ok = bool(res["correct"]), bool(res["parse_ok"])
        except Exception as exc:  # 单题异常不拖垮整轮实验
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [警告] 第 {idx} 题推理失败：{error}")

        seconds = time.time() - t0
        _append_row(out_path, [
            Path(model_path).stem, bench_name, item_id, output,
            parse_ok, correct, _answer_text(item),
            tokens, round(seconds, 2), tag, error,
        ])
        ran += 1
        if ran % 10 == 0 or idx == len(items):
            print(f"  进度 {idx}/{len(items)}  用时 {seconds:.1f}s/题")

    # 汇总以文件为准（含续跑前已完成的部分）
    final_rows = _read_rows(out_path)
    # 同一题若因重跑出现多行，保留最后一行
    latest = {}
    for row in final_rows:
        latest[str(row.get("item_id"))] = row
    final_rows = list(latest.values())
    total = len(final_rows)
    correct_n = sum(1 for r in final_rows if _as_bool(r.get("correct")))
    parse_n = sum(1 for r in final_rows if _as_bool(r.get("parse_ok")))
    errors_n = sum(1 for r in final_rows if str(r.get("error", "")).strip())
    acc = correct_n / total if total else 0.0
    parse_rate = parse_n / total if total else 0.0

    print(f"结果已写入：{out_path}")
    print(f"本次补跑 {ran} 题；文件累计 {total} 题")
    print(f"正确率：{correct_n}/{total} = {acc:.2%}")
    print(f"解析成功率：{parse_n}/{total} = {parse_rate:.2%}")
    if errors_n:
        print(f"推理异常：{errors_n} 题（可用 --resume 重跑）")

    meta = {
        "model_path": str(Path(model_path).resolve()),
        "model_file": Path(model_path).name,
        "bench": bench_name,
        "n_requested": n,
        "seed": seed,
        "n_ctx": n_ctx,
        "tag": tag,
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "rows_total": total,
        "correct": correct_n,
        "parse_ok": parse_n,
        "accuracy": round(acc, 6),
        "parse_rate": round(parse_rate, 6),
        "errors": errors_n,
    }
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"运行元数据：{meta_path}")


def main():
    ap = argparse.ArgumentParser(description="QuantCliff 核心评测循环")
    ap.add_argument("--model", required=True, help="GGUF 模型文件路径")
    ap.add_argument("--bench", required=True,
                    help="基准模块名：mmlu_pro/truthfulqa/ifeval/ceval/needle/chinese_facts")
    ap.add_argument("--n", type=int, default=50, help="题目数量")
    ap.add_argument("--out", required=True, help="结果 CSV 路径")
    ap.add_argument("--tag", default="", help="实验标签（写入 CSV，便于溯源）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_ctx", type=int, default=4096,
                    help="上下文长度；needle 基准请用 8192")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：跳过已完成题目，只补跑剩余/报错题目")
    args = ap.parse_args()
    run(args.model, args.bench, args.n, args.out, args.tag,
        args.seed, args.n_ctx, args.resume)


if __name__ == "__main__":
    main()
