"""结果汇总与分析（窗口 B 初版）。

功能：
    1. 扫描 results/ 下全部原始 CSV，按 model + bench 汇总正确率、解析成功率、
       Wilson 95% 置信区间、平均耗时；
    2. 计算相对于同基座 F16 档的"相对退化率"，隔离模型基础能力差异；
    3. 对中文基准（ceval / chinese_facts）计算文本质量指标：
         - 乱码率：非常用字符占比（本身是低比特退化的证据）
         - 3-gram 重复率：1 - 不重复字符三元组 / 全部三元组
         - 重复行率：1 - 不重复非空行 / 全部非空行
    4. 输出 summary_by_bench.csv、summary_overall.csv、summary.md。

用法：
    python scripts/analyze.py --results-dir results --out-prefix results/summary

注意：默认对同一 (model, bench, item_id) 只保留最后写入的一行，
这样窗口 A 的 pilot 结果会被后续全量结果覆盖，不会重复计数；
需要保留全部原始行时加 --no-dedupe。
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 位宽从低到高（用于排序与曲线）
QUANT_ORDER = ["Q2_0", "Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

# 常见写法统一（大小写不敏感）
QUANT_ALIASES = {
    "q2_0": "Q2_0",
    "q2_k": "Q2_K",
    "q3_k_m": "Q3_K_M",
    "q4_k_m": "Q4_K_M",
    "q5_k_m": "Q5_K_M",
    "q6_k": "Q6_K",
    "q8_0": "Q8_0",
    "f16": "F16",
    "fp16": "F16",
    # Qwen3 / Gemma 的 GGUF 仓库通常只提供 BF16 作为高精度基线；
    # BF16 与 F16 在本实验里都视为"未量化基线"，统一记作 F16。
    "bf16": "F16",
}

ALLOWED_PUNCT = set(
    "，。！？；：、（）【】《》〈〉“”‘’—…·"
    ",.!?;:()[]{}<>\"'`~@#$%^&*+-=/_\\|"
)

CJK_START, CJK_END = "\u4e00", "\u9fff"


def _to_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def quant_of(model: str) -> str:
    """从模型文件名中识别量化档位。"""
    s = str(model).upper()
    # 长关键字优先，避免 Q2_0 被 Q2_K 之类的短词误匹配
    for key in sorted(QUANT_ALIASES, key=len, reverse=True):
        if key.upper() in s:
            return QUANT_ALIASES[key]
    return "UNKNOWN"


def base_of(model: str) -> str:
    """去掉量化后缀，得到基座模型名（用于配对 F16 基线）。"""
    s = str(model)
    for key in sorted(QUANT_ALIASES, key=len, reverse=True):
        s = re.sub(re.escape(key), "", s, flags=re.IGNORECASE)
    return re.sub(r"[-_]+$", "", s).strip()


def text_quality(text: str) -> dict:
    """计算单条输出的乱码率与重复度。"""
    if not isinstance(text, str) or not text.strip():
        return {"garble_rate": np.nan, "repeat_3gram_rate": np.nan,
                "dup_line_rate": np.nan}

    total = len(text)
    bad = 0
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch in ALLOWED_PUNCT:
            continue
        if CJK_START <= ch <= CJK_END:
            continue
        bad += 1

    cjk = "".join(ch for ch in text if CJK_START <= ch <= CJK_END)
    if len(cjk) >= 3:
        grams = [cjk[i:i + 3] for i in range(len(cjk) - 2)]
        repeat = 1 - len(set(grams)) / len(grams)
    else:
        repeat = np.nan

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dup_line = 1 - len(set(lines)) / len(lines) if len(lines) >= 2 else 0.0

    return {
        "garble_rate": bad / total,
        "repeat_3gram_rate": repeat,
        "dup_line_rate": dup_line,
    }


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson 95% 置信区间。"""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def load_results(results_dir: Path, include_backups=False, dedupe=True):
    paths = sorted(results_dir.glob("*.csv"))
    frames = []
    for path in paths:
        name = path.name
        if name.startswith("summary") or name.startswith("curve"):
            continue
        if not include_backups and "_bak_" in name:
            continue
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[跳过] {name}：{exc}")
            continue
        if "model" not in df.columns or "bench" not in df.columns:
            continue
        df["source_file"] = name
        df["source_mtime"] = path.stat().st_mtime
        frames.append(df)

    if not frames:
        raise SystemExit(f"{results_dir} 下没有可分析的结果 CSV")

    df = pd.concat(frames, ignore_index=True)
    for col in ["correct", "parse_ok"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_to_bool)
    for col in ["tokens", "seconds"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].map(_to_float)
    if "error" not in df.columns:
        df["error"] = ""
    if "output" not in df.columns:
        df["output"] = ""

    df["model"] = df["model"].astype(str)
    df["bench"] = df["bench"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["quant"] = df["model"].map(quant_of)
    df["base"] = df["model"].map(base_of)

    # 同一题多次运行时保留最后写入的一行
    if dedupe:
        df = (df.sort_values("source_mtime")
                .drop_duplicates(["model", "bench", "item_id"], keep="last"))

    quality = df["output"].map(text_quality).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), quality.reset_index(drop=True)],
                   axis=1)
    return df


def summarize(df: pd.DataFrame):
    group_cols = ["base", "model", "quant", "bench"]
    g = df.groupby(group_cols, dropna=False)

    out = g.agg(
        n=("correct", "size"),
        correct=("correct", "sum"),
        parse_ok=("parse_ok", "sum"),
        mean_seconds=("seconds", "mean"),
        mean_tokens=("tokens", "mean"),
        garble_rate=("garble_rate", "mean"),
        repeat_3gram_rate=("repeat_3gram_rate", "mean"),
        dup_line_rate=("dup_line_rate", "mean"),
    ).reset_index()

    out["accuracy"] = out["correct"] / out["n"].replace(0, np.nan)
    out["parse_rate"] = out["parse_ok"] / out["n"].replace(0, np.nan)
    ci = out.apply(lambda r: wilson_ci(int(r["correct"]), int(r["n"])), axis=1)
    out["wilson_lo"] = [c[0] for c in ci]
    out["wilson_hi"] = [c[1] for c in ci]

    # 同一 base + bench 内，以 F16 为基线算相对退化率
    out["rel_degradation"] = np.nan
    for (base, bench), idx in out.groupby(["base", "bench"]).groups.items():
        sub = out.loc[idx]
        f16 = sub[sub["quant"] == "F16"]
        if f16.empty or not np.isfinite(f16.iloc[0]["accuracy"]) \
                or f16.iloc[0]["accuracy"] <= 0:
            continue
        baseline = f16.iloc[0]["accuracy"]
        out.loc[idx, "rel_degradation"] = (
            baseline - sub["accuracy"]) / baseline

    out["quant_order"] = out["quant"].map(
        {q: i for i, q in enumerate(QUANT_ORDER)}).fillna(99)
    out = out.sort_values(["base", "bench", "quant_order"]).drop(
        columns=["quant_order"])
    return out


def overall(df: pd.DataFrame):
    g = df.groupby(["base", "model", "quant"], dropna=False)
    out = g.agg(
        n=("correct", "size"),
        correct=("correct", "sum"),
        parse_ok=("parse_ok", "sum"),
        mean_seconds=("seconds", "mean"),
        garble_rate=("garble_rate", "mean"),
        repeat_3gram_rate=("repeat_3gram_rate", "mean"),
        dup_line_rate=("dup_line_rate", "mean"),
    ).reset_index()

    out["accuracy"] = out["correct"] / out["n"].replace(0, np.nan)
    out["parse_rate"] = out["parse_ok"] / out["n"].replace(0, np.nan)
    ci = out.apply(lambda r: wilson_ci(int(r["correct"]), int(r["n"])), axis=1)
    out["wilson_lo"] = [c[0] for c in ci]
    out["wilson_hi"] = [c[1] for c in ci]

    # 总体相对退化率：只有当前档位与 F16 覆盖完全相同的基准集合时才计算，
    # 否则不同题集混合会让数字失真（例如某档多跑了一个基准）。
    bench_sets = (df.groupby(["base", "model", "quant"])["bench"]
                    .agg(lambda s: tuple(sorted(set(s)))).to_dict())
    out["rel_degradation"] = np.nan
    for base, idx in out.groupby("base").groups.items():
        sub = out.loc[idx]
        f16 = sub[sub["quant"] == "F16"]
        if f16.empty or not np.isfinite(f16.iloc[0]["accuracy"]) \
                or f16.iloc[0]["accuracy"] <= 0:
            continue
        f16_row = f16.iloc[0]
        f16_key = bench_sets.get((base, f16_row["model"], "F16"))
        baseline = f16_row["accuracy"]
        for row_idx, row in sub.iterrows():
            key = bench_sets.get((base, row["model"], row["quant"]))
            if key == f16_key:
                out.loc[row_idx, "rel_degradation"] = (
                    baseline - row["accuracy"]) / baseline

    out["quant_order"] = out["quant"].map(
        {q: i for i, q in enumerate(QUANT_ORDER)}).fillna(99)
    out = out.sort_values(["base", "quant_order"]).drop(
        columns=["quant_order"])
    return out


def _fmt_pct(value):
    return "" if pd.isna(value) else f"{value:.2%}"


def write_markdown(by_bench: pd.DataFrame, overall_df: pd.DataFrame, path: Path):
    lines = [
        "# 结果汇总（自动生成）",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "> 相对退化率 = (同基座 F16 正确率 − 当前档位正确率) / F16 正确率。",
        "> 乱码率与重复度仅对中文输出有解释意义，分基准表会如实列出。",
        "",
        "## 总体（跨基准合并）",
        "",
        "| 基座 | 档位 | 题数 | 正确率 | 解析率 | Wilson 95% | 相对退化 |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for _, r in overall_df.iterrows():
        ci = "" if pd.isna(r["wilson_lo"]) else (
            f"[{r['wilson_lo']:.1%}, {r['wilson_hi']:.1%}]")
        lines.append(
            f"| {r['base']} | {r['quant']} | {int(r['n'])} | "
            f"{_fmt_pct(r['accuracy'])} | {_fmt_pct(r['parse_rate'])} | "
            f"{ci} | {_fmt_pct(r['rel_degradation'])} |"
        )

    lines += [
        "",
        "## 分基准",
        "",
        "| 模型 | 基准 | 题数 | 正确率 | 解析率 | 相对退化 | 乱码率 | 3-gram重复率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in by_bench.iterrows():
        lines.append(
            f"| {r['model']} | {r['bench']} | {int(r['n'])} | "
            f"{_fmt_pct(r['accuracy'])} | {_fmt_pct(r['parse_rate'])} | "
            f"{_fmt_pct(r['rel_degradation'])} | "
            f"{_fmt_pct(r['garble_rate'])} | "
            f"{_fmt_pct(r['repeat_3gram_rate'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入：{path}")


def main():
    ap = argparse.ArgumentParser(description="QuantCliff 结果汇总")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-prefix", default="results/summary",
                    help="输出文件前缀，如 results/summary")
    ap.add_argument("--include-backups", action="store_true",
                    help="把 _bak_ 归档文件也纳入分析")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="不按 (model, bench, item_id) 去重")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    df = load_results(results_dir, args.include_backups, not args.no_dedupe)
    print(f"读取 {df['source_file'].nunique()} 个结果文件，{len(df)} 条记录")

    by_bench = summarize(df)
    overall_df = overall(df)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    by_bench.to_csv(prefix.with_name(prefix.name + "_by_bench.csv"),
                    index=False, encoding="utf-8-sig")
    overall_df.to_csv(prefix.with_name(prefix.name + "_overall.csv"),
                      index=False, encoding="utf-8-sig")
    write_markdown(by_bench, overall_df,
                   prefix.with_name(prefix.name + ".md"))
    print("完成。")


if __name__ == "__main__":
    main()
