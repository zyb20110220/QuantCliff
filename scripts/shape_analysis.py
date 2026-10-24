"""窗口 B 任务 B4：B2 结果的形态分析。

回答 RQ1（衰减形态）与 RQ2（维度差异）：
    1. 对每个基准的"位宽—正确率"曲线，比较线性模型与分段（平台+斜坡）模型，
       用 AIC/BIC 判断更像"均匀衰减"还是"悬崖/平台"；
    2. 用二阶差分定位斜率变化最大的位置；
    3. 对相邻量化档做配对 McNemar 检验，判断差异是否显著；
    4. 输出 CSV 与 Markdown 报告。

输入：results/b2_*.csv（run_b2.py 生成）
输出：results/shape_analysis_b2.csv
      results/mcnemar_b2.csv
      results/shape_analysis_b2.md

用法：
    python scripts/shape_analysis.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import analyze

# 名义位宽（用于横轴）。K-quant 的实际有效位宽略有差异，这里是近似刻度。
BITS = {
    "Q2_K": 2.5,
    "Q3_K_M": 3.5,
    "Q4_K_M": 4.5,
    "Q5_K_M": 5.5,
    "Q6_K": 6.5,
    "Q8_0": 8.0,
    "F16": 16.0,
}
QUANT_ORDER = list(BITS)

# 相邻档对
PAIRS = list(zip(QUANT_ORDER[:-1], QUANT_ORDER[1:]))


def _aic_bic(sse: float, n: int, k: int):
    sse = max(sse, 1e-12)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return aic, bic


def fit_linear(x: np.ndarray, y: np.ndarray):
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    sse = float(np.sum((y - pred) ** 2))
    aic, bic = _aic_bic(sse, len(y), 2)
    return {"sse": sse, "aic": aic, "bic": bic,
            "slope": float(coef[1]), "intercept": float(coef[0])}


def fit_piecewise(x: np.ndarray, y: np.ndarray):
    """平台 + 斜坡模型：y = a (x>=c)，y = a + b*(x-c) (x<c)。

    在相邻 x 的中点网格上搜索变化点 c，用最小二乘拟合 a,b。
    """
    xs = np.unique(x)
    candidates = (xs[:-1] + xs[1:]) / 2
    best = None
    for c in candidates:
        design = np.column_stack([np.ones_like(x), np.minimum(0, x - c)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if b < 0:  # 物理上斜率应为正；退化时按常数模型处理
            a, b = float(np.mean(y)), 0.0
        pred = a + b * np.minimum(0, x - c)
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best["sse"]:
            aic, bic = _aic_bic(sse, len(y), 3)
            best = {"sse": sse, "aic": aic, "bic": bic,
                    "change_log2": float(c), "change_bits": float(2 ** c),
                    "intercept": a, "slope": b}
    return best


def second_difference(x: np.ndarray, y: np.ndarray):
    """返回斜率变化最大的位置（二阶差分）。"""
    order = np.argsort(x)
    x, y = x[order], y[order]
    slopes = np.diff(y) / np.diff(x)
    if len(slopes) < 2:
        return {"max_drop": 0.0, "at_bits": None, "at_quant": None}
    accel = np.diff(slopes)
    idx = int(np.argmin(accel))          # 斜率下降最多
    at_bits = float(x[idx + 1])
    nearest = min(QUANT_ORDER, key=lambda q: abs(BITS[q] - at_bits))
    return {"max_drop": float(-accel[idx]),
            "at_bits": at_bits, "at_quant": nearest}


def mcnemar(low: pd.Series, high: pd.Series):
    """配对 McNemar（连续性校正）。low/high 为同一批题的 bool 正确性。"""
    mask = low.notna() & high.notna()
    low = low[mask].astype(bool)
    high = high[mask].astype(bool)
    b = int(np.sum(low & ~high))   # 低档对、高档错
    c = int(np.sum(~low & high))   # 低档错、高档对
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(stat),
            "p": float(chi2.sf(stat, 1))}


def analyse_curve(sub: pd.DataFrame):
    """sub: 某基准下每个 quant 一行的 accuracy。"""
    x = np.log2(np.array([BITS[q] for q in sub["quant"]]))
    y = sub["accuracy"].to_numpy()
    linear = fit_linear(x, y)
    piecewise = fit_piecewise(x, y)
    delta_aic = piecewise["aic"] - linear["aic"]   # <0 支持分段
    preferred = "piecewise" if delta_aic < 0 else "linear"
    second = second_difference(x, y)
    return {
        "linear_aic": linear["aic"],
        "linear_bic": linear["bic"],
        "linear_slope": linear["slope"],
        "piecewise_aic": piecewise["aic"],
        "piecewise_bic": piecewise["bic"],
        "piecewise_slope": piecewise["slope"],
        "delta_aic": delta_aic,
        "delta_bic": piecewise["bic"] - linear["bic"],
        "preferred": preferred,
        "change_bits": piecewise["change_bits"],
        "change_quant": min(QUANT_ORDER,
                            key=lambda q: abs(BITS[q] - piecewise["change_bits"])),
        "max_slope_drop": second["max_drop"],
        "drop_at_quant": second["at_quant"],
    }


def main():
    ap = argparse.ArgumentParser(description="B2 形态分析")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--pattern", default="b2_*.csv")
    ap.add_argument("--out-prefix", default="results/shape_analysis_b2")
    args = ap.parse_args()

    df = analyze.load_results(Path(args.results_dir), pattern=args.pattern)
    # 只保留七个主档位
    df = df[df["quant"].isin(QUANT_ORDER)]

    # 每个基准的 accuracy 曲线
    curves = (df.groupby(["bench", "quant"])["correct"].mean()
                .reset_index().rename(columns={"correct": "accuracy"}))
    # 追加 ALL（跨基准合并）
    all_curve = (df.groupby("quant")["correct"].mean()
                   .reset_index().rename(columns={"correct": "accuracy"}))
    all_curve["bench"] = "ALL"
    # 稳健性检查：needle 全对会抬高平台，单独给一个剔除 needle 的版本
    no_needle = (df[df["bench"] != "needle"].groupby("quant")["correct"].mean()
                   .reset_index().rename(columns={"correct": "accuracy"}))
    no_needle["bench"] = "ALL_no_needle"
    curves = pd.concat([curves, all_curve, no_needle], ignore_index=True)

    rows = []
    for bench, sub in curves.groupby("bench"):
        sub = sub.set_index("quant").reindex(QUANT_ORDER).dropna().reset_index()
        if len(sub) < 3:
            continue
        row = analyse_curve(sub)
        row["bench"] = bench
        rows.append(row)
    shape = pd.DataFrame(rows)[
        ["bench", "preferred", "delta_aic", "delta_bic",
         "change_bits", "change_quant", "max_slope_drop", "drop_at_quant",
         "linear_aic", "linear_bic", "piecewise_aic", "piecewise_bic",
         "linear_slope", "piecewise_slope"]]

    # McNemar：逐基准 + ALL
    pivot = df.pivot_table(index=["bench", "item_id"], columns="quant",
                           values="correct", aggfunc="first")
    mrows = []
    for bench in list(df["bench"].unique()) + ["ALL", "ALL_no_needle"]:
        if bench == "ALL":
            sub = pivot
        elif bench == "ALL_no_needle":
            sub = pivot.drop(index="needle", level="bench")
        else:
            sub = pivot.xs(bench, level="bench")
        for low_q, high_q in PAIRS:
            if low_q not in sub.columns or high_q not in sub.columns:
                continue
            res = mcnemar(sub[low_q], sub[high_q])
            mrows.append({"bench": bench, "low": low_q, "high": high_q,
                          **res})
    mcnemar_df = pd.DataFrame(mrows)
    mcnemar_df["significant_0.05"] = mcnemar_df["p"] < 0.05
    # 基准内校正：每个基准做 6 次比较
    mcnemar_df["significant_bonferroni_bench"] = (
        mcnemar_df["p"] < (0.05 / len(PAIRS)))
    # 全局校正：所有基准 + ALL 家族合计的比较次数
    mcnemar_df["significant_bonferroni_global"] = (
        mcnemar_df["p"] < (0.05 / len(mcnemar_df)))

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    shape.to_csv(prefix.with_name(prefix.name + "_curves.csv"),
                 index=False, encoding="utf-8-sig")
    mcnemar_df.to_csv(prefix.with_name(prefix.name + "_mcnemar.csv"),
                      index=False, encoding="utf-8-sig")

    # Markdown
    lines = ["# B2 形态分析（RQ1 / RQ2）", ""]
    lines.append("## 1. 曲线形态：线性 vs 分段")
    lines.append("")
    lines.append("ΔAIC = 分段 AIC − 线性 AIC；**ΔAIC < 0 支持分段（平台/悬崖）**，"
                 "ΔAIC > 0 支持线性（均匀衰减）。")
    lines.append("")
    lines.append("| 基准 | 偏好 | ΔAIC | 变化点(bits) | 最近档位 | 最大斜率下降 | 下降位置 |")
    lines.append("|---|---|---:|---:|---|---:|---|")
    for _, r in shape.iterrows():
        lines.append(
            f"| {r['bench']} | {r['preferred']} | {r['delta_aic']:.2f} | "
            f"{r['change_bits']:.2f} | {r['change_quant']} | "
            f"{r['max_slope_drop']:.3f} | {r['drop_at_quant']} |")
    lines += ["", "## 2. 相邻档 McNemar 检验", "",
              "| 基准 | 低档 | 高档 | 低对高错 b | 低错高对 c | χ² | p | "
              "0.05 | 基准内校正 | 全局校正 |",
              "|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for _, r in mcnemar_df.iterrows():
        lines.append(
            f"| {r['bench']} | {r['low']} | {r['high']} | {int(r['b'])} | "
            f"{int(r['c'])} | {r['chi2']:.2f} | {r['p']:.2e} | "
            f"{'是' if r['significant_0.05'] else '否'} | "
            f"{'是' if r['significant_bonferroni_bench'] else '否'} | "
            f"{'是' if r['significant_bonferroni_global'] else '否'} |")
    prefix.with_name(prefix.name + ".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(f"已写入：{prefix.with_name(prefix.name + '_curves.csv')}")
    print(f"已写入：{prefix.with_name(prefix.name + '_mcnemar.csv')}")
    print(f"已写入：{prefix.with_name(prefix.name + '.md')}")
    print("\n形态偏好：")
    for _, r in shape.iterrows():
        print(f"  {r['bench']:15s} {r['preferred']:10s} "
              f"ΔAIC={r['delta_aic']:+.2f} 变化点≈{r['change_quant']}")


if __name__ == "__main__":
    main()
