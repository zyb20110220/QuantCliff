"""窗口 C 任务 C5：RQ5 分析 —— 三元（1.58-bit）落点 vs PTQ 曲线延长线。

核心问题（RQ5 / 假设 H4）：
    训练式 1.58-bit 三元量化的能力落点，位于 PTQ 位宽曲线的延长线上，
    还是显著高于延长线——即"训练式量化能否绕过悬崖"？

做法：
    1. 用主实验 Qwen3-1.7B 的 7 个 PTQ 档位拟合"log2(位宽) → 正确率"直线；
    2. 把这条线外推到 1.58 bit，得到"如果三元只是普通 PTQ，应该有多少正确率"；
    3. 与实测的三元正确率比较，算偏差；
    4. 另外做三元 vs Qwen3-1.7B Q2_K 的配对 McNemar（同一批题）。

注意：
    - 三元模型与 Qwen3-1.7B 是同一底模的不同量化路线，题目完全一致，
      因此题目层面可以配对。
    - 外推只有 0.9 bit 的距离（2.5 → 1.58），属于轻度外推，但仍要在报告中说明。
    - 直线外推只作为参考基线，不作为"应该值"的强断言；外推区间越远越不可靠。

用法：
    python scripts/rq5_ternary.py --results-dir results --out-prefix results/rq5_ternary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze

# PTQ 档位的实际位宽（与 shape_analysis.py 一致）
PTQ_BITS = {
    "Q2_K": 2.5,
    "Q3_K_M": 3.5,
    "Q4_K_M": 4.5,
    "Q5_K_M": 5.5,
    "Q6_K": 6.5,
    "Q8_0": 8.0,
    "F16": 16.0,
}
PTQ_ORDER = list(PTQ_BITS)

# 三元模型的标称位宽：三值 {-1, 0, +1} → log2(3) ≈ 1.585 bit
TERNARY_BITS = np.log2(3.0)

# 主实验模型（三元模型的对照底模）
BASE_MODEL = "Qwen3-1.7B"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 置信区间（结果夹到 [0,1]，避免浮点误差给出微小负下界）。"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小二乘拟合 y = a + b x，返回 (a, b)。"""
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def equivalent_bits(acc: float, curve: pd.Series) -> float:
    """把一个准确率反查到 PTQ 曲线上，返回等效位宽（对数轴线性插值）。

    curve: index 为档位名、值为准确率，需按位宽升序。
    用途：把"三元实测 44.20%"翻译成"相当于 PTQ 的多少 bit"，比成堆的
    百分点更容易理解。超出曲线范围时返回对应端点位宽。
    """
    if curve.empty or not np.isfinite(acc):
        return float("nan")
    xs = np.log2(np.array([PTQ_BITS[q] for q in curve.index], dtype=float))
    ys = curve.to_numpy(dtype=float)
    if acc <= ys.min():
        return float(2 ** xs[int(np.argmin(ys))])
    if acc >= ys.max():
        return float(2 ** xs[int(np.argmax(ys))])
    for i in range(len(ys) - 1):
        lo, hi = ys[i], ys[i + 1]
        if hi == lo:
            continue
        if min(lo, hi) <= acc <= max(lo, hi):
            t = (acc - lo) / (hi - lo)
            return float(2 ** (xs[i] + t * (xs[i + 1] - xs[i])))
    return float("nan")


def mcnemar(low: pd.Series, high: pd.Series) -> dict:
    """配对 McNemar（连续性校正），口径与 shape_analysis.py 一致。"""
    mask = low.notna() & high.notna()
    low = low[mask].astype(bool)
    high = high[mask].astype(bool)
    b = int(np.sum(low & ~high))
    c = int(np.sum(~low & high))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(stat),
            "p": float(stats.chi2.sf(stat, 1))}


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ5 三元 vs PTQ 曲线分析")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-prefix", default="results/rq5_ternary")
    ap.add_argument("--ptq-pattern", default="b2_*.csv",
                    help="主实验 PTQ 结果的通配符")
    ap.add_argument("--ternary-pattern", default="c2_*.csv",
                    help="三元实验结果（不含 pilot）的通配符")
    args = ap.parse_args()

    results = Path(args.results_dir)
    ptq = analyze.load_results(results, pattern=args.ptq_pattern)
    ter = analyze.load_results(results, pattern=args.ternary_pattern)

    # ---- 只保留需要的模型与档位 ----
    ptq = ptq[ptq["base"].astype(str).str.contains("Qwen3", case=False, na=False)]
    ptq = ptq[ptq["quant"].isin(PTQ_ORDER)]
    ter = ter[ter["quant"] == "Q2_0"]
    if ptq.empty:
        raise SystemExit("没有找到 Qwen3-1.7B 的 PTQ 结果（先跑完 B2）")
    if ter.empty:
        raise SystemExit("没有找到三元结果（先跑完 C2）")

    benches = [b for b in ptq["bench"].unique() if b in set(ter["bench"])]
    # 按固定顺序排列，便于报告阅读；未列出的基准排在最后
    order = ["mmlu_pro", "truthfulqa", "ifeval", "ceval", "needle",
             "chinese_facts"]
    benches = sorted(benches, key=lambda b: order.index(b) if b in order else 99)

    # ---- 1. 逐基准：实测三元值 + PTQ 各档 ----
    ptq_acc = (ptq.groupby(["bench", "quant"])
                  .agg(n=("correct", "size"), acc=("correct", "mean"))
                  .reset_index())
    ter_acc = (ter.groupby("bench")
                  .agg(n_ter=("correct", "size"), acc_ter=("correct", "mean"),
                       k_ter=("correct", "sum"))
                  .reset_index())

    rows = []
    for bench in benches:
        sub = ptq_acc[ptq_acc["bench"] == bench].set_index("quant")
        sub = sub.reindex(PTQ_ORDER).dropna()
        if len(sub) < 3:
            continue
        t_row = ter_acc[ter_acc["bench"] == bench]
        if t_row.empty:
            continue
        x_all = np.log2(np.array([PTQ_BITS[q] for q in sub.index]))
        y_all = sub["acc"].to_numpy()
        a_all, b_all = fit_line(x_all, y_all)

        # 只用低比特三点（Q2_K/Q3_K_M/Q4_K_M）拟合：外推点离它们更近，更讲理
        low_idx = [q for q in sub.index if q in ("Q2_K", "Q3_K_M", "Q4_K_M")]
        a_low = b_low = np.nan
        if len(low_idx) >= 3:
            x_low = np.log2(np.array([PTQ_BITS[q] for q in low_idx]))
            y_low = sub.loc[low_idx, "acc"].to_numpy()
            a_low, b_low = fit_line(x_low, y_low)

        xt = np.log2(TERNARY_BITS)
        pred_all = a_all + b_all * xt
        pred_low = a_low + b_low * xt if np.isfinite(a_low) else np.nan

        acc_ter = float(t_row["acc_ter"].iloc[0])
        n_ter = int(t_row["n_ter"].iloc[0])
        lo, hi = wilson(int(t_row["k_ter"].iloc[0]), n_ter)
        eq_bits = equivalent_bits(acc_ter, sub["acc"])

        rows.append({
            "bench": bench,
            "n_ternary": n_ter,
            "acc_ternary": acc_ter,
            "ci_low": lo,
            "ci_high": hi,
            "eq_ptq_bits": eq_bits,
            "acc_ptq_q2k": float(sub.loc["Q2_K", "acc"]) if "Q2_K" in sub.index else np.nan,
            "acc_ptq_f16": float(sub.loc["F16", "acc"]) if "F16" in sub.index else np.nan,
            "pred_all_fit": pred_all,
            "pred_low_fit": pred_low,
            "dev_vs_all_fit": acc_ter - pred_all,
            "dev_vs_low_fit": acc_ter - pred_low if np.isfinite(pred_low) else np.nan,
        })
    per_bench = pd.DataFrame(rows)

    # ---- 2. 合并口径（跨基准）----
    pooled = pd.DataFrame([{
        "bench": "ALL",
        "n_ternary": int(len(ter)),
        "acc_ternary": float(ter["correct"].mean()),
        "ci_low": wilson(int(ter["correct"].sum()), len(ter))[0],
        "ci_high": wilson(int(ter["correct"].sum()), len(ter))[1],
        "acc_ptq_q2k": float(ptq[ptq["quant"] == "Q2_K"]["correct"].mean()),
        "acc_ptq_f16": float(ptq[ptq["quant"] == "F16"]["correct"].mean()),
    }])
    # 合并口径的预测值：把每道题的预测正确率加总平均（按各基准样本量加权）
    w = per_bench.set_index("bench")["n_ternary"]
    for col, src in (("pred_all_fit", "pred_all_fit"),
                     ("pred_low_fit", "pred_low_fit")):
        vals = per_bench.set_index("bench")[src]
        pooled[col] = float((vals * w).sum() / w.sum())
    pooled["dev_vs_all_fit"] = pooled["acc_ternary"] - pooled["pred_all_fit"]
    pooled["dev_vs_low_fit"] = pooled["acc_ternary"] - pooled["pred_low_fit"]

    # 等效位宽：用合并后的 PTQ 曲线反查
    ptq_curve = (ptq.groupby("quant")["correct"].mean()
                    .reindex(PTQ_ORDER).dropna())
    pooled["eq_ptq_bits"] = equivalent_bits(
        float(pooled["acc_ternary"].iloc[0]), ptq_curve)

    full = pd.concat([per_bench, pooled], ignore_index=True)

    # ---- 3. 三元 vs Qwen3 Q2_K 配对 McNemar ----
    pivot = ter.assign(base="ternary").pivot_table(
        index=["bench", "item_id"], values="correct", aggfunc="first")
    q2k = ptq[ptq["quant"] == "Q2_K"].pivot_table(
        index=["bench", "item_id"], values="correct", aggfunc="first")
    both = pivot.join(q2k, how="inner", lsuffix="_ter", rsuffix="_q2k")

    mrows = []
    bench_levels = set(both.index.get_level_values("bench"))
    for bench in benches + ["ALL"]:
        if bench == "ALL":
            sub = both
        elif bench in bench_levels:
            sub = both.loc[[bench]]
        else:
            continue
        if sub.empty:
            continue
        res = mcnemar(sub["correct_ter"], sub["correct_q2k"])
        mrows.append({"bench": bench, **res})
    mcn = pd.DataFrame(mrows)
    if not mcn.empty:
        mcn["significant_0.05"] = mcn["p"] < 0.05
        mcn["significant_bonferroni"] = mcn["p"] < (0.05 / len(mcn))

    # ---- 输出 ----
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(prefix.with_name(prefix.name + "_table.csv"),
                index=False, encoding="utf-8-sig")
    if not mcn.empty:
        mcn.to_csv(prefix.with_name(prefix.name + "_mcnemar.csv"),
                   index=False, encoding="utf-8-sig")

    def pct(v):
        return "—" if not np.isfinite(v) else f"{v * 100:.2f}%"

    def pp(v):
        return "—" if not np.isfinite(v) else f"{v * 100:+.2f}pp"

    lines = ["# RQ5：三元（1.58-bit）落点 vs PTQ 曲线延长线", ""]
    lines.append(f"> 三元模型 `Ternary-Bonsai-1.7B`（Q2_0 g64，训练式三值量化）"
                 f"标称位宽 log2(3) ≈ {TERNARY_BITS:.2f} bit；"
                 f"对照 PTQ 曲线来自 {BASE_MODEL}（同一底模）。")
    lines.append("> 外推基线用两种拟合：全部 7 点外推（`pred_all_fit`）"
                 "与只用低比特 3 点外推（`pred_low_fit`，更贴近外推区间）。")
    lines.append("")
    lines += ["| 基准 | 三元题数 | 三元实测 | 95% CI | 等效 PTQ 位宽 | PTQ Q2_K | PTQ F16 | "
              "全点外推 | 低点外推 | 偏离(低点) |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in full.iterrows():
        eb = r.get("eq_ptq_bits")
        eb_txt = "—" if eb is None or not np.isfinite(eb) else f"{eb:.2f} bit"
        lines.append(
            f"| {r['bench']} | {int(r['n_ternary'])} | {pct(r['acc_ternary'])} | "
            f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}] | {eb_txt} | "
            f"{pct(r['acc_ptq_q2k'])} | {pct(r['acc_ptq_f16'])} | "
            f"{pct(r['pred_all_fit'])} | {pct(r['pred_low_fit'])} | "
            f"{pp(r['dev_vs_low_fit'])} |")
    lines.append("")
    lines.append("> **等效 PTQ 位宽**：把三元实测准确率反查到 PTQ 曲线上得到的位宽。"
                 "例如合并口径 44.20% 落在 Q3_K_M 与 Q4_K_M 之间，"
                 "说明三元模型的表现相当于 PTQ 的约 3.9 bit。")
    lines.append("> 若某基准的三元值低于曲线最低锚点（Q2_K），该列会被夹紧到 "
                 "Q2_K 的位宽并显示为下限，**不代表精确等效**。")
    lines.append("> 另外：等效位宽是按准确率反查得到的产物，"
                 "不同能力维度的等效位宽差异很大，不宜只引用合计值。")

    lines += ["", "## 三元 vs Qwen3-1.7B Q2_K 配对 McNemar", "",
              "| 基准 | 三元对/Q2_K错 (b) | 三元错/Q2_K对 (c) | χ² | p | 0.05 | Bonferroni |",
              "|---|---:|---:|---:|---:|---|---|"]
    if mcn.empty:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        for _, r in mcn.iterrows():
            lines.append(f"| {r['bench']} | {int(r['b'])} | {int(r['c'])} | "
                         f"{r['chi2']:.2f} | {r['p']:.3g} | "
                         f"{'是' if r['significant_0.05'] else '否'} | "
                         f"{'是' if r['significant_bonferroni'] else '否'} |")

    lines += ["", "## 判读规则（预注册口径，先定后看数据）", "",
              "**主证据：三元 vs Qwen3-1.7B Q2_K 的配对 McNemar**（上表）。",
              "这一比较不需要任何外推假设——两者是同底模、同一批题，"
              "只是量化路线不同，是 RQ5 最干净的对照。", "",
              "**辅助证据：与 PTQ 曲线外推线的偏离**（表一最后一列）。"
              "外推只有 0.9 bit 距离，但仍是外推，不能单独作为结论依据。", "",
              "1. 若三元 **显著优于** Q2_K（McNemar 显著且 b>c 方向一致）→ "
              "支持 H4：训练式量化确实绕过了悬崖。",
              "2. 若三元与 Q2_K 无显著差异 → 三元只是\"另一种 Q2 水平\"，H4 不成立"
              "（同样是有效结论）。",
              "3. 若三元 **显著差于** Q2_K → 说明低比特训练式量化并没有带来预期收益，"
              "需要检查是否 Prism 运行时与主线的提示模板/停止条件不同。",
              "4. 任何结论都要同时报告每基准结果——前两部分已证明能力维度之间差异很大；"
              "只报合并值可能掩盖某一维度的崩塌。", "",
              "**关于低点外推的注意事项**：只用 Q2_K/Q3_K_M/Q4_K_M 三点做线性外推时，"
              "斜率被低比特陡降段主导，外推值会明显低于 Q2_K 本身。"
              "这只是\"若退化线性继续\"的假设值，不是三元模型\"应该\"达到的水平，"
              "报告中不得把它当作及格线。"]

    out_md = prefix.with_name(prefix.name + ".md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"已写入：{prefix.with_name(prefix.name + '_table.csv')}")
    if not mcn.empty:
        print(f"已写入：{prefix.with_name(prefix.name + '_mcnemar.csv')}")
    print(f"已写入：{out_md}")
    print()
    print("合并口径：")
    r = pooled.iloc[0]
    print(f"  三元实测      {pct(r['acc_ternary'])}  "
          f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}]")
    print(f"  PTQ Q2_K      {pct(r['acc_ptq_q2k'])}")
    print(f"  PTQ F16       {pct(r['acc_ptq_f16'])}")
    print(f"  全点外推预测  {pct(r['pred_all_fit'])}  → 偏离 {pp(r['dev_vs_all_fit'])}")
    print(f"  低点外推预测  {pct(r['pred_low_fit'])}  → 偏离 {pp(r['dev_vs_low_fit'])}")


if __name__ == "__main__":
    main()
