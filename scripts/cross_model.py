"""跨模型一致性分析（RQ3）。

回答的问题：
  1. 同一个量化档位在不同模型上损失多少？悬崖陡峭度是否一致？
  2. 哪些基准对量化最敏感？这个"敏感度排序"在不同模型之间是否一致？

输入：results/ 下的 b2_*.csv（Qwen3-1.7B）与 b3_*.csv（Qwen2.5-1.5B / Gemma-3-1B）
输出：<out-prefix>_overall.csv     每个模型 × 档位的总正确率与相对退化
      <out-prefix>_bench.csv       每个模型 × 基准的 Q2_K / Q3_K_M 相对退化
      <out-prefix>_ranking.csv     基准敏感度排名 + Kendall's W
      <out-prefix>.md              可读报告

用法：
  python scripts/cross_model.py --results-dir results --out-prefix results/cross_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze

# 展示顺序：位宽从低到高
QUANT_ORDER = [q for q in analyze.QUANT_ORDER if q != "Q2_0"]

# 编号短名，报告里更好读
SHORT_NAME = {
    "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "qwen2.5-1.5b-instruct": "Qwen2.5-1.5B",
    "gemma-3-1b-it": "Gemma-3-1B",
}

# 敏感性重点档位
FOCUS_QUANTS = ["Q2_K", "Q3_K_M", "Q4_K_M"]


def short_base(base: str) -> str:
    """把基座名映射成报告用短名。"""
    return SHORT_NAME.get(base, SHORT_NAME.get(base.lower(), base))


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W 一致性系数（含并列秩校正）。

    rank_matrix: 行 = 评审者(模型)，列 = 被评对象(基准)，值为秩。
    返回 0~1，1 表示完全一致。
    """
    m, n = rank_matrix.shape
    rank_sum = rank_matrix.sum(axis=0)
    s = float(np.sum((rank_sum - rank_sum.mean()) ** 2))

    # 并列校正项 T
    t_correction = 0.0
    for row in rank_matrix:
        _, counts = np.unique(row, return_counts=True)
        t_correction += float(np.sum(counts ** 3 - counts))

    denom = m ** 2 * (n ** 3 - n) - m * t_correction
    if denom <= 0:
        return float("nan")
    return 12.0 * s / denom


def main() -> None:
    ap = argparse.ArgumentParser(description="跨模型一致性分析（RQ3）")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-prefix", default="results/cross_model")
    args = ap.parse_args()

    # 同时读取 B2（主模型）与 B3（验证模型）
    frames = []
    for pattern in ("b2_*.csv", "b3_*.csv"):
        sub = analyze.load_results(Path(args.results_dir), pattern=pattern)
        sub["batch"] = pattern[:2]
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["quant"].isin(QUANT_ORDER)].copy()
    df["model_short"] = df["base"].map(short_base)

    # ---------- 1. 总体：模型 × 档位 ----------
    overall = (df.groupby(["model_short", "quant"])
                 .agg(n=("correct", "size"), acc=("correct", "mean"))
                 .reset_index())
    # 每个模型自己的 F16 基线（同基准集合，可直接比较）
    base_acc = (overall[overall["quant"] == "F16"]
                .set_index("model_short")["acc"].to_dict())
    overall["rel_degradation"] = overall.apply(
        lambda r: (base_acc[r["model_short"]] - r["acc"])
        / base_acc[r["model_short"]] if base_acc.get(r["model_short"]) else np.nan,
        axis=1)

    # 陡峭度指标：档位间准确率的极差与标准差（越大说明曲线越不平）
    flat = (overall.groupby("model_short")["acc"]
            .agg(["min", "max", "std"]).reset_index())
    flat["range_pp"] = (flat["max"] - flat["min"]) * 100
    flat["std_pp"] = flat["std"] * 100
    flat["q2k_rel"] = flat["model_short"].map(
        overall[overall["quant"] == "Q2_K"]
        .set_index("model_short")["rel_degradation"])

    # ---------- 2. 分基准：重点档位的相对退化 ----------
    bench = (df.groupby(["model_short", "bench", "quant"])["correct"].mean()
               .reset_index().rename(columns={"correct": "acc"}))
    bench_base = (bench[bench["quant"] == "F16"]
                  .set_index(["model_short", "bench"])["acc"].to_dict())
    bench["rel_degradation"] = bench.apply(
        lambda r: (bench_base.get((r["model_short"], r["bench"]), np.nan)
                   - r["acc"])
        / bench_base.get((r["model_short"], r["bench"]), np.nan)
        if bench_base.get((r["model_short"], r["bench"])) else np.nan,
        axis=1)
    bench_focus = bench[bench["quant"].isin(FOCUS_QUANTS)].copy()

    # ---------- 3. 敏感度排名一致性（Kendall's W） ----------
    # 用 Q2_K 的相对退化给基准排"越敏感越靠前"（秩 1 = 最敏感）
    q2 = bench[bench["quant"] == "Q2_K"].pivot(
        index="bench", columns="model_short", values="rel_degradation")
    q2 = q2.dropna()
    rank_block = []
    for col in q2.columns:
        # 退化越大 → 秩越小（越敏感）
        rank_block.append(stats.rankdata(-q2[col].to_numpy(), method="average"))
    rank_matrix = np.vstack(rank_block)
    w = kendall_w(rank_matrix)

    ranking = q2.copy()
    ranking.columns = [f"rel_deg_{c}" for c in ranking.columns]
    for col, ranks in zip(q2.columns, rank_matrix):
        ranking[f"rank_{col}"] = ranks.astype(int)
    ranking = ranking.reset_index().sort_values(
        ranking.columns[1], ascending=False)

    # 两两 Spearman：模型之间排序是否一致
    pairwise = []
    cols = list(q2.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if len(q2) < 3:
                continue
            rho, p = stats.spearmanr(q2[cols[i]], q2[cols[j]])
            pairwise.append({"model_a": cols[i], "model_b": cols[j],
                             "spearman_rho": rho, "p_value": p})
    pairwise_df = pd.DataFrame(pairwise)

    # ---------- 输出 ----------
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(prefix.with_name(prefix.name + "_overall.csv"),
                   index=False, encoding="utf-8-sig")
    flat.to_csv(prefix.with_name(prefix.name + "_flatness.csv"),
                index=False, encoding="utf-8-sig")
    bench_focus.to_csv(prefix.with_name(prefix.name + "_bench.csv"),
                       index=False, encoding="utf-8-sig")
    ranking.to_csv(prefix.with_name(prefix.name + "_ranking.csv"),
                   index=False, encoding="utf-8-sig")

    # Markdown 报告
    models = sorted(flat["model_short"].unique())
    lines = ["# 跨模型一致性分析（RQ3）", ""]
    lines.append("> 相对退化 = (本模型 F16 正确率 − 当前档位正确率) / F16 正确率。"
                 "三个模型使用同一套 6 个基准、每个档位 500 题。")
    lines.append("")

    lines += ["## 1. 总体准确率（%）", "",
              "| 档位 | " + " | ".join(models) + " |",
              "|---|" + "---:|" * len(models)]
    for q in QUANT_ORDER:
        cells = []
        for m in models:
            v = overall[(overall["model_short"] == m)
                        & (overall["quant"] == q)]["acc"]
            cells.append(f"{v.iloc[0] * 100:.2f}" if len(v) else "—")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")

    lines += ["", "## 2. 相对退化率（%）", "",
              "| 档位 | " + " | ".join(models) + " |",
              "|---|" + "---:|" * len(models)]
    for q in QUANT_ORDER:
        cells = []
        for m in models:
            v = overall[(overall["model_short"] == m)
                        & (overall["quant"] == q)]["rel_degradation"]
            cells.append(f"{v.iloc[0] * 100:+.2f}" if len(v) else "—")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")

    lines += ["", "## 3. 悬崖陡峭度", "",
              "| 模型 | 最低档准确率 | 最高档准确率 | 极差(pp) | 标准差(pp) | Q2_K 相对退化 |",
              "|---|---:|---:|---:|---:|---:|"]
    for _, r in flat.iterrows():
        lines.append(
            f"| {r['model_short']} | {r['min'] * 100:.2f}% | "
            f"{r['max'] * 100:.2f}% | {r['range_pp']:.2f} | "
            f"{r['std_pp']:.2f} | "
            + (f"{r['q2k_rel'] * 100:+.2f}%" if np.isfinite(r["q2k_rel"])
               else "—") + " |")

    lines += ["", "## 4. 各基准在 Q2_K 的相对退化（%）", "",
              "| 基准 | " + " | ".join(q2.columns) + " |",
              "|---|" + "---:|" * len(q2.columns)]
    for bench_name, r in q2.sort_values(q2.columns[0], ascending=False).iterrows():
        lines.append(f"| {bench_name} | "
                     + " | ".join(f"{r[c] * 100:+.2f}" for c in q2.columns)
                     + " |")

    lines += ["", "## 5. 敏感度排序一致性", "",
              f"Kendall's W = **{w:.3f}**"
              f"（0 = 完全不一致，1 = 完全一致；n={len(q2)} 个基准，"
              f"m={len(q2.columns)} 个模型）", ""]
    if not pairwise_df.empty:
        lines += ["| 模型 A | 模型 B | Spearman ρ | p |",
                  "|---|---|---:|---:|"]
        for _, r in pairwise_df.iterrows():
            lines.append(f"| {r['model_a']} | {r['model_b']} | "
                         f"{r['spearman_rho']:+.3f} | {r['p_value']:.3f} |")
    lines += ["", "> 排名规则：Q2_K 相对退化越大 → 秩越小（越敏感）。"]

    prefix.with_name(prefix.name + ".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(f"已写入：{prefix.with_name(prefix.name + '_overall.csv')}")
    print(f"已写入：{prefix.with_name(prefix.name + '_flatness.csv')}")
    print(f"已写入：{prefix.with_name(prefix.name + '_bench.csv')}")
    print(f"已写入：{prefix.with_name(prefix.name + '_ranking.csv')}")
    print(f"已写入：{prefix.with_name(prefix.name + '.md')}")
    print()
    print("悬崖陡峭度：")
    for _, r in flat.iterrows():
        print(f"  {r['model_short']:16s} 极差={r['range_pp']:5.2f}pp  "
              f"Q2_K相对退化={r['q2k_rel'] * 100:+.2f}%")
    print(f"\nKendall's W = {w:.3f}")


if __name__ == "__main__":
    main()
