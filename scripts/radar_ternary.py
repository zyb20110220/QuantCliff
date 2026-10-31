"""窗口 C：五维能力雷达图（三元 vs PTQ）。

以 Qwen3-1.7B F16（未量化基线）为满圈基准，把每个基准的正确率换算成
"相对基线的保留比例"，画在同一张雷达图上，便于一眼看出哪种量化
路线在哪一维掉得最多。

之所以用"相对基线的比例"而不是原始正确率：
    六个基准难度不同（needle 100%、TruthfulQA 56%），
    直接画原始正确率会让高基线基准主导图形，低基线基准被压成一点。

用法：
    python scripts/radar_ternary.py --results-dir results --out docs/figures/radar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import analyze

# 雷达图的轴顺序（与报告表格一致）
BENCH_ORDER = ["mmlu_pro", "truthfulqa", "ifeval", "ceval", "needle",
               "chinese_facts"]
BENCH_LABEL = {
    "mmlu_pro": "学科知识\n(MMLU-Pro)",
    "truthfulqa": "事实真实性\n(TruthfulQA)",
    "ifeval": "指令跟随\n(IFEval)",
    "ceval": "中文知识\n(C-Eval)",
    "needle": "长上下文\n(needle 4K)",
    "chinese_facts": "中文常识\n(自建 30 题)",
}


def radial_upper_bound(curves) -> float:
    """算出雷达图的半径上限，保证所有数据点都落在极坐标边界之内。

    历史 bug：曾经写死 ax.set_ylim(0, 1.15)，而三元 IFEval = 1.325，
    超出的那一个角会被裁到坐标区外（图上表现为虚线/顶点"突出边界"）。

    规则：取所有曲线的最大值，乘 1.10 留白；最低不小于 1.0，
    这样即使全部低于基线也能完整画出基线参考圈。
    """
    values = []
    for series in curves.values():
        arr = np.asarray(series, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(float(arr.max()))
    data_max = max(values) if values else 1.0
    return max(1.0, data_max) * 1.10


def radial_ticks(top: float) -> list[float]:
    """选择径向刻度：只保留不会贴到外圈（<= 0.94×上限）的刻度。"""
    return [t for t in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
            if t <= top * 0.94]


def _setup_font() -> None:
    """让 matplotlib 能显示中文；找不到中文字体时退回英文标签。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "Arial Unicode MS"]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 没有中文字体时不要让图变成方框，直接用英文标签
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    ap = argparse.ArgumentParser(description="三元 vs PTQ 五维雷达图")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="docs/figures/radar_ternary.svg")
    ap.add_argument("--formats", default="svg",
                    help="导出格式，逗号分隔；默认矢量图 svg。"
                         "需要位图时用 --formats png，或 png,svg 两者都要")
    ap.add_argument("--ptq-pattern", default="b2_*.csv")
    ap.add_argument("--ternary-pattern", default="c2_*.csv")
    ap.add_argument("--baseline-quant", default="F16",
                    help="作为满圈基准的档位")
    args = ap.parse_args()

    df = analyze.load_results(Path(args.results_dir), pattern=args.ptq_pattern)
    df = df[df["base"].astype(str).str.contains("Qwen3", case=False, na=False)]

    # 基线：每个基准的原始正确率
    base = (df[df["quant"] == args.baseline_quant]
            .groupby("bench")["correct"].mean())
    if base.empty:
        raise SystemExit(f"没有找到基线档位 {args.baseline_quant} 的结果")

    def ratios(sub: pd.DataFrame) -> pd.Series:
        """把某档位的正确率换算成相对基线的比例。"""
        acc = sub.groupby("bench")["correct"].mean()
        out = {}
        for b in BENCH_ORDER:
            if b in acc.index and b in base.index and base[b] > 0:
                out[b] = acc[b] / base[b]
            else:
                out[b] = np.nan
        return pd.Series(out)

    curves = {}
    for q in ("Q2_K", "Q4_K_M", args.baseline_quant):
        sub = df[df["quant"] == q]
        if not sub.empty:
            curves[f"PTQ {q}"] = ratios(sub)

    # 三元结果（如果已经跑完）
    ter_paths = sorted(Path(args.results_dir).glob(args.ternary_pattern))
    if ter_paths:
        ter = analyze.load_results(Path(args.results_dir),
                                   pattern=args.ternary_pattern)
        if not ter.empty:
            curves["三元 1.58bit"] = ratios(ter)

    if not curves:
        raise SystemExit("没有任何可画的曲线")

    # ---- 画图 ----
    _setup_font()
    labels = [BENCH_LABEL.get(b, b) for b in BENCH_ORDER]
    n = len(BENCH_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8.2, 8.2),
                           subplot_kw={"projection": "polar"})
    colors = {"PTQ Q2_K": "#d62728", "PTQ Q4_K_M": "#ff9f43",
              f"PTQ {args.baseline_quant}": "#2c7fb8",
              "三元 1.58bit": "#2ca02c"}

    for name, series in curves.items():
        vals = [series[b] if np.isfinite(series[b]) else 0 for b in BENCH_ORDER]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=name,
                color=colors.get(name))
        ax.fill(angles, vals, alpha=0.10, color=colors.get(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", pad=14)
    # 极坐标默认会再补一圈 30° 的次刻度，导致 6 轴图上多出无关辐条。
    # 这里干脆把角度网格全部关掉，改成手动画 6 条，结果才完全可控。
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(False, axis="x", which="both")
    ax.grid(True, axis="y", which="major")
    # 径向刻度标签放到两个轴之间的空白扇区，避免压住数据线
    ax.set_rlabel_position(30)
    # 角度标签：把 0 度放在正上方，逆时针排列
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 半径上限必须覆盖所有数据点（详见 radial_upper_bound 的说明）
    top = radial_upper_bound(curves)
    ax.set_ylim(0, top)
    ticks = radial_ticks(top)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=8, color="grey")

    # 手动画 6 条辐条（已在上面关掉自动角度网格）
    for a in angles[:-1]:
        ax.plot([a, a], [0, top], linewidth=0.6, color="#b8b8b8", zorder=0)

    # 基线参考圈（=1.0）：解释"在圈外 = 比未量化基线好"。
    # 必须比网格圈深且明显虚线，否则会和 1.0 处的网格圈糊在一起看不出区别。
    # 用 361 个点画真正的圆，而不是只连 6 个轴的六边形。
    circle = np.linspace(0, 2 * np.pi, 361)
    ax.plot(circle, np.full(circle.shape, 1.0), linestyle=(0, (7, 4)),
            linewidth=1.8, color="#111111", alpha=1.0, zorder=3)

    ax.set_title(
        f"能力保留率雷达图（以 Qwen3-1.7B {args.baseline_quant} = 1.0 为基准）\n"
        "虚线圈 = 未量化基线；圈外 = 超过基线，圈内 = 低于基线",
        fontsize=12, pad=26)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    written = []
    for fmt in formats:
        target = out.with_suffix(f".{fmt}")
        fig.savefig(target, format=fmt, dpi=200, bbox_inches="tight")
        written.append(target)
    plt.close(fig)

    for target in written:
        print(f"已写入：{target}")
    print("\n各轴数值（相对基线比例）：")
    print(pd.DataFrame(curves).reindex(BENCH_ORDER).round(3).to_string())


if __name__ == "__main__":
    main()
