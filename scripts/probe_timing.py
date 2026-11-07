"""窗口 D 任务 D2：预警时机曲线分析（RQ7）。

输入
    results/probe_think_<bench>.npz（D1 采集）
    - think_L{层}_p{0,25,50,75,100}：思考过程 5 个观察点的末位激活
    - nothink_L{层}：非思考臂提示词末位激活
    - correct / nothink_correct：双臂各自生成答案的正确性
    - mean_logprob：思考臂答案段平均对数概率（置信度基线）

口径（与 RQ6 完全一致，直接复用 probe_train 的组件）
    - 每个位置一个逻辑回归探针（L2, C=0.1, 类权重平衡, 标准缩放）
    - 嵌套 CV：外层 5 折出折外分数算 AUROC，内层 5 折选层
    - 1000 次 bootstrap CI；DeLong 配对检验；题型难度折外基线
    - 思考长度混杂 → 相对/绝对双口径子集各报一遍（探索性，标注）

输出
    results/probe_timing_<bench>.png   时机曲线（探针线+CI / 置信度 / 难度 / 非思考）
    results/probe_timing.md            表格与结论文本
    results/probe_timing_<bench>.json  机器可读明细
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from probe_train import (_pipeline, bootstrap_auc_ci, delong_test,
                         load_groups, nested_cv_oof, oof_group_baseline)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
POS_KEYS = ["p0", "p25", "p50", "p75", "p100"]
LAYERS = [7, 14, 21, 28]

DEFAULT_N = {"truthfulqa": 250, "mmlu_pro": 250}


def load_think(bench: str, results_dir: Path = RESULTS) -> dict | None:
    path = results_dir / f"probe_think_{bench}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    layers = [int(x) for x in d["layers"]]
    out = {
        "bench": bench,
        "layers": layers,
        "correct": d["correct"].astype(bool),
        "nothink_correct": d["nothink_correct"].astype(bool),
        "mean_logprob": d["mean_logprob"].astype(float),
        "n_think": d["n_think"].astype(int),
        "n_answer": d["n_answer"].astype(int),
        "think_triggered": d["think_triggered"].astype(bool),
        "item_id": d["item_id"].astype(str),
        "n": int(len(d["correct"])),
    }
    for key in POS_KEYS:
        for L in layers:
            out[f"L{L}_{key}"] = d[f"think_L{L}_{key}"].astype(np.float64)
    for L in layers:
        out[f"L{L}_nothink"] = d[f"nothink_L{L}"].astype(np.float64)
    return out


def _layers_of(data: dict, key: str) -> dict:
    return {L: data[f"L{L}_{key}"] for L in data["layers"]}


def _auc(data: dict, key: str, seed: int) -> dict:
    """对某观察点跑嵌套 CV，返回 AUROC + CI + 每折选层 + 折外分数。"""
    y = data["correct"]
    oof, chosen = nested_cv_oof(_layers_of(data, key), y, data["layers"],
                                5, seed)
    auc = float(roc_auc_score(y, oof))
    lo, hi = bootstrap_auc_ci(y, oof, seed=seed)
    return {"oof": oof, "auc": auc, "ci": (lo, hi), "chosen": chosen,
            "key": key}


def mcnemar(y1: np.ndarray, y2: np.ndarray) -> tuple[float, float, int, int]:
    """配对二值标签的 McNemar 精确检验。返回 (p, 差值, b, c)。"""
    b = int(((y1 == 1) & (y2 == 0)).sum())
    c = int(((y1 == 0) & (y2 == 1)).sum())
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return p, b - c, b, c


def subset_rows(data: dict, mask: np.ndarray) -> None:
    """按掩码就地缩小子集（探索性分析用）。"""
    data = data.copy()
    for k in list(data):
        v = data[k]
        if isinstance(v, np.ndarray) and len(v) == data["n"]:
            data[k] = v[mask]
        elif isinstance(v, list) and len(v) == data["n"]:
            data[k] = [x for i, x in enumerate(v) if mask[i]]
    data["n"] = int(mask.sum())
    return data


def analyse_bench(data: dict, seed: int) -> dict:
    bench = data["bench"]
    y = data["correct"]
    y_nt = data["nothink_correct"]
    n = data["n"]

    # ---- 1. 五个位置的探针 ----
    positions = {}
    for key in POS_KEYS:
        r = _auc(data, key, seed)
        positions[key] = r

    # ---- 2. 非思考臂探针 ----
    oof_nt, chosen_nt = nested_cv_oof(_layers_of(data, "nothink"), y_nt,
                                      data["layers"], 5, seed)
    auc_nt = float(roc_auc_score(y_nt, oof_nt))
    ci_nt = bootstrap_auc_ci(y_nt, oof_nt, seed=seed)
    nothink = {"oof": oof_nt, "auc": auc_nt, "ci": ci_nt,
               "chosen": chosen_nt, "y": y_nt}

    # ---- 3. 置信度基线（同一批题的答案段平均对数概率）----
    conf = data["mean_logprob"]
    ok = np.isfinite(conf)
    auc_conf = float(roc_auc_score(y[ok], conf[ok])) if ok.sum() > 10 else float("nan")
    ci_conf = bootstrap_auc_ci(y, conf, seed=seed)

    # ---- 4. 题型难度基线（折外）----
    groups = load_groups(bench, n, seed, data["item_id"])
    auc_diff = float("nan")
    if groups is not None:
        oof_grp = oof_group_baseline(groups, y, 5, seed)
        auc_diff = float(roc_auc_score(y, oof_grp))

    # ---- 5. 统计检验 ----
    # 每个位置 vs 置信度（DeLong）；相邻位置两两（DeLong，配对口径）
    ok_conf = np.isfinite(conf)
    rows = []
    prev = None
    for key in POS_KEYS:
        r = positions[key]
        p_conf = delong_test(y[ok_conf], r["oof"][ok_conf],
                             conf[ok_conf])["p"]
        p_nt = delong_test(y, r["oof"], nothink["oof"])["p"] \
            if key == "p100" else float("nan")
        p_prev = delong_test(y, r["oof"], prev["oof"])["p"] \
            if prev else float("nan")
        rows.append({
            "位置": key, "观察点": f"{int(POSITIONS[POS_KEYS.index(key)] * 100)}%",
            "AUROC": round(r["auc"], 3),
            "95% CI": f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]",
            "选层(多数)": pd.Series(r["chosen"]).mode().iloc[0],
            "vs 置信度 p": f"{p_conf:.3f}" if np.isfinite(p_conf) else "—",
            "vs 前一位置 p": f"{p_prev:.3f}" if np.isfinite(p_prev) else "—",
            "vs 非思考 p": f"{p_nt:.3f}" if np.isfinite(p_nt) else "—",
        })
        prev = r

    # ---- 6. 正确率对比：thinking 是否真的有用 ----
    acc = float(y.mean())
    acc_nt = float(y_nt.mean())
    p_mc, d_mc, b, c = mcnemar(y, y_nt)
    trig = float(data["think_triggered"].mean())
    n_t = data["n_think"]

    # ---- 7. 双口径子集（探索性）----
    # 7a. 绝对思考长度三分位
    sub_abs = []
    qs = np.percentile(n_t[n_t > 0], [33, 67]) if (n_t > 0).sum() >= 10 \
        else [np.nan, np.nan]
    for name, mask in (
        ("短思考", (n_t <= qs[0])),
        ("中思考", (n_t > qs[0]) & (n_t <= qs[1])),
        ("长思考", (n_t > qs[1])),
    ):
        if mask.sum() < 30 or not np.isfinite(qs[0]):
            sub_abs.append({"组": name, "n": int(mask.sum()), "可用": False})
            continue
        sub = subset_rows(data, mask)
        r100 = _auc(sub, "p100", seed)
        sub_abs.append({"组": name, "n": int(mask.sum()), "可用": True,
                        "正确率": round(float(sub["correct"].mean()), 3),
                        "p100 AUROC": round(r100["auc"], 3)})

    # 7b. 思考占比 r = n_think/(n_think+n_answer) 三分位
    sub_rel = []
    tot = n_t + data["n_answer"]
    frac = n_t / np.maximum(tot, 1)
    qs2 = np.percentile(frac[(n_t > 0)], [33, 67]) \
        if (n_t > 0).sum() >= 10 else [np.nan, np.nan]
    for name, mask in (
        ("低思考占比", (frac <= qs2[0])),
        ("中思考占比", (frac > qs2[0]) & (frac <= qs2[1])),
        ("高思考占比", (frac > qs2[1])),
    ):
        if mask.sum() < 30 or not np.isfinite(qs2[0]):
            sub_rel.append({"组": name, "n": int(mask.sum()), "可用": False})
            continue
        sub = subset_rows(data, mask)
        r100 = _auc(sub, "p100", seed)
        sub_rel.append({"组": name, "n": int(mask.sum()), "可用": True,
                        "正确率": round(float(sub["correct"].mean()), 3),
                        "p100 AUROC": round(r100["auc"], 3)})

    summary = {
        "bench": bench, "n": n,
        "accuracy_think": acc, "accuracy_nothink": acc_nt,
        "mcnemar_p": p_mc, "delta": d_mc, "b": b, "c": c,
        "think_trigger_rate": trig,
        "n_think_mean": float(n_t.mean()),
        "n_think_median": float(np.median(n_t)),
        "conf_auc": auc_conf, "conf_ci": list(ci_conf),
        "diff_auc": auc_diff,
        "nothink": {"auc": auc_nt, "ci": list(ci_nt),
                    "chosen": chosen_nt},
        "positions": {k: {"auc": v["auc"], "ci": list(v["ci"]),
                          "chosen": v["chosen"]}
                      for k, v in positions.items()},
        "subsets_abs": sub_abs,
        "subsets_rel": sub_rel,
    }
    return summary, rows, positions, nothink


def save_models(data: dict, summary: dict, positions: dict,
                nothink: dict) -> Path:
    """在全量数据上重训每个观察位置的探针并落盘，供 Gradio 思考模式演示。

    层取嵌套 CV 的多数选层；阈值取该位置折外分数的 30/80 分位，
    与 RQ6 演示（probe_train._save_model）同口径。非思考臂一并保存。
    """
    out = {}
    for key in POS_KEYS:
        r = positions[key]
        layer = int(pd.Series(r["chosen"]).mode().iloc[0])
        pipe = _pipeline()
        pipe.fit(data[f"L{layer}_{key}"], data["correct"])
        q_lo, q_hi = np.percentile(r["oof"], [30.0, 80.0])
        out[key] = {"layer": layer, "pipe": pipe,
                    "thr_yellow": float(q_lo), "thr_green": float(q_hi),
                    "auc": r["auc"], "ci": list(r["ci"])}
    layer_nt = int(pd.Series(nothink["chosen"]).mode().iloc[0])
    pipe_nt = _pipeline()
    pipe_nt.fit(data[f"L{layer_nt}_nothink"], data["nothink_correct"])
    q_lo, q_hi = np.percentile(nothink["oof"], [30.0, 80.0])
    out["nothink"] = {"layer": layer_nt, "pipe": pipe_nt,
                       "thr_yellow": float(q_lo), "thr_green": float(q_hi),
                       "auc": nothink["auc"], "ci": list(nothink["ci"])}
    path = RESULTS / f"probe_timing_models_{data['bench']}.joblib"
    joblib.dump(out, path)
    return path


def plot_curve(summary: dict, rows: list, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    xs = [0, 25, 50, 75, 100]
    aucs = [summary["positions"][k]["auc"] for k in POS_KEYS]
    los = [summary["positions"][k]["ci"][0] for k in POS_KEYS]
    his = [summary["positions"][k]["ci"][1] for k in POS_KEYS]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(xs, los, his, alpha=0.15, color="tab:blue")
    ax.plot(xs, aucs, "o-", color="tab:blue", label="探针（思考观察点）")

    ca = summary["conf_auc"]
    if np.isfinite(ca):
        ax.axhline(ca, ls="--", color="tab:orange",
                   label=f"置信度基线 {ca:.3f}")
    da = summary["diff_auc"]
    if np.isfinite(da):
        ax.axhline(da, ls=":", color="tab:green",
                   label=f"题型难度基线 {da:.3f}")
    na = summary["nothink"]["auc"]
    if np.isfinite(na):
        ax.axhline(na, ls="-.", color="tab:red",
                   label=f"非思考探针 {na:.3f}")

    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set_xlabel("观察点在思考过程中的位置（% 思考 token）")
    ax.set_ylabel("AUROC（折外）")
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks(xs)
    ax.set_title(f"RQ7 预警时机曲线：{summary['bench']} "
                 f"（n={summary['n']}，think 正确率 "
                 f"{summary['accuracy_think']:.1%} vs 非思考 "
                 f"{summary['accuracy_nothink']:.1%}）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def render_md(summary: dict, rows: list) -> str:
    bench = summary["bench"]
    df = pd.DataFrame(rows).to_markdown(index=False)
    n_t = summary["n_think_mean"]
    lines = [
        f"# RQ7 预警时机曲线：{bench}（拓展实验）",
        "",
        f"> n={summary['n']} · 思考触发率 {summary['think_trigger_rate']:.1%}"
        f" · 平均思考 {n_t:.0f} token（中位 {summary['n_think_median']:.0f}）",
        "",
        "## 1. 各观察点的预警能力",
        "",
        df,
        "",
        f"- 置信度基线（答案段平均对数概率）：AUROC "
        f"{summary['conf_auc']:.3f} {summary['conf_ci']}",
        f"- 题型难度基线（折外）：AUROC {summary['diff_auc']:.3f}"
        if np.isfinite(summary["diff_auc"]) else "- 题型难度基线：无分组字段，未报",
        f"- 非思考臂探针（模板未插思考块）：AUROC "
        f"{summary['nothink']['auc']:.3f} {summary['nothink']['ci']}",
        "",
        "## 2. 思考有没有用（正确率）",
        "",
        f"- thinking 正确率 {summary['accuracy_think']:.3f}，"
        f"非思考正确率 {summary['accuracy_nothink']:.3f}",
        f"- McNemar p={summary['mcnemar_p']:.4f}，"
        f"差值 {summary['delta']:+.0f} 题（b={summary['b']}, c={summary['c']}）",
        "",
        "## 3. 探索性子集（p100 探针）",
        "",
    ]
    if any(s.get("可用") for s in summary["subsets_abs"]):
        lines.append("### 绝对思考长度（token 三分位）")
        for s in summary["subsets_abs"]:
            if s.get("可用"):
                lines.append(f"- {s['组']}（n={s['n']}）：正确率 "
                             f"{s['正确率']}，p100 AUROC {s['p100 AUROC']}")
            else:
                lines.append(f"- {s['组']}：样本不足，未报")
    else:
        lines.append("- 思考长度分布不足以分组，未报")
    if any(s.get("可用") for s in summary["subsets_rel"]):
        lines.append("### 思考占比（n_think/(n_think+n_answer) 三分位）")
        for s in summary["subsets_rel"]:
            if s.get("可用"):
                lines.append(f"- {s['组']}（n={s['n']}）：正确率 "
                             f"{s['正确率']}，p100 AUROC {s['p100 AUROC']}")
            else:
                lines.append(f"- {s['组']}：样本不足，未报")
    else:
        lines.append("- 思考占比分布不足以分组，未报")
    lines += ["", "## 4. 结论", "", "（待补充：按曲线走向二选一撰写）", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ7 时机曲线分析")
    ap.add_argument("--bench", default="truthfulqa",
                    help="基准名，逗号分隔")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true",
                    help="同时保存各观察位置的探针模型（供演示）")
    args = ap.parse_args()

    benches = [b.strip() for b in args.bench.split(",") if b.strip()]
    for bench in benches:
        data = load_think(bench)
        if data is None:
            print(f"跳过（无数据）：{bench}")
            continue
        summary, rows, positions, nothink = analyse_bench(data, args.seed)
        png = RESULTS / f"probe_timing_{bench}.png"
        plot_curve(summary, rows, png)
        md = render_md(summary, rows)
        (RESULTS / "probe_timing.md").write_text(md, encoding="utf-8")
        (RESULTS / f"probe_timing_{bench}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2,
                       default=str), encoding="utf-8")
        if args.save_models:
            mp = save_models(data, summary, positions, nothink)
            print(f"探针模型已保存：{mp}")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"图：{png}")


if __name__ == "__main__":
    main()
