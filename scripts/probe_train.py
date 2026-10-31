"""窗口 C 任务 C5：失效预警探针的训练、评估与校准（RQ6 第二步）。

核心问题（RQ6 / 假设 H5）
    模型"开口前"的隐藏层激活，能在多大程度上预测本轮答案是否正确？
    它是否显著优于"模型自身置信度"这个零成本基线？

方法
    - 特征：提示词末位 token 在若干层的隐藏状态（每层 1024 维）
    - 模型：L2 正则的逻辑回归（Pipeline 内标准化，避免泄漏）
    - 评估：**嵌套交叉验证**——外层 5 折算 AUROC，内层 5 折只用外层训练集选层。
      得到全样本的折外（out-of-fold）预测后再算 AUROC 与 DeLong。
      最初用的是单次 30% 切分，但在 MMLU-Pro 上出现训练集 CV-AUROC 0.65、
      测试集 0.48 的巨大反差——120 个测试样本的方差太大，改成嵌套 CV 更稳。
    - 区间：AUROC 用 bootstrap（1000 次）给 95% 置信区间，避免只看点估计
    - 基线：生成长度的平均对数概率（经典置信度），同样算 AUROC/ECE
    - 显著性：DeLong 检验比较两个 AUROC（同一样本上的相关比较）

用法
    python scripts/probe_train.py --results-dir results --out-prefix results/probe
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BENCH_ORDER = ["mmlu_pro", "truthfulqa", "ifeval", "ceval", "chinese_facts"]
N_BINS = 10

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# 可用于"题型难度基线"的分组字段。
# 为什么需要这个基线：IFEval 的约束类型就写在提示词里，而各类型的正确率
# 从 0% 到 100% 不等。如果只用"这题属于哪一类"就能拿到同样的 AUROC，
# 那探针就不是在读不确定性，只是在认题型。
GROUP_FIELD = {
    "mmlu_pro": "category",
    "truthfulqa": "category",
    "ceval": "subject",
    "ifeval": "type",
}


# ---------------------------------------------------------------- 统计工具

def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """期望校准误差：把预测概率分箱，算各箱 |准确率 − 平均置信度| 的加权和。"""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(prob)
    if ok.sum() == 0:
        return float("nan")
    y_true, prob = y_true[ok], prob[ok]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y_true[m].mean() - prob[m].mean())
    return float(total)


def _midrank(x: np.ndarray) -> np.ndarray:
    """处理并列的中位秩。"""
    order = np.argsort(x)
    z = x[order]
    n = len(z)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_test(y_true: np.ndarray, score_a: np.ndarray,
                score_b: np.ndarray) -> dict:
    """DeLong 检验：比较同一批样本上两个预测分数的 AUROC 是否有显著差异。

    实现参考 Sun & Xu (2014) 的快速 DeLong。要求分数越大表示越可能"正确"。
    """
    y_true = np.asarray(y_true, dtype=bool)
    pos = y_true.sum()
    m = int(pos)
    n = int(len(y_true) - pos)
    if m < 2 or n < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "z": float("nan"), "p": float("nan"),
                "note": "正负样本太少"}

    preds = np.vstack([np.asarray(score_a, dtype=float),
                       np.asarray(score_b, dtype=float)])
    # 按"正样本在前"排序
    order = np.argsort(-y_true.astype(int), kind="stable")
    preds_sorted = preds[:, order]

    tx = np.empty([2, m])
    ty = np.empty([2, n])
    tz = np.empty([2, m + n])
    for r in range(2):
        tx[r, :] = _midrank(preds_sorted[r, :m])
        ty[r, :] = _midrank(preds_sorted[r, m:])
        tz[r, :] = _midrank(preds_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    # 只有 2 个预测器，协方差退化成 2x2，直接手算避免 np.cov 的单样本警告
    s01 = np.array([[np.var(v01[0], ddof=1), np.cov(v01[0], v01[1], ddof=1)[0, 1]],
                    [np.cov(v01[0], v01[1], ddof=1)[0, 1], np.var(v01[1], ddof=1)]])
    s10 = np.array([[np.var(v10[0], ddof=1), np.cov(v10[0], v10[1], ddof=1)[0, 1]],
                    [np.cov(v10[0], v10[1], ddof=1)[0, 1], np.var(v10[1], ddof=1)]])
    cov = s01 / m + s10 / n

    diff = aucs[0] - aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        # 分数存在极端并列（例如几乎全是 0/1 的分数）时，放置值方差会归零。
        # 这时 DeLong 的正态近似失效，**不能**返回 p=1.0（那等于声称"无差异"），
        # 应返回 NaN 并在 note 里说明，让调用方知道这次检验不可用。
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
                "z": float("nan"), "p": float("nan"),
                "note": "方差为 0（分数极端并列），DeLong 不可用"}
    z = diff / np.sqrt(var_diff)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z))),
            "note": ""}


# ---------------------------------------------------------------- 数据加载

def oof_group_baseline(groups: np.ndarray, y: np.ndarray,
                       n_splits: int, seed: int) -> np.ndarray:
    """折外的"题型难度"基线：用该题所属分组在训练集里的平均正确率当分数。

    这个基线不接触任何模型内部信号，只用"这题是哪一类"。
    如果探针打不过它，说明探针的价值只是认题型。
    """
    n = len(y)
    oof = np.full(n, np.nan)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((n, 1))
    for tr, te in outer.split(dummy, y):
        means = pd.Series(y[tr]).groupby(groups[tr]).mean()
        fallback = float(y[tr].mean())
        oof[te] = [float(means.get(g, fallback)) for g in groups[te]]
    return oof


def load_groups(bench: str, n: int, seed: int,
                item_ids: np.ndarray) -> np.ndarray | None:
    """按同一 seed 重新加载题目，取分组键；顺序对不上就放弃（宁可不报）。"""
    field = GROUP_FIELD.get(bench)
    if field is None:
        return None
    try:
        b = importlib.import_module(f"benchmarks.{bench}")
        items = b.load(n, seed)
    except Exception:  # noqa: BLE001
        return None
    ids = [str(it.get("id")) for it in items]
    if len(ids) != len(item_ids) or ids != [str(x) for x in item_ids]:
        return None
    return np.array([str(it.get(field, "")) for it in items])


def load_bench(results: Path, bench: str):
    """读一个基准的 npz；不存在则返回 None。"""
    path = results / f"probe_{bench}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    layers = [int(x) for x in d["layers"]]
    out = {
        "bench": bench,
        "layers": layers,
        "correct": d["correct"].astype(bool),
        "mean_logprob": d["mean_logprob"].astype(float),
        "first_token_prob": d["first_token_prob"].astype(float),
        "item_id": d["item_id"].astype(str),
        "n": int(len(d["correct"])),
    }
    for L in layers:
        out[f"L{L}"] = d[f"hidden_L{L}"].astype(np.float64)
    return out


def _pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        # 1024 维 + 上千样本，L2 强正则更稳；C 不细调，避免过拟合测试集
        ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
    ])


def _pick_layer(train: dict, layers: list[int], seed: int) -> tuple[int, dict]:
    """只用训练集做 5 折 CV，挑 CV-AUROC 最高的层。"""
    y = train["correct"]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if min(n_pos, n_neg) < 5:
        return layers[-1], {L: float("nan") for L in layers}
    scores = {}
    for L in layers:
        s = cross_val_score(_pipeline(), train[f"L{L}"], y, cv=cv,
                            scoring="roc_auc")
        scores[L] = float(np.mean(s))
    return max(scores, key=scores.get), scores


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray,
                     n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """AUROC 的 95% bootstrap 置信区间。"""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n = len(y)
    if n < 10 or len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, score[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def nested_cv_oof(layer_data: dict, y: np.ndarray, layers: list[int],
                  n_splits: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """嵌套交叉验证：返回全样本的折外探针概率，以及每折选中的层。

    外层选层完全避开外层测试集，所以折外 AUROC 是无偏的。
    """
    y = np.asarray(y, dtype=bool)
    n = len(y)
    oof = np.full(n, np.nan)
    chosen = []
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((n, 1))
    for tr, te in outer.split(dummy, y):
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
        best_L, best_score = layers[-1], -np.inf
        for L in layers:
            s = cross_val_score(_pipeline(), layer_data[L][tr], y[tr],
                                cv=inner, scoring="roc_auc")
            m = float(np.mean(s))
            if m > best_score:
                best_L, best_score = L, m
        pipe = _pipeline()
        pipe.fit(layer_data[best_L][tr], y[tr])
        oof[te] = pipe.predict_proba(layer_data[best_L][te])[:, 1]
        chosen.append(best_L)
    return oof, chosen


def analyse_bench(data: dict, seed: int, n_splits: int,
                  groups: np.ndarray | None = None) -> dict:
    """单基准内：嵌套 CV 出折外预测 → 与两个基线做 AUROC/ECE/DeLong 对比。"""
    y = data["correct"]
    layers = data["layers"]
    if len(np.unique(y)) < 2:
        return {"bench": data["bench"], "n": data["n"],
                "status": f"标签全为 {bool(y[0])}，无法评估"}
    if min(int(y.sum()), int((~y).sum())) < n_splits:
        return {"bench": data["bench"], "n": data["n"],
                "status": "正/负样本过少，无法做分层 CV"}

    layer_data = {L: data[f"L{L}"] for L in layers}
    oof, chosen = nested_cv_oof(layer_data, y, layers, n_splits, seed)

    # 每层单独做 5 折 CV 的 AUROC，用于展示"层深与可预测性的关系"
    cv_by_layer = {}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for L in layers:
        s = cross_val_score(_pipeline(), layer_data[L], y, cv=cv,
                            scoring="roc_auc")
        cv_by_layer[L] = float(np.mean(s))

    conf_score = data["mean_logprob"].astype(float)
    conf_prob = np.exp(np.clip(conf_score, -50, 0))

    auc_probe = roc_auc_score(y, oof)
    auc_conf = (roc_auc_score(y, conf_score)
                if np.isfinite(conf_score).all() else float("nan"))
    ci_probe = bootstrap_auc_ci(y, oof, seed=seed)
    ci_conf = (bootstrap_auc_ci(y, conf_score, seed=seed)
               if np.isfinite(conf_score).all()
               else (float("nan"), float("nan")))
    dl = delong_test(y, oof, conf_score)

    # 题型难度基线：只用"这题属于哪一类"，不含任何模型信号
    auc_group = float("nan")
    ci_group = (float("nan"), float("nan"))
    dl_group = {"p": float("nan")}
    if groups is not None and len(groups) == len(y):
        oof_group = oof_group_baseline(groups, y, n_splits, seed)
        if np.isfinite(oof_group).all() and len(np.unique(oof_group)) > 1:
            auc_group = float(roc_auc_score(y, oof_group))
            ci_group = bootstrap_auc_ci(y, oof_group, seed=seed)
            dl_group = delong_test(y, oof, oof_group)

    # 探针选中的层的众数（每折可能不同）
    vals, counts = np.unique(chosen, return_counts=True)
    mode_layer = int(vals[int(np.argmax(counts))])

    return {
        "bench": data["bench"],
        "n": data["n"],
        "n_splits": n_splits,
        "acc": float(y.mean()),
        "layer_mode": mode_layer,
        "layer_choices": "、".join(f"L{L}×{c}" for L, c in
                                  zip(vals.tolist(), counts.tolist())),
        "cv_auc_by_layer": {str(k): round(v, 4) for k, v in cv_by_layer.items()},
        "auc_probe": float(auc_probe),
        "auc_probe_lo": ci_probe[0],
        "auc_probe_hi": ci_probe[1],
        "auc_conf": float(auc_conf),
        "auc_conf_lo": ci_conf[0],
        "auc_conf_hi": ci_conf[1],
        "delong_z": dl["z"],
        "delong_p": dl["p"],
        "auc_group": auc_group,
        "auc_group_lo": ci_group[0],
        "auc_group_hi": ci_group[1],
        "delong_group_p": dl_group["p"],
        "ece_probe": ece(y, oof),
        "ece_conf": ece(y, conf_prob),
        "status": "ok",
        "_oof": {"y": y, "probe": oof, "conf": conf_prob,
                 "conf_raw": conf_score},
    }


def cross_benchmark(all_data: dict, seed: int) -> pd.DataFrame:
    """跨基准泛化：在 mmlu_pro 上选层+训练，直接用到其他基准上测试。"""
    src = all_data.get("mmlu_pro")
    if src is None:
        return pd.DataFrame()
    y = src["correct"]
    if len(np.unique(y)) < 2:
        return pd.DataFrame()
    best_L, _ = _pick_layer({"correct": y,
                             **{f"L{L}": src[f"L{L}"] for L in src["layers"]}},
                            src["layers"], seed)
    pipe = _pipeline()
    pipe.fit(src[f"L{best_L}"], y)

    rows = []
    for bench, d in all_data.items():
        if bench == "mmlu_pro":
            continue
        yy = d["correct"]
        if len(np.unique(yy)) < 2:
            continue
        score = pipe.predict_proba(d[f"L{best_L}"])[:, 1]
        auc = roc_auc_score(yy, score)
        ac = (roc_auc_score(yy, d["mean_logprob"])
              if np.isfinite(d["mean_logprob"]).all() else float("nan"))
        rows.append({"train_bench": "mmlu_pro", "test_bench": bench,
                     "layer": best_L, "n": d["n"],
                     "test_acc": float(yy.mean()),
                     "auc_probe": float(auc), "auc_conf": float(ac)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="失效预警探针训练与评估")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-prefix", default="results/probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5,
                    help="嵌套 CV 的折数（外层评估、内层选层）")
    args = ap.parse_args()

    results = Path(args.results_dir)
    all_data = {}
    for bench in BENCH_ORDER:
        d = load_bench(results, bench)
        if d is not None:
            all_data[bench] = d
    if not all_data:
        raise SystemExit("没有找到任何 probe_*.npz，请先跑 probe_collect.py")

    print(f"读取 {len(all_data)} 个基准："
          + "、".join(f"{b}({d['n']})" for b, d in all_data.items()))

    rows = [analyse_bench(d, args.seed, args.n_splits,
                          load_groups(b, d["n"], args.seed, d["item_id"]))
            for b, d in all_data.items()]
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])

    cross = cross_benchmark(all_data, args.seed)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prefix.with_name(prefix.name + "_metrics.csv"),
              index=False, encoding="utf-8-sig")
    if not cross.empty:
        cross.to_csv(prefix.with_name(prefix.name + "_cross_bench.csv"),
                     index=False, encoding="utf-8-sig")

    # 保存折外预测，供画校准曲线 / ROC，以及 Gradio 演示取阈值
    pred_rows = []
    for r in ok_rows:
        t = r["_oof"]
        for i in range(len(t["y"])):
            pred_rows.append({
                "bench": r["bench"],
                "y_true": int(t["y"][i]),
                "probe_prob": float(t["probe"][i]),
                "conf_prob": float(t["conf"][i]),
            })
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(
            prefix.with_name(prefix.name + "_preds.csv"),
            index=False, encoding="utf-8-sig")

    # ---- Markdown 报告 ----
    def f3(v):
        return "—" if v is None or not np.isfinite(v) else f"{v:.3f}"

    def fp(v):
        return "不适用" if v is None or not np.isfinite(v) else f"{v:.3g}"

    lines = ["# RQ6：失效预警探针", ""]
    lines.append("> 特征 = 提示词末位 token 的隐藏层激活；标签 = 该题是否答对。"
                 f"评估用**嵌套交叉验证**（外层 {args.n_splits} 折出 AUROC，"
                 f"内层 {args.n_splits} 折只用外层训练集选层），"
                 "AUROC 区间为 1000 次 bootstrap 的 95% CI。")
    lines.append("")
    lines += ["## 1. 各基准内的探针表现", "",
              "| 基准 | 题数 | 正确率 | 选中层 | 探针 AUROC [95% CI] | "
              "置信度 AUROC | 题型难度 AUROC | p(探针 vs 置信度) | p(探针 vs 难度) |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['bench']} | {r['n']} | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['bench']} | {r['n']} | {r['acc'] * 100:.1f}% | "
            f"{r['layer_choices']} | "
            f"**{f3(r['auc_probe'])}** [{f3(r['auc_probe_lo'])}, {f3(r['auc_probe_hi'])}] | "
            f"{f3(r['auc_conf'])} | {f3(r['auc_group'])} | "
            f"{fp(r['delong_p'])} | {fp(r['delong_group_p'])} |")

    if ok_rows:
        pooled_auc = np.mean([r["auc_probe"] for r in ok_rows])
        pooled_conf = np.mean([r["auc_conf"] for r in ok_rows
                               if np.isfinite(r["auc_conf"])])
        n_sig = sum(1 for r in ok_rows
                    if np.isfinite(r["delong_p"]) and r["delong_p"] < 0.05)
        lines += ["", f"跨基准平均：探针 AUROC **{pooled_auc:.3f}**，"
                      f"置信度基线 **{pooled_conf:.3f}**；"
                      f"{len(ok_rows)} 个基准中 **{n_sig} 个**的 DeLong p<0.05。"]

    lines += ["", "## 2. 每层单独训练的效果（5 折 CV-AUROC，全样本）", ""]
    header = "| 基准 | " + " | ".join(f"L{L}" for L in
                                     (list(all_data.values())[0]["layers"])) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(list(all_data.values())[0]["layers"]))
    for r in rows:
        if r.get("status") != "ok":
            continue
        cells = [f3(r["cv_auc_by_layer"].get(str(L), float("nan")))
                 for L in all_data[r["bench"]]["layers"]]
        lines.append(f"| {r['bench']} | " + " | ".join(cells) + " |")

    if not cross.empty:
        lines += ["", "## 3. 跨基准泛化（在 mmlu_pro 上训练，直接用到其他基准）", "",
                  "| 训练 | 测试 | 层 | 题数 | 测试正确率 | 探针 AUROC | 置信度 AUROC |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for _, r in cross.iterrows():
            lines.append(
                f"| {r['train_bench']} | {r['test_bench']} | L{r['layer']} | "
                f"{int(r['n'])} | {r['test_acc'] * 100:.1f}% | "
                f"{f3(r['auc_probe'])} | {f3(r['auc_conf'])} |")

    lines += ["", "## 4. 题型难度基线的作用（重要）", "",
              "题目所属类别几乎写在了提示词里（IFEval 的约束类型、C-Eval 的学科、"
              "MMLU-Pro 的类目）。如果只用「这题属于哪一类」的历史正确率就能得到"
              "同样的 AUROC，那探针就不是在读不确定性，只是在认题型。", "",
              "因此表中额外列出 **题型难度 AUROC**（完全不用模型信号，只用类别均值）"
              "以及 **p(探针 vs 难度)**。只有当探针显著优于难度基线时，"
              "才能说它真的用上了隐藏层的实例级信息。"]

    lines += ["", "## 5. 判读口径（先定后看数据）", "",
              "- **H5 成立的判据**：探针 AUROC 显著高于**两个**基线"
              "（置信度与题型难度，DeLong p<0.05），且绝对水平有实用价值"
              "（经验上 AUROC ≥ 0.70 才算可用）。",
              "- 若探针只赢置信度、却输给题型难度基线 → 探针学到的其实是"
              "「这题难不难」，而不是「这次它知不知道」，H5 不成立。",
              "- 若探针与置信度**无显著差异**：说明隐藏层没有提供超出"
              "模型自身置信度的信息，这也是有价值的负结果——"
              "意味着零成本的置信度过滤就够了。",
              "- ECE 越小说明概率越可信；探针 ECE 高意味着"
              "「红绿灯」的阈值需要重新校准，不能直接用 0.5。",
              "- 跨基准泛化差是正常现象：探针学到的是特定题型下的"
              "「我知道/我不知道」信号，换题型后未必成立。"]

    out_md = prefix.with_name(prefix.name + "_report.md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n已写入：{prefix.with_name(prefix.name + '_metrics.csv')}")
    if not cross.empty:
        print(f"已写入：{prefix.with_name(prefix.name + '_cross_bench.csv')}")
    print(f"已写入：{out_md}")
    print()
    for r in ok_rows:
        print(f"  {r['bench']:15s} 层{r['layer_choices']:<14s} "
              f"探针AUROC={r['auc_probe']:.3f} [{r['auc_probe_lo']:.3f},{r['auc_probe_hi']:.3f}]  "
              f"置信度={r['auc_conf']:.3f}  "
              f"DeLong p={fp(r['delong_p'])}")


if __name__ == "__main__":
    main()
