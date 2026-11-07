"""probe_timing.py（RQ7 时机曲线）的单元测试。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import probe_timing as pt  # noqa: E402


def _synthetic(tmp_path: Path, n: int = 80, seed: int = 0):
    """构造一个带信号的合成 npz：p100 特征可分，p0 特征接近噪声。"""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, n // 3, replace=False)] = True
    d = {"layers": np.array(pt.LAYERS), "correct": y,
         "nothink_correct": y, "mean_logprob": y.astype(float) * 0.5 + 0.3,
         "n_think": rng.integers(50, 400, n).astype(np.int32),
         "n_answer": rng.integers(2, 20, n).astype(np.int32),
         "think_triggered": np.ones(n, dtype=bool),
         "item_id": np.array([str(i) for i in range(n)]),
         "bench": np.array(["syn"] * n)}
    for key in pt.POS_KEYS:
        signal = key == "p100"
        for L in pt.LAYERS:
            x = rng.normal(0, 1, (n, 1024))
            if signal:
                x += y[:, None] * 1.5
            d[f"think_L{L}_{key}"] = x.astype(np.float32)
    for L in pt.LAYERS:
        d[f"nothink_L{L}"] = rng.normal(0, 1, (n, 1024)).astype(np.float32)
    np.savez_compressed(tmp_path / "probe_think_syn.npz", **d)
    return d


def test_load_missing_returns_none(tmp_path):
    assert pt.load_think("nonexistent", tmp_path) is None


def test_load_roundtrip(tmp_path):
    _synthetic(tmp_path, n=12)
    data = pt.load_think("syn", tmp_path)
    assert data["n"] == 12
    assert data["layers"] == pt.LAYERS
    assert data["L21_p50"].shape == (12, 1024)
    assert data["L7_nothink"].shape == (12, 1024)
    assert data["correct"].dtype == bool


def test_mcnemar_exact():
    y1 = np.array([1, 1, 0, 0])
    y2 = np.array([1, 0, 1, 0])
    p, delta, b, c = pt.mcnemar(y1, y2)
    assert b == 1 and c == 1 and delta == 0 and p == 1.0

    y3 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y4 = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    p2, d2, b2, c2 = pt.mcnemar(y3, y4)
    assert b2 == 1 and c2 == 1 and d2 == 0 and p2 == 1.0


def test_subset_rows_keeps_consistency(tmp_path):
    _synthetic(tmp_path, n=20)
    data = pt.load_think("syn", tmp_path)
    mask = data["n_think"] > 0
    sub = pt.subset_rows(data, mask)
    assert sub["n"] == int(mask.sum())
    assert len(sub["correct"]) == sub["n"]
    assert sub["L14_p25"].shape[0] == sub["n"]


def test_analyse_bench_runs(tmp_path):
    _synthetic(tmp_path, n=80, seed=1)
    data = pt.load_think("syn", tmp_path)
    summary, rows, positions, nothink = pt.analyse_bench(data, seed=2)
    assert len(rows) == 5
    assert "p100" in positions and "p0" in positions
    # 合成数据里 p100 信号强：AUROC 应明显高于 0.5
    assert positions["p100"]["auc"] > 0.7
    assert positions["p0"]["auc"] < 0.9
    assert summary["bench"] == "syn"
    # 置信度基线与题型难度字段存在
    assert "conf_auc" in summary and "diff_auc" in summary


def test_render_md_contains_sections(tmp_path):
    _synthetic(tmp_path, n=80)
    data = pt.load_think("syn", tmp_path)
    summary, rows, _, _ = pt.analyse_bench(data, seed=3)
    md = pt.render_md(summary, rows)
    assert "预警时机曲线" in md
    assert "AUROC" in md
    assert "McNemar" in md


def test_plot_curve_saves(tmp_path):
    _synthetic(tmp_path, n=80)
    data = pt.load_think("syn", tmp_path)
    summary, rows, _, _ = pt.analyse_bench(data, seed=4)
    png = tmp_path / "curve.png"
    pt.plot_curve(summary, rows, png)
    assert png.exists() and png.stat().st_size > 1000


def test_save_models(tmp_path, monkeypatch):
    import joblib
    _synthetic(tmp_path, n=80)
    data = pt.load_think("syn", tmp_path)
    summary, rows, positions, nothink = pt.analyse_bench(data, seed=5)
    out = tmp_path / "probe_timing_models_syn.joblib"
    monkeypatch.setattr(pt, "RESULTS", tmp_path)
    saved = pt.save_models(data, summary, positions, nothink)
    assert saved == out
    bundle = joblib.load(out)
    assert set(bundle) == set(pt.POS_KEYS) | {"nothink"}
    for key in pt.POS_KEYS:
        m = bundle[key]
        assert 0.0 <= m["thr_yellow"] <= m["thr_green"] <= 1.0
        assert m["layer"] in pt.LAYERS
    p = bundle["p100"]["pipe"].predict_proba(
        data["L21_p100"][:5])
    assert p.shape == (5, 2)
