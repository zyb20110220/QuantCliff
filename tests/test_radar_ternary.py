"""radar_ternary.py 的单元测试：半径上限与刻度选择。

守护的问题：雷达图曾经把半径上限写死成 1.15，而三元 IFEval 的保留率是 1.325，
导致那个顶点被裁到极坐标边界之外（图上表现为"有一角超出边界"）。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import radar_ternary as rt  # noqa: E402


def _series(**kw) -> pd.Series:
    base = {b: 1.0 for b in rt.BENCH_ORDER}
    base.update(kw)
    return pd.Series(base)


class TestRadialUpperBound(unittest.TestCase):

    def test_covers_value_above_one(self):
        """1.325 的顶点必须落在上限之内——这就是当初的 bug。"""
        curves = {"三元": _series(ifeval=1.325), "F16": _series()}
        top = rt.radial_upper_bound(curves)
        self.assertGreater(top, 1.325)
        self.assertGreaterEqual(top, 1.325 * 1.05)

    def test_fixed_ylim_115_would_have_clipped(self):
        """显式记录旧写法的失败条件，防止有人把上限改回常量。"""
        curves = {"三元": _series(ifeval=1.325)}
        self.assertGreater(1.325, 1.15)
        self.assertGreater(rt.radial_upper_bound(curves), 1.15)

    def test_never_below_one(self):
        """全部低于基线时，仍要能完整画出 1.0 参考圈。"""
        curves = {"Q2_K": _series(**{b: 0.5 for b in rt.BENCH_ORDER})}
        self.assertGreaterEqual(rt.radial_upper_bound(curves), 1.0)

    def test_ignores_nan(self):
        curves = {"x": pd.Series([0.5, np.nan, 0.9, np.nan])}
        # NaN 必须被忽略（否则最大值会是 NaN）；有效最大值 0.9 低于下限 1.0，
        # 所以结果是 1.0 × 1.10 = 1.1
        self.assertAlmostEqual(rt.radial_upper_bound(curves), 1.10, places=6)

    def test_empty_curves_falls_back_to_one(self):
        self.assertGreaterEqual(rt.radial_upper_bound({}), 1.0)


class TestRadialTicks(unittest.TestCase):

    def test_ticks_stay_inside_upper_bound(self):
        top = rt.radial_upper_bound({"x": _series(ifeval=1.325)})  # ≈1.4575
        ticks = rt.radial_ticks(top)
        self.assertTrue(all(t <= top for t in ticks))
        # 最外一圈不贴边：1.4 应被排除，避免标签被轴名遮住
        self.assertNotIn(1.4, ticks)
        self.assertIn(1.2, ticks)

    def test_zero_values_produce_no_ticks(self):
        self.assertEqual(rt.radial_ticks(0.1), [])


if __name__ == "__main__":
    unittest.main()
