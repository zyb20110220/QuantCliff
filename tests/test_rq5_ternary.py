"""rq5_ternary.py 的单元测试：Wilson 区间、直线拟合、McNemar。

这些函数决定 RQ5 的核心判读（三元落点是否显著偏离 PTQ 外推线），
算错会直接让结论反向，所以单独测。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import rq5_ternary as rq5  # noqa: E402


class TestWilson(unittest.TestCase):

    def test_half_success_is_centred_near_half(self):
        lo, hi = rq5.wilson(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        # 对称区间应大致以 0.5 为中心
        self.assertAlmostEqual((lo + hi) / 2, 0.5, places=2)

    def test_zero_success_lower_bound_is_zero(self):
        lo, hi = rq5.wilson(0, 30)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.15)

    def test_all_success_upper_bound_is_one(self):
        lo, hi = rq5.wilson(30, 30)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

    def test_more_samples_narrower_interval(self):
        narrow = rq5.wilson(500, 1000)
        wide = rq5.wilson(5, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_empty_sample_is_nan(self):
        lo, hi = rq5.wilson(0, 0)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))


class TestFitLine(unittest.TestCase):

    def test_recovers_known_coefficients(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.0 + 3.0 * x
        a, b = rq5.fit_line(x, y)
        self.assertAlmostEqual(a, 2.0, places=6)
        self.assertAlmostEqual(b, 3.0, places=6)

    def test_ternary_bit_width_is_below_q2k(self):
        # 三值量化 = log2(3) ≈ 1.585，应低于 Q2_K 的 2.5
        self.assertAlmostEqual(rq5.TERNARY_BITS, 1.5849625, places=5)
        self.assertLess(rq5.TERNARY_BITS, rq5.PTQ_BITS["Q2_K"])


class TestMcNemar(unittest.TestCase):

    def test_discordant_counts(self):
        a = pd.Series([True, True, True, False])
        b = pd.Series([False, False, True, False])
        res = rq5.mcnemar(a, b)
        # a 对 b 错：第 1、2 题 = 2；a 错 b 对：0
        self.assertEqual(res["b"], 2)
        self.assertEqual(res["c"], 0)

    def test_identical_is_not_significant(self):
        s = pd.Series([True, False, True])
        res = rq5.mcnemar(s, s)
        self.assertEqual(res["p"], 1.0)

    def test_pairing_drops_missing_items(self):
        a = pd.Series([True, True, np.nan])
        b = pd.Series([False, False, True])
        res = rq5.mcnemar(a, b)
        # 第 3 题有一侧缺失，应被排除，不参与计数
        self.assertEqual(res["b"] + res["c"], 2)


if __name__ == "__main__":
    unittest.main()
