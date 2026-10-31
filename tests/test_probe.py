"""probe_train.py 的单元测试：ECE 与 DeLong 检验。

DeLong 是手写实现，且直接决定"探针是否显著优于置信度"这一核心结论，
一旦算错会让 RQ6 的判断反向，所以必须用已知结论的构造样本来验证。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import probe_train as pt  # noqa: E402


class TestECE(unittest.TestCase):

    def test_perfectly_calibrated_is_zero(self):
        # 一半样本置信度 0.5 且其中一半正确 → 完全校准
        y = np.array([1, 0, 1, 0])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(pt.ece(y, p), 0.0, places=6)

    def test_overconfident_is_positive(self):
        # 全部置信 1.0，但只有一半正确 → ECE 应约为 0.5
        y = np.array([1, 0, 1, 0])
        p = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(pt.ece(y, p), 0.5, places=6)

    def test_underconfident_is_positive(self):
        y = np.array([1, 1, 1, 1])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(pt.ece(y, p), 0.5, places=6)

    def test_nan_is_ignored(self):
        y = np.array([1, 0, 1])
        p = np.array([0.5, 0.5, np.nan])
        val = pt.ece(y, p)
        self.assertTrue(np.isfinite(val))

    def test_all_nan_returns_nan(self):
        y = np.array([1, 0])
        p = np.array([np.nan, np.nan])
        self.assertTrue(np.isnan(pt.ece(y, p)))


class TestDeLong(unittest.TestCase):

    def _toy(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.random(n) < 0.5
        # 正样本分数整体偏高 = 有效预测器
        good = rng.normal(loc=np.where(y, 1.0, 0.0), scale=1.0)
        # 与标签无关 = 随机预测器
        noise = rng.normal(size=n)
        # 完美反向
        perfect = y.astype(float)
        return y, good, noise, perfect

    def test_auc_matches_sklearn(self):
        y, good, noise, _ = self._toy()
        res = pt.delong_test(y, good, noise)
        self.assertAlmostEqual(res["auc_a"], roc_auc_score(y, good), places=6)
        self.assertAlmostEqual(res["auc_b"], roc_auc_score(y, noise), places=6)

    def test_identical_scores_not_significant(self):
        y, good, _, _ = self._toy()
        res = pt.delong_test(y, good, good.copy())
        self.assertGreater(res["p"], 0.9)

    def test_clearly_better_detects_difference(self):
        y, good, noise, _ = self._toy()
        res = pt.delong_test(y, good, noise)
        self.assertGreater(res["auc_a"], res["auc_b"])
        self.assertLess(res["p"], 0.05)

    def test_perfect_vs_anti_perfect_variance_collapses(self):
        """完美/完全反向的 0-1 分数会让 DeLong 方差归零 —— 应返回 NaN 而非 p=1。

        这是一个真实的局限：分数极端并列时正态近似失效。返回 p=1.0 会
        错误地宣称"无差异"，所以约定返回 NaN 并附 note。
        """
        y, _, _, perfect = self._toy()
        res = pt.delong_test(y, perfect, 1.0 - perfect)
        self.assertAlmostEqual(res["auc_a"], 1.0, places=6)
        self.assertAlmostEqual(res["auc_b"], 0.0, places=6)
        self.assertTrue(np.isnan(res["p"]))
        self.assertIn("方差为 0", res["note"])

    def test_strong_vs_anti_correlated_continuous(self):
        """连续且无并列的强/反强预测量：差异应显著。"""
        rng = np.random.default_rng(7)
        n = 120
        y = rng.random(n) < 0.5
        centre = np.where(y, 1.5, -1.5)
        strong = centre + rng.normal(scale=1.0, size=n)
        anti = -strong + rng.normal(scale=0.1, size=n)
        res = pt.delong_test(y, strong, anti)
        self.assertGreater(res["auc_a"], 0.8)
        self.assertLess(res["auc_b"], 0.2)
        self.assertLess(res["p"], 0.01)

    def test_degenerate_labels_returns_nan(self):
        y = np.array([1, 1, 1, 1])
        s = np.arange(4, dtype=float)
        res = pt.delong_test(y, s, s)
        self.assertTrue(np.isnan(res["auc_a"]))
        self.assertTrue(np.isnan(res["p"]))

    def test_ties_are_handled(self):
        # 大量并列分数不应崩溃，且 AUC 应约等于 0.5（无区分度）
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        s = np.ones(8)
        res = pt.delong_test(y, s, s + 0.0)
        self.assertAlmostEqual(res["auc_a"], 0.5, places=6)


class TestLayerSelection(unittest.TestCase):

    def test_picks_layer_with_signal(self):
        """构造"只有 L28 有信号"的数据，应选中 L28。"""
        rng = np.random.default_rng(1)
        n = 200
        y = rng.random(n) < 0.5
        noise1 = rng.normal(size=(n, 16))
        noise2 = rng.normal(size=(n, 16))
        signal = rng.normal(size=(n, 16)) + np.where(y, 2.0, 0.0)[:, None]
        train = {"correct": y, "L7": noise1, "L28": signal,
                 "L14": noise2, "L21": noise2.copy()}
        best, scores = pt._pick_layer(train, [7, 14, 21, 28], seed=0)
        self.assertEqual(best, 28)


class TestGroupBaseline(unittest.TestCase):
    """题型难度基线：只用「这题属于哪一类」，不含任何模型信号。"""

    def test_perfect_separation_by_group(self):
        # A 类全对、B 类全错 → 只用类别均值就能完美预测
        groups = np.array(["A"] * 20 + ["B"] * 20)
        y = np.array([True] * 20 + [False] * 20)
        score = pt.oof_group_baseline(groups, y, n_splits=5, seed=0)
        self.assertEqual(len(score), 40)
        self.assertFalse(np.isnan(score).any())
        self.assertGreater(score[:20].mean(), score[20:].mean())

    def test_no_group_signal_is_uninformative(self):
        # 两个类别的正确率相同 → 该基线不应有明显区分度
        rng = np.random.default_rng(3)
        n = 200
        groups = np.array(["A", "B"] * (n // 2))
        y = rng.random(n) < 0.5
        score = pt.oof_group_baseline(groups, y, n_splits=5, seed=0)
        auc = roc_auc_score(y, score)
        self.assertLess(auc, 0.65)

    def test_unseen_group_falls_back_to_global_mean(self):
        # 某个组只在测试折出现，应回退到全局均值而不是崩溃
        rng = np.random.default_rng(4)
        n = 100
        groups = np.array([f"g{i}" for i in range(n)])  # 每组仅 1 个样本
        y = rng.random(n) < 0.5
        score = pt.oof_group_baseline(groups, y, n_splits=5, seed=0)
        self.assertEqual(len(score), n)
        self.assertFalse(np.isnan(score).any())


if __name__ == "__main__":
    unittest.main()
