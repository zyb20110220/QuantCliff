"""shape_analysis.py 的单元测试：McNemar 配对索引与形态拟合。

这里重点守护一个曾经踩过的坑：多模型合并分析时，如果 McNemar 的透视表索引
不带模型维度，同一 item_id 会在多个模型下重复出现，`aggfunc="first"` 会静默
只保留第一个模型的数据，把"三模型合并检验"变成"单模型检验"，而且不会报错。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import shape_analysis as sa  # noqa: E402


def _toy_frame(model: str, deltas: dict) -> pd.DataFrame:
    """构造一个假的结果表：每个档位 10 道题，按 deltas 指定答对几道。"""
    rows = []
    for quant, n_correct in deltas.items():
        for i in range(10):
            rows.append({
                "model": f"{model}-{quant}",
                "base": model,
                "bench": "toy",
                "item_id": i,
                "quant": quant,
                "correct": i < n_correct,
            })
    return pd.DataFrame(rows)


class TestMcNemar(unittest.TestCase):

    def test_counts_and_pvalue(self):
        low = pd.Series([True, True, False, False, True])
        high = pd.Series([False, False, True, True, False])
        # b = 低对高错 = 第 1、2、5 题 = 3
        # c = 低错高对 = 第 3、4 题   = 2
        res = sa.mcnemar(low, high)
        self.assertEqual(res["b"], 3)
        self.assertEqual(res["c"], 2)
        # |3-2|-1 = 0 → 统计量为 0 → p = 1，不应声称显著
        self.assertEqual(res["p"], 1.0)

    def test_identical_series_is_not_significant(self):
        s = pd.Series([True, False, True, True, False])
        res = sa.mcnemar(s, s)
        self.assertEqual(res["b"], 0)
        self.assertEqual(res["c"], 0)
        self.assertEqual(res["p"], 1.0)


class TestMultiModelPivot(unittest.TestCase):
    """核心回归测试：合并两个模型后，b/c 计数必须是两者之和。"""

    def test_pooled_counts_do_not_collapse_to_first_model(self):
        # 模型 A：低档全对、高档全错 → 贡献 b=10, c=0
        a = _toy_frame("model-a", {"Q2_K": 10, "Q3_K_M": 0})
        # 模型 B：低档全错、高档全对 → 贡献 b=0, c=10
        b = _toy_frame("model-b", {"Q2_K": 0, "Q3_K_M": 10})
        df = pd.concat([a, b], ignore_index=True)

        # 修复后的透视表：索引带 base，两个模型各自成行
        pivot = df.pivot_table(index=["base", "bench", "item_id"],
                               columns="quant", values="correct",
                               aggfunc="first")
        res = sa.mcnemar(pivot["Q2_K"], pivot["Q3_K_M"])
        self.assertEqual(res["b"], 10, "应合并两个模型的计数，而不是只看第一个")
        self.assertEqual(res["c"], 10)
        # b 与 c 相互抵消 → 不应显著
        self.assertGreater(res["p"], 0.05)

    def test_old_buggy_pivot_would_have_hidden_second_model(self):
        """显式验证旧写法确实会丢数据，作为文档说明。"""
        a = _toy_frame("model-a", {"Q2_K": 10, "Q3_K_M": 0})
        b = _toy_frame("model-b", {"Q2_K": 0, "Q3_K_M": 10})
        df = pd.concat([a, b], ignore_index=True)

        buggy = df.pivot_table(index=["bench", "item_id"], columns="quant",
                               values="correct", aggfunc="first")
        res = sa.mcnemar(buggy["Q2_K"], buggy["Q3_K_M"])
        # 旧写法只剩 model-a 的 10 条记录，b=10, c=0，会错误地报出极显著
        self.assertEqual(int(res["b"]) + int(res["c"]), 10)
        self.assertLess(res["p"], 0.05)


class TestCurveFitting(unittest.TestCase):

    def test_linear_data_prefers_linear(self):
        # 严格线性：分段模型没有优势，AIC 应判定为 linear
        x = np.log2(np.array([sa.BITS[q] for q in sa.QUANT_ORDER]))
        y = 0.5 + 0.02 * x
        linear = sa.fit_linear(x, y)
        piece = sa.fit_piecewise(x, y)
        self.assertGreater(piece["aic"], linear["aic"])

    def test_flat_then_cliff_prefers_piecewise(self):
        # 先平后陡：分段模型应明显更优
        x = np.log2(np.array([sa.BITS[q] for q in sa.QUANT_ORDER]))
        y = np.where(x < 2.2, 0.30, 0.50)
        linear = sa.fit_linear(x, y)
        piece = sa.fit_piecewise(x, y)
        self.assertLess(piece["aic"], linear["aic"])


if __name__ == "__main__":
    unittest.main()
