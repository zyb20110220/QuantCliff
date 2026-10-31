"""app.py 的单元测试：红黄绿分档与页面渲染。

不加载真实模型——只测纯逻辑，避免测试变慢。真实推理链路由
`_smoke_app.py` 手工冒烟验证（需要 0.6B 权重）。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import app  # noqa: E402

META = {
    "layer": 21,
    "auroc_oof": 0.693,
    "auc_conf": 0.575,
    "auc_group": 0.587,
    "thr_green": 0.80,
    "thr_yellow": 0.07,
    "bands": {
        "green": {"coverage": 0.20, "n": 100, "accuracy": 0.62},
        "yellow": {"coverage": 0.50, "n": 250, "accuracy": 0.308},
        "red": {"coverage": 0.30, "n": 150, "accuracy": 0.187},
    },
}


def _dummy_interceptor():
    """只挂上 meta，不加载模型。"""
    obj = object.__new__(app.Interceptor)
    obj.meta = META
    return obj


class TestBandOf(unittest.TestCase):

    def setUp(self):
        self.inter = _dummy_interceptor()

    def test_green_at_threshold(self):
        self.assertEqual(self.inter.band_of(0.80), "green")

    def test_green_above(self):
        self.assertEqual(self.inter.band_of(0.95), "green")

    def test_yellow_between(self):
        self.assertEqual(self.inter.band_of(0.07), "yellow")
        self.assertEqual(self.inter.band_of(0.5), "yellow")

    def test_red_below(self):
        self.assertEqual(self.inter.band_of(0.069), "red")
        self.assertEqual(self.inter.band_of(0.0), "red")


class TestLightHtml(unittest.TestCase):

    def test_uses_band_colour(self):
        html = app._light_html("green", 0.9)
        self.assertIn(app.LIGHT_STYLE["green"][0], html)
        self.assertIn("可信", html)

    def test_red_band(self):
        html = app._light_html("red", 0.01)
        self.assertIn(app.LIGHT_STYLE["red"][0], html)
        self.assertIn("高风险", html)

    def test_shows_probe_score(self):
        html = app._light_html("yellow", 0.4321)
        self.assertIn("0.432", html)

    def test_three_bulbs_rendered(self):
        html = app._light_html("green", 0.9)
        self.assertEqual(html.count("border-radius:50%"), 3)


class TestDetailMarkdown(unittest.TestCase):

    def test_reports_band_accuracy_and_coverage(self):
        r = {"probe": 0.9, "conf": 0.5, "answer": "A", "correct": True,
             "parse_ok": True, "prompt_tokens": 120}
        md = app._detail_md(r, META, None)
        self.assertIn("62%", md)      # 绿灯档实际正确率
        self.assertIn("20%", md)      # 覆盖率
        self.assertIn("0.900", md)    # 探针分

    def test_red_band_uses_red_stats(self):
        r = {"probe": 0.01, "conf": 0.5, "answer": "B", "correct": False,
             "parse_ok": True, "prompt_tokens": 100}
        md = app._detail_md(r, META, None)
        self.assertIn("19%", md)

    def test_shows_truth_only_when_provided(self):
        r = {"probe": 0.9, "conf": 0.5, "answer": "A", "correct": True,
             "parse_ok": True, "prompt_tokens": 10}
        self.assertNotIn("标准答案", app._detail_md(r, META, None))
        self.assertIn("标准答案", app._detail_md(r, META, "A"))


class TestWarnMarkdown(unittest.TestCase):
    """第一步（只出预警）的文案：必须明确说明没有调用生成。"""

    def test_contains_probe_score_and_no_generation_note(self):
        md = app._warn_md({"probe": 0.123, "prompt_tokens": 88},
                          "red", META)
        self.assertIn("0.123", md)
        self.assertIn("88 tokens", md)
        self.assertIn("没有调用任何生成", md)

    def test_shows_band_accuracy(self):
        md = app._warn_md({"probe": 0.9, "prompt_tokens": 50}, "green", META)
        self.assertIn("62%", md)
        self.assertIn("20%", md)

    def test_does_not_leak_answer_text(self):
        # 第一步的文案里不应出现任何答案字段
        md = app._warn_md({"probe": 0.5, "prompt_tokens": 10}, "yellow", META)
        self.assertNotIn("模型答案", md)
        self.assertNotIn("标准答案", md)


class TestRenderItem(unittest.TestCase):

    ITEM = {"id": 0, "question": "Where are taxes much higher on income?",
            "options": ["Sweden", "Italy", "Japan", "Brazil"], "answer": "A"}

    def test_includes_full_question(self):
        md = app._render_item(self.ITEM)
        self.assertIn("Where are taxes much higher on income?", md)

    def test_lists_all_four_options_with_letters(self):
        md = app._render_item(self.ITEM)
        for letter, opt in zip("ABCD", self.ITEM["options"]):
            self.assertIn(f"**{letter}.** {opt}", md)

    def test_shows_item_id(self):
        self.assertIn("#0", app._render_item(self.ITEM))

    def test_empty_item_renders_nothing(self):
        self.assertEqual(app._render_item({}), "")
        self.assertEqual(app._render_item(None), "")

    def test_item_without_options_still_shows_question(self):
        md = app._render_item({"id": 7, "question": "What is 2+2?"})
        self.assertIn("What is 2+2?", md)
        self.assertIn("#7", md)


if __name__ == "__main__":
    unittest.main()
