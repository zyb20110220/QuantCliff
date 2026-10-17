"""ifeval.judge 单元测试：10 类约束逐类验证。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmarks import ifeval  # noqa: E402


class TestIFEval(unittest.TestCase):

    def setUp(self):
        self.items = {item["type"]: item for item in ifeval.load(100)}
        self.assertEqual(len(ifeval.load(100)), 100)

    def _judge(self, kind, output):
        return ifeval.judge(output, self.items[kind])

    def test_include(self):
        self.assertTrue(self._judge("include", "太阳的历史很久。")["correct"])
        self.assertFalse(self._judge("include", "太阳很亮。")["correct"])

    def test_exclude(self):
        self.assertTrue(self._judge("exclude", "太阳很亮。")["correct"])
        self.assertFalse(self._judge("exclude", "太阳很亮，但是很热。")["correct"])

    def test_prefix(self):
        self.assertTrue(self._judge("prefix", "首先，太阳很亮。")["correct"])
        self.assertFalse(self._judge("prefix", "太阳很亮。")["correct"])

    def test_suffix(self):
        self.assertTrue(self._judge("suffix", "太阳很亮。谢谢")["correct"])
        self.assertFalse(self._judge("suffix", "谢谢，太阳很亮。")["correct"])

    def test_bullets(self):
        ok = "- 太阳发光\n- 太阳发热\n- 太阳很大"
        self.assertTrue(self._judge("bullets", ok)["correct"])
        self.assertFalse(self._judge("bullets", "- 只有一条")["correct"])

    def test_sentences(self):
        self.assertTrue(self._judge("sentences", "太阳发光。太阳发热。")["correct"])
        self.assertFalse(self._judge("sentences", "太阳发光。")["correct"])

    def test_json(self):
        ok = '```json\n{"主题": "太阳", "介绍": "恒星"}\n```'
        self.assertTrue(self._judge("json", ok)["correct"])
        self.assertFalse(self._judge("json", "太阳是一颗恒星")["correct"])

    def test_no_comma(self):
        self.assertTrue(self._judge("no_comma", "太阳很亮很热")["correct"])
        self.assertFalse(self._judge("no_comma", "太阳很亮，很热")["correct"])

    def test_min_chars(self):
        self.assertTrue(self._judge("min_chars", "太" * 30)["correct"])
        self.assertFalse(self._judge("min_chars", "太" * 29)["correct"])

    def test_max_chars(self):
        self.assertTrue(self._judge("max_chars", "太" * 40)["correct"])
        self.assertFalse(self._judge("max_chars", "太" * 41)["correct"])

    def test_empty_is_parse_fail(self):
        r = self._judge("include", "   ")
        self.assertFalse(r["correct"])
        self.assertFalse(r["parse_ok"])


if __name__ == "__main__":
    unittest.main()
