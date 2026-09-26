"""mmlu_pro.judge 判分逻辑的单元测试（不加载模型，纯离线）。

运行：python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

# 使仓库根目录下运行时也能 import benchmarks 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmarks.mmlu_pro import judge  # noqa: E402


class TestJudge(unittest.TestCase):

    def setUp(self):
        self.item = {"answer": "B"}

    def test_pure_letter(self):
        r = judge("B", self.item)
        self.assertTrue(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_chinese_answer_prefix(self):
        r = judge("答案：B", self.item)
        self.assertTrue(r["correct"])

    def test_explanation_with_letter(self):
        r = judge("我认为正确答案是 B，因为其他选项都不符合。", self.item)
        self.assertTrue(r["correct"])

    def test_wrong_letter(self):
        r = judge("A", self.item)
        self.assertFalse(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_lowercase(self):
        r = judge("b", self.item)
        self.assertTrue(r["correct"])

    def test_no_letter_parse_fail(self):
        r = judge("我不知道这道题的答案。", self.item)
        self.assertFalse(r["parse_ok"])

    def test_letter_after_answer_word_only(self):
        # "答案" 后没有字母、但整段有 C → 应回退整段搜索得 C
        r = judge("答案：无法确定。C 可能正确", self.item)
        self.assertTrue(r["parse_ok"])
        self.assertFalse(r["correct"])  # 得 C 而非 B


if __name__ == "__main__":
    unittest.main()
