"""chinese_facts.judge 单元测试（纯离线，不加载模型）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmarks import chinese_facts  # noqa: E402


class TestChineseFacts(unittest.TestCase):

    def setUp(self):
        self.items = {item["id"]: item for item in chinese_facts.load(30)}

    def test_has_30_questions(self):
        self.assertEqual(len(self.items), 30)
        self.assertEqual(sorted(self.items), list(range(1, 31)))

    def test_exact_answer(self):
        r = chinese_facts.judge("北京", self.items[1])
        self.assertTrue(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_answer_in_sentence(self):
        r = chinese_facts.judge("答案是北京。", self.items[1])
        self.assertTrue(r["correct"])

    def test_wrong_answer(self):
        r = chinese_facts.judge("上海", self.items[1])
        self.assertFalse(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_unknown_is_parse_fail(self):
        r = chinese_facts.judge("我不知道。", self.items[1])
        self.assertFalse(r["correct"])
        self.assertFalse(r["parse_ok"])

    def test_last_position_wins_correct(self):
        r = chinese_facts.judge("不是上海，是北京", self.items[1])
        self.assertTrue(r["correct"])

    def test_last_position_wins_wrong(self):
        r = chinese_facts.judge("不是北京，是上海", self.items[1])
        self.assertFalse(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_number_boundary(self):
        # 第 26 题答案是 4；输出 14 不应被当成 4
        r = chinese_facts.judge("14", self.items[26])
        self.assertFalse(r["correct"])
        self.assertFalse(r["parse_ok"])
        r2 = chinese_facts.judge("4", self.items[26])
        self.assertTrue(r2["correct"])

    def test_h2o_not_match_h2o2(self):
        # 第 21 题答案是 h2o；h2o2 是错误答案
        r = chinese_facts.judge("H2O2", self.items[21])
        self.assertFalse(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_fullwidth_normalization(self):
        # 全角数字应被 NFKC 归一化
        r = chinese_facts.judge("１００", self.items[2])
        self.assertTrue(r["correct"])

    def test_oxygen_question_is_mass_based(self):
        # 2026-10-17 审核修订：明确"按质量计"，避免与按原子数计的氢歧义
        self.assertIn("按质量计", self.items[13]["question"])


if __name__ == "__main__":
    unittest.main()
