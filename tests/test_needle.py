"""needle.judge 与题目构造的单元测试（不加载模型）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmarks import needle  # noqa: E402


class TestNeedle(unittest.TestCase):

    def setUp(self):
        self.items = needle.load(20, seed=42)

    def test_count_and_determinism(self):
        self.assertEqual(len(self.items), 20)
        again = needle.load(20, seed=42)
        self.assertEqual([x["passkey"] for x in self.items],
                         [x["passkey"] for x in again])

    def test_haystack_long_enough(self):
        for item in self.items[:3]:
            self.assertGreaterEqual(len(item["haystack"]), needle.TARGET_CHARS)

    def test_needle_inside_haystack(self):
        for item in self.items[:3]:
            self.assertIn(item["passkey"], item["haystack"])
            self.assertIn(str(item["serial"]), item["haystack"])

    def test_prompt_contains_serial(self):
        item = self.items[0]
        prompt = needle.build_prompt(item)
        self.assertIn(str(item["serial"]), prompt)
        self.assertIn("保险箱密码", prompt)

    def test_judge_exact(self):
        item = self.items[0]
        r = needle.judge(f"密码是 {item['passkey']}", item)
        self.assertTrue(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_judge_spaced_digits(self):
        item = {"passkey": "123456"}
        r = needle.judge("1 2 3 4 5 6", item)
        self.assertTrue(r["correct"])

    def test_judge_wrong_number(self):
        item = self.items[0]
        r = needle.judge("000000", item)
        self.assertFalse(r["correct"])
        self.assertTrue(r["parse_ok"])

    def test_judge_no_number(self):
        item = self.items[0]
        r = needle.judge("我不知道", item)
        self.assertFalse(r["correct"])
        self.assertFalse(r["parse_ok"])


if __name__ == "__main__":
    unittest.main()
