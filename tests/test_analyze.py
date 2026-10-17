"""analyze.py 的单元测试：位宽解析、文本质量指标、端到端汇总。"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import analyze  # noqa: E402


class TestQuantParsing(unittest.TestCase):

    def test_quant_of(self):
        self.assertEqual(analyze.quant_of("Qwen3-1.7B-Q4_K_M"), "Q4_K_M")
        self.assertEqual(analyze.quant_of("Qwen3-1.7B-f16"), "F16")
        self.assertEqual(analyze.quant_of("Ternary-Bonsai-1.7B-Q2_0"), "Q2_0")
        self.assertEqual(analyze.quant_of("some-model"), "UNKNOWN")

    def test_base_of(self):
        self.assertEqual(analyze.base_of("Qwen3-1.7B-Q4_K_M"), "Qwen3-1.7B")
        self.assertEqual(analyze.base_of("Qwen3-1.7B-f16"), "Qwen3-1.7B")
        self.assertEqual(analyze.base_of("Ternary-Bonsai-1.7B-Q2_0"),
                         "Ternary-Bonsai-1.7B")


class TestTextQuality(unittest.TestCase):

    def test_clean_chinese(self):
        q = analyze.text_quality("太阳是一颗恒星，地球绕着它转。")
        self.assertEqual(q["garble_rate"], 0.0)
        self.assertLess(q["repeat_3gram_rate"], 0.2)

    def test_garbled_text(self):
        q = analyze.text_quality("□□□□□□")
        self.assertEqual(q["garble_rate"], 1.0)

    def test_repetition(self):
        q = analyze.text_quality("哈哈哈" * 20)
        self.assertGreater(q["repeat_3gram_rate"], 0.8)

    def test_empty(self):
        q = analyze.text_quality("")
        self.assertTrue(pd.isna(q["garble_rate"]))


class TestWilson(unittest.TestCase):

    def test_bounds(self):
        lo, hi = analyze.wilson_ci(22, 50)
        self.assertLess(lo, 0.44)
        self.assertGreater(hi, 0.44)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


class TestEndToEnd(unittest.TestCase):

    def _row(self, model, item_id, correct, output, bench="mmlu_pro"):
        return {
            "model": model, "bench": bench, "item_id": item_id,
            "output": output, "parse_ok": "True",
            "correct": str(correct), "answer": "A",
            "tokens": 3, "seconds": 1.0, "tag": "test", "error": "",
        }

    def test_summary(self):
        rows = [
            self._row("Qwen3-1.7B-F16", 0, True, "A"),
            self._row("Qwen3-1.7B-F16", 1, True, "A"),
            self._row("Qwen3-1.7B-Q4_K_M", 0, True, "A"),
            self._row("Qwen3-1.7B-Q4_K_M", 1, False, "B"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            df = analyze.load_results(Path(td))
            self.assertEqual(len(df), 4)
            self.assertTrue(set(df["base"]) == {"Qwen3-1.7B"})

            by_bench = analyze.summarize(df)
            overall = analyze.overall(df)

            f16 = overall[overall["quant"] == "F16"].iloc[0]
            q4 = overall[overall["quant"] == "Q4_K_M"].iloc[0]
            self.assertAlmostEqual(f16["accuracy"], 1.0)
            self.assertAlmostEqual(q4["accuracy"], 0.5)
            self.assertAlmostEqual(q4["rel_degradation"], 0.5)

            self.assertEqual(len(by_bench), 2)

    def test_dedupe_keeps_latest(self):
        rows = [
            self._row("Qwen3-1.7B-F16", 0, False, "B"),
            self._row("Qwen3-1.7B-F16", 0, True, "A"),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            df = analyze.load_results(Path(td))
            self.assertEqual(len(df), 1)
            self.assertTrue(bool(df.iloc[0]["correct"]))


if __name__ == "__main__":
    unittest.main()
