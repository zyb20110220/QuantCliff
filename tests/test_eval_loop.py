"""eval_loop 辅助逻辑的单元测试（不加载模型）。"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import eval_loop  # noqa: E402


class TestEvalLoopHelpers(unittest.TestCase):

    def test_answer_text_list(self):
        item = {"answer": ["北京", "Beijing"]}
        self.assertEqual(eval_loop._answer_text(item), "北京 | Beijing")

    def test_answer_text_scalar(self):
        self.assertEqual(eval_loop._answer_text({"answer": "B"}), "B")
        self.assertEqual(eval_loop._answer_text({}), "")

    def test_old_header_migration(self):
        old_header = ["model", "bench", "item_id", "output", "parse_ok",
                      "correct", "answer", "tokens", "seconds", "tag"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(old_header)
                w.writerow(["m", "b", "0", "A", "True", "True",
                            "A", "3", "1.0", "t"])
            rows = eval_loop._read_rows(path)
            self.assertIn("error", set(eval_loop.HEADER) - set(rows[0].keys()))

            eval_loop._write_rows(path, rows)
            migrated = eval_loop._read_rows(path)
            self.assertEqual(list(migrated[0].keys()), eval_loop.HEADER)
            self.assertEqual(migrated[0]["error"], "")

    def test_archive_keeps_old_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.csv"
            path.write_text("x", encoding="utf-8")
            dest = eval_loop._archive(path)
            self.assertFalse(path.exists())
            self.assertTrue(dest.exists())
            self.assertIn("_bak_", dest.name)


if __name__ == "__main__":
    unittest.main()
