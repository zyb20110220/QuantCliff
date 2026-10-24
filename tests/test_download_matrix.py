"""下载矩阵的离线校验（不联网、不下载）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import download_matrix  # noqa: E402

from analyze import quant_of  # noqa: E402


class TestDownloadMatrix(unittest.TestCase):

    def test_three_main_models_have_seven_levels(self):
        expected = {"Q2_K", "Q3_K_M", "Q4_K_M",
                    "Q5_K_M", "Q6_K", "Q8_0", "F16"}
        for model in ["qwen3-1.7b", "qwen2.5-1.5b", "gemma-3-1b"]:
            self.assertEqual(set(download_matrix.MATRIX[model]["files"]),
                             expected, model)

    def test_ternary_has_q2_0(self):
        files = download_matrix.MATRIX["ternary-bonsai-1.7b"]["files"]
        self.assertEqual(list(files), ["Q2_0"])
        # 最新 Prism 运行时要求 group-64，不能退回旧的 g128 文件
        self.assertIn("g64", files["Q2_0"])

    def test_all_filenames_are_gguf(self):
        for entry in download_matrix.MATRIX.values():
            for filename in entry["files"].values():
                self.assertTrue(filename.endswith(".gguf"), filename)

    def test_bf16_maps_to_f16_baseline(self):
        self.assertEqual(quant_of("Qwen3-1.7B-BF16"), "F16")
        self.assertEqual(quant_of("gemma-3-1b-it-BF16"), "F16")
        self.assertEqual(quant_of("qwen2.5-1.5b-instruct-fp16"), "F16")


if __name__ == "__main__":
    unittest.main()
