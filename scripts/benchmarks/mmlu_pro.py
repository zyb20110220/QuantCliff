"""MMLU-Pro 基准（4 选 1 裁剪版）。

原始 MMLU-Pro 是 10 选 1（TIGER-Lab/MMLU-Pro，test 集 12,032 题），
但实际数据中每题选项数不固定（3~10 个不等，仅约 83% 的题是 10 选项）。
处理策略（在报告中如实说明）：
    1. 选择 MMLU-Pro 而非原版 MMLU：题目更新更难，降低"模型训练时见过题"的污染风险；
    2. 按每题实际选项数，随机保留正确答案 + 至多 3 个干扰项 → 4 选 1：
       保证小模型基线（F16）也有足够分辨率，否则 10 选 1 对 1.7B 模型接近随机，
       测不出"退化曲线"；seed 固定，三个模型、七个档位抽到完全相同的选项。
"""
import random
import re

from datasets import load_dataset

# 输出很短（一个字母），给少量解释空间即可
MAX_TOKENS = 32

_LETTERS = "ABCD"


def load(n: int = 150, seed: int = 42):
    # 走环境变量镜像（download.py 已设 HF_ENDPOINT；这里兜底再设一次）
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    ds = load_dataset("TIGER-Lab/MMLU-Pro", "default", split="test")
    all_items = list(ds)
    rng = random.Random(seed)
    rng.shuffle(all_items)

    out = []
    for idx, raw in enumerate(all_items[:n]):
        correct_i = raw["answer_index"]          # 0 ~ 选项数-1
        # 选项数不固定（3~10）：随机保留正确答案 + 至多 3 个干扰项
        n_opts = len(raw["options"])
        others = [i for i in range(n_opts) if i != correct_i]
        picked = rng.sample(others, min(3, n_opts - 1)) + [correct_i]
        rng.shuffle(picked)
        # 重建 4 个选项并记录正确项的新字母
        options = []
        answer_new = None
        for new_i, old_i in enumerate(picked):
            options.append(raw["options"][old_i])
            if old_i == correct_i:
                answer_new = _LETTERS[new_i]
        out.append({
            "id": idx,
            "question": raw["question"],
            "options": options,          # 4 个选项文本
            "answer": answer_new,        # 新字母
            "category": raw.get("category", "unknown"),
        })
    return out


def build_prompt(item) -> str:
    lines = [
        "请从 A-D 中选择唯一正确的答案，只输出字母（例如：A）。",
        f"问题：{item['question']}",
        "选项：",
    ]
    for letter, opt in zip(_LETTERS, item["options"]):
        lines.append(f"{letter}. {opt}")
    lines.append("答案：")
    return "\n".join(lines)


def judge(output: str, item) -> dict:
    """从模型输出中解析字母并判分。

    解析规则：优先找"答案"字样后的第一个 A-D；找不到则在整段中找第一个 A-D。
    找不到任何字母 → parse_ok=False（记入"解析失败"，本身是退化证据，不算答错）。
    """
    m = re.search(r"[Aa][Nn][Ss][Ww][Ee][Rr][:：]?\s*([A-Da-d])", output)
    if not m:
        m = re.search(r"\b([A-Da-d])\b", output)
    if not m:
        return {"correct": False, "parse_ok": False}
    letter = m.group(1).upper()
    return {"correct": (letter == item["answer"]), "parse_ok": True}
