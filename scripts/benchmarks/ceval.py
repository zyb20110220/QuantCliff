"""C-Eval 基准（中文 4 选 1，跨 4 个学科抽样）。

数据源：`ceval/ceval-exam` 的 test 划分。原始数据每个学科一个配置，
这里取 4 个有代表性的学科（高中语文、高中数学、高中物理、计算机网络），
按 seed 打乱后轮流抽样，凑满 n 题（默认 100）。

选项顺序按 seed 打乱，七个量化档位抽到完全相同的题和选项。
"""
import random
import re

from datasets import load_dataset

MAX_TOKENS = 32
_LETTERS = "ABCD"

SUBJECTS = [
    "high_school_chinese",
    "high_school_mathematics",
    "high_school_physics",
    "computer_network",
]

_CACHE = {}


def _get_subject(subject: str):
    if subject not in _CACHE:
        _CACHE[subject] = load_dataset(
            "ceval/ceval-exam", subject, split="test")
    return _CACHE[subject]


def _to_item(raw, item_id: int, rng: random.Random, subject: str = ""):
    """把一条 C-Eval 原始数据转成统一的 4 选 1 item。"""
    # 跳过选项不全的题
    if not all(str(raw.get(k, "")).strip() for k in _LETTERS):
        return None
    ans = str(raw.get("answer", "")).strip().upper()
    if ans not in _LETTERS:
        return None

    options = [str(raw[k]).strip() for k in _LETTERS]
    order = list(range(4))
    rng.shuffle(order)                      # order[新位置] = 旧位置
    new_options = [options[i] for i in order]
    answer_new = _LETTERS[order.index(_LETTERS.index(ans))]

    return {
        "id": item_id,
        "question": str(raw.get("question", "")).strip(),
        "options": new_options,
        "answer": answer_new,
        "subject": subject or raw.get("subject", ""),
    }


def load(n: int = 100, seed: int = 42):
    pools = []
    for si, subject in enumerate(SUBJECTS):
        rows = list(_get_subject(subject))
        rng = random.Random(seed * 100 + si)
        rng.shuffle(rows)
        pool = []
        for raw in rows:
            item = _to_item(raw, -1, rng, subject)
            if item is not None:
                pool.append(item)
        pools.append(pool)

    # 学科间轮流取题，保证覆盖面；某学科不够时由其他学科补足
    out = []
    while len(out) < n and any(pools):
        progressed = False
        for pool in pools:
            if pool and len(out) < n:
                out.append(pool.pop(0))
                progressed = True
        if not progressed:
            break

    for i, item in enumerate(out):
        item["id"] = i
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
    # 优先取"答案"后的字母；否则取最后一个独立字母（模型常先分析后给结论）
    m = re.search(r"[Aa][Nn][Ss][Ww][Ee][Rr][:：]?\s*([A-Da-d])", str(output))
    if m:
        letter = m.group(1).upper()
    else:
        matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", str(output))
        if not matches:
            return {"correct": False, "parse_ok": False, "match": "no_letter"}
        letter = matches[-1].upper()
    return {"correct": letter == item["answer"], "parse_ok": True,
            "match": "letter"}
