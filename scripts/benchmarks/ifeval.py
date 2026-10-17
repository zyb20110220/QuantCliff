"""IFEval 风格的自构造指令跟随基准（100 题）。

开题方案要求"自写 8 类可判指令解析器"。这里实现 10 类可机器判定的约束，
用 10 个主题 × 10 类约束生成 100 道题，全部离线、结果确定：

    1. include     回答必须包含指定词
    2. exclude     回答不得包含指定词
    3. prefix      回答必须以指定字符串开头
    4. suffix      回答必须以指定字符串结尾
    5. bullets     恰好 N 条以 "- " 开头的列表项
    6. sentences   恰好 N 个句子（以。！？!? 结尾）
    7. json        输出合法 JSON，且包含指定键
    8. no_comma    不得出现逗号（，或 ,）
    9. min_chars   至少 N 个汉字
   10. max_chars   至多 N 个汉字

说明：这套题测的是"听不听话"，不是知识正确性；因此只要满足约束即判对。
"""
import json
import re

MAX_TOKENS = 160

TOPICS = ["太阳", "长江", "长城", "故宫", "春节", "熊猫", "水稻", "高铁", "月亮", "海洋"]

# 每类约束的模板与判分参数
CONSTRAINTS = [
    {
        "type": "include",
        "instruction": "回答中必须包含词语「历史」。",
        "param": {"keyword": "历史"},
    },
    {
        "type": "exclude",
        "instruction": "回答中不得出现词语「但是」。",
        "param": {"keyword": "但是"},
    },
    {
        "type": "prefix",
        "instruction": "回答必须以「首先，」开头。",
        "param": {"prefix": "首先，"},
    },
    {
        "type": "suffix",
        "instruction": "回答必须以「谢谢」结尾。",
        "param": {"suffix": "谢谢"},
    },
    {
        "type": "bullets",
        "instruction": "请用恰好 3 条以「- 」开头的列表项回答。",
        "param": {"n": 3},
    },
    {
        "type": "sentences",
        "instruction": "请用恰好 2 句话回答（以句号、问号或感叹号结尾）。",
        "param": {"n": 2},
    },
    {
        "type": "json",
        "instruction": "请只输出一个 JSON 对象，包含键「主题」和「介绍」。",
        "param": {"keys": ["主题", "介绍"]},
    },
    {
        "type": "no_comma",
        "instruction": "回答中不得出现逗号（中文逗号或英文逗号）。",
        "param": {},
    },
    {
        "type": "min_chars",
        "instruction": "回答至少 30 个汉字。",
        "param": {"n": 30},
    },
    {
        "type": "max_chars",
        "instruction": "回答不超过 40 个汉字。",
        "param": {"n": 40},
    },
]

_CJK = re.compile(r"[\u4e00-\u9fff]")
_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_SENT_END = re.compile(r"[。！？!?]")


def load(n: int = 100, seed: int = 42):
    """生成 100 道题；seed 目前不影响内容（保持接口统一）。"""
    items = []
    idx = 0
    for topic in TOPICS:
        for cons in CONSTRAINTS:
            items.append({
                "id": idx,
                "topic": topic,
                "question": f"请介绍{topic}。",
                "instruction": cons["instruction"],
                "type": cons["type"],
                "param": dict(cons["param"]),
            })
            idx += 1
    return items[:max(0, min(n, len(items)))]


def build_prompt(item) -> str:
    return (
        "请完成下面的写作任务，并严格遵守所有格式要求。\n"
        f"任务：{item['question']}\n"
        f"要求：{item['instruction']}\n"
        "回答："
    )


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 包裹，便于 JSON 判分。"""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def judge(output: str, item) -> dict:
    text = str(output)
    stripped = text.strip()
    if not stripped:
        return {"correct": False, "parse_ok": False, "match": "empty"}

    t = item["type"]
    p = item["param"]

    if t == "include":
        ok = p["keyword"] in text
    elif t == "exclude":
        ok = p["keyword"] not in text
    elif t == "prefix":
        ok = stripped.startswith(p["prefix"])
    elif t == "suffix":
        ok = stripped.endswith(p["suffix"])
    elif t == "bullets":
        ok = len(_BULLET.findall(text)) == p["n"]
    elif t == "sentences":
        ok = len(_SENT_END.findall(text)) == p["n"]
    elif t == "json":
        ok = False
        candidate = _strip_code_fence(text)
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(candidate[start:end + 1])
                ok = isinstance(obj, dict) and all(k in obj for k in p["keys"])
            except json.JSONDecodeError:
                ok = False
    elif t == "no_comma":
        ok = ("，" not in text) and ("," not in text)
    elif t == "min_chars":
        ok = len(_CJK.findall(text)) >= p["n"]
    elif t == "max_chars":
        ok = len(_CJK.findall(text)) <= p["n"]
    else:  # 未知类型直接判错，避免静默通过
        ok = False

    return {"correct": bool(ok), "parse_ok": True, "match": t}
