"""Needle-in-a-haystack（4K 上下文）基准，自实现约 60 行。

做法：用固定的中文句子拼出约 5500 字的"长记录"（Qwen 分词器约 4K tokens），
在随机深度插入一条含 6 位密码的"针"，然后问模型该密码是多少。

注意：调用 eval_loop.py 时必须显式给足上下文，例如 `--n_ctx 8192`，
否则 4K 提示词会被截断，测的就不是长上下文能力了。

判分：输出中出现该密码即算对；出现数字但不对 → 答错；完全没有数字 → 解析失败。
"""
import random
import re

MAX_TOKENS = 32

# 目标长度：约 4K tokens（中文约 1.3~1.6 字/token）
TARGET_CHARS = 5500

FILLER = [
    "山谷里的风从清晨一直吹到傍晚，带来草木和泥土的气味。",
    "村子旁边有一条安静的小河，河面上偶尔漂过几片落叶。",
    "值班人员沿着仓库外的围墙慢慢走了一圈，确认门窗都已经锁好。",
    "远处的灯塔每隔一段时间就会亮起，为夜行的船只指引方向。",
    "屋檐下的燕子来回穿梭，忙着把新衔来的草茎放进巢里。",
    "桌面上的台灯发出温暖的光，照亮了摊开的笔记本和铅笔。",
    "雨停之后，石板路上留下浅浅的水洼，倒映着灰蓝色的天空。",
    "清晨的集市渐渐热闹起来，摊主们把新鲜的蔬菜整齐地摆开。",
    "列车穿过长长的隧道，窗外的景色从田野变成了连绵的山丘。",
    "图书馆里非常安静，只能听见翻书页和笔尖划过纸面的声音。",
    "海边的风带着咸味，浪花一遍遍拍打着黑色的礁石。",
    "花园角落里的老树抽出新芽，几只麻雀在枝头跳来跳去。",
    "厨房里飘出米饭的香气，锅盖被蒸汽顶得轻轻作响。",
    "操场边的看台上坐着几个人，他们正在等待傍晚的比赛开始。",
    "天空中的云层缓慢移动，阳光时而明亮时而柔和。",
    "邮递员把信件放进门口的邮箱，又骑上车继续赶往下一户人家。",
    "钟表店的橱窗里摆着各式各样的钟表，指针发出细小的滴答声。",
    "山坡上的羊群低头吃草，牧羊犬安静地趴在旁边的石头上。",
    "夜色渐深，街上的行人越来越少，路灯把影子拉得很长。",
    "实验室里的仪器保持运转，显示屏上的曲线平稳地向前延伸。",
]

_DIGITS = re.compile(r"\d+")


def _build_haystack(target_chars: int, rng: random.Random) -> str:
    """用固定句子拼出指定长度的长文本（句子之间用句号分隔）。"""
    sentences = []
    total = 0
    while total < target_chars:
        s = rng.choice(FILLER)
        sentences.append(s)
        total += len(s)
    return "".join(sentences)


def load(n: int = 20, seed: int = 42):
    """生成 n 道长上下文题；深度在 10%~90% 之间均匀分布。"""
    rng = random.Random(seed)
    depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    items = []
    for i in range(n):
        serial = 100000 + i
        passkey = str(rng.randrange(100000, 1000000))
        depth = depths[i % len(depths)]

        haystack = _build_haystack(TARGET_CHARS, rng)
        # 按句号切分后插入，避免把句子从中间截断
        sentences = [s + "。" for s in haystack.split("。") if s]
        pos = max(0, min(len(sentences) - 1, int(len(sentences) * depth)))
        needle = f"特别记录：编号 {serial} 的保险箱密码是 {passkey}。"
        sentences.insert(pos, needle)

        items.append({
            "id": i,
            "serial": serial,
            "passkey": passkey,
            "depth": depth,
            "haystack": "".join(sentences),
        })
    return items


def build_prompt(item) -> str:
    return (
        "下面是一份很长的值班记录。请通读全文后回答问题。\n\n"
        f"{item['haystack']}\n\n"
        f"问题：记录中编号 {item['serial']} 的保险箱密码是多少？"
        "只输出 6 位数字密码。\n"
        "答案："
    )


def judge(output: str, item) -> dict:
    text = str(output)
    passkey = item["passkey"]

    # 先直接匹配；再退化为"只保留数字"后匹配（应对 1 2 3 4 5 6 这类输出）
    if passkey in text:
        return {"correct": True, "parse_ok": True, "match": "exact"}
    digits = "".join(_DIGITS.findall(text))
    if passkey in digits:
        return {"correct": True, "parse_ok": True, "match": "digits"}
    if digits:
        return {"correct": False, "parse_ok": True, "match": "wrong_number"}
    return {"correct": False, "parse_ok": False, "match": "no_number"}
