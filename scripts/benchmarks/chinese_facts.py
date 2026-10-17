"""中文事实基准（30 条自构造题，见 docs/proposal.md 附录）。

判分规则（与开题方案一致）：
    1. 输出中出现标准答案词 → 对；
    2. 输出中出现已知错误答案词 → 错；
    3. 两者都没有 → parse_ok=False（记为解析失败，留待人工抽检）。

为了降低"答案词出现在无关上下文"造成的误判，采用**最后出现位置**规则：
如果正确答案和错误答案同时出现，以更靠后的那个为准（模型通常把结论放在最后），
例如"不是上海，是北京"判对，"不是北京，是上海"判错。

数值答案使用数字边界匹配，避免 "5" 命中 "15"、"340" 命中 "3400"。
"""
import re
import unicodedata

# 输出很短（一个词或一个数字）
MAX_TOKENS = 24

# 每题：问题、可接受答案（同义词/不同写法）、已知错误答案、备注
FACTS = [
    {
        "id": 1,
        "question": "中华人民共和国的首都是哪座城市？",
        "answer": ["北京"],
        "wrong": ["上海", "南京", "广州", "西安", "天津", "重庆", "深圳", "洛阳", "开封"],
        "note": "",
    },
    {
        "id": 2,
        "question": "标准大气压下水的沸点约多少摄氏度？",
        "answer": ["100", "一百"],
        "wrong": ["0", "80", "90", "120", "212"],
        "note": "题干已限定摄氏度",
    },
    {
        "id": 3,
        "question": "地球绕太阳公转一周约需多久？",
        "answer": ["一年", "365", "365天", "约365", "365.25"],
        "wrong": ["一天", "24小时", "一个月", "30天", "12小时", "自转"],
        "note": "",
    },
    {
        "id": 4,
        "question": "中国最长的河流是哪条？",
        "answer": ["长江"],
        "wrong": ["黄河", "珠江", "黑龙江", "雅鲁藏布江", "淮河", "松花江"],
        "note": "",
    },
    {
        "id": 5,
        "question": "人体正常体温约多少摄氏度？",
        "answer": ["37", "36.5", "36-37", "36.5-37", "37.2"],
        "wrong": ["25", "30", "35", "38", "39", "40", "42"],
        "note": "腋温常用 36.5~37.2 摄氏度",
    },
    {
        "id": 6,
        "question": "光在真空中的速度约为每秒多少万千米？",
        "answer": ["30万", "三十万", "299792", "3×10^8", "3*10^8", "30万千米"],
        "wrong": ["3万", "300万", "340", "1000", "30米"],
        "note": "约 29.98 万千米/秒",
    },
    {
        "id": 7,
        "question": "《红楼梦》的作者是谁？",
        "answer": ["曹雪芹"],
        "wrong": ["罗贯中", "施耐庵", "吴承恩", "鲁迅", "巴金", "老舍"],
        "note": "高鹗续写后四十回，故不列入错误答案",
    },
    {
        "id": 8,
        "question": "北半球白昼最长的一天是哪个节气？",
        "answer": ["夏至", "夏至日"],
        "wrong": ["冬至", "春分", "秋分", "立夏", "立春"],
        "note": "",
    },
    {
        "id": 9,
        "question": "原子序数为 1 的元素是什么？",
        "answer": ["氢", "hydrogen"],
        "wrong": ["氦", "氧", "碳", "氮", "钠"],
        "note": "",
    },
    {
        "id": 10,
        "question": "直角边分别为 3 和 4 的直角三角形，斜边是多少？",
        "answer": ["5", "五"],
        "wrong": ["6", "7", "4", "25", "12", "50"],
        "note": "",
    },
    {
        "id": 11,
        "question": "中国国旗上有几颗五角星？",
        "answer": ["五颗", "5颗", "五个", "5个", "五", "5"],
        "wrong": ["四", "4", "六", "6", "一", "1", "三", "3"],
        "note": "",
    },
    {
        "id": 12,
        "question": "地球唯一的天然卫星是什么？",
        "answer": ["月球", "月亮"],
        "wrong": ["火星", "太阳", "金星", "木星", "土星"],
        "note": "",
    },
    {
        "id": 13,
        "question": "人体中按质量计含量最多的元素是什么？",
        "answer": ["氧", "oxygen"],
        "wrong": ["氢", "碳", "钙", "氮", "铁"],
        "note": "2026-10-17 审核：补充'按质量计'，避免与按原子数计的氢产生歧义",
    },
    {
        "id": 14,
        "question": "中国历史上第一个大一统王朝是什么？",
        "answer": ["秦朝", "秦"],
        "wrong": ["汉朝", "商朝", "周朝", "唐朝", "夏朝", "明朝", "清朝"],
        "note": "",
    },
    {
        "id": 15,
        "question": "声音在空气中的传播速度约为每秒多少米？",
        "answer": ["340", "约340", "三百四十"],
        "wrong": ["1500", "1000", "100", "30", "3400"],
        "note": "15 摄氏度空气中约 340 m/s",
    },
    {
        "id": 16,
        "question": "1 千克等于多少克？",
        "answer": ["1000", "一千"],
        "wrong": ["100", "10", "500", "10000", "1024"],
        "note": "",
    },
    {
        "id": 17,
        "question": "太阳系中体积最大的行星是哪一颗？",
        "answer": ["木星"],
        "wrong": ["土星", "地球", "火星", "海王星", "天王星", "太阳"],
        "note": "",
    },
    {
        "id": 18,
        "question": "中国陆地面积约多少万平方公里？",
        "answer": ["960", "九百六十"],
        "wrong": ["300", "500", "1200", "1400", "9600"],
        "note": "约 960 万平方公里",
    },
    {
        "id": 19,
        "question": "光合作用释放的气体是什么？",
        "answer": ["氧气", "氧"],
        "wrong": ["二氧化碳", "氮气", "氢气", "一氧化碳"],
        "note": "",
    },
    {
        "id": 20,
        "question": "珠穆朗玛峰的海拔约多少米？",
        "answer": ["8848", "8844", "8848.86", "八千八百"],
        "wrong": ["8000", "7000", "9000", "6000"],
        "note": "2020 年中尼联合公布 8848.86 米",
    },
    {
        "id": 21,
        "question": "水的化学式是什么？",
        "answer": ["h2o", "水分子"],
        "wrong": ["co2", "o2", "h2o2", "ch4"],
        "note": "",
    },
    {
        "id": 22,
        "question": "圆周率 π 保留两位小数是多少？",
        "answer": ["3.14", "3.14159", "3.1416"],
        "wrong": ["3.12", "3.16", "3.41", "2.14", "3.15"],
        "note": "",
    },
    {
        "id": 23,
        "question": "中国第一部诗歌总集是什么？",
        "answer": ["诗经"],
        "wrong": ["楚辞", "唐诗三百首", "论语", "离骚", "乐府"],
        "note": "",
    },
    {
        "id": 24,
        "question": "血液中红细胞主要运输什么？",
        "answer": ["氧气", "氧"],
        "wrong": ["二氧化碳", "养分", "激素", "白细胞", "血小板"],
        "note": "",
    },
    {
        "id": 25,
        "question": "长江与黄河都发源于哪个高原？",
        "answer": ["青藏", "青藏高原"],
        "wrong": ["内蒙古", "黄土高原", "云贵高原", "帕米尔", "巴西高原"],
        "note": "",
    },
    {
        "id": 26,
        "question": "一年有几个季度？",
        "answer": ["四", "4"],
        "wrong": ["三", "3", "五", "5", "二", "2", "六", "6"],
        "note": "",
    },
    {
        "id": 27,
        "question": "计算机中 1 字节等于多少 bit？",
        "answer": ["8", "八"],
        "wrong": ["4", "16", "32", "1024", "2", "10"],
        "note": "",
    },
    {
        "id": 28,
        "question": "中国农历正月初一的节日是什么？",
        "answer": ["春节", "过年", "农历新年"],
        "wrong": ["元宵节", "中秋节", "端午节", "元旦", "清明节"],
        "note": "",
    },
    {
        "id": 29,
        "question": "成年人体内骨骼共有约多少块？",
        "answer": ["206", "二百零六"],
        "wrong": ["300", "100", "150", "210", "205", "306"],
        "note": "",
    },
    {
        "id": 30,
        "question": "地球大气中含量最多的气体是什么？",
        "answer": ["氮气", "氮"],
        "wrong": ["氧气", "二氧化碳", "氢气", "氩气", "水蒸气"],
        "note": "按体积计约 78%",
    },
]


def _normalize(text: str) -> str:
    """全角转半角、转小写，便于匹配。"""
    return unicodedata.normalize("NFKC", str(text)).lower()


def _alias_pattern(alias: str):
    """把答案词编译成正则。

    - 纯数字/小数：加数字边界，避免 5 命中 15、340 命中 3400；
    - 纯 ASCII 字母数字（如 h2o）：加字母数字边界，避免 h2o 命中 h2o2；
    - 中文：直接子串匹配。
    """
    a = _normalize(alias)
    if re.fullmatch(r"\d+(?:\.\d+)?", a):
        return re.compile(r"(?<![\d.])" + re.escape(a) + r"(?![\d.])")
    if re.fullmatch(r"[a-z0-9.+*^]+", a):
        return re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])")
    return re.compile(re.escape(a))


def _last_pos(text: str, aliases):
    """返回任一答案词最后一次出现的起始位置；没有则 -1。"""
    pos = -1
    for alias in aliases:
        for m in _alias_pattern(alias).finditer(text):
            pos = max(pos, m.start())
    return pos


def load(n: int = 30, seed: int = 42):
    """返回 30 条题；n 可小于 30（按固定顺序取前 n 条，便于冒烟）。"""
    items = []
    for row in FACTS[:max(0, min(n, len(FACTS)))]:
        items.append(dict(row))
    return items


def build_prompt(item) -> str:
    return (
        "请直接回答下面的中文事实问题，只输出答案本身，不要解释。\n"
        f"问题：{item['question']}\n"
        "答案："
    )


def judge(output: str, item) -> dict:
    text = _normalize(output)
    if not text.strip():
        return {"correct": False, "parse_ok": False, "match": "empty"}

    correct_pos = _last_pos(text, item["answer"])
    wrong_pos = _last_pos(text, item["wrong"])

    if correct_pos >= 0 and correct_pos >= wrong_pos:
        return {"correct": True, "parse_ok": True, "match": "answer"}
    if wrong_pos >= 0:
        return {"correct": False, "parse_ok": True, "match": "wrong"}
    # 既没有命中标准答案，也没有命中已知错误答案 → 留待人工抽检
    return {"correct": False, "parse_ok": False, "match": "unknown"}
