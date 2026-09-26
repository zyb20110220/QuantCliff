"""基准模块包。

每个基准模块必须提供：
    load(n, seed)      -> list[item]   题目加载（seed 固定保证各档位抽到同一批题）
    build_prompt(item) -> str          完整提示词（含指令与选项）
    judge(output, item)-> dict         判分，返回 {"correct": bool, "parse_ok": bool}
    MAX_TOKENS         -> int          输出长度上限

item 建议字段：id（题号）、question（题干）、answer（标准答案）、其他基准自定。
"""
