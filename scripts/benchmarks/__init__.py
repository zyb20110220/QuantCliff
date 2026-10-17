"""基准模块包。

每个基准模块必须提供：
    load(n, seed)      -> list[item]   题目加载（seed 固定保证各档位抽到同一批题）
    build_prompt(item) -> str          完整提示词（含指令与选项）
    judge(output, item)-> dict         判分，返回 {"correct": bool, "parse_ok": bool}
    MAX_TOKENS         -> int          输出长度上限

item 建议字段：id（题号）、question（题干）、answer（标准答案）、其他基准自定。

环境约定：数据与模型缓存统一放在仓库根目录（data/、hf_cache/、models/），
因此这里在导入 datasets 之前先把 HF_ENDPOINT / HF_HOME 设好。
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / "hf_cache"))
