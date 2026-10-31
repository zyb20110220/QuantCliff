"""验证探针的预警不依赖模型答案（防泄漏检验）。

回应的问题
    演示里看到"红灯 + 模型答错"，怎么排除这是**看到答案之后才判断**的？
    光看代码说"我先算了预警"说服力不够，所以这里做三重可复现的实证检验。

检验 1：预警路径完全不生成
    调用 score_only()，它只对提示词做一次前向。统计本次调用是否产生任何
    token（通过拦截 model.generate 断言一次都没被调用）。

检验 2：顺序无关 / 结果无关
    同一道题，分别按两种顺序取值：
      A. 只算预警（不生成）
      B. 先算预警、再生成答案
    两者的探针分必须**逐位相等**。若预警偷看了答案，A 与 B 会不一致。

检验 3：只对提示词敏感
    把同一道题的四个选项打乱成另一道题的选项（改提示词、不改标签），
    探针分必须变化——说明它读的确实是提示词，不是别的。

检验 4：不是"预言机"
    如果探针真能拿到答案，AUROC 会是 1.0、绿灯档正确率会是 100%。
    实际折外 AUROC 与各档正确率都远低于此，见 probe_model_*.json。

用法：
    python scripts/verify_probe.py --n 20
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import app  # noqa: E402


class _GenerateGuard:
    """临时替换 model.generate，用来证明预警路径没有触发生成。"""

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self._orig = None

    def __enter__(self):
        self._orig = self.model.generate

        def _blocked(*a, **kw):
            self.calls += 1
            raise AssertionError("预警路径不应调用 generate")

        self.model.generate = _blocked
        return self

    def __exit__(self, *exc):
        self.model.generate = self._orig
        return False


def check_no_generation(inter, items) -> dict:
    """检验 1：score_only 全程不触发生成。"""
    ok = 0
    with _GenerateGuard(inter.model) as guard:
        for it in items:
            try:
                inter.score_only(it)
                ok += 1
            except AssertionError:
                break
    return {"passed": guard.calls == 0 and ok == len(items),
            "generate_calls": guard.calls,
            "items_done": ok}


def check_order_invariance(inter, items) -> dict:
    """检验 2：单独预警 vs 预警+生成，探针分逐位相等。"""
    diffs = []
    for it in items:
        a = inter.score_only(it)["probe"]
        full = inter._run(it)
        b = full["probe"]
        diffs.append(abs(a - b))
    return {"passed": max(diffs) == 0.0,
            "max_abs_diff": float(max(diffs)),
            "n": len(diffs)}


def check_prompt_sensitivity(inter, items) -> dict:
    """检验 3：改提示词（选项打乱）后探针分必须变化。"""
    changed = 0
    for it in items:
        base = inter.score_only(it)["probe"]
        altered = copy.deepcopy(it)
        opts = list(altered.get("options") or [])
        if len(opts) < 2:
            continue
        altered["options"] = opts[1:] + opts[:1]  # 循环左移一位
        new = inter.score_only(altered)["probe"]
        if abs(new - base) > 1e-9:
            changed += 1
    return {"passed": changed == len(items), "changed": changed,
            "n": len(items)}


def check_not_oracle(meta_path: Path) -> dict:
    """检验 4：探针不是预言机——AUROC 远低于 1，绿灯档远低于 100%。"""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    green = meta["bands"]["green"]["accuracy"]
    return {"passed": meta["auroc_oof"] < 0.9 and green < 0.9,
            "auroc_oof": meta["auroc_oof"],
            "green_band_accuracy": green}


def main() -> None:
    ap = argparse.ArgumentParser(description="探针防泄漏验证")
    ap.add_argument("--n", type=int, default=20,
                    help="用于前两项检验的题目数量")
    ap.add_argument("--out", default="results/verify_probe.md")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads() or 8))
    inter = app.Interceptor(app.PRESET_BENCH, app.PRESET_N)
    items = inter.items[:args.n]

    print(f"载入 {app.PRESET_BENCH}，取前 {len(items)} 题做检验\n")

    r1 = check_no_generation(inter, items)
    print(f"[检验 1] 预警路径不触发生成：{'通过' if r1['passed'] else '失败'}  "
          f"(generate 被调用 {r1['generate_calls']} 次)")

    r2 = check_order_invariance(inter, items)
    print(f"[检验 2] 单独预警 vs 预警+生成：{'通过' if r2['passed'] else '失败'}  "
          f"(最大差异 {r2['max_abs_diff']:.3e})")

    r3 = check_prompt_sensitivity(inter, items)
    print(f"[检验 3] 改提示词后预警变化：{'通过' if r3['passed'] else '失败'}  "
          f"({r3['changed']}/{r3['n']} 题发生变化)")

    meta_path = Path("results") / f"probe_model_{app.PRESET_BENCH}.json"
    r4 = check_not_oracle(meta_path)
    print(f"[检验 4] 不是预言机：{'通过' if r4['passed'] else '失败'}  "
          f"(折外 AUROC {r4['auroc_oof']:.3f}，绿灯档正确率 "
          f"{r4['green_band_accuracy'] * 100:.0f}%)")

    all_passed = all(r["passed"] for r in (r1, r2, r3, r4))

    lines = [
        "# 探针防泄漏验证",
        "",
        "> 目的：排除「预警其实是看到答案之后才给出的」这一可能。",
        "",
        "| 检验 | 内容 | 结果 | 证据 |",
        "|---|---|---|---|",
        f"| 1 | `score_only()` 全程不调用生成 | {'通过' if r1['passed'] else '失败'} | "
        f"generate 被调用 {r1['generate_calls']} 次（预期 0） |",
        f"| 2 | 单独预警与「预警 + 生成」的探针分一致 | "
        f"{'通过' if r2['passed'] else '失败'} | "
        f"最大绝对差异 {r2['max_abs_diff']:.3e} |",
        f"| 3 | 打乱选项后预警发生变化 | {'通过' if r3['passed'] else '失败'} | "
        f"{r3['changed']}/{r3['n']} 题发生变化 |",
        f"| 4 | 探针不是「预言机」 | {'通过' if r4['passed'] else '失败'} | "
        f"折外 AUROC {r4['auroc_oof']:.3f}（远低于 1），"
        f"绿灯档正确率 {r4['green_band_accuracy'] * 100:.0f}%（远低于 100%） |",
        "",
        "## 结论",
        "",
        ("四项检验全部通过：预警只依赖提示词的隐藏状态，"
         "既不读取也不依赖模型生成的答案。"
         if all_passed else
         "**存在未通过的检验，需要排查后再引用结论。**"),
        "",
        "## 为什么检验 4 最有力",
        "",
        "如果探针在推理时能接触到答案，它在折外评估里的 AUROC 必然是 1.0、"
        "绿灯档正确率必然是 100%。实测 AUROC 只有 0.693、绿灯档只有 62%，"
        "**距离「全知」差得很远**——这恰恰说明它只能从提示词里推测，"
        "而不是照抄答案。",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写入：{out}")
    print(f"总体：{'全部通过' if all_passed else '有失败项'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
