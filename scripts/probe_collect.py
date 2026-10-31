"""窗口 C 任务 C4：失效预警探针的数据采集（RQ6 第一步）。

目的
    在模型"开口之前"，只看提示词末位的隐藏层激活，判断这一题它会不会答错。

与主实验的一致性
    - 题目、提示词模板、判分器全部复用 scripts/benchmarks/ 下的同一套实现，
      保证探针面对的是与 B2/B3 完全相同的输入。
    - 提示词不加 chat template（主实验用 llama.cpp 也是裸提示词），
      这一点在报告里必须写明。

每题做两件事
    1. 对提示词做一次前向，取出若干层"最后一个 token"的隐藏状态；
    2. 贪婪生成答案 → 判分得到 correct，同时从 generate 的 scores 里
       算出"生成 token 的平均对数概率"，作为置信度基线。

输出
    results/probe_<bench>.npz   特征矩阵 + 标签 + 基线 + 元信息
    results/probe_<bench>.csv   逐题明细（便于人工抽查）
    results/probe_<bench>.meta.json

用法
    # 先小规模验证（约 2 分钟）
    python scripts/probe_collect.py --bench mmlu_pro --n 40

    # 正式采集（每个基准各自保存，可断点）
    python scripts/probe_collect.py --all
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results"
MODEL_DIR = (REPO_ROOT / "hf_cache" / "hub" / "models--Qwen--Qwen3-0.6B"
             / "snapshots" / "c1899de289a04d12100db370d81485cdf75e47ca")

# 让 `import benchmarks.xxx` 能找到包；benchmarks/__init__.py 会顺带
# 把 HF_ENDPOINT / HF_HOME 指到仓库内的 hf_cache
sys.path.insert(0, str(SCRIPTS))

# 取哪几层的激活。Qwen3-0.6B 共 28 层，hidden_states[0] 是 embedding 输出，
# hidden_states[28] 是最终层。这里取四分之一、一半、四分之三、最后一层，
# 训练时再比较哪一层最有预测力。
LAYERS = [7, 14, 21, 28]

# 各基准默认采集量。chinese_facts / ifeval 是固定题池（30 / 100），
# needle 长上下文单题太慢，默认不采。
DEFAULT_N = {
    "mmlu_pro": 400,
    "truthfulqa": 500,
    "ceval": 400,
    "chinese_facts": 30,
    "ifeval": 100,
}


def _model_path() -> Path:
    """优先用显式目录，找不到就交给 transformers 走 HF 缓存。"""
    return MODEL_DIR if MODEL_DIR.exists() else Path("Qwen/Qwen3-0.6B")


def _features_path(bench: str) -> Path:
    return RESULTS / f"probe_{bench}.npz"


def _is_done(bench: str, n: int) -> bool:
    meta = RESULTS / f"probe_{bench}.meta.json"
    if not meta.exists() or not _features_path(bench).exists():
        return False
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return d.get("n_items", 0) >= n and d.get("errors", 1) == 0


def collect_bench(model, tok, bench_name: str, n: int, seed: int,
                  max_len: int) -> None:
    bench = importlib.import_module(f"benchmarks.{bench_name}")
    items = bench.load(n, seed)
    print(f"\n基准 {bench_name}：采集 {len(items)} 题"
          f"（MAX_TOKENS={bench.MAX_TOKENS}）")

    rows = []
    feats = {f"hidden_L{L}": [] for L in LAYERS}
    corrects, confs, conf_first, ptok = [], [], [], []
    errors = 0
    started = time.time()

    for idx, item in enumerate(items, 1):
        item_id = str(item.get("id", idx))
        prompt = bench.build_prompt(item)
        t0 = time.time()
        row = {"item_id": item_id, "bench": bench_name}
        try:
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=max_len)
            n_prompt = int(enc["input_ids"].shape[1])

            # ---- 1. 前向取隐藏状态（不生成）----
            with torch.inference_mode():
                fwd = model(**enc, output_hidden_states=True)
            hs = fwd.hidden_states  # tuple，长度 = 层数 + 1
            for L in LAYERS:
                feats[f"hidden_L{L}"].append(
                    hs[L][0, -1, :].float().numpy().copy())
            del fwd, hs

            # ---- 2. 贪婪生成 + 逐 token 分数 ----
            with torch.inference_mode():
                gen = model.generate(
                    **enc, max_new_tokens=bench.MAX_TOKENS, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                    pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                )
            new_ids = gen.sequences[0][n_prompt:]
            output = tok.decode(new_ids, skip_special_tokens=True).strip()

            # 生成的 token 的平均对数概率 = 经典置信度基线
            logp = []
            for step, score in enumerate(gen.scores):
                if step >= len(new_ids):
                    break
                lp = torch.log_softmax(score[0].float(), dim=-1)
                logp.append(float(lp[int(new_ids[step])]))
            mean_logp = float(np.mean(logp)) if logp else float("nan")
            first_prob = float(np.exp(logp[0])) if logp else float("nan")

            res = bench.judge(output, item)
            correct = bool(res["correct"])
            parse_ok = bool(res["parse_ok"])

            corrects.append(correct)
            confs.append(mean_logp)
            conf_first.append(first_prob)
            ptok.append(n_prompt)
            row.update({
                "answer": output,
                "correct": correct,
                "parse_ok": parse_ok,
                "mean_logprob": round(mean_logp, 6),
                "first_token_prob": round(first_prob, 6),
                "prompt_tokens": n_prompt,
                "gen_tokens": len(new_ids),
                "seconds": round(time.time() - t0, 2),
                "error": "",
            })
        except Exception as exc:  # 单题异常不拖垮整批
            errors += 1
            # 保持数组对齐：异常题也占一行，特征用 NaN
            for L in LAYERS:
                feats[f"hidden_L{L}"].append(np.full(1024, np.nan,
                                                     dtype=np.float32))
            corrects.append(False)
            confs.append(float("nan"))
            conf_first.append(float("nan"))
            ptok.append(0)
            row.update({"answer": "", "correct": False, "parse_ok": False,
                        "mean_logprob": float("nan"),
                        "first_token_prob": float("nan"),
                        "prompt_tokens": 0, "gen_tokens": 0,
                        "seconds": round(time.time() - t0, 2),
                        "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [警告] 第 {idx} 题失败：{row['error']}")

        rows.append(row)
        if idx % 25 == 0 or idx == len(items):
            el = (time.time() - started) / idx
            print(f"  进度 {idx}/{len(items)}  {el:.2f}s/题  "
                  f"预计剩余 {el * (len(items) - idx) / 60:.1f} 分钟")

    # ---- 落盘 ----
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {k: np.stack(v) for k, v in feats.items()}
    payload["correct"] = np.array(corrects, dtype=bool)
    payload["mean_logprob"] = np.array(confs, dtype=np.float32)
    payload["first_token_prob"] = np.array(conf_first, dtype=np.float32)
    payload["prompt_tokens"] = np.array(ptok, dtype=np.int32)
    payload["bench"] = np.array([bench_name] * len(rows))
    payload["item_id"] = np.array([r["item_id"] for r in rows])
    payload["layers"] = np.array(LAYERS)
    np.savez_compressed(_features_path(bench_name), **payload)

    csv_path = RESULTS / f"probe_{bench_name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    acc = float(np.mean(corrects)) if corrects else 0.0
    meta = {
        "bench": bench_name,
        "n_items": len(rows),
        "errors": errors,
        "accuracy": acc,
        "layers": LAYERS,
        "max_len": max_len,
        "seed": seed,
        "model": str(_model_path()),
        "chat_template": False,
        "started": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - started) / 60, 2),
    }
    (RESULTS / f"probe_{bench_name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  完成：{len(rows)} 题，正确率 {acc * 100:.1f}%，错误 {errors} 条")
    print(f"  已写入：{_features_path(bench_name)}、{csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="失效预警探针：隐藏层激活采集")
    ap.add_argument("--bench", default="mmlu_pro",
                    help="基准名，逗号分隔；或用 --all")
    ap.add_argument("--n", type=int, default=0,
                    help="每个基准采集多少题；不填则用 DEFAULT_N")
    ap.add_argument("--all", action="store_true",
                    help="采集 DEFAULT_N 里的全部基准")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="提示词最大 token 数（超出截断）")
    ap.add_argument("--force", action="store_true", help="忽略已完成，重新采集")
    args = ap.parse_args()

    benches = list(DEFAULT_N) if args.all else [
        b.strip() for b in args.bench.split(",") if b.strip()]

    # 线程数：CPU 推理用满物理核心附近的线程数收益最好
    torch.set_num_threads(min(8, torch.get_num_threads() or 8))

    path = _model_path()
    print(f"模型：{path}")
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.float32)
    model.eval()
    print(f"加载完成：{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M 参数")

    for bench_name in benches:
        n = args.n or DEFAULT_N.get(bench_name, 100)
        if not args.force and _is_done(bench_name, n):
            print(f"\n跳过（已完成）：{bench_name}")
            continue
        collect_bench(model, tok, bench_name, n, args.seed, args.max_len)


if __name__ == "__main__":
    main()
