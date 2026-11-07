"""窗口 D 任务 D1：thinking 模式下的探针特征采集（预警时机曲线）。

与 C4 采集的区别
    1. 提示词改用 Qwen3 的 chat template，并开启 enable_thinking：
       模型先生成 <think>...</think> 思考段，再给出答案。
    2. 探针观察点不再只有"提示词末位"，而是沿思考段取 5 个位置：
       0% / 25% / 50% / 75% / 100%（相对思考段 token 数）。
       位置越靠后，离"决策时刻"越近——这正是"时机曲线"的横轴。
    3. 同时采集一个"非思考"臂：同一 chat template 但 enable_thinking=False
       （模板会插入空的 <think></think> 块），在提示词末位取激活。
       这样可以把"chat template 本身"与"思考过程"两个因素拆开：
       原始裸提示词（C4）vs 非思考模板（D1）vs 思考模板（D1）。

输出
    results/probe_think_<基准>.npz   5 位置 × 4 层特征 + 标签 + 基线
    results/probe_think_<基准>.csv   逐题明细（含思考/答案文本片段）
    results/probe_think_<基准>.meta.json

用法
    python scripts/probe_collect_thinking.py --bench truthfulqa --n 20    # 试跑
    python scripts/probe_collect_thinking.py --bench truthfulqa,mmlu_pro  # 正式
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

sys.path.insert(0, str(SCRIPTS))

LAYERS = [7, 14, 21, 28]
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
POS_KEYS = ["p0", "p25", "p50", "p75", "p100"]

DEFAULT_N = {"truthfulqa": 250, "mmlu_pro": 250}
MAX_NEW = 1024          # 思考可能很长，给足空间
MAX_LEN = 2048          # 提示词最大长度


def _model_path() -> Path:
    return MODEL_DIR if MODEL_DIR.exists() else Path("Qwen/Qwen3-0.6B")


def _features_path(bench: str) -> Path:
    return RESULTS / f"probe_think_{bench}.npz"


def _is_done(bench: str, n: int) -> bool:
    meta = RESULTS / f"probe_think_{bench}.meta.json"
    if not meta.exists() or not _features_path(bench).exists():
        return False
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return d.get("n_items", 0) >= n and d.get("errors", 1) == 0


def _split_think(new_ids: torch.Tensor, tok, open_id: int, close_id: int):
    """把生成序列切成 (思考段, 答案段)。无思考块时思考段为空。"""
    ids = new_ids.tolist()
    if open_id in ids and close_id in ids:
        o = ids.index(open_id)
        c = ids.index(close_id, o + 1)
        return new_ids[o + 1:c], new_ids[c + 1:]
    return new_ids[:0], new_ids


def collect_bench(model, tok, bench_name: str, n: int, seed: int,
                  open_id: int, close_id: int, max_new: int) -> None:
    bench = importlib.import_module(f"benchmarks.{bench_name}")
    items = bench.load(n, seed)
    print(f"\n基准 {bench_name}：采集 {len(items)} 题（thinking 模式）")

    rows = []
    feats = {}
    for key in POS_KEYS:
        for L in LAYERS:
            feats[f"think_L{L}_{key}"] = []
    for L in LAYERS:
        feats[f"nothink_L{L}"] = []
    corrects, confs, n_think, n_answer, triggered = [], [], [], [], []
    nt_corrects, nt_ans = [], []
    errors = 0
    started = time.time()

    pad_id = tok.pad_token_id or tok.eos_token_id

    for idx, item in enumerate(items, 1):
        item_id = str(item.get("id", idx))
        prompt = bench.build_prompt(item)
        msgs = [{"role": "user", "content": prompt}]
        t0 = time.time()
        row = {"item_id": item_id, "bench": bench_name}
        try:
            # ---- 1. thinking 生成 ----
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=True,
                return_tensors="pt", return_dict=True)
            ids = enc["input_ids"]
            n_prompt = int(ids.shape[1])

            with torch.inference_mode():
                out = model.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                    pad_token_id=pad_id)
            new_ids = out.sequences[0][n_prompt:]
            think_tok, ans_tok = _split_think(new_ids, tok, open_id, close_id)
            n_t = int(think_tok.shape[0])
            n_a = int(ans_tok.shape[0])
            answer = tok.decode(ans_tok, skip_special_tokens=True).strip()

            # 答案段的平均对数概率（与 C4 同口径）
            logp = []
            for step, score in enumerate(out.scores):
                if step < n_t + (1 if n_t else 0):
                    continue  # 跳过思考段与 </think>
                if step - (n_t + (1 if n_t else 0)) >= n_a:
                    break
                lp = torch.log_softmax(score[0].float(), dim=-1)
                logp.append(
                    float(lp[int(ans_tok[step - (n_t + (1 if n_t else 0))])]))
            conf = float(np.exp(np.mean(logp))) if logp else float("nan")
            del out

            # ---- 2. 五个观察点：对 prompt + 思考前缀做前向 ----
            base = ids[0]
            for key, p in zip(POS_KEYS, POSITIONS):
                k = int(round(p * n_t))
                prefix = torch.cat([base, think_tok[:k]]) if k else base
                with torch.inference_mode():
                    fwd = model(prefix.unsqueeze(0), output_hidden_states=True)
                hs = fwd.hidden_states
                for L in LAYERS:
                    feats[f"think_L{L}_{key}"].append(
                        hs[L][0, -1, :].float().numpy().copy())
                del fwd

            # ---- 3. 非思考臂：enable_thinking=False，取激活 + 生成答案 ----
            enc_n = tok.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt", return_dict=True)
            ids_n = enc_n["input_ids"]
            with torch.inference_mode():
                fwd_n = model(ids_n, output_hidden_states=True)
            for L in LAYERS:
                feats[f"nothink_L{L}"].append(
                    fwd_n.hidden_states[L][0, -1, :].float().numpy().copy())
            del fwd_n
            with torch.inference_mode():
                out_n = model.generate(ids_n, max_new_tokens=64, do_sample=False,
                                       pad_token_id=pad_id)
            ans_n = tok.decode(out_n[0][ids_n.shape[1]:],
                               skip_special_tokens=True).strip()
            res_n = bench.judge(ans_n, item)
            nt_corrects.append(bool(res_n["correct"]))
            nt_ans.append(ans_n[:120])
            del out_n

            res = bench.judge(answer, item)
            correct = bool(res["correct"])
            parse_ok = bool(res["parse_ok"])

            corrects.append(correct)
            confs.append(conf)
            n_think.append(n_t)
            n_answer.append(n_a)
            triggered.append(n_t > 0)
            row.update({
                "think_triggered": n_t > 0,
                "n_think": n_t,
                "n_answer": n_a,
                "correct": correct,
                "parse_ok": parse_ok,
                "nothink_correct": bool(res_n["correct"]),
                "nothink_parse_ok": bool(res_n["parse_ok"]),
                "nothink_answer": nt_ans[-1],
                "mean_logprob": round(conf, 6) if np.isfinite(conf) else "",
                "prompt_tokens": n_prompt,
                "seconds": round(time.time() - t0, 2),
                "answer": answer[:120],
                "thinking": tok.decode(think_tok,
                                       skip_special_tokens=True)[:120],
                "error": "",
            })
        except Exception as exc:  # 单题异常不拖垮整批
            errors += 1
            for key in POS_KEYS:
                for L in LAYERS:
                    feats[f"think_L{L}_{key}"].append(
                        np.full(1024, np.nan, dtype=np.float32))
            for L in LAYERS:
                feats[f"nothink_L{L}"].append(
                    np.full(1024, np.nan, dtype=np.float32))
            corrects.append(False)
            confs.append(float("nan"))
            n_think.append(0)
            n_answer.append(0)
            triggered.append(False)
            nt_corrects.append(False)
            nt_ans.append("")
            row.update({"think_triggered": False, "n_think": 0, "n_answer": 0,
                        "correct": False, "parse_ok": False,
                        "nothink_correct": False, "nothink_parse_ok": False,
                        "nothink_answer": "",
                        "mean_logprob": "", "prompt_tokens": 0,
                        "seconds": round(time.time() - t0, 2),
                        "answer": "", "thinking": "",
                        "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [警告] 第 {idx} 题失败：{row['error']}")

        rows.append(row)
        if idx % 10 == 0 or idx == len(items):
            el = (time.time() - started) / idx
            print(f"  进度 {idx}/{len(items)}  {el:.1f}s/题  "
                  f"预计剩余 {el * (len(items) - idx) / 60:.1f} 分钟")

    # ---- 落盘 ----
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {k: np.stack(v) for k, v in feats.items()}
    payload["correct"] = np.array(corrects, dtype=bool)
    payload["nothink_correct"] = np.array(nt_corrects, dtype=bool)
    payload["mean_logprob"] = np.array(confs, dtype=np.float32)
    payload["n_think"] = np.array(n_think, dtype=np.int32)
    payload["n_answer"] = np.array(n_answer, dtype=np.int32)
    payload["think_triggered"] = np.array(triggered, dtype=bool)
    payload["bench"] = np.array([bench_name] * len(rows))
    payload["item_id"] = np.array([r["item_id"] for r in rows])
    payload["layers"] = np.array(LAYERS)
    payload["positions"] = np.array(POSITIONS)
    np.savez_compressed(_features_path(bench_name), **payload)

    csv_path = RESULTS / f"probe_think_{bench_name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    acc = float(np.mean(corrects)) if corrects else 0.0
    trig = float(np.mean(triggered)) if triggered else 0.0
    nt_acc = float(np.mean(nt_corrects)) if nt_corrects else 0.0
    meta = {
        "bench": bench_name,
        "n_items": len(rows),
        "errors": errors,
        "accuracy": acc,
        "nothink_accuracy": nt_acc,
        "think_trigger_rate": trig,
        "think_tokens_mean": float(np.mean(n_think)),
        "layers": LAYERS,
        "positions": POSITIONS,
        "max_new": max_new,
        "seed": seed,
        "model": str(_model_path()),
        "chat_template": True,
        "enable_thinking": True,
        "started": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - started) / 60, 2),
    }
    (RESULTS / f"probe_think_{bench_name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  完成：{len(rows)} 题，思考正确率 {acc * 100:.1f}% / 非思考 {nt_acc * 100:.1f}%，"
          f"思考触发率 {trig * 100:.1f}%，平均思考 {meta['think_tokens_mean']:.0f} token，"
          f"错误 {errors} 条")
    print(f"  已写入：{_features_path(bench_name)}、{csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="thinking 模式探针采集")
    ap.add_argument("--bench", default="truthfulqa",
                    help="基准名，逗号分隔")
    ap.add_argument("--n", type=int, default=0,
                    help="每个基准采集多少题；不填则用 DEFAULT_N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    benches = [b.strip() for b in args.bench.split(",") if b.strip()]
    torch.set_num_threads(min(8, torch.get_num_threads() or 8))

    path = _model_path()
    print(f"模型：{path}")
    tok = AutoTokenizer.from_pretrained(str(path))
    open_id = tok.convert_tokens_to_ids("<think>")
    close_id = tok.convert_tokens_to_ids("</think>")
    if open_id is None or close_id is None:
        raise SystemExit("tokenizer 里找不到 <think>/</think> 特殊 token")
    model = AutoModelForCausalLM.from_pretrained(str(path),
                                                 dtype=torch.float32)
    model.eval()
    print(f"加载完成：{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M 参数；"
          f"<think>={open_id} </think>={close_id}")

    for bench_name in benches:
        n = args.n or DEFAULT_N.get(bench_name, 100)
        if not args.force and _is_done(bench_name, n):
            print(f"\n跳过（已完成）：{bench_name}")
            continue
        collect_bench(model, tok, bench_name, n, args.seed, open_id, close_id,
                      args.max_new)


if __name__ == "__main__":
    main()
