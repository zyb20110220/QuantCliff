"""窗口 C 任务 C7：失效预警拦截器（Gradio 演示）。

做什么
    输入一道 4 选 1 题 → 模型先"不输出答案"，只用提示词末位隐藏激活给出
    红/黄/绿预警 → 再把答案输出出来。用来演示 RQ6 的结论。

诚实性设计
    1. 探针只在 TruthfulQA 上显著有效（折外 AUROC 0.693），所以本演示
       **默认加载 TruthfulQA 的题目**，并在界面上注明这一点。
    2. 三档阈值不是拍脑袋的 0.5，而是折外预测的 30/80 分位；界面上直接
       显示每一档的**覆盖率与实际正确率**，听众可以自己判断可信度。
    3. 自定义题目属于分布外输入，界面上会给出警告。

关于"实时输出"
    第二步支持**流式逐 token 输出**（transformers 的 TextIteratorStreamer），
    界面上能看到模型一字一句写完答案。注意：这是**生成过程**，不是内部推理过程。

    为什么不开 Qwen3 的 thinking 模式（`enable_thinking=True`）？
    开了之后必须走 chat template，提示词格式从此改变；而探针是在
    "裸提示词末位激活"这个分布上训练的。两者的隐藏状态分布不同，
    把旧探针的分打在新提示上会得到无效结果。要用 thinking 就得重采数据、
    重训探针，属于另一个实验，不能在这里混用。

启动
    python scripts/app.py
    然后浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from threading import Thread

import gradio as gr
import joblib
import numpy as np
import torch
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODEL_DIR = (REPO_ROOT / "hf_cache" / "hub" / "models--Qwen--Qwen3-0.6B"
             / "snapshots" / "c1899de289a04d12100db370d81485cdf75e47ca")

# 演示用的预设题库与对应探针
PRESET_BENCH = "truthfulqa"
PRESET_N = 500
SEED = 42
MAX_LEN = 2048

LIGHT_STYLE = {
    "green": ("#1f9d55", "可信", "模型大概率能答对"),
    "yellow": ("#d69e2e", "不确定", "不好说，建议人工复核"),
    "red": ("#e53e3e", "高风险", "模型大概率答错"),
}


def _model_path() -> Path:
    p = MODEL_DIR
    return p if p.exists() else Path("Qwen/Qwen3-0.6B")


# ---------------------------------------------------------------- 载入资源

class Interceptor:
    """把模型、探针、判分器打包在一起，页面回调只调它。"""

    def __init__(self, probe_bench: str, n_preset: int):
        self.bench_name = probe_bench
        self.bench = importlib.import_module(f"benchmarks.{probe_bench}")

        mp = RESULTS / f"probe_model_{probe_bench}.joblib"
        jp = RESULTS / f"probe_model_{probe_bench}.json"
        if not mp.exists() or not jp.exists():
            raise SystemExit(
                f"找不到探针模型：{mp}\n"
                "请先运行：python scripts/probe_train.py --save-model")
        bundle = joblib.load(mp)
        self.pipe = bundle["pipeline"]
        self.layer = int(bundle["layer"])
        self.meta = json.loads(jp.read_text(encoding="utf-8"))

        path = _model_path()
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.float32)
        self.model.eval()
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id

        # 预设题目：同一 seed 重新生成，与探针训练时完全一致
        self.items = self.bench.load(n_preset, SEED)
        self.by_id = {str(it.get("id")): it for it in self.items}

    # ---- 第一步：只算预警（完全不生成任何 token）----
    def score_only(self, item: dict) -> dict:
        """只对提示词做一次前向，取末位隐藏激活算探针分。

        这里**不做任何生成**：编码器输入的只有提示词，返回值里也没有答案。
        界面上"先出灯"用的就是这条路径，所以"看到答案再判断"在物理上不可能。
        """
        prompt = self.bench.build_prompt(item)
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_LEN)
        with torch.inference_mode():
            fwd = self.model(**enc, output_hidden_states=True)
        feat = (fwd.hidden_states[self.layer][0, -1, :]
                .float().numpy().reshape(1, -1))
        probe_p = float(self.pipe.predict_proba(feat)[0, 1])
        return {"probe": probe_p,
                "prompt_tokens": int(enc["input_ids"].shape[1])}

    # ---- 第二步：生成答案（与预警完全独立）----
    def answer_only(self, item: dict) -> dict:
        prompt = self.bench.build_prompt(item)
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_LEN)
        n_prompt = int(enc["input_ids"].shape[1])

        with torch.inference_mode():
            gen = self.model.generate(
                **enc, max_new_tokens=self.bench.MAX_TOKENS, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.pad_id)
        new_ids = gen.sequences[0][n_prompt:]
        output = self.tok.decode(new_ids, skip_special_tokens=True).strip()

        logp = []
        for step, score in enumerate(gen.scores):
            if step >= len(new_ids):
                break
            lp = torch.log_softmax(score[0].float(), dim=-1)
            logp.append(float(lp[int(new_ids[step])]))
        conf = float(np.exp(np.mean(logp))) if logp else float("nan")

        res = self.bench.judge(output, item)
        return {"conf": conf, "answer": output,
                "correct": bool(res["correct"]),
                "parse_ok": bool(res["parse_ok"]),
                "prompt_tokens": n_prompt}

    # ---- 第二步（流式）：逐 token 吐出答案 ----
    def answer_stream(self, item: dict):
        """生成答案并逐个 token 返回。

        产出 (累积文本, 最终结果)：
          - 生成过程中：最终结果为 None，累积文本不断变长；
          - 生成结束后：产出一份带 conf / correct 的完整结果。
        这叫"生成过程"，**不是**模型的内部思考过程——本模型走裸提示词，
        不会输出  thinking 段落（原因见模块文档）。
        """
        prompt = self.bench.build_prompt(item)
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_LEN)
        n_prompt = int(enc["input_ids"].shape[1])
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True,
                                        skip_special_tokens=True)

        holder: dict = {}

        def _worker():
            with torch.inference_mode():
                holder["out"] = self.model.generate(
                    **enc, max_new_tokens=self.bench.MAX_TOKENS,
                    do_sample=False, streamer=streamer,
                    output_scores=True, return_dict_in_generate=True,
                    pad_token_id=self.pad_id)

        thread = Thread(target=_worker, daemon=True)
        thread.start()

        partial = ""
        for chunk in streamer:
            partial += chunk
            yield partial, None
        thread.join()

        out = holder.get("out")
        if out is None:  # 生成异常时给出可读的保底结果
            yield partial, {"conf": float("nan"), "answer": partial.strip(),
                            "correct": False, "parse_ok": False,
                            "prompt_tokens": n_prompt}
            return

        new_ids = out.sequences[0][n_prompt:]
        answer = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        logp = []
        for step, score in enumerate(out.scores):
            if step >= len(new_ids):
                break
            lp = torch.log_softmax(score[0].float(), dim=-1)
            logp.append(float(lp[int(new_ids[step])]))
        conf = float(np.exp(np.mean(logp))) if logp else float("nan")
        res = self.bench.judge(answer, item)
        yield answer, {"conf": conf, "answer": answer,
                       "correct": bool(res["correct"]),
                       "parse_ok": bool(res["parse_ok"]),
                       "prompt_tokens": n_prompt}

    # ---- 一次跑完（内部就是上面两步，供脚本调用）----
    def _run(self, item: dict) -> dict:
        warn = self.score_only(item)
        ans = self.answer_only(item)
        return {**warn, **ans}

    def band_of(self, p: float) -> str:
        if p >= self.meta["thr_green"]:
            return "green"
        if p < self.meta["thr_yellow"]:
            return "red"
        return "yellow"


INTER: Interceptor | None = None


# ---------------------------------------------------------------- 页面

def _light_html(band: str, p: float) -> str:
    color, label, hint = LIGHT_STYLE[band]
    dim = "#d9d9d9"
    circles = []
    for key in ("red", "yellow", "green"):
        c = color if key == band else dim
        circles.append(
            f"<div style='width:52px;height:52px;border-radius:50%;"
            f"background:{c};margin:0 8px;display:inline-block;"
            f"box-shadow:{'0 0 14px ' + c if key == band else 'none'}'></div>")
    return (
        "<div style='text-align:center;padding:12px 0'>"
        + "".join(circles)
        + f"<div style='margin-top:10px;font-size:20px;font-weight:700;"
          f"color:{color}'>{label}</div>"
        + f"<div style='color:#555;font-size:13px'>{hint}</div>"
        + f"<div style='color:#888;font-size:12px;margin-top:6px'>"
          f"探针分 P(答对) = {p:.3f}</div>"
        "</div>")


def _clean_answer(output: str) -> str:
    """把模型生出的脏文本压成一行好看的答案。

    0.6B 模型用裸提示词时经常复读模板（例如 'C\\nAnswer:\\nC'），
    直接展示很难看。这里只取第一个字母选项，拿不到就返回前 60 字。
    原始文本同时保留在界面上（折叠区），不做任何隐藏。
    """
    if not output or not output.strip():
        return "（空）"
    for token in output.replace("\n", " ").split():
        t = token.strip(" .,:;`'\"()[]")
        if len(t) == 1 and t.upper() in "ABCD":
            return t.upper()
    first = output.split("\n")[0].strip()
    return first[:60] if first else "（空）"


def _detail_md(r: dict, meta: dict, show_truth: str | None) -> str:
    band = "green" if r["probe"] >= meta["thr_green"] else (
        "red" if r["probe"] < meta["thr_yellow"] else "yellow")
    b = meta["bands"][band]
    lines = [
        f"**模型答案**：{_clean_answer(r['answer'])}",
        "",
        f"- 探针分：**{r['probe']:.3f}**　置信度（生成 token 平均概率）："
        f"{r['conf']:.3f}",
        f"- 提示词长度：{r['prompt_tokens']} tokens　"
        f"解析成功：{'是' if r['parse_ok'] else '否'}",
        "",
        f"> 本档（{'绿' if band == 'green' else '黄' if band == 'yellow' else '红'}灯）"
        f"在折外评估中覆盖 {b['coverage'] * 100:.0f}% 的题（{b['n']} 道），"
        f"**实际正确率 {b['accuracy'] * 100:.0f}%**。"
        f"题库整体基线为 33%。",
    ]
    if show_truth is not None:
        lines += ["", f"**标准答案**：{show_truth}　"
                      f"模型本题{'答对 ✅' if r['correct'] else '答错 ❌'}"]
    # 原始输出不隐藏，只在折叠区展示
    raw = (r["answer"] or "").replace("\n", " ")[:200]
    lines += ["", "<details><summary>模型的原始输出（未清洗）</summary>", "",
              f"<code>{raw}</code>", "", "</details>"]
    return "\n".join(lines)


def _warn_md(warn: dict, band: str, meta: dict) -> str:
    """第一步的结果：只含预警，不含任何答案信息。"""
    b = meta["bands"][band]
    label = {"green": "绿", "yellow": "黄", "red": "红"}[band]
    return "\n".join([
        f"**预警档位**：{label}灯（{LIGHT_STYLE[band][1]}）",
        "",
        f"- 探针分 P(答对)：**{warn['probe']:.3f}**",
        f"- 提示词长度：{warn['prompt_tokens']} tokens",
        "",
        f"> 本档在折外评估中覆盖 {b['coverage'] * 100:.0f}% 的题（{b['n']} 道），"
        f"**实际正确率 {b['accuracy'] * 100:.0f}%**。题库整体基线为 33%。",
        "",
        "> ⚙️ 这一步**只对提示词做了一次前向**，没有调用任何生成，"
        "返回值里不包含答案——所以预警不可能依据答案得出。"
        "点下面的第二步按钮，再让模型把它的话说出来。",
    ])


def _render_item(item: dict) -> str:
    """把一道题完整渲染成 Markdown（问题 + 四个选项）。

    下拉框里只能放短标签，长题目会被截断；所以单独开一块预览区
    展示完整内容，避免"看不到题目是什么"。
    """
    if not item or item.get("question") is None:
        return ""
    lines = [f"**题目 #{item.get('id')}**", "", str(item["question"])]
    opts = list(item.get("options") or [])
    if opts:
        lines.append("")
        for letter, opt in zip("ABCD", opts):
            lines.append(f"- **{letter}.** {opt}")
    return "\n".join(lines)


# ---- 第一步：只出预警 ----

def warn_preset(item_id: str):
    if INTER is None:
        return "<p>模型未加载</p>", ""
    item = INTER.by_id.get(str(item_id))
    if item is None:
        return "<p>找不到该题</p>", ""
    w = INTER.score_only(item)
    band = INTER.band_of(w["probe"])
    return _light_html(band, w["probe"]), _warn_md(w, band, INTER.meta)


def warn_custom(question: str, a: str, b: str, c: str, d: str):
    item, err = _custom_item(question, a, b, c, d)
    if err:
        return f"<p>{err}</p>", ""
    w = INTER.score_only(item)
    band = INTER.band_of(w["probe"])
    note = ("\n\n> ⚠️ 自定义题目属于**分布外输入**：探针在 TruthfulQA 上训练，"
            "换题型后预警可能失效（跨基准泛化 AUROC 仅 0.46）。")
    return _light_html(band, w["probe"]), _warn_md(w, band, INTER.meta) + note


def _streaming_md(text: str) -> str:
    """生成过程中的临时展示：只显示最近一段，避免越长越刷屏。"""
    shown = text.strip()
    if not shown:
        return "**模型正在生成（逐 token）…**"
    tail = shown[-300:]
    return ("**模型正在生成（逐 token）…**\n\n"
            f"```text\n{tail}\n```")


def _answer_head(answer: str, correct: bool | None) -> str:
    """最终答案那一行的标题文案。"""
    head = f"**模型答案**：{_clean_answer(answer)}"
    if correct is not None:
        head += "　" + ("答对 ✅" if correct else "答错 ❌")
    return head


# ---- 第二步：流式生成答案 ----

def answer_preset(item_id: str):
    if INTER is None:
        yield "模型未加载", ""
        return
    item = INTER.by_id.get(str(item_id))
    if item is None:
        yield "找不到该题", ""
        return
    warn = INTER.score_only(item)  # 复核一次，保证展示的档位与探针一致
    truth = str(item.get("answer"))
    for text, final in INTER.answer_stream(item):
        if final is None:
            yield _streaming_md(text), ""
        else:
            r = {**warn, **final}
            yield (_answer_head(final["answer"], final["correct"]),
                   _detail_md(r, INTER.meta, truth))


def answer_custom(question: str, a: str, b: str, c: str, d: str):
    item, err = _custom_item(question, a, b, c, d)
    if err:
        yield f"{err}", ""
        return
    warn = INTER.score_only(item)
    for text, final in INTER.answer_stream(item):
        if final is None:
            yield _streaming_md(text), ""
        else:
            r = {**warn, **final}
            yield (_answer_head(final["answer"], None),
                   _detail_md(r, INTER.meta, None))


def _custom_item(question: str, a: str, b: str, c: str, d: str):
    if INTER is None:
        return None, "模型未加载"
    opts = [x.strip() for x in (a, b, c, d)]
    if not question.strip() or any(not o for o in opts):
        return None, "请填完整问题和四个选项"
    return {"question": question.strip(), "options": opts}, ""


def build_ui(probe_bench: str, n_preset: int) -> gr.Blocks:
    global INTER
    INTER = Interceptor(probe_bench, n_preset)
    meta = INTER.meta

    choices = []
    for it in INTER.items:
        q = str(it.get("question", ""))
        # 超过 52 字加省略号，让"被截断"这件事在界面上看得出来
        label = q if len(q) <= 52 else q[:52] + "…"
        choices.append((f"#{it.get('id')}  {label}", str(it.get("id"))))

    summary = (
        f"### 当前探针\n"
        f"- 训练题池：`{probe_bench}`（{meta['n_train']} 题），"
        f"使用第 **L{meta['layer']}** 层末位 token 激活\n"
        f"- 折外 AUROC：**{meta['auroc_oof']:.3f}** "
        f"[{meta['auroc_ci'][0]:.3f}, {meta['auroc_ci'][1]:.3f}]；"
        f"对照：置信度基线 {meta['auc_conf']:.3f}、题型难度基线 {meta['auc_group']:.3f}\n"
        f"- 阈值：绿灯 p≥{meta['thr_green']:.3f}，红灯 p<{meta['thr_yellow']:.3f}\n\n"
        "| 档位 | 覆盖率 | 实际正确率 |\n|---|---:|---:|\n"
        f"| 绿 | {meta['bands']['green']['coverage'] * 100:.0f}% | "
        f"{meta['bands']['green']['accuracy'] * 100:.0f}% |\n"
        f"| 黄 | {meta['bands']['yellow']['coverage'] * 100:.0f}% | "
        f"{meta['bands']['yellow']['accuracy'] * 100:.0f}% |\n"
        f"| 红 | {meta['bands']['red']['coverage'] * 100:.0f}% | "
        f"{meta['bands']['red']['accuracy'] * 100:.0f}% |\n\n"
        "基线（全题库正确率）约 33%。"
    )

    with gr.Blocks(title="QuantCliff 失效预警拦截器") as demo:
        gr.Markdown("# 失效预警拦截器\n"
                    "模型先**不输出答案**，只用提示词末位的隐藏层激活判断"
                    "「这题它大概率会不会答错」。")
        with gr.Accordion("当前探针的统计表现", open=False):
            gr.Markdown(summary)

        with gr.Tab("预设题目（推荐）"):
            dd = gr.Dropdown(choices=choices,
                             value=choices[0][1] if choices else None,
                             label="从 TruthfulQA 题库中选择")
            # 完整题目预览：下拉框只能放短标签，这里展示全文：
            preview = gr.Markdown(
                _render_item(INTER.items[0]) if INTER.items else "")
            with gr.Row():
                btn1 = gr.Button("① 先给预警（不输出答案）", variant="primary")
                btn1b = gr.Button("② 再显示模型答案（流式）")
            out1 = gr.HTML()
            det1 = gr.Markdown()
            ans1 = gr.Markdown()
            det1b = gr.Markdown()
            dd.change(lambda i: _render_item(INTER.by_id.get(str(i), {})),
                      inputs=[dd], outputs=[preview])
            btn1.click(warn_preset, inputs=[dd], outputs=[out1, det1])
            btn1b.click(answer_preset, inputs=[dd], outputs=[ans1, det1b])

        with gr.Tab("自定义题目（分布外）"):
            q = gr.Textbox(label="问题", lines=2,
                           placeholder="请输入 4 选 1 的问题")
            with gr.Row():
                a = gr.Textbox(label="A")
                b = gr.Textbox(label="B")
            with gr.Row():
                c = gr.Textbox(label="C")
                d = gr.Textbox(label="D")
            with gr.Row():
                btn2 = gr.Button("① 先给预警（不输出答案）", variant="primary")
                btn2b = gr.Button("② 再显示模型答案（流式）")
            out2 = gr.HTML()
            det2 = gr.Markdown()
            ans2 = gr.Markdown()
            det2b = gr.Markdown()
            btn2.click(warn_custom, inputs=[q, a, b, c, d],
                       outputs=[out2, det2])
            btn2b.click(answer_custom, inputs=[q, a, b, c, d],
                        outputs=[ans2, det2b])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="失效预警拦截器演示")
    ap.add_argument("--bench", default=PRESET_BENCH,
                    help="用哪个基准的题库与探针")
    ap.add_argument("--n-preset", type=int, default=PRESET_N)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成临时公网链接")
    args = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads() or 8))
    demo = build_ui(args.bench, args.n_preset)
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                share=args.share, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
