"""probe_collect_thinking.py 的单元测试（重点：答案 token 与 scores 的对齐）。"""
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import probe_collect_thinking as pc  # noqa: E402

V = 10
OPEN, CLOSE = 8, 9


def test_split_with_think_block():
    ids = torch.tensor([OPEN, 1, 2, 3, CLOSE, 4, 5])
    think, ans, off = pc._split_think(ids, None, OPEN, CLOSE)
    assert think.tolist() == [1, 2, 3]
    assert ans.tolist() == [4, 5]
    # 答案第一个 token 的绝对索引 = </think> 位置 + 1
    assert off == 5


def test_split_without_think_block():
    ids = torch.tensor([4, 5])
    think, ans, off = pc._split_think(ids, None, OPEN, CLOSE)
    assert think.tolist() == []
    assert ans.tolist() == [4, 5]
    assert off == 0


def test_split_empty_answer():
    ids = torch.tensor([OPEN, 1, CLOSE])
    think, ans, off = pc._split_think(ids, None, OPEN, CLOSE)
    assert think.tolist() == [1]
    assert ans.tolist() == []
    assert off == 3


def _one_hot_scores(gen_ids, vocab=V, peak=20.0):
    """模拟 generate 的 scores：每一步给实际生成的 token 最高 logit。"""
    out = []
    for g in gen_ids:
        v = torch.zeros(vocab)
        v[g] = peak
        out.append(v.unsqueeze(0))
    return out


def test_conf_alignment_uses_absolute_index():
    """回归测试：旧实现对“思考长度+1”取偏移，会错位到 </think> 那一步。

    这里答案 token 在真实位置上有满分概率；只有索引正确时对数概率才接近 0。
    旧公式 off = n_t + 1 = 2 → 会去 scores[2]（即 </think> 的分布）里查
    答案 token 4，得到约 -20 的对数概率。
    """
    gen = [OPEN, 1, 2, CLOSE, 4, 5]
    ids = torch.tensor(gen)
    think, ans, off = pc._split_think(ids, None, OPEN, CLOSE)
    scores = _one_hot_scores(gen)

    logp = []
    for j in range(len(ans)):
        lp = torch.log_softmax(scores[off + j][0].float(), dim=-1)
        logp.append(float(lp[int(ans[j])]))
    assert all(p > -1e-3 for p in logp), logp

    wrong_off = len(think) + 1  # 旧实现的偏移
    wrong = float(
        torch.log_softmax(scores[wrong_off][0].float(), dim=-1)[int(ans[0])])
    assert wrong < -10, wrong


def test_model_path_prefers_existing_dir():
    p = pc._model_path()
    assert isinstance(p, Path)
