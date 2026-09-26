# QuantCliff

## 《小模型量化能力悬崖的定位研究》

在三个开源小模型（Qwen3-1.7B / Qwen2.5-1.5B / Gemma-3-1B）上，用统一的 **7 个量化档位 × 5 类能力 × 500 题** 的对照实验，研究量化位宽与能力退化之间的函数形态：是否存在"悬崖式"崩塌、哪类能力更敏感、规律是否跨模型成立。

## 文档

- [开题方案](docs/proposal.md)
- [实验报告](docs/report.md)（寒假窗口产出）
- [过程证据](docs/evidence.md)

## 目录结构

```
QuantCliff/
├── docs/          # 全部 Markdown 文档
├── scripts/       # 实验与演示代码
├── results/       # 原始结果 CSV（过程证据，全部提交）
├── tests/         # 判分逻辑单元测试
├── data/          # 基准数据（git 排除，D 盘）
└── models/        # GGUF 模型（git 排除，D 盘）
```

## 环境

- Windows + CPU 推理（llama-cpp-python，AVX2）
- Python 3.11，依赖见 `requirements.txt`
- 数据/模型下载走 hf-mirror.com 镜像

## 进度

| 窗口 | 时间 | 任务 | 状态 |
|---|---|---|---|
| A | 2026 国庆 1 周 | 建仓、环境、最小闭环 | 进行中 |
| B | 2027 寒假 4 周 | 全量实验 + 分析 | 未开始 |
| C | 2027 暑假 8 周 | 网页、实验2、报告、答辩 | 未开始 |
