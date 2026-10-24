# 过程证据记录

> 本文件记录项目全过程的关键决策、修订与阶段性小结，作为综评过程性材料。每阶段结束必须追加小结（带日期、耗时、截图、遇到的问题）。

## 决策记录（2026-09-26 定稿）

- 项目：QuantCliff《小模型量化能力悬崖的定位研究》，面向综评材料 + 答辩演示
- 主体模型 Qwen3-1.7B；验证模型 Qwen2.5-1.5B、Gemma-3-1B（目的是证实规律而非对比差异）
- 7 档全跑：Q2_K/Q3_K_M/Q4_K_M/Q5_K_M/Q6_K/Q8_0/F16
- 5 类能力 × 500 题/档：MMLU-Pro 150 + TruthfulQA 100 + IFEval 100 + C-Eval 100 + needle 4K×20 + 中文事实 30
- greedy 采样全局固定；口径用相对退化率
- 自写评测循环（不用 lm-eval-harness）；数据走 hf-mirror.com
- 成品：Gradio 网页 + docs/report.md；论文最后写
- 云端：Kaggle 备用（仅 Bonsai 探索章节）
- 时间窗口：国庆 1 周（A）+ 寒假 4 周（B）+ 暑假 8 周（C）
- 代码分工：协作者代写全部代码，本人负责运行、改参数、理解

## 三项融合修订（2026-09-27 定稿）

- 主题升级：QuantCliff《小模型的能力边界与失效预警》（仓库名不变，副标题与文档更新）
- 三部分：主研究=量化临界点（原方案不变，新增中文错字率指标）；实验 2=三元能力退化图谱；实验 3=开口前失效预警探针
- 实验 2 定稿：`prism-ml/Ternary-Bonsai-1.7B`（底模 Qwen3-1.7B，Q2_0 436 MiB）全套 500 题必做；8B 版 180 题（MMLU-Pro 150 + 中文事实 30）加分；dense 8B 基线用 Q8_0
- 实验 3 定稿：冻结 Qwen3-0.6B，隐藏层激活 + sklearn 逻辑回归探针，2000 样本，按题切分 + 跨基准泛化；指标 AUROC/ECE + DeLong vs 置信度基线；1.7B 探针窗口 C 末加分
- 窗口 A 新增 A5：transformers 加载 0.6B smoke test（5 分钟排雷）
- 2026-09-27 HF 事实核查：Bonsai 官方 F16 与 Q2_0 是同一组三元权重的两种存储（KL≈0，互比无效）；Q2_0 不在主线 llama.cpp（需 Prism fork，备选 `onnx-community/Ternary-Bonsai-1.7B-ONNX`）；官方 8B 数据（avg 79.3→75.5）显示训练式三元无悬崖，为 H4 提供先验
- 降级路径（窗口 C 紧张时依次执行）：三元题量砍半 → 弃 8B 加分 → 弃 1.7B 探针；主线三部分不可砍

## 研究问题修订记录

- 初版 RQ 直接问"悬崖在哪"，预设了"存在悬崖"
- 2026-09-26 修订：RQ1 改为开放的"衰减形态"问题（均匀 vs 悬崖，两种结果均为发现）；"先崩"改为"敏感性差异"；配套统计方法升级为形态检验（线性 vs 分段拟合 AIC/BIC + 二阶差分）→ 变化点检测 → 维度排序 → Kendall W 一致性
- 意义：设计阶段即修正了提问预设，保证无论实验结果如何均有结论

## 阶段小结模板

```markdown
## 窗口X 任务X.N 小结（日期）

- 做了什么：
- 耗时：
- 结果/截图：
- 遇到的问题与解决：
- 下一步：
```

## 会话恢复要点（2026-09-26）

> 若对话丢失，新会话阅读本节 + docs/proposal.md 即可完全恢复上下文。

**项目身份**：QuantCliff《小模型的能力边界与失效预警》（2026-09-27 由《小模型量化能力悬崖的定位研究》扩展）；仓库 https://github.com/zyb20110220/QuantCliff；本地 C:\Users\zyb\source\repos\QuantCliff。

**已定决策（grill 三轮全部落定）**：
- 主模型 Qwen3-1.7B；验证 Qwen2.5-1.5B、Gemma-3-1B（目的：证实规律而非对比差异；口径：相对退化率）
- 7 档全跑：Q2_K/Q3_K_M/Q4_K_M/Q5_K_M/Q6_K/Q8_0/F16
- 每档 500 题：MMLU-Pro 150（10 选 1 裁剪为 4 选 1，seed=42）+ TruthfulQA 100 + IFEval 100 + C-Eval 100 + needle 4K×20 + 中文事实 30
- greedy 采样全局固定；自写评测循环（不用 lm-eval-harness）；下载走 hf-mirror.com
- 成品：Gradio 网页 + docs/report.md；论文最后写
- 时间：国庆 1 周（A）+ 寒假 4 周（B）+ 暑假 8 周（C）
- 代码分工：协作者代写全部代码，本人运行/改参/理解
- .gitignore 排除 data/ models/（+ pycache/venv 技术排除），其余全提交；每次改动即 commit（纯标题：`type: 窗口X任务X.N - 描述`）
- 三项融合（2026-09-27）：主题升级《小模型的能力边界与失效预警》；实验 2 三元图谱（Ternary-Bonsai-1.7B 全套 500 题必做，8B 版 180 题加分，dense 基线 Q8_0）；实验 3 探针（Qwen3-0.6B 冻结 + 隐藏层激活 + 逻辑回归，2000 样本按题切分，1.7B 加分）；主研究新增中文错字率指标；窗口 A 新增 A5 探针路线排雷

**RQ 体系（2026-09-27 扩充为 6 项）**：RQ1 衰减形态（开放，不预设悬崖）/ RQ2 维度敏感性差异 / RQ3 跨模型一致性 / RQ4 实用边界 / RQ5 三元 vs PTQ 曲线 / RQ6 探针预警能力与尺度。
统计：线性 vs 分段拟合（AIC/BIC）+ 二阶差分 → PELT 变化点 + McNemar → 退化速率排序 → Kendall W；RQ5 三元点 vs 延长线偏差 + 配对 McNemar；RQ6 AUROC/ECE + DeLong。

**当前进度**：
- 完成：窗口 A 全部任务（A1~A5 + 小结）✅。2026-10-01 实测：环境 Python 3.11.15（uv 管理）+ 92 包；模型 Q4_K_M 1.19GB / f16 3.79GB 存仓库根目录 models/；最小闭环 Q4 44% vs F16 56%（各 50 题，解析成功率 100%）；单元测试 7/7；探针路线验证通过（transformers 5.18.0 + Qwen3-0.6B）。
- 完成：窗口 B 准备期（2026-10-17）✅。6 个基准模块齐全；eval_loop 支持断点续跑/容错/归档；analyze.py 可出汇总表与中文文本质量指标；30 条中文事实题审核完毕；下载矩阵 22 个文件远端核对通过；单元测试 46/46。
- 完成：窗口 B 任务 B1（2026-10-24）✅。22 个模型文件全部下载并核对（27.41 GiB）；6 个代表档位兼容性冒烟全过；Bonsai Q2_0 g64 通过 Prism CLI 跑通端到端。
- 下一步（寒假窗口 B）：B2 主模型 7 档 × 500 题 → B3 两个验证模型 → B4 用 analyze.py 出汇总 + 补形态检验（线性 vs 分段 AIC/BIC、PELT 变化点、相邻档 McNemar）+ 初版 report.md。
- 暑假窗口 C：三元图谱（1.7B 必做/8B 加分）+ 探针（0.6B 必做/1.7B 加分）+ 网页 + 报告答辩。

**环境要点**：git push 走仓库本地代理 127.0.0.1:7897（勿改全局）；模型/数据/HF 缓存统一放仓库根目录（models/、data/、hf_cache/，均已 git 排除；C 盘空间已足够，2026-09-27 决定，不再拆 D 盘）；用户 Python 不佳（代码需中文注释、可运行优先）。

## 各阶段小结

### 窗口A 小结（2026-10-01）

- 做了什么：A2 环境（Python 3.11.15 venv + 92 包，uv 管理）、A3 下载两档模型（Q4_K_M 1.19GB + f16 3.79GB，hf-mirror + Xet 高速）、A4 最小闭环（2 档 × 50 题 + 单元测试 7/7 全绿）、A5 探针排雷（transformers 5.18.0 加载 Qwen3-0.6B 成功，28 层）
- 结果：Q4_K_M 22/50 = 44%（解析成功 50/50）；F16 28/50 = 56%（解析成功 50/50）；F16 明显高于 Q4，验收通过 ✅
- 耗时：环境约 15 分钟、下载约 10 分钟（含一次中断重试）、最小闭环约 4 分钟（Q4 1.9s/题、F16 3.5s/题）、探针排雷约 2 分钟
- 遇到的问题与解决：
  1. 首次模型下载被中断（~119MB）→ 重启 + 开启 HF_XET_HIGH_PERFORMANCE，速度 0.8MB/s → 14~20MB/s
  2. MMLU-Pro 数据越界：代码硬编码 10 选项，实际数据集选项数 3~10 不等（仅 83% 是 10 选项）→ 改为按每题实际选项数动态裁剪 4 选 1，冒烟验证通过
  3. 实际文件名 `Qwen3-1.7B-f16.gguf`（小写 f16），与文档示例 F16 不一致 → 已按实际文件名执行，后续全量实验照此
  4. transformers 装到 5.18.0（远新于预期 4.44）→ A5 验证兼容，探针路线可行
- 截图：（待补充：环境安装成功、模型文件列表、两档正确率、测试全绿、0.6B 加载成功）
- 下一步：进入冻结期（课业优先）；轻量任务=审核 proposal.md 附录 30 条中文事实题；寒假窗口 B 前待办=eval_loop 断点续跑、补齐 5 个基准模块 + analyze.py（含中文错字率）

### 窗口B 准备期小结（2026-10-17）

- 做了什么：
  1. 补齐 5 个基准模块：`truthfulqa`（817 题转 4 选 1）、`ifeval`（10 类可判指令 × 10 主题 = 100 题）、`ceval`（4 学科轮流抽 100 题）、`needle`（约 4K tokens 长上下文 × 20 题）、`chinese_facts`（30 题）；加上已有 `mmlu_pro`，六个基准接口统一。
  2. `eval_loop.py` 增强：`--resume` 断点续跑；逐题异常兜底（新增 `error` 列）；不加 `--resume` 时旧 CSV 自动改名归档；每个结果旁生成 `.meta.json` 运行元数据。
  3. 新增 `analyze.py`：按基座 × 档位 × 基准汇总正确率、解析率、Wilson 95% CI、相对退化率；对中文输出计算乱码率、3-gram 重复率、重复行率；输出 CSV + Markdown。
  4. 审核 30 条中文事实题：仅第 13 题需修订（补充"按质量计"，避免与按原子数计的氢歧义），其余事实无误；判分器支持同义词、全角/半角、数字边界与"最后出现位置"规则。
  5. 新增 `download_matrix.py`：3 个主模型 × 7 档 + 三元 Q2_0，共 22 个文件；默认只预览，`--verify-remote` 已全部通过，`--execute` 留给寒假 B1。
- 关键发现（重要）：
  1. `ggml-org/Qwen3-1.7B-GGUF` 与 `ggml-org/gemma-3-1b-it-GGUF` 实际只有 Q4_K_M / Q8_0 / F16 三档，`ggml-org/Qwen2.5-1.5B-Instruct-GGUF` 不存在 → 正式实验改用 `unsloth/Qwen3-1.7B-GGUF`、`Qwen/Qwen2.5-1.5B-Instruct-GGUF`、`unsloth/gemma-3-1b-it-GGUF`（均 7 档）。
  2. 不同仓库的同名 Q4_K_M 大小/哈希不同（ggml-org 1223 MiB vs unsloth 1056 MiB）→ 同一模型只用一个仓库，下载脚本按模型建子目录，避免混用。
  3. Qwen3 / Gemma 的全档位仓库只提供 BF16 作为高精度基线 → 分析脚本把 BF16/F16/fp16 统一记作 F16 档，报告注明。
  4. `datasets 5.0.1` 已不支持脚本型数据集；`truthfulqa/truthful_qa` 与 `ceval/ceval-exam` 的 parquet 版本可直接加载。C-Eval 学科名是 `high_school_mathematics`（易误写成 `high_school_math`）。
- 验证结果：
  - 单元测试 46/46 全绿（窗口 A 为 7 项，本次新增 39 项）
  - 真实数据 smoke：chinese_facts 30 / ifeval 100 / needle 3 / truthfulqa 3 / ceval 100 全部加载成功
  - Q4 中文事实 2 题端到端跑通；第二次 `--resume` 正确跳过，补跑 0 题
  - `analyze.py` 汇总窗口 A 结果：Q4 44.00% vs F16 56.00%，Q4 在 MMLU-Pro 上相对退化 21.43%
- 遇到的问题与解决：
  1. C-Eval 学科名 `high_school_math` 不存在 → 改为 `high_school_mathematics`
  2. chinese_facts 的标准答案是列表，直接写 CSV 不整洁 → 统一用 ` | ` 拼接
  3. 总体相对退化率在"某档多跑了别的基准"时会失真 → 改为仅当该档位与 F16 的基准集合完全一致时才计算总体值，分基准值不受影响
- 下一步：寒假 B1 下载全量矩阵；B2 主模型 7 档全量；B3 两个验证模型；B4 分析与初版报告。

### 窗口B 任务B1 小结（2026-10-24）

- 做了什么：下载全部 22 个模型文件（Qwen3-1.7B / Qwen2.5-1.5B / Gemma-3-1B 各 7 档 + Ternary-Bonsai Q2_0 g64），合计 **27.41 GiB**；逐文件与远端大小比对 **22/22 通过**，生成 `results/model_manifest.csv`。
- 下载问题与解决：
  1. 直连 `hf-mirror.com` 出现 SSL 中断（`UNEXPECTED_EOF_WHILE_READING`），Q3 卡在 510MB。
  2. 改用本地代理 `127.0.0.1:7897` + 官方 `huggingface.co`，关闭 Xet，Q3 从断点续传到 896 MiB 成功。
  3. `download_matrix.py` 已加入失败重试、断点续传、`--verify-local` 清单核对。
- 兼容性冒烟（6 个代表档位）：
  - Qwen3 Q2_K / Q4_K_M / BF16、Qwen2.5 Q2_K、Gemma-3 Q2_K 全部通过；
  - Bonsai Q2_0 主线 llama.cpp 不支持；最新 Prism 构建又要求 **group-64**，旧 g128 文件加载失败；
  - 改用 Prism 官方预编译 Windows CPU 运行时（build 10787）+ `Ternary-Bonsai-1.7B-Q2_0_g64.gguf`，通过。
- 新增能力：`eval_loop.py --backend prism_cli`，用 `llama-completion.exe` 生成；stdout 取文本，stderr 取 token 数。Bonsai 2 题端到端 **2/2 正确**，约 4.5s/题（每题重新加载模型）。
- 关键数字：22/22 文件通过；27.41 GiB；代表档位冒烟 6/6；全 22 档加载测试 0 失败；Bonsai 2/2。
- 下一步：B2 主模型 7 档 × 500 题全量实验。
