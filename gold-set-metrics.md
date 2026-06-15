# 核心指标速查 · Gold Set（10 项）

> **Gold Set**：研究者按环节预定义的该栈官方依据 + 必备检查项（如环节 1 CUDA：torch.add 文档、cuBLAS L1、Custom Op 教程；CANN：torch_npu 清单、CANN 配套表、Ascend C 流程）。配对任务共用**同构 checklist**，栈特有项分开列。每条配对建议必报 **Δ（CANN − CUDA）**。

**自动化程度：** **全自动** = 规则+日志/script 可复算 · **半自动** = 需预定义 Gold/checklist · **需标注** = 依赖人工或专家 rubric

**三层权重：** 执行层 30% · 内容层 45% · 输出层 25%

完整触点 tp-id 见 [touchpoint-metrics.md](touchpoint-metrics.md) · 实测见 [页面四 · s1-q1](llm-metrics-results.html?pair=s1-q1)

---

## 完整速查表（9 字段）

| 指标名称 | 中文名称 | 指标定义 | 计算方法 | 自动化程度 | 所属模块 | 所属层 | 触点指标 (tp-id) | 文献依据 |
|---------|---------|---------|---------|-----------|---------|--------|-----------------|---------|
| Recall@k | 检索召回率 | 前 k 次检索/引用中，命中**环节 Gold 文档集**的比例；衡量是否找全官方依据。 | Recall@k = \|Relevant ∩ Top-k\| / \|Relevant\| | 半自动 — 检索 log 可自动解析；Gold 文档集与相关性需预定义 | 检索深度与精准度 | 执行层 | **主对照** tp-id-001 资料/文档 · 检索命中率；同维度：tp-id-003 多入口可达率（资料）、tp-id-046 有效获取渠道数（工具） | Manning et al., 2008 经典 IR；Pirolli & Card, 1999 信息觅食 |
| MRR | 平均倒数排名 | 首个相关 Gold 文档排名的**倒数均值**；衡量检索排序是否把官方文档排前面。 | MRR = mean(1 / rank_first relevant) | 半自动 — 同 Recall@k，依赖 Gold 相关性判定 | 检索深度与精准度 | 执行层 | tp-id-002 文档跳转次数 · tp-id-004 单次任务文档跳转浏览次数 · tp-id-047 检索平均轮次（工具） | Manning et al., 2008 Mean Reciprocal Rank 标准定义 |
| NPL | 归一化路径长度 | 从提问到环节产出的**有效步骤数**相对专家最短路径的归一化长度。 | NPL = STS / STS_baseline | 半自动 — 步数可计 log；「有效步骤」与 baseline 需专家路径 | 路径效率 | 执行层 | tp-id-050 平均操作步数 · tp-id-067 版本部署命令数 · tp-id-068 版本切换配置修改量 · tp-id-071 反馈信息调取命令数 | Card et al., 1980 GOMS/KLM 最小操作路径；HCI 任务效率测度 |
| TTC | 到达结论耗时 | 从提问到出现**可停止结论**的时间（秒）或对话轮次。 | TTC = t_conclusion − t_query（或轮次数） | 半自动 — 时间戳/轮次客观；「结论出现」需 slot 规则或标注 | 多轮收敛 | 执行层 | 触点无时长型 tp-id；间接：tp-id-001 含搜索时长、tp-id-004 浏览次数 | Young et al., 2013 任务型对话；Mehri et al., 2019 USR；Miller, 1968 响应时间 |
| OSR | 官方信息占比 | 引用中来自**该栈官方一手源**的比例（非 CUDA 博客充当 CANN 依据）。 | OSR = #official_cites / #total_cites | 全自动 — 栈级域名白名单 + URL 抽取即可复算 | 官方信息占比 | 内容层 | tp-id-011 内/外链有效率；间接：有效链指向官方域比例 | CRAAP 信源评估框架；Epistemic trust；引用权威分析（bibliometrics） |
| VerAcc | 版本准确率 | 回答中**版本号、CANN/CUDA 配套关系**是否与官方 release notes、检索页一致；任务要求注明版本时必须给出可核查陈述。 | VerAcc = #correct_version_claims / #version_claims | 半自动 — 版本串可 NER 抽取；真伪对照 release notes / 检索 URL，深度分级常需标注 | 信息时效性与 API 幻觉 | 内容层 | tp-id-012 文档错误点位密度（版本类）· 版本配套关系准确性（易部署） | 软件版本兼容性理论；release notes 对照；实体级 P/R 评估范式 |
| Cov@checklist | 检查清单覆盖率 | 回答覆盖环节**栈特有检查项**（如 torch_npu 清单、CANN 配套）的比例。 | Cov@checklist = \|hit ∩ required\| / \|required\| | 半自动 — checklist 固定后可关键词/NER；深度分级（DOK）常需标注 | 知识覆盖率 | 内容层 | tp-id-006 FAQ 命中率 · tp-id-007 文档结构风格一致率 · tp-id-030 接口自解释读懂率 · tp-id-043 编程模型标准对齐率 · tp-id-075 错误码配套信息完整性 | Es et al., 2023 RAGAS Context Recall；集合覆盖率；Webb, 2002 DOK 深度分级 |
| pass@1 | 一次试跑成功率 | 回答中代码块复制到 sandbox **一次试跑**是否成功；衡量能否少改即跑。 | pass@1 = #run_ok / #code_blocks | 全自动 — sandbox 试跑，exit code 客观 | 可执行性与完整性 | 输出层 | tp-id-017 一次跑通成功率 · tp-id-048 首次安装/部署一次性成功率 · tp-id-049 前置依赖手动补充率（反向） | Chen et al., 2021 HumanEval pass@k；JTBD 任务完成度 |
| TTFD | 决策句前置度 | 单次回答中，**决策句**（自研/组合/等）首次出现的位置占全文比例。 | TTFD = tokens_decision / tokens_total | 半自动 — token 计数客观；决策句可用规则或双盲标注 | 快速给答 | 输出层 | 触点无 tp-id（LLM 回答结构专有） | Miller, 1968 人机对话响应时间；Grice 合作原则·量准则 |
| CSC | 代码简洁系数 | 同功能下，回答代码相对**参考最简实现**是否足够短、足够简单。 | CSC = LoC_ref / LoC_answer | 全自动 — 参考实现固定后 LoC 可脚本统计 | 代码简洁性 | 输出层 | tp-id-067 版本部署命令数 · tp-id-068 版本切换配置修改量 · tp-id-050 平均操作步数 | McCabe, 1976 圈复杂度；Clean Code 简洁原则；ISO/IEC 25010 可维护性 |

---

## 执行层（4 项 · 30%）

### Recall@k · 检索召回率

- **指标定义：** 前 k 次检索/引用中，命中**环节 Gold 文档集**的比例；衡量是否找全官方依据。
- **计算方法：** `Recall@k = |Relevant ∩ Top-k| / |Relevant|`
- **自动化程度：** 半自动 — 检索 log 可自动解析；Gold 文档集与相关性需预定义
- **所属模块：** 检索深度与精准度
- **所属层：** 执行层
- **触点指标 (tp-id)：** **主对照** tp-id-001 资料/文档 · 检索命中率；同维度：tp-id-003 多入口可达率（资料）、tp-id-046 有效获取渠道数（工具）
- **文献依据：** Manning et al., 2008 经典 IR；Pirolli & Card, 1999 信息觅食

### MRR · 平均倒数排名

- **指标定义：** 首个相关 Gold 文档排名的**倒数均值**；衡量检索排序是否把官方文档排前面。
- **计算方法：** `MRR = mean(1 / rank_first relevant)`
- **自动化程度：** 半自动 — 同 Recall@k，依赖 Gold 相关性判定
- **所属模块：** 检索深度与精准度
- **所属层：** 执行层
- **触点指标 (tp-id)：** tp-id-002 文档跳转次数 · tp-id-004 单次任务文档跳转浏览次数 · tp-id-047 检索平均轮次（工具）
- **文献依据：** Manning et al., 2008 Mean Reciprocal Rank 标准定义

### NPL · 归一化路径长度

- **指标定义：** 从提问到环节产出的**有效步骤数**相对专家最短路径的归一化长度。
- **计算方法：** `NPL = STS / STS_baseline`
- **自动化程度：** 半自动 — 步数可计 log；「有效步骤」与 baseline 需专家路径
- **所属模块：** 路径效率
- **所属层：** 执行层
- **触点指标 (tp-id)：** tp-id-050 平均操作步数 · tp-id-067 版本部署命令数 · tp-id-068 版本切换配置修改量 · tp-id-071 反馈信息调取命令数
- **文献依据：** Card et al., 1980 GOMS/KLM 最小操作路径；HCI 任务效率测度

### TTC · 到达结论耗时

- **指标定义：** 从提问到出现**可停止结论**的时间（秒）或对话轮次。
- **计算方法：** `TTC = t_conclusion − t_query`（或轮次数）
- **自动化程度：** 半自动 — 时间戳/轮次客观；「结论出现」需 slot 规则或标注
- **所属模块：** 多轮收敛
- **所属层：** 执行层
- **触点指标 (tp-id)：** 触点无时长型 tp-id；间接：tp-id-001 含搜索时长、tp-id-004 浏览次数
- **文献依据：** Young et al., 2013 任务型对话；Mehri et al., 2019 USR；Miller, 1968 响应时间

---

## 内容层（3 项 · 45%）

### OSR · 官方信息占比

- **指标定义：** 引用中来自**该栈官方一手源**的比例（非 CUDA 博客充当 CANN 依据）。
- **计算方法：** `OSR = #official_cites / #total_cites`
- **自动化程度：** 全自动 — 栈级域名白名单 + URL 抽取即可复算
- **所属模块：** 官方信息占比
- **所属层：** 内容层
- **触点指标 (tp-id)：** tp-id-011 内/外链有效率；间接：有效链指向官方域比例
- **文献依据：** CRAAP 信源评估框架；Epistemic trust；引用权威分析（bibliometrics）

### VerAcc · 版本准确率

- **指标定义：** 回答中**版本号、CANN/CUDA 配套关系**是否与官方 release notes、检索页一致；任务要求注明版本时必须给出可核查陈述。
- **计算方法：** `VerAcc = #correct_version_claims / #version_claims`
- **自动化程度：** 半自动 — 版本串可 NER 抽取；真伪对照 release notes / 检索 URL，深度分级常需标注
- **所属模块：** 信息时效性与 API 幻觉
- **所属层：** 内容层
- **触点指标 (tp-id)：** tp-id-012 文档错误点位密度（版本类）· 版本配套关系准确性（易部署）
- **文献依据：** 软件版本兼容性理论；release notes 对照；实体级 P/R 评估范式

### Cov@checklist · 检查清单覆盖率

- **指标定义：** 回答覆盖环节**栈特有检查项**（如 torch_npu 清单、CANN 配套）的比例。
- **计算方法：** `Cov@checklist = |hit ∩ required| / |required|`
- **自动化程度：** 半自动 — checklist 固定后可关键词/NER；深度分级（DOK）常需标注
- **所属模块：** 知识覆盖率
- **所属层：** 内容层
- **触点指标 (tp-id)：** tp-id-006 FAQ 命中率 · tp-id-007 文档结构风格一致率 · tp-id-030 接口自解释读懂率 · tp-id-043 编程模型标准对齐率 · tp-id-075 错误码配套信息完整性
- **文献依据：** Es et al., 2023 RAGAS Context Recall；集合覆盖率；Webb, 2002 DOK 深度分级

---

## 输出层（3 项 · 25%）

### pass@1 · 一次试跑成功率

- **指标定义：** 回答中代码块复制到 sandbox **一次试跑**是否成功；衡量能否少改即跑。
- **计算方法：** `pass@1 = #run_ok / #code_blocks`
- **自动化程度：** 全自动 — sandbox 试跑，exit code 客观
- **所属模块：** 可执行性与完整性
- **所属层：** 输出层
- **触点指标 (tp-id)：** tp-id-017 一次跑通成功率 · tp-id-048 首次安装/部署一次性成功率 · tp-id-049 前置依赖手动补充率（反向）
- **文献依据：** Chen et al., 2021 HumanEval pass@k；JTBD 任务完成度
- **备注：** ask 型环节（如环节 1）无代码块时常为 N/A

### TTFD · 决策句前置度

- **指标定义：** 单次回答中，**决策句**（自研/组合/等）首次出现的位置占全文比例。
- **计算方法：** `TTFD = tokens_decision / tokens_total`
- **自动化程度：** 半自动 — token 计数客观；决策句可用规则或双盲标注
- **所属模块：** 快速给答
- **所属层：** 输出层
- **触点指标 (tp-id)：** 触点无 tp-id（LLM 回答结构专有）
- **文献依据：** Miller, 1968 人机对话响应时间；Grice 合作原则·量准则
- **阈值参考：** TTFD ≤ 0.25 为优（前 25% 即给决策）

### CSC · 代码简洁系数

- **指标定义：** 同功能下，回答代码相对**参考最简实现**是否足够短、足够简单。
- **计算方法：** `CSC = LoC_ref / LoC_answer`
- **自动化程度：** 全自动 — 参考实现固定后 LoC 可脚本统计
- **所属模块：** 代码简洁性
- **所属层：** 输出层
- **触点指标 (tp-id)：** tp-id-067 版本部署命令数 · tp-id-068 版本切换配置修改量 · tp-id-050 平均操作步数
- **文献依据：** McCabe, 1976 圈复杂度；Clean Code 简洁原则；ISO/IEC 25010 可维护性
- **备注：** 环节 5–6 必评；无代码则 N/A

---

## 配对 Δ 报告建议

| 指标 | Δ 方向解读（CANN − CUDA） |
|------|---------------------------|
| Recall@k / MRR / OSR / VerAcc / Cov@checklist / pass@1 / CSC | 正值 = CANN 更优 |
| NPL / TTFD | 正值 = CANN 更差（路径更长 / 决策更晚） |
| TTC | 正值 = CANN 更慢 |
| VerAcc | 负值 = CANN 版本陈述更不准确或漏写 |

计分脚本：`python3 -m runner.metrics` · API：`GET /api/metrics/pair/s1-q1`
