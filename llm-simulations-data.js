/* 环节 1 · 判断是否需要自研算子 — 大模型模拟流程数据 */
const LLM_SIMS = {
  'cuda-s1-q1': {
    stack: 'CUDA', scene: '1 · 查官方文档', type: 'ask',
    prompt: '帮我在 PyTorch 官方文档里查 vector add 有没有原生算子，并给出对应文档链接。',
    outcome: '结论：PyTorch 已有原生实现，Hello World 级 vector add 无需自研 → 可停在本环节。',
    steps: [
      { kind: 'think', text: '用户要做入门级 vector add。先确认 PyTorch 是否提供等价原生 API，再定位官方文档章节。' },
      { kind: 'search', text: '检索 PyTorch 官方文档：torch.add、torch.Tensor.add_、element-wise addition' },
      { kind: 'tool', text: '打开 https://pytorch.org/docs/stable/generated/torch.add.html' },
      { kind: 'reply', text: 'PyTorch 原生支持逐元素加法：\n\n• torch.add(input, other) — 官方 API\n• Tensor.add_ / Tensor.__add__ — 运算符重载\n• 底层可 dispatch 到 CUDA kernel，无需用户手写\n\n文档链接：\n→ torch.add: pytorch.org/docs/stable/generated/torch.add.html\n→ Tensor 运算: pytorch.org/docs/stable/tensors.html' },
      { kind: 'outcome', text: '✅ 有原生算子。入门级 vector add 直接用 torch.add 即可，不必进入 Custom Op 开发链路。' }
    ]
  },
  'cuda-s1-q2': {
    stack: 'CUDA', scene: '1 · 查官方文档', type: 'ask',
    prompt: '请对照 cuDNN/cuBLAS 算子列表，判断我这个算子有没有高效内置实现。',
    outcome: '结论：vector add 有 cuBLAS axpy/saxpy 等底层实现，PyTorch 已封装 → 无需为 Hello World 自研。',
    steps: [
      { kind: 'think', text: 'vector add 属于 Level-1 BLAS（axpy：y = a*x + y）。需对照 cuBLAS 与 PyTorch 是否已接入。' },
      { kind: 'search', text: '检索 NVIDIA cuBLAS API：cublasSaxpy、cublasDaxpy；cuDNN 是否单独暴露 element-wise add' },
      { kind: 'tool', text: '查阅 cuBLAS 文档 Level-1 Functions；PyTorch ATen 中 add 的 CUDA 实现路径' },
      { kind: 'reply', text: '对照结果：\n\n| 层级 | vector add 支持 |\n|------|----------------|\n| cuBLAS | ✅ saxpy/daxpy（标量乘加，含纯 add 场景） |\n| cuDNN | 无独立「add」算子页，element-wise 由 PyTorch 自有 CUDA kernel 或 cuBLAS 路径处理 |\n| PyTorch | ✅ torch.add 已绑定高效 CUDA 实现 |\n\n结论：不存在「必须手写 CUDA kernel 才有高效实现」的情况。' },
      { kind: 'outcome', text: '✅ 高效内置路径已存在。仅当需要融合、特殊 dtype、或非标准内存布局时才考虑自研。' }
    ]
  },
  'cuda-s1-t1': {
    stack: 'CUDA', scene: '1 · 查官方文档', type: 'task',
    prompt: '总结 Custom Op 教程要点：哪些情况才必须写 CUDA kernel？输出一张决策清单。',
    outcome: '产出：决策清单 — Hello World vector add 落在「用原生算子」象限，不自研。',
    steps: [
      { kind: 'think', text: '用户需要可执行的「要不要写 kernel」决策表。基于 PyTorch Custom C++ and CUDA Operators 教程归纳。' },
      { kind: 'search', text: '读取 PyTorch 教程：Custom C++ and CUDA Extensions、When to write custom ops' },
      { kind: 'tool', text: '整理教程中的典型触发条件与反例' },
      { kind: 'reply', text: '**Custom Op 决策清单**\n\n| 情况 | 是否必须自研 CUDA |\n|------|------------------|\n| PyTorch 原生算子已覆盖（如 add/matmul） | ❌ 否 |\n| cuDNN/cuBLAS 已高效实现 | ❌ 否，优先用内置 |\n| 可用多个 torch 算子组合且性能可接受 | ❌ 否 |\n| 需要算子融合（如 fused bias+relu） | ⚠️ 视情况，可 Triton/自定义 |\n| 全新数学运算、无内置等价 | ✅ 是 |\n| 特殊稀疏/量化/自定义内存布局 | ✅ 通常是 |\n| 训练需要自定义 backward 且无法 autograd 组合 | ✅ 通常是 |\n\n**Hello World vector add** → ❌ 不必自研' },
      { kind: 'outcome', text: '📋 决策清单已生成。当前任务可直接用 torch.add，workflow 在环节 1 结束。' }
    ]
  },
  'cuda-s2-q1': {
    stack: 'CUDA', scene: '2 · 问 AI 助手', type: 'ask',
    prompt: '我是应用开发者，什么情况下必须自研 CUDA 算子？请给排查顺序。',
    outcome: '排查顺序已给出；vector add 场景在第 1 步即可排除自研。',
    steps: [
      { kind: 'think', text: '应用开发者视角：先排除内置与组合方案，再谈自研。输出可逐步执行的排查顺序。' },
      { kind: 'reply', text: '**推荐排查顺序（应用开发者）**\n\n1️⃣ 查 PyTorch 是否有原生 API（docs/search torch.xxx）\n2️⃣ 查 cuDNN/cuBLAS 是否覆盖\n3️⃣ 尝试用 2–3 个内置算子组合实现\n4️⃣ 用 profiler 看瓶颈是否在算子本身\n5️⃣ 若以上均不满足 → 考虑 Triton 或 Custom CUDA\n\n**必须自研的典型信号：**\n• 模型论文里的新算子，框架里没有\n• 融合算子，组合调用性能差一个数量级\n• 需要自定义 backward 且 autograd 无法表达' },
      { kind: 'outcome', text: '✅ 按顺序执行：vector add 在第 1 步命中 torch.add → 停止，不自研。' }
    ]
  },
  'cuda-s2-q2': {
    stack: 'CUDA', scene: '2 · 问 AI 助手', type: 'ask',
    prompt: 'cuDNN 能覆盖时还有必要手写 kernel 吗？帮我对比利弊并给建议。',
    outcome: '建议：Hello World vector add 不必手写；维护成本远高于收益。',
    steps: [
      { kind: 'think', text: '对比手写 kernel vs 用 cuDNN/PyTorch 内置的利弊，给出场景化建议。' },
      { kind: 'reply', text: '**对比**\n\n| 维度 | 用内置/cuDNN | 手写 CUDA kernel |\n|------|-------------|------------------|\n| 开发成本 | 低，一行 torch.add | 高，kernel+binding+build |\n| 性能 | 已高度优化 | 新手很难超越 |\n| 维护 | PyTorch 升级自动跟进 | 需跟进 API/sm 架构变化 |\n| 调试 | 成熟 | 编译/launch/精度排查复杂 |\n| 适用 | 标准算子 | 融合/新算子/极致优化 |\n\n**建议：** cuDNN/PyTorch 能覆盖时，默认不要手写。仅 profiling 证明内置不够时才自研。' },
      { kind: 'outcome', text: '✅ vector add 属于「内置足够」→ 不建议为学习目的以外的原因手写 kernel。' }
    ]
  },
  'cuda-s2-t1': {
    stack: 'CUDA', scene: '2 · 问 AI 助手', type: 'task',
    prompt: '能否用几个 torch 内置算子组合实现？请给等价写法和判断依据。',
    outcome: '等价写法：torch.add(a, b) 或 a + b；组合方案已满足需求。',
    steps: [
      { kind: 'think', text: 'vector add 是最基础运算，组合方案即原生 API 本身。给出代码与判断依据。' },
      { kind: 'tool', text: '生成等价实现代码片段' },
      { kind: 'reply', text: '**等价写法（vector add）**\n\n```python\n# 方式 1：API\nc = torch.add(a, b)\n\n# 方式 2：运算符\nc = a + b\n\n# 方式 3：in-place\nc = a.clone(); c.add_(b)\n```\n\n**判断依据：**\n• 语义完全等价于 z = x + y\n• 无需自定义 backward（autograd 自动处理）\n• GPU 上自动 dispatch CUDA kernel\n\n→ 不存在「必须组合多个算子才能模拟 add」的情况，原生即最优组合。' },
      { kind: 'outcome', text: '✅ 组合路径 = 直接用 torch.add。无需自研。' }
    ]
  },
  'cuda-s3-q1': {
    stack: 'CUDA', scene: '3 · IDE 内 Agent', type: 'ask',
    prompt: '分析我选中的 forward，慢在哪个 op？有没有走 cuDNN？',
    outcome: '模拟分析：示例 forward 中无自定义算子瓶颈；add 走 PyTorch 原生 CUDA kernel。',
    steps: [
      { kind: 'think', text: '用户选中了一段示例 forward。需要模拟 profiler 解读与 cuDNN 路径判断。' },
      { kind: 'tool', text: '读取选中代码：\n```python\ndef forward(self, x, y):\n    return self.custom_add(x, y)  # 自定义 wrapper\n```' },
      { kind: 'tool', text: '模拟 torch.profiler 输出：custom_add 12ms（87%）| 其中 aten::add 11.8ms' },
      { kind: 'reply', text: '**Profiler 解读**\n\n• 热点：custom_add → 底层 aten::add 占 87% 时间\n• cuDNN 路径：element-wise add **不经过** cuDNN，走 PyTorch 自有 CUDA kernel（正常）\n• 瓶颈性质：并非「缺 cuDNN 实现」，而是重复 wrapper + 未必要的 Python 开销\n\n**建议：** 去掉 custom_add wrapper，直接 `return x + y`，预计耗时可忽略。' },
      { kind: 'outcome', text: '✅ 慢因是多余封装，不是「必须自研 kernel」。换内置即可。' }
    ]
  },
  'cuda-s3-t1': {
    stack: 'CUDA', scene: '3 · IDE 内 Agent', type: 'task',
    prompt: '扫描当前仓库，列出可替换为 PyTorch 内置的自定义算子。',
    outcome: '扫描结果：发现 1 处 custom_add 可替换为 torch.add。',
    steps: [
      { kind: 'think', text: '模拟 Agent 扫描仓库中 @torch.library、CUDAExtension、class *Function 等自定义算子定义。' },
      { kind: 'tool', text: 'grep -r "custom_add\\|CUDAExtension\\|torch.autograd.Function" ./src' },
      { kind: 'tool', text: '分析 hits：\n• src/ops/custom_add.py — CustomAddFunction，实现 x+y\n• src/models/demo.py — 调用 custom_add' },
      { kind: 'reply', text: '**扫描结果**\n\n| 文件 | 自定义算子 | 可替换为 | 理由 |\n|------|-----------|---------|------|\n| src/ops/custom_add.py | CustomAddFunction | torch.add / x+y | 语义完全等价 |\n| src/models/demo.py | 调用 custom_add | 直接 x+y | 同上 |\n\n**不可替换：** 无（本仓库仅 Hello World 级自定义）' },
      { kind: 'outcome', text: '✅ 1 处可替换。替换后环节 1 结论：不需要继续自研链路。' }
    ]
  },
  'cuda-s3-t2': {
    stack: 'CUDA', scene: '3 · IDE 内 Agent', type: 'task',
    prompt: '生成对比脚本：torch.add vs 我现在的实现，比对输出精度与耗时。',
    outcome: '脚本已生成；模拟运行显示输出一致、原生实现更快。',
    steps: [
      { kind: 'think', text: '生成可运行的 benchmark 脚本，对比 custom_add 与 torch.add 的精度与 latency。' },
      { kind: 'tool', text: '创建 scripts/benchmark_add.py' },
      { kind: 'reply', text: '```python\nimport torch, time\n\ndef custom_add(a, b): return a + b  # 或你的 CustomAdd\n\ndevice = "cuda"\na = torch.randn(1024, 1024, device=device)\nb = torch.randn(1024, 1024, device=device)\n\n# 精度\nout1 = torch.add(a, b)\nout2 = custom_add(a, b)\nprint("max diff:", (out1 - out2).abs().max().item())  # → 0.0\n\n# 耗时\nfor fn in [torch.add, custom_add]:\n    torch.cuda.synchronize()\n    t0 = time.perf_counter()\n    for _ in range(1000): fn(a, b)\n    torch.cuda.synchronize()\n    print(fn.__name__, (time.perf_counter()-t0)/1000*1000, "ms/iter")\n```' },
      { kind: 'tool', text: '模拟运行输出：\nmax diff: 0.0\ntorch.add: 0.008 ms/iter\ncustom_add: 0.009 ms/iter（wrapper 略慢）' },
      { kind: 'outcome', text: '✅ 精度一致，原生不慢于自定义。结论：无需自研 CUDA kernel。' }
    ]
  },

  'cann-s1-q1': {
    stack: 'CANN', scene: '1 · 查官方文档', type: 'ask',
    prompt: '帮我在 torch_npu 支持清单里查目标算子是否支持，并注明版本要求。',
    outcome: '结论：torch.add / 逐元素加在 torch_npu 支持清单内 → 无需 Ascend C 自研。',
    steps: [
      { kind: 'think', text: '目标算子：vector add（torch.add）。需在 torch_npu 算子支持列表中检索并标注 CANN 版本。' },
      { kind: 'search', text: '检索 torch_npu 官方算子支持清单：add、add_、__add__' },
      { kind: 'tool', text: '打开 torch_npu 算子适配列表（Ascend Extension for PyTorch 文档）' },
      { kind: 'reply', text: '**清单检索结果（vector add）**\n\n| PyTorch API | torch_npu 支持 | 备注 |\n|-------------|---------------|------|\n| torch.add | ✅ 已支持 | 逐元素加 |\n| Tensor.add_ | ✅ 已支持 | in-place |\n| Tensor.__add__ | ✅ 已支持 | 运算符 |\n\n版本：随 torch_npu 版本迭代持续支持，需 CANN 与驱动匹配（查兼容性矩阵）。\n\n→ 入门级 vector add **不在**「需自研 Ascend C」范围。' },
      { kind: 'outcome', text: '✅ 已支持。可直接 torch.add(..., device=npu)，workflow 停在环节 1。' }
    ]
  },
  'cann-s1-q2': {
    stack: 'CANN', scene: '1 · 查官方文档', type: 'ask',
    prompt: '根据我的 CANN + torch_npu 版本，查兼容性列表并判断能否直接跑。',
    outcome: '模拟：CANN 8.0 + torch_npu 2.1 组合兼容 → 可直接跑 torch.add。',
    steps: [
      { kind: 'think', text: '用户未指定版本，模拟常见组合 CANN 8.0.RC1 + torch_npu 2.1.0 + PyTorch 2.1。查兼容性矩阵。' },
      { kind: 'search', text: '检索 Ascend CANN 版本配套表：CANN ↔ torch_npu ↔ PyTorch ↔ 驱动' },
      { kind: 'tool', text: '对照兼容性列表：CANN 8.0.x + torch_npu 2.1.x → 配套 PyTorch 2.1' },
      { kind: 'reply', text: '**兼容性检查（模拟输入）**\n\n• CANN：8.0.RC1\n• torch_npu：2.1.0\n• PyTorch：2.1.0\n• 驱动：23.0.3\n\n| 检查项 | 结果 |\n|--------|------|\n| CANN ↔ torch_npu 配套 | ✅ 匹配 |\n| 驱动 ↔ CANN | ✅ 需 ≥ 最低驱动版本 |\n| torch.add on NPU | ✅ 支持清单内 |\n\n**验证命令：**\n`python -c "import torch; import torch_npu; print(torch.npu.is_available())"`' },
      { kind: 'outcome', text: '✅ 版本组合可跑。vector add 无需自研，直接 NPU 上调用即可。' }
    ]
  },
  'cann-s1-t1': {
    stack: 'CANN', scene: '1 · 查官方文档', type: 'task',
    prompt: '在 CANN 内置算子库文档里找等价 vector add 实现，整理对照表。',
    outcome: '对照表：CANN 有 Add 类内置算子，torch_npu 已映射 torch.add。',
    steps: [
      { kind: 'think', text: '在 CANN 算子库中找 Add/Eltwise 类算子，与 PyTorch torch.add 做映射对照。' },
      { kind: 'search', text: '检索 CANN 内置算子：Add、Adds、Eltwise（SUM）' },
      { kind: 'tool', text: '整理 CANN 算子库 ↔ PyTorch API 映射表' },
      { kind: 'reply', text: '**CANN ↔ PyTorch 对照（vector add）**\n\n| CANN 内置算子 | PyTorch 等价 | torch_npu 映射 |\n|--------------|-------------|---------------|\n| Add | 逐元素 z=x+y | torch.add ✅ |\n| Adds | z=x+alpha | torch.add(input, other, alpha=α) ✅ |\n\n框架层：用户调用 torch.add，torch_npu 下发 CANN Add 算子，**无需用户写 Ascend C**。\n\n自研 Ascend C 仅当：CANN 无对应算子且 torch_npu 未适配时。' },
      { kind: 'outcome', text: '📋 对照表完成。Hello World vector add 走内置映射，不自研。' }
    ]
  },
  'cann-s2-q1': {
    stack: 'CANN', scene: '2 · 问 AI 助手', type: 'ask',
    prompt: '如何查某 PyTorch 算子在 Ascend 上是否已支持？请给逐步排查流程。',
    outcome: '逐步排查流程已输出；vector add 在第 2 步即可确认支持。',
    steps: [
      { kind: 'think', text: '给出 Ascend 侧可复用的标准排查 SOP，便于用户逐步执行。' },
      { kind: 'reply', text: '**Ascend 算子支持排查流程**\n\n1️⃣ 查 torch_npu 算子支持清单（文档/GitHub README）\n2️⃣ 查 CANN 版本配套表，确认环境版本匹配\n3️⃣ NPU 上最小脚本试跑：`t = torch.add(a, b)`，看是否报错\n4️⃣ 若报错，读报错类型：\n   • NotImplemented → 未适配，考虑组合/自研/等版本\n   • RuntimeError 驱动/CANN → 环境问题，先修环境\n5️⃣ 查昇腾社区 Issue 是否有 workaround\n6️⃣ 仍不支持 → 评估 Ascend C 自研或 ATB 等替代' },
      { kind: 'outcome', text: '✅ 对 vector add：步骤 1–2 即确认支持，无需走到步骤 6。' }
    ]
  },
  'cann-s2-q2': {
    stack: 'CANN', scene: '2 · 问 AI 助手', type: 'ask',
    prompt: '不支持时，等框架 / 组合算子 / 自研 Ascend C 各适合什么场景？帮我选型。',
    outcome: '选型指南已给出；vector add 不适用（已支持），供后续不支持算子参考。',
    steps: [
      { kind: 'think', text: '三种路径的场景边界，帮助用户在「不支持」时做决策。' },
      { kind: 'reply', text: '**不支持时的选型**\n\n| 路径 | 适合场景 | 不适合 |\n|------|---------|--------|\n| 等框架 | 官方 roadmap 近期会支持；非阻塞 | 上线 deadline 紧 |\n| 组合算子 | 可用现有 torch_npu 算子拼出等价语义 | 性能差、语义无法等价 |\n| 自研 Ascend C | 新算子、融合、性能关键且无内置 | Hello World、已有原生支持 |\n\n**决策问句：**\n• 下一个 CANN 版本会支持吗？→ 等\n• 用 add+mul 能拼吗？→ 组合\n• 拼出来慢 10x 且是核心路径？→ 自研' },
      { kind: 'outcome', text: '✅ vector add 已支持，三条路径均不需要。本回答供「不支持算子」场景复用。' }
    ]
  },
  'cann-s2-t1': {
    stack: 'CANN', scene: '2 · 问 AI 助手', type: 'task',
    prompt: '从 CUDA 迁到 Ascend，这个算子常见坑有哪些？输出检查清单。',
    outcome: '迁移检查清单已生成；vector add 仅需确认 torch_npu 映射与 dtype 一致。',
    steps: [
      { kind: 'think', text: 'CUDA→Ascend 迁移常见坑：设备 API、dtype、不支持算子、精度差异。针对 vector add 定制清单。' },
      { kind: 'tool', text: '生成迁移检查清单 Markdown' },
      { kind: 'reply', text: '**CUDA → Ascend 迁移检查清单（vector add）**\n\n- [ ] `cuda` → `npu` 设备迁移：`tensor.to("npu")`\n- [ ] import torch_npu 并初始化\n- [ ] CANN/驱动/torch_npu 版本配套\n- [ ] torch.add 在支持清单内（✅ 是）\n- [ ] dtype 一致：float16 在 NPU 上是否有精度策略\n- [ ] 不要硬编码 `.cuda()`，改用 `.to(device)`\n- [ ] 对比 CPU golden / CUDA 输出的 rtol/atol\n- [ ] profiler 确认无 silent CPU fallback\n\n**vector add 特有：** 通常无额外坑，属「低风险迁移」算子。' },
      { kind: 'outcome', text: '📋 清单已输出。add 迁移成本低，无需为此自研 Ascend C。' }
    ]
  },
  'cann-s3-q1': {
    stack: 'CANN', scene: '3 · IDE 内 Agent', type: 'ask',
    prompt: '分析选中的 NPU 代码：是否走加速还是 fallback 到 CPU？标出可疑算子。',
    outcome: '模拟分析：add 走 NPU；可疑点仅为硬编码 .cuda() 未改 .npu()。',
    steps: [
      { kind: 'think', text: '分析选中 NPU 代码的设备 placement 与算子 dispatch。' },
      { kind: 'tool', text: '读取选中代码：\n```python\ndef forward(self, x, y):\n    x = x.cuda()  # ⚠️ 迁移遗留\n    return x + y\n```' },
      { kind: 'tool', text: '模拟 NPU trace：x.cuda() 报错或 tensor 不在 NPU；x+y 若 tensor 在 NPU 则走 Add 算子' },
      { kind: 'reply', text: '**分析结果**\n\n| 行 | 问题 | 影响 |\n|----|------|------|\n| x.cuda() | 迁移遗留，NPU 环境应 .npu() | 可能直接报错或设备不一致 |\n| x + y | 若两端在 NPU → ✅ CANN Add 加速 | 正常路径 |\n\n**可疑算子：** 无「不支持算子」，问题是 **设备 API 未迁移**。\n\n修复：`device = x.device` 或 `.to("npu")`，无需自研算子。' },
      { kind: 'outcome', text: '✅ 非算子不支持问题，改设备代码即可。不必 Ascend C 自研。' }
    ]
  },
  'cann-s3-t1': {
    stack: 'CANN', scene: '3 · IDE 内 Agent', type: 'task',
    prompt: '扫描项目，列出所有 torch_npu 调用及可能不支持的算子。',
    outcome: '扫描完成：主要为 torch.add；1 处 .cuda() 需改为 NPU。',
    steps: [
      { kind: 'think', text: '扫描 import torch_npu、.npu()、以及可能在 NPU 上运行的 torch 算子调用。' },
      { kind: 'tool', text: 'grep -rn "torch_npu\\|\\.npu(\\|torch\\." ./src' },
      { kind: 'tool', text: '交叉对照 torch_npu 支持清单' },
      { kind: 'reply', text: '**项目扫描结果**\n\n| 文件 | 调用 | 支持状态 |\n|------|------|----------|\n| src/train.py | import torch_npu | 环境依赖 ✅ |\n| src/model.py | x + y (NPU tensor) | torch.add ✅ |\n| src/model.py | x.cuda() | ⚠️ 非算子问题，API 遗留 |\n| src/ops/custom_loss.py | torch.logsumexp | 需查清单 ⚠️ |\n\n**可能不支持：** custom_loss 中的 logsumexp 需单独核实（非本 Hello World 范围）。' },
      { kind: 'outcome', text: '✅ vector add 路径全部支持。环节 1 对 add 结论：不自研。' }
    ]
  },
  'cann-s3-t2': {
    stack: 'CANN', scene: '3 · IDE 内 Agent', type: 'task',
    prompt: '生成 NPU 探测脚本并试跑，整理支持情况与报错信息。',
    outcome: '探测脚本已生成；模拟试跑 torch.add 通过，无报错。',
    steps: [
      { kind: 'think', text: '生成 NPU 探测脚本，批量测试目标算子并捕获报错。' },
      { kind: 'tool', text: '创建 scripts/npu_probe.py' },
      { kind: 'reply', text: '```python\nimport torch\nimport torch_npu\n\ndevice = "npu:0"\ntests = {\n    "torch.add": lambda: torch.add(\n        torch.randn(4, device=device),\n        torch.randn(4, device=device)),\n}\n\nfor name, fn in tests.items():\n    try:\n        fn()\n        print(f"✅ {name}")\n    except Exception as e:\n        print(f"❌ {name}: {e}")\n```' },
      { kind: 'tool', text: '模拟试跑输出：\n✅ torch.add\n\n支持情况：vector add 在 NPU 上正常执行，无 fallback 报错。' },
      { kind: 'outcome', text: '✅ 探测通过。无需进入 Ascend C 自研 workflow。' }
    ]
  }
};
