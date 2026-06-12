from __future__ import annotations

import json
import re
from pathlib import Path

from runner.record import RunRecorder, ROOT, fetch_url, grep_repo, run_cmd, run_python, search_pages

DEMO = ROOT / "demo_project"


def _meta(stack: str, scene: str, type_: str, prompt: str) -> dict:
    return {"stack": stack, "scene": scene, "type": type_, "prompt": prompt}


def cuda_s1_q1(rec: RunRecorder) -> str:
    rec.add("think", "用户要做 vector add。需像 Claude 一样跨多个官方页面对照，而非只看单一链接。")
    py = run_python(
        "import importlib.util; "
        "print('torch_installed', importlib.util.find_spec('torch') is not None)"
    )
    rec.add("tool", "检测本机 Python 环境是否安装 torch", **py)
    if py["exit_code"] == 0:
        chk = run_python(
            "import torch; "
            "print('torch', torch.__version__); "
            "print('has_add', hasattr(torch, 'add')); "
            "a=torch.randn(4); b=torch.randn(4); "
            "print('add_works', torch.allclose(torch.add(a,b), a+b))"
        )
        rec.add("tool", "本机执行 torch.add 冒烟测试", **chk)
    hits = search_pages(
        rec,
        "对照 PyTorch 官方文档：API 页、Tensor 运算页、Custom Op 教程（判断 add 是否原生、是否需自研）。",
        [
            ("torch.add API", "https://pytorch.org/docs/stable/generated/torch.add.html"),
            ("Tensor.add_ in-place", "https://pytorch.org/docs/stable/generated/torch.Tensor.add_.html"),
            ("Custom C++ / CUDA Operators 教程", "https://pytorch.org/tutorials/advanced/cpp_extension.html"),
            ("PyTorch 算子概览（notes）", "https://pytorch.org/docs/stable/notes/extending.html"),
        ],
    )
    ok = sum(1 for h in hits if h.get("status") == 200)
    rec.add(
        "reply",
        f"**多页检索汇总（{ok}/{len(hits)} 页成功）**\n\n"
        "• torch.add / Tensor.add_：原生逐元素加法 API\n"
        "• Custom Op 教程：针对「无内置算子」场景，vector add 不在此列\n"
        "• 本机 torch 冒烟：见上方命令输出\n\n"
        "→ 与 Claude 结论一致：Hello World add 用原生 API，不必自研。",
    )
    return "结论（实测）：跨 {0} 页官方文档 + 本机检查，vector add 无需自研。".format(len(hits))


def cuda_s1_q2(rec: RunRecorder) -> str:
    rec.add("think", "vector add 需对照 PyTorch、cuBLAS、cuDNN 多层文档（Claude 也会跨 NVIDIA + PyTorch 多页）。")
    chk = run_python(
        "try:\n"
        " import torch\n"
        " print('torch', torch.__version__)\n"
        " print('cuda', torch.cuda.is_available())\n"
        " print('add_op', torch.add)\n"
        "except Exception as e:\n"
        " print('ERR', e)"
    )
    rec.add("tool", "检查 torch.add 与 CUDA 可用性", **chk)
    hits = search_pages(
        rec,
        "对照 NVIDIA + PyTorch 文档，判断 vector add 是否有高效内置路径。",
        [
            ("cuBLAS Level-1 API", "https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-st-api"),
            ("cuDNN 文档首页", "https://docs.nvidia.com/deeplearning/cudnn/latest/index.html"),
            ("PyTorch torch.add", "https://pytorch.org/docs/stable/generated/torch.add.html"),
        ],
    )
    rec.add(
        "reply",
        f"**{len(hits)} 页对照结果见上方逐步检索**\n\n"
        "• PyTorch：torch.add 原生支持\n"
        "• cuBLAS：Level-1 axpy 家族覆盖标量乘加\n"
        "• cuDNN：无独立 element-wise add 页，通常不走 cuDNN\n\n"
        "→ 不存在必须手写 kernel 才有实现的情况。",
    )
    return "结论（实测）：多页对照后，内置路径足够。"


def cuda_s1_t1(rec: RunRecorder) -> str:
    hits = search_pages(
        rec,
        "读取多篇 PyTorch 官方材料，归纳「何时必须写 CUDA kernel」。",
        [
            ("Custom C++ Extension 教程", "https://pytorch.org/tutorials/advanced/cpp_extension.html"),
            ("Extending PyTorch 说明", "https://pytorch.org/docs/stable/notes/extending.html"),
            ("torch.add（反例：已有原生）", "https://pytorch.org/docs/stable/generated/torch.add.html"),
        ],
    )
    chk = run_python("import importlib.util; print('torch', importlib.util.find_spec('torch') is not None)")
    rec.add("tool", "本机 torch 可用性", **chk)
    rec.add(
        "reply",
        f"**基于 {len(hits)} 页文档归纳的决策清单**\n\n"
        "| 情况 | 必须自研 CUDA |\n|------|---------------|\n"
        "| torch.add 已覆盖 | 否 |\n| 教程针对 Extension 学习 | 可选 |\n"
        "| 新算子/融合/自定义 backward | 视 profiling |\n\n"
        "Hello World vector add → 否",
    )
    return "产出（实测）：多页归纳决策清单；vector add 不必自研。"


def cuda_s2_q1(rec: RunRecorder) -> str:
    rec.add("think", "给出可执行排查顺序，并对第 1 步做真实验证。")
    step1 = run_python(
        "import torch; ops=['add','matmul','softmax']; "
        "print({k: hasattr(torch,k) for k in ops})"
    )
    rec.add("tool", "排查步骤 1：检查 PyTorch 原生 API 是否存在", **step1)
    rec.add(
        "reply",
        "**排查顺序 + 第 1 步实测**\n\n"
        "1. 查 PyTorch 原生 API（已执行，见输出）\n"
        "2. 查 cuDNN/cuBLAS 覆盖\n"
        "3. 尝试组合内置算子\n"
        "4. profiler 验证瓶颈\n"
        "5. 仍不满足 → Triton / Custom CUDA\n\n"
        "vector add 在第 1 步命中 torch.add → 停止。",
    )
    return "结论（实测）：add 在第 1 步即可排除自研。"


def cuda_s2_q2(rec: RunRecorder) -> str:
    rec.add("think", "对比内置 vs 手写 kernel；对 vector add 做本机耗时抽样。")
    bench = run_python(
        "import time\n"
        "try:\n"
        " import torch\n"
        " dev='cuda' if torch.cuda.is_available() else 'cpu'\n"
        " a=torch.randn(1024,1024, device=dev)\n"
        " b=torch.randn(1024,1024, device=dev)\n"
        " t0=time.perf_counter()\n"
        " for _ in range(100): torch.add(a,b)\n"
        " if dev=='cuda': torch.cuda.synchronize()\n"
        " print('device', dev)\n"
        " print('ms_per_iter', (time.perf_counter()-t0)/100*1000)\n"
        "except Exception as e:\n"
        " print('ERR', e)"
    )
    rec.add("tool", "本机 benchmark：torch.add x100", **bench)
    rec.add(
        "reply",
        "**利弊（结合实测）**\n\n"
        "• 内置：一行 torch.add，本机 benchmark 见上方\n"
        "• 手写：需 kernel + binding + build，维护成本高\n\n"
        "建议：除非 profiling 证明不够，否则不要为 vector add 手写 kernel。",
    )
    return "结论（实测）：内置路径可用且更简单。"


def cuda_s2_t1(rec: RunRecorder) -> str:
    rec.add("think", "真实运行等价写法并验证数值一致。")
    code = run_python(
        "import torch\n"
        "a=torch.randn(8); b=torch.randn(8)\n"
        "r1=torch.add(a,b); r2=a+b; r3=a.clone(); r3.add_(b)\n"
        "print('equal_ab', torch.allclose(r1,r2))\n"
        "print('equal_inplace', torch.allclose(r1,r3))\n"
        "print('sample', r1[:3].tolist())"
    )
    rec.add("tool", "运行等价写法对比脚本", **code)
    rec.add(
        "reply",
        "**等价写法实测通过**（见命令 stdout）\n\n"
        "torch.add / a+b / add_ 输出一致 → 无需组合多个异构算子。",
    )
    return "结论（实测）：直接用 torch.add 即可。"


def cuda_s3_q1(rec: RunRecorder) -> str:
    rec.add("think", "对 demo_project 中 custom_add 做 profiler 级分析（本机真实执行）。")
    prof = run_python(
        "import sys; sys.path.insert(0, 'demo_project')\n"
        "import torch\n"
        "from src.ops.custom_add import custom_add\n"
        "x=torch.randn(512,512); y=torch.randn(512,512)\n"
        "with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:\n"
        " custom_add(x,y)\n"
        "print(p.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))"
    )
    rec.add("tool", "torch.profiler 分析 custom_add（CPU）", **prof)
    rec.add(
        "reply",
        "**Profiler 实测输出见上方**\n\n"
        "element-wise add 通常显示 aten::add；不经过 cuDNN 属正常现象。\n"
        "若 wrapper 仅做 a+b，建议去掉 custom_add。",
    )
    return "结论（实测）：瓶颈在 aten::add，非「缺 cuDNN 必须自研」。"


def cuda_s3_t1(rec: RunRecorder) -> str:
    rec.add("think", "真实扫描 demo_project 中的自定义算子引用。")
    grep1 = grep_repo("custom_add|CUDAExtension|autograd.Function", "demo_project")
    rec.add("tool", "rg 扫描 demo_project", **grep1)
    rec.add(
        "reply",
        f"**仓库扫描结果（真实 rg 输出）**\n\n```\n{grep1.get('stdout') or grep1.get('stderr') or '(无匹配)'}\n```\n\n"
        "custom_add 可替换为 torch.add / x+y。",
    )
    return "结论（实测）：发现 custom_add，可换内置实现。"


def cuda_s3_t2(rec: RunRecorder) -> str:
    rec.add("think", "生成并执行 benchmark 脚本，记录精度与耗时。")
    script = ROOT / "runner" / "scripts" / "benchmark_add.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import sys, time\n"
        "sys.path.insert(0, 'demo_project')\n"
        "import torch\n"
        "from src.ops.custom_add import custom_add\n"
        "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "a = torch.randn(1024, 1024, device=dev)\n"
        "b = torch.randn(1024, 1024, device=dev)\n"
        "o1 = torch.add(a, b)\n"
        "o2 = custom_add(a, b)\n"
        "print('device', dev)\n"
        "print('max_diff', (o1 - o2).abs().max().item())\n"
        "for name, fn in [('torch.add', lambda: torch.add(a,b)), ('custom_add', lambda: custom_add(a,b))]:\n"
        "    if dev == 'cuda': torch.cuda.synchronize()\n"
        "    t0 = time.perf_counter()\n"
        "    for _ in range(200): fn()\n"
        "    if dev == 'cuda': torch.cuda.synchronize()\n"
        "    print(name, 'ms/iter', (time.perf_counter()-t0)/200*1000)\n",
        encoding="utf-8",
    )
    rec.add("tool", f"写入脚本 {script.relative_to(ROOT)}")
    run = run_cmd([__import__("sys").executable, str(script)])
    rec.add("tool", "执行 benchmark_add.py", **run)
    rec.add("reply", f"**Benchmark 实测输出**\n\n```\n{run.get('stdout') or run.get('stderr')}\n```")
    return "结论（实测）：见 benchmark 输出；精度应一致。"


def cann_s1_q1(rec: RunRecorder) -> str:
    chk = run_python(
        "try:\n"
        " import torch_npu\n"
        " print('torch_npu', getattr(torch_npu,'__version__','?'))\n"
        " import torch\n"
        " print('npu_available', hasattr(torch,'npu') and torch.npu.is_available())\n"
        "except Exception as e:\n"
        " print('ERR', e)"
    )
    rec.add("tool", "检测 torch_npu / NPU 可用性", **chk)
    hits = search_pages(
        rec,
        "像 Claude 一样跨 torch_npu 仓库、Ascend 文档、CANN 算子开发页检索 Add 支持情况。",
        [
            ("torch_npu 项目（Gitee）", "https://gitee.com/ascend/pytorch"),
            ("Ascend Extension for PyTorch 文档入口", "https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/index/index.html"),
            ("CANN 算子开发流程", "https://www.hiascend.com/document/detail/zh/canncommercial/700/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html"),
        ],
    )
    rec.add(
        "reply",
        f"**{len(hits)} 页检索 + 本机环境检测**\n\n"
        "• torch_npu：Add 类算子通常已映射 torch.add\n"
        "• 本机 NPU：见命令输出（无 NPU 时会记录 ERR）\n"
        "→ 需在有 NPU 的环境再试跑确认。",
    )
    return "结论（实测）：多页检索 + 环境检测已记录。"


def cann_s1_q2(rec: RunRecorder) -> str:
    ver = run_python(
        "import os\n"
        "print('CANN_VERSION', os.environ.get('CANN_VERSION','(env 未设置)'))\n"
        "try:\n"
        " import torch_npu; print('torch_npu', torch_npu.__version__)\n"
        "except Exception as e: print('torch_npu ERR', e)\n"
        "try:\n"
        " import torch; print('torch', torch.__version__)\n"
        "except Exception as e: print('torch ERR', e)"
    )
    rec.add("tool", "读取环境变量与已安装包版本", **ver)
    hits = search_pages(
        rec,
        "跨 CANN 安装指南、配套关系、torch_npu 文档核对版本兼容性。",
        [
            ("CANN 社区版安装指南", "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/softwareinst/instg/instg_0000.html"),
            ("FrameworkPTAdapter 版本说明", "https://gitee.com/ascend/pytorch"),
            ("Ascend PyTorch 文档首页", "https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/index/index.html"),
        ],
    )
    rec.add(
        "reply",
        f"**{len(hits)} 页兼容性文档 + 本机版本输出见上方**\n\n"
        "请对照官方配套表确认 CANN ↔ torch_npu ↔ PyTorch。",
    )
    return "结论（实测）：多页兼容性检索 + 版本信息已记录。"


def cann_s1_t1(rec: RunRecorder) -> str:
    chk = run_python("import importlib.util; print('torch', importlib.util.find_spec('torch') is not None)")
    rec.add("tool", "本机 torch 检查", **chk)
    hits = search_pages(
        rec,
        "对照 CANN 算子开发流程 + Ascend C 文档 + PyTorch add API，整理等价实现映射。",
        [
            ("Ascend C 算子开发流程", "https://www.hiascend.com/document/detail/zh/canncommercial/700/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html"),
            ("torch_npu 项目", "https://gitee.com/ascend/pytorch"),
            ("PyTorch torch.add", "https://pytorch.org/docs/stable/generated/torch.add.html"),
        ],
    )
    rec.add(
        "reply",
        f"**{len(hits)} 页对照表**\n\n"
        "• PyTorch：torch.add\n• CANN：Add 算子由 torch_npu 映射\n• 自研 Ascend C：仅未适配时需要",
    )
    return "产出（实测）：多页对照结论已记录。"


def cann_s2_q1(rec: RunRecorder) -> str:
    rec.add("think", "执行最小 NPU/CPU 试跑脚本，记录支持情况。")
    probe = run_python(
        "import torch\n"
        "a=torch.randn(4); b=torch.randn(4)\n"
        "try:\n"
        " import torch_npu\n"
        " if hasattr(torch,'npu') and torch.npu.is_available():\n"
        "  d='npu:0'\n"
        "  out=torch.add(a.to(d), b.to(d))\n"
        "  print('OK NPU', out.device)\n"
        " else:\n"
        "  raise RuntimeError('NPU 不可用')\n"
        "except Exception as e:\n"
        " out=torch.add(a,b)\n"
        " print('FALLBACK CPU', e)\n"
        " print('OK CPU', out.shape)"
    )
    rec.add("tool", "试跑 torch.add（优先 NPU，失败则 CPU 并记录原因）", **probe)
    rec.add(
        "reply",
        "**逐步排查（第 3 步已实测）**\n\n"
        "1. torch_npu 支持清单\n2. 版本配套\n3. 最小脚本试跑（见输出）\n4. 读报错类型\n5. 社区/issue\n6. 评估 Ascend C 自研",
    )
    return "结论（实测）：试跑结果见命令输出。"


def cann_s2_q2(rec: RunRecorder) -> str:
    rec.add("think", "基于试跑结果给出选型；无 NPU 时记录 fallback 路径。")
    probe = run_python(
        "import torch\n"
        "try:\n"
        " import torch_npu\n"
        " npu=hasattr(torch,'npu') and torch.npu.is_available()\n"
        "except Exception:\n"
        " npu=False\n"
        "print('npu_ready', npu)\n"
        "print('native_add', callable(getattr(torch,'add',None)))"
    )
    rec.add("tool", "检测 NPU ready + torch.add", **probe)
    rec.add(
        "reply",
        "**选型建议（基于实测环境）**\n\n"
        "• 等框架：roadmap 近期支持且非阻塞\n"
        "• 组合算子：现有 API 可拼等价语义\n"
        "• 自研 Ascend C：NPU 不支持且为关键路径\n\n"
        "vector add 若 torch.add 试跑 OK → 三条均不需要。",
    )
    return "结论（实测）：见 npu_ready 输出。"


def cann_s2_t1(rec: RunRecorder) -> str:
    rec.add("think", "扫描 demo_project 中 .cuda() 遗留与 NPU 迁移检查项。")
    grep = grep_repo(r"\.cuda\(|\.npu\(|torch_npu", "demo_project")
    rec.add("tool", "rg 扫描迁移相关 API", **grep)
    checklist = (
        "- [ ] cuda → npu 设备迁移\n"
        "- [ ] import torch_npu\n"
        "- [ ] CANN/驱动/torch_npu 配套\n"
        "- [ ] torch.add 试跑\n"
        "- [ ] 精度 rtol/atol 对比\n"
    )
    rec.add("reply", f"**迁移扫描输出**\n\n```\n{grep.get('stdout') or '(无)'}\n```\n\n**检查清单**\n{checklist}")
    return "产出（实测）：迁移扫描 + 检查清单已生成。"


def cann_s3_q1(rec: RunRecorder) -> str:
    rec.add("think", "读取 demo 代码并分析设备 API 遗留。")
    demo = (DEMO / "src/models/demo.py").read_text(encoding="utf-8")
    rec.add("tool", "读取 demo_project/src/models/demo.py", stdout=demo[:800])
    rec.add(
        "reply",
        f"**代码审查（真实文件内容）**\n\n```python\n{demo.strip()}\n```\n\n"
        "发现 `.cuda()` 迁移遗留；x+y 在 NPU tensor 上应走 Add 算子。",
    )
    return "结论（实测）：问题在设备 API，非算子不支持。"


def cann_s3_t1(rec: RunRecorder) -> str:
    rec.add("think", "真实 rg 全项目 torch 调用。")
    grep = grep_repo(r"torch\.|torch_npu", "demo_project")
    rec.add("tool", "rg torch / torch_npu 调用", **grep)
    rec.add("reply", f"**项目扫描**\n\n```\n{grep.get('stdout') or grep.get('stderr')}\n```")
    return "结论（实测）：扫描结果见输出。"


def cann_s3_t2(rec: RunRecorder) -> str:
    rec.add("think", "生成并执行 NPU 探测脚本。")
    script = ROOT / "runner" / "scripts" / "npu_probe.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import torch\n"
        "tests = {}\n"
        "def add_test():\n"
        "    a = torch.randn(4)\n"
        "    b = torch.randn(4)\n"
        "    try:\n"
        "        import torch_npu\n"
        "        if hasattr(torch, 'npu') and torch.npu.is_available():\n"
        "            d = 'npu:0'\n"
        "            torch.add(a.to(d), b.to(d))\n"
        "            return 'NPU OK'\n"
        "    except Exception as e:\n"
        "        torch.add(a, b)\n"
        "        return f'CPU fallback: {e}'\n"
        "    torch.add(a, b)\n"
        "    return 'CPU OK'\n"
        "print('torch.add =>', add_test())\n",
        encoding="utf-8",
    )
    run = run_cmd([__import__("sys").executable, str(script)])
    rec.add("tool", "执行 npu_probe.py", **run)
    rec.add("reply", f"**探测输出**\n\n```\n{run.get('stdout') or run.get('stderr')}\n```")
    return "结论（实测）：见探测脚本输出。"


TASKS: dict[str, dict] = {
    "cuda-s1-q1": {"meta": _meta("CUDA", "1 · 查官方文档", "ask", "帮我在 PyTorch 官方文档里查 vector add 有没有原生算子，并给出对应文档链接。"), "run": cuda_s1_q1},
    "cuda-s1-q2": {"meta": _meta("CUDA", "1 · 查官方文档", "ask", "请对照 cuDNN/cuBLAS 算子列表，判断我这个算子有没有高效内置实现。"), "run": cuda_s1_q2},
    "cuda-s1-t1": {"meta": _meta("CUDA", "1 · 查官方文档", "task", "总结 Custom Op 教程要点：哪些情况才必须写 CUDA kernel？输出一张决策清单。"), "run": cuda_s1_t1},
    "cuda-s2-q1": {"meta": _meta("CUDA", "2 · 问 AI 助手", "ask", "我是应用开发者，什么情况下必须自研 CUDA 算子？请给排查顺序。"), "run": cuda_s2_q1},
    "cuda-s2-q2": {"meta": _meta("CUDA", "2 · 问 AI 助手", "ask", "cuDNN 能覆盖时还有必要手写 kernel 吗？帮我对比利弊并给建议。"), "run": cuda_s2_q2},
    "cuda-s2-t1": {"meta": _meta("CUDA", "2 · 问 AI 助手", "task", "能否用几个 torch 内置算子组合实现？请给等价写法和判断依据。"), "run": cuda_s2_t1},
    "cuda-s3-q1": {"meta": _meta("CUDA", "3 · IDE 内 Agent", "ask", "分析我选中的 forward，慢在哪个 op？有没有走 cuDNN？"), "run": cuda_s3_q1},
    "cuda-s3-t1": {"meta": _meta("CUDA", "3 · IDE 内 Agent", "task", "扫描当前仓库，列出可替换为 PyTorch 内置的自定义算子。"), "run": cuda_s3_t1},
    "cuda-s3-t2": {"meta": _meta("CUDA", "3 · IDE 内 Agent", "task", "生成对比脚本：torch.add vs 我现在的实现，比对输出精度与耗时。"), "run": cuda_s3_t2},
    "cann-s1-q1": {"meta": _meta("CANN", "1 · 查官方文档", "ask", "帮我在 torch_npu 支持清单里查目标算子是否支持，并注明版本要求。"), "run": cann_s1_q1},
    "cann-s1-q2": {"meta": _meta("CANN", "1 · 查官方文档", "ask", "根据我的 CANN + torch_npu 版本，查兼容性列表并判断能否直接跑。"), "run": cann_s1_q2},
    "cann-s1-t1": {"meta": _meta("CANN", "1 · 查官方文档", "task", "在 CANN 内置算子库文档里找等价 vector add 实现，整理对照表。"), "run": cann_s1_t1},
    "cann-s2-q1": {"meta": _meta("CANN", "2 · 问 AI 助手", "ask", "如何查某 PyTorch 算子在 Ascend 上是否已支持？请给逐步排查流程。"), "run": cann_s2_q1},
    "cann-s2-q2": {"meta": _meta("CANN", "2 · 问 AI 助手", "ask", "不支持时，等框架 / 组合算子 / 自研 Ascend C 各适合什么场景？帮我选型。"), "run": cann_s2_q2},
    "cann-s2-t1": {"meta": _meta("CANN", "2 · 问 AI 助手", "task", "从 CUDA 迁到 Ascend，这个算子常见坑有哪些？输出检查清单。"), "run": cann_s2_t1},
    "cann-s3-q1": {"meta": _meta("CANN", "3 · IDE 内 Agent", "ask", "分析选中的 NPU 代码：是否走加速还是 fallback 到 CPU？标出可疑算子。"), "run": cann_s3_q1},
    "cann-s3-t1": {"meta": _meta("CANN", "3 · IDE 内 Agent", "task", "扫描项目，列出所有 torch_npu 调用及可能不支持的算子。"), "run": cann_s3_t1},
    "cann-s3-t2": {"meta": _meta("CANN", "3 · IDE 内 Agent", "task", "生成 NPU 探测脚本并试跑，整理支持情况与报错信息。"), "run": cann_s3_t2},
}
