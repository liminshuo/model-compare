/* 配对问题库 · 对齐 opknow 26 任务 / 7 工作流阶段 */
function pair(scene, goal, cuda, cann) {
  return { scene, goal, cuda, cann };
}

const JOURNEY_CATS = [
  { name: '1 环境与安装', short: '环境与安装', tasks: [
    { k: 'I', t: '版本兼容排查' },
    { k: 'J', t: '安装与环境变量' },
    { k: 'K', t: '容器 / 镜像搭建' }
  ]},
  { name: '2 算子开发', short: '算子开发', tasks: [
    { k: 'D', t: '自定义算子开发' },
    { k: 'L', t: '算子精度排查' },
    { k: 'M', t: '动态 shape / Tiling' },
    { k: 'N', t: '算子融合' },
    { k: 'O', t: '注册与框架集成' },
    { k: 'C', t: '算子库选型' }
  ]},
  { name: '3 训练', short: '训练', tasks: [
    { k: 'F', t: '分布式训练配置' },
    { k: 'P', t: '混合精度训练' },
    { k: 'Q', t: '显存 / OOM 优化' },
    { k: 'R', t: '精度 / 收敛排查' }
  ]},
  { name: '4 推理与部署', short: '推理与部署', tasks: [
    { k: 'A', t: '模型转换 / 导出' },
    { k: 'S', t: '推理服务部署' },
    { k: 'H', t: '量化' },
    { k: 'T', t: '动态 batch / shape 推理' }
  ]},
  { name: '5 性能优化', short: '性能优化', tasks: [
    { k: 'B', t: 'Profiling 定位瓶颈' },
    { k: 'U', t: '访存 / occupancy 优化' },
    { k: 'V', t: '计算与传输重叠' },
    { k: 'W', t: '多卡通信优化' }
  ]},
  { name: '6 调试', short: '调试', tasks: [
    { k: 'X', t: '内存越界定位' },
    { k: 'E', t: '报错码 / 异常排查' }
  ]},
  { name: '7 迁移', short: '迁移', tasks: [
    { k: 'G', t: 'CUDA → 昇腾 迁移' },
    { k: 'Y', t: '跨芯片迁移' },
    { k: 'Z', t: '概念心智模型对照' }
  ]}
];

const PAIRED_BASE = { novice: {} };
const ALL_TASK_KEYS = JOURNEY_CATS.flatMap(c => c.tasks.map(t => t.k));

function setPairs(level, key, pairs) {
  PAIRED_BASE[level][key] = pairs;
}

// —— 1 环境与安装 ——
setPairs('novice', 'I', [
  pair('版本矩阵查询', '确认驱动、运行时、框架三者版本兼容',
    'CUDA / cuDNN / NVIDIA 驱动 / PyTorch 的版本兼容矩阵去哪查？如何对照我当前机器？',
    'CANN / Ascend 驱动 / 固件 / torch_npu 的版本兼容矩阵去哪查？如何对照？'),
  pair('不匹配诊断', '识别版本不匹配的典型报错并修复',
    'cuDNN 与 CUDA 版本不匹配时典型报错是什么？修复步骤？',
    'CANN Toolkit 与 Ascend 驱动/固件不匹配时典型报错是什么？修复步骤？'),
  pair('多组件对齐', '一次性核对所有相关组件版本',
    '训练环境需对齐 CUDA Toolkit、驱动、cuDNN、NCCL、PyTorch——检查清单？',
    '训练环境需对齐 CANN、驱动、固件、HCCL、torch_npu——检查清单？'),
  pair('升级决策', '评估升级某一组件对全栈的影响',
    '只升级 NVIDIA 驱动不升级 CUDA Toolkit 可以吗？判断依据？',
    '只升级 CANN 不升级 Ascend 驱动/固件可以吗？判断依据？')
]);
setPairs('novice', 'J', [
  pair('首次安装', '按正确顺序完成 Toolkit 安装',
    'Ubuntu 22.04 首次安装 CUDA Toolkit：与驱动的顺序？完整步骤？',
    'Ubuntu 22.04 首次安装 CANN Toolkit：与驱动、固件的顺序？完整步骤？'),
  pair('环境变量', '配置 PATH/LD_LIBRARY_PATH 并持久化',
    'CUDA 的 PATH、LD_LIBRARY_PATH、CUDA_HOME 怎么配？写入 ~/.bashrc？',
    'CANN 的 set_env.sh、ASCEND_HOME、LD_LIBRARY_PATH 怎么配？如何永久生效？'),
  pair('安装后验证', '确认 nvcc / cann 工具链可用',
    'nvidia-smi 正常但 nvcc 找不到，如何排查？',
    'npu-smi 正常但 atc_msft_run 等命令不可用，如何排查？'),
  pair('框架安装', '安装与 Toolkit 匹配的 PyTorch',
    'PyTorch 2.x 对应 CUDA 12 的 pip 安装命令？',
    'torch_npu 对应 CANN 8.x 的 pip 安装命令？')
]);
setPairs('novice', 'K', [
  pair('容器运行时', '配置 Docker 访问加速设备',
    'Docker 使用 GPU：nvidia-container-toolkit 安装与配置步骤？',
    'Docker 使用 Ascend：Ascend Docker Runtime、/dev/davinci* 挂载方式？'),
  pair('基础镜像', '选择官方推荐的基础镜像',
    'NGC CUDA 基础镜像怎么选版本？PyTorch NGC 镜像？',
    '昇腾官方容器镜像怎么选？与 CANN 版本如何对应？'),
  pair('K8s 部署', '在 K8s 中调度加速卡 Pod',
    'K8s GPU Pod：Device Plugin、资源声明、YAML 示例？',
    'K8s Ascend Pod：Device Plugin、资源声明、YAML 示例？'),
  pair('多机一致性', '多节点容器环境版本一致',
    '多机 GPU 训练前如何批量验证各节点 CUDA/驱动版本一致？',
    '多机 Ascend 训练前如何批量验证各节点 CANN/驱动版本一致？')
]);

// —— 2 算子开发 ——
setPairs('novice', 'D', [
  pair('Hello World 算子', '完成第一个自定义算子',
    'CUDA vector add 最简 kernel + PyTorch C++ Extension 完整示例？',
    'Ascend C vector add + msOpGen 最简算子完整示例？'),
  pair('工程结构', '理解算子工程目录与编译流程',
    'PyTorch CUDAExtension 的 setup.py、编译、加载流程？',
    'CANN 算子工程目录、CMake、TBE/Ascend C 编译流程？'),
  pair('前向反向', '实现带梯度的自定义算子',
    'PyTorch 自定义 CUDA 算子如何实现 backward？',
    'Ascend C 自定义算子如何实现梯度？aclnn 接口？'),
  pair('调试入门', '算子编译失败时定位问题',
    'nvcc 编译 CUDA 算子常见报错（sm_XX、C++标准）？',
    'Ascend C/msOpGen 编译算子常见报错与日志位置？')
]);
setPairs('novice', 'L', [
  pair('精度对比', '自定义算子与参考实现数值对齐',
    'CUDA 自定义算子如何与 PyTorch 原生/cuBLAS 对比精度？rtol/atol？',
    'CANN 自定义算子如何用 msaccucmp 与 CPU golden 对比精度？'),
  pair('边界 shape', '测试非常规 shape 的精度',
    'CUDA 算子在空 tensor、超大维、非对齐 shape 下如何测精度？',
    'CANN 算子在 ND/NZ format、动态 shape 下如何测精度？'),
  pair('低精度', 'fp16/bf16 算子精度验收',
    'CUDA fp16 算子精度标准？累加顺序影响？',
    'CANN fp16 算子精度标准？与 CUDA 侧阈值如何对齐？'),
  pair('逐层定位', '模型中算子精度异常定位',
    '模型某层输出分叉，如何 dump 对比定位首个精度异常算子？',
    'NPU 模型某层输出分叉，如何 dump 对比定位首个精度异常算子？')
]);
setPairs('novice', 'M', [
  pair('动态 shape 概念', '理解动态 shape 对算子/推理的影响',
    'TensorRT dynamic shape profile 怎么设？min/opt/max 含义？',
    'Ascend C Tiling 策略如何处理动态 shape？ATC 动态维度参数？'),
  pair('Tiling 策略', '为不同 shape 选择 tiling 参数',
    'CUDA kernel tiling/block size 如何根据 shape 调整？',
    'Ascend C Tiling 数据结构如何根据 shape 填写？'),
  pair('导出配置', '导出支持动态 shape 的模型',
    'PyTorch 导出 ONNX 时 dynamic_axes 怎么设？',
    'PyTorch 导出 OM 时 ATC --input_shape 动态 batch 怎么配？'),
  pair('性能验证', '动态 shape 下 benchmark',
    'TensorRT 多 profile 下 latency 如何 benchmark？',
    'ais_bench 动态 batch 推理如何 benchmark？')
]);
setPairs('novice', 'N', [
  pair('框架层融合', '启用框架算子融合加速',
    'torch.compile / cuDNN fusion / SDPA 后端怎么启用算子融合？',
    'torch_npu 图模式 / 昇腾算子融合 pass 怎么启用？'),
  pair('推理引擎融合', '推理阶段算子融合',
    'TensorRT builder 的 layer fusion 策略？哪些算子会被融合？',
    'ATC 图优化 / 算子融合规则？如何控制 fusion 开关？'),
  pair('融合验证', '确认融合生效且精度不变',
    '如何验证 TensorRT fusion 已生效？精度如何对比？',
    '如何验证 CANN 图融合已生效？精度如何对比？'),
  pair('融合回退', '融合导致精度/性能问题时拆分',
    'TensorRT 某 fusion 导致精度下降，如何禁用特定 pass？',
    'ATC 某 fusion 导致精度下降，如何禁用或拆分 subgraph？')
]);
setPairs('novice', 'O', [
  pair('框架注册', '将自定义算子注册到 PyTorch 计算图',
    'PyTorch TORCH_LIBRARY 注册自定义 CUDA 算子步骤？',
    'torch_npu / aclnn 注册自定义 Ascend C 算子步骤？'),
  pair('autograd 集成', '算子支持自动求导',
    'PyTorch Autograd Function 包装 CUDA 算子？',
    'torch_npu 自定义算子如何接入 autograd？'),
  pair('ONNX/OM 导出', '含自定义算子的模型导出',
    '含自定义 CUDA 算子的模型导出 ONNX 如何处理？',
    '含自定义 CANN 算子的模型导出 OM 如何处理？'),
  pair('serving 集成', '推理服务加载含自定义算子的模型',
    'TensorRT plugin 或 ONNX Runtime custom op 部署流程？',
    'OM 含自定义算子节点的 ACL 推理部署流程？')
]);
setPairs('novice', 'C', [
  pair('算子库概览', '了解平台提供的算子库体系',
    'cuBLAS、cuDNN、cuFFT、NCCL 各自适用什么场景？',
    'AOL、ATB、TBE、aclnn 各自适用什么场景？'),
  pair('算子覆盖查询', '查模型所需算子是否被支持',
    '如何查 PyTorch 算子在 CUDA/cuDNN 上是否有高效实现？',
    '如何查 PyTorch 算子在 torch_npu/CANN 上是否支持？'),
  pair('选型决策', '在多个算子库/API 间选择',
    '做 Attention：用 cuDNN SDPA、flash-attn 还是手写 CUDA？',
    '做 Attention：用 ATB、内置 flash attention 还是 Ascend C？'),
  pair('性能预期', '评估内置算子库 vs 自研',
    '什么情况下必须用自定义 CUDA 算子而非 cuDNN？',
    '什么情况下必须用 Ascend C 而非 ATB/AOL？')
]);

// —— 3 训练 ——
setPairs('novice', 'F', [
  pair('DDP 配置', '配置单机/多机数据并行训练',
    'PyTorch DDP：init_process_group(backend=nccl)、torchrun 最小配置？',
    'torch_npu DDP：init_process_group(backend=hccl)、torchrun 最小配置？'),
  pair('多机网络', '多机训练网络与环境配置',
    '多机 NCCL：MASTER_ADDR、NCCL_IB_DISABLE、网络拓扑检查？',
    '多机 HCCL：hccn_tool 配 IP、RoCE 配置、网络检查？'),
  pair('通信验证', '训练前验证集合通信可用',
    '如何用 nccl-tests 或简单脚本验证 NCCL allreduce？',
    '如何用脚本验证 HCCL allreduce 正常？'),
  pair('启动方式', '选择正确的分布式启动器',
    'torchrun vs slurm vs MPI 启动 GPU 分布式训练？',
    'torchrun vs 昇腾多机启动脚本启动 NPU 分布式训练？')
]);
setPairs('novice', 'P', [
  pair('AMP 开启', '启用混合精度训练',
    'torch.cuda.amp.autocast + GradScaler 完整训练循环？',
    'torch_npu 混合精度 / apex 在昇腾上怎么开？'),
  pair('精度格式', '选择 fp16/bf16',
    'NVIDIA GPU 训练选 fp16 还是 bf16？对 loss scale 影响？',
    'Ascend NPU 训练选 fp16 还是 bf16？支持情况？'),
  pair('数值稳定', '混合精度下 loss NaN 排查',
    'AMP 训练 loss 变 NaN，GradScaler 怎么调？',
    'NPU AMP 训练 loss 变 NaN，排查顺序？'),
  pair('性能收益', '评估混合精度带来的加速',
    'GPU fp16 训练相对 fp32 典型加速比？哪些层不适合 fp16？',
    'NPU fp16 训练相对 fp32 典型加速比？哪些层不适合 fp16？')
]);
setPairs('novice', 'Q', [
  pair('OOM 定位', '快速定位显存溢出原因',
    'CUDA OOM 如何定位是哪一层、哪个 tensor？',
    'NPU OOM 如何定位是哪一层、哪个 tensor？'),
  pair('显存优化', '系统性减少显存占用',
    'gradient checkpointing、offload、ZeRO 在 GPU 上怎么用？',
    'gradient checkpointing、offload 在 NPU 上怎么用？'),
  pair('batch 策略', '在显存限制下调整 batch',
    '显存不够时 gradient accumulation 与 batch size 如何组合？',
    'NPU 显存不够时 gradient accumulation 与 batch size 如何组合？'),
  pair('显存监控', '训练时监控显存使用',
    'nvidia-smi dmon / PyTorch memory snapshot 怎么用？',
    'npu-smi / torch_npu 显存监控怎么用？')
]);
setPairs('novice', 'R', [
  pair('loss 不下降', '训练 loss 长期不收敛排查',
    'GPU 训练 loss 不下降：lr、数据、BN、精度、设备 排查顺序？',
    'NPU 训练 loss 不下降：lr、数据、BN、精度、设备 排查顺序？'),
  pair('精度异常', '训练精度明显低于预期',
    'GPU 训练精度低：数据增强、label、fp16 溢出怎么查？',
    'NPU 训练精度低：format、算子精度、fp16 溢出怎么查？'),
  pair('收敛曲线', '对比 CUDA 基线收敛行为',
    '同一模型 GPU 与 NPU 训练 loss 曲线差异大，如何对比？',
    '迁移训练后 loss 曲线与 CUDA 基线不对齐，如何定位？'),
  pair('超参调优', '在目标平台上重新调参',
    'GPU 上调好的 lr/warmup 迁到 NPU 要调整吗？经验法则？',
    'NPU 训练超参与 CUDA 侧如何对齐起点？')
]);

// —— 4 推理与部署 ——
setPairs('novice', 'A', [
  pair('导出流程', '完成训练模型到推理格式的转换',
    'PyTorch → ONNX → TensorRT engine 完整步骤？',
    'PyTorch → ONNX → ATC → OM 完整步骤？'),
  pair('导出验证', '确认转换后模型输出一致',
    'ONNX/TensorRT 与 PyTorch 输出对比方法？误差阈值？',
    'OM 与 PyTorch 输出对比方法？误差阈值？'),
  pair('动态 shape', '导出支持变长输入的模型',
    'ONNX dynamic_axes + TensorRT optimization profile 配置？',
    'ATC 动态 batch / 动态 shape 编译参数？'),
  pair('不支持算子', '导出时遇到不支持算子',
    'ONNX 导出报错 unsupported op，如何处理？',
    'ATC 转换报错 unsupported op，如何拆分 subgraph？')
]);
setPairs('novice', 'S', [
  pair('服务框架', '选择推理服务框架并部署',
    'Triton Inference Server 部署 TensorRT/ONNX 模型步骤？',
    'MindIE / ACL serving 部署 OM 模型步骤？'),
  pair('API 封装', 'HTTP/gRPC 推理接口',
    'FastAPI 封装 GPU 推理服务最小示例？',
    'FastAPI 封装 NPU ACL 推理服务最小示例？'),
  pair('并发 batching', '提升服务吞吐量',
    'Triton dynamic batcher 配置？延迟与吞吐权衡？',
    '昇腾推理服务 dynamic batching 配置？'),
  pair('生产监控', '线上推理服务监控',
    'GPU 推理服务 p99 latency、GPU util、OOM 监控？',
    'NPU 推理服务 p99 latency、NPU util、plog 告警？')
]);
setPairs('novice', 'H', [
  pair('PTQ 量化', '训练后量化压缩模型',
    'TensorRT INT8 PTQ：校准数据集、builder flag 配置？',
    'msmodelslim W8A8：校准流程、ATC 量化参数？'),
  pair('精度评估', '量化后精度损失评估',
    'TensorRT INT8 相对 FP16 精度如何评估？acceptable drop？',
    'OM INT8 相对 FP16 精度如何评估？acceptable drop？'),
  pair('量化导出', '导出量化模型用于部署',
    '量化后 ONNX/TensorRT engine 导出流程？',
    '量化后 OM 导出流程？scale 参数如何携带？'),
  pair('QAT 可选', '量化感知训练（如需要）',
    'PyTorch QAT 后导出 TensorRT INT8 流程？',
    'PyTorch QAT 后导出量化 OM 流程？')
]);
setPairs('novice', 'T', [
  pair('动态 batch', '推理支持可变 batch size',
    'TensorRT dynamic batch dimension 怎么设？',
    'ATC dynamic_batch_size_range 怎么配？'),
  pair('动态 shape', '推理支持可变输入尺寸',
    'TensorRT 多 profile 处理不同 input shape？',
    'OM 动态 shape 推理 ACL API 怎么用？'),
  pair('benchmark', '动态场景下性能测试',
    'TensorRT 不同 batch size 下 throughput/latency 测试？',
    'ais_bench 动态 batch NPU 推理 benchmark？'),
  pair('padding 策略', '动态 batch 的 padding 处理',
    '推理服务 dynamic batching 的 padding 与有效计算比例？',
    'NPU 推理 dynamic batching 的 padding 策略？')
]);

// —— 5 性能优化 ——
setPairs('novice', 'B', [
  pair('Profiler 入门', '使用 profiling 工具定位瓶颈',
    'Nsight Systems 分析 GPU 训练 timeline 怎么看？',
    'msprof / Ascend PyTorch Profiler 分析 NPU 训练怎么看？'),
  pair('利用率解读', '设备利用率低的含义',
    'nvidia-smi GPU-Util 低但训练慢说明什么？',
    'npu-smi 利用率低但训练慢说明什么？'),
  pair('瓶颈分类', '区分计算/通信/数据瓶颈',
    '如何判断瓶颈在 DataLoader、GPU 计算还是 NCCL？',
    '如何判断瓶颈在 DataLoader、NPU 计算还是 HCCL？'),
  pair('报告输出', '将 profiling 结论交付团队',
    'GPU profiling 报告应包含哪些指标和优化建议？',
    'NPU profiling 报告应包含哪些指标和优化建议？')
]);
setPairs('novice', 'U', [
  pair('访存分析', '判断 memory bound 并优化',
    'ncu memory throughput 如何判断 kernel 是 memory bound？',
    'Ascend C double-buffer / UB 访存优化怎么做？'),
  pair('occupancy', '提升 SM/核利用率',
    'CUDA occupancy calculator 怎么用？block size 怎么调？',
    'Ascend Vector/Cube 核利用率如何分析和提升？'),
  pair('内存布局', '优化 tensor 布局提升带宽',
    'contiguous / coalesced access 对 GPU kernel 影响？',
    'ND/NZ format 对 NPU 算子性能影响？如何转换？'),
  pair('kernel 优化', '算子级访存优化',
    'CUDA shared memory tile 优化示例？',
    'Ascend C L0/L1 cache 友好访问模式？')
]);
setPairs('novice', 'V', [
  pair('异步拷贝', 'Host-Device 异步传输',
    'cudaMemcpyAsync + stream 隐藏拷贝延迟？',
    'aclrtMemcpyAsync + stream 隐藏拷贝延迟？'),
  pair('多 stream', '计算与传输流水线并行',
    'CUDA 多 stream 实现 compute/copy overlap？',
    'NPU 多 stream 实现 compute/copy overlap？'),
  pair('pinned memory', '减少 H2D 拷贝开销',
    'pinned memory + prefetch 怎么配 DataLoader？',
    'NPU 侧 Host-Device 拷贝优化策略？'),
  pair('推理流水线', '推理阶段 overlap 优化',
    'GPU 推理 prefill/decode 与 H2D 如何 overlap？',
    'NPU 推理 AIPP 预处理与推理如何 overlap？')
]);
setPairs('novice', 'W', [
  pair('通信调优', '优化多卡集合通信性能',
    'NCCL 环境变量（NCCL_IB、NCCL_TREE）调优指南？',
    'HCCL 环境变量、通信算法选择调优指南？'),
  pair('拓扑感知', '根据硬件拓扑优化通信',
    'NCCL 如何感知 NVLink/PCIe 拓扑？',
    'HCCL 如何感知 Ascend 卡间拓扑？'),
  pair('通信占比', '量化通信在训练中的开销',
    'profiler 显示 allreduce 占比过高，如何优化？',
    'profiler 显示 HCCL allreduce 占比过高，如何优化？'),
  pair('多机通信', '跨节点通信优化',
    'NCCL 多机 IB/RoCE 配置与带宽测试？',
    'HCCL 多机 RDMA 配置与带宽测试？')
]);

// —— 6 调试 ——
setPairs('novice', 'X', [
  pair('内存检测工具', '使用 sanitizer 定位越界',
    'compute-sanitizer 检测 CUDA 内存越界怎么用？',
    'msSanitizer 检测 Ascend 内存越界怎么用？'),
  pair('非法访问', 'CUDA illegal memory access 排查',
    'CUDA illegal memory access 异步报错如何定位到 kernel？',
    'Ascend 内存访问异常 plog/日志如何定位？'),
  pair('同步调试', '用同步模式缩小排查范围',
    'CUDA_LAUNCH_BLOCKING=1 调试异步错误？',
    'NPU 同步调试模式如何开启？'),
  pair('dump 分析', '结合 core dump 分析',
    'CUDA core dump 环境变量配置与分析？',
    'Ascend 训练/推理崩溃后 plog 分析流程？')
]);
setPairs('novice', 'E', [
  pair('错误码查询', '根据错误码定位根因',
    'CUDA 错误码（如 2、700、719）含义与排查？',
    'ACL 错误码（如 507008）、EZ9999 含义与排查？'),
  pair('日志定位', '从日志找到关键错误信息',
    'GPU 训练/推理错误从哪些日志查（CUDA、NCCL）？',
    'NPU 训练/推理错误从哪些日志查（plog、HCCL）？'),
  pair('逐步复现', '缩小问题复现范围',
    'GPU 报错后如何最小化复现脚本？',
    'NPU 报错后如何最小化复现脚本？'),
  pair('社区案例', '查找类似错误的解决方案',
    'CUDA 某报错在 Stack Overflow/GitHub 怎么高效搜？',
    'Ascend 某报错在昇腾社区/CSDN 怎么高效搜？')
]);

// —— 7 迁移 ——
setPairs('novice', 'G', [
  pair('自动迁移', '用框架工具一键迁移',
    'CUDA PyTorch 代码迁 ROCm/HIP 有哪些工具/模式？',
    'CUDA PyTorch 代码用 transfer_to_npu 一行迁移要改什么？'),
  pair('算子扫描', '扫描不支持的算子/API',
    '如何扫描 PyTorch 模型中 CUDA 特有、不易迁移的算子？',
    '如何用 ms_fmk_transplt 扫描 torch_npu 不支持的 API？'),
  pair('通信迁移', '分布式代码改写',
    'NCCL 后端代码迁 HCCL 要改哪些部分？',
    'ProcessGroup 从 nccl 改 hccl 的完整 diff？'),
  pair('验收标准', '迁移完成后的性能精度验收',
    'CUDA→昇腾迁移：性能达到 CUDA 多少比例算达标？',
    '迁移精度验收：逐层 cosine / 端到端 metric 阈值？')
]);
setPairs('novice', 'Y', [
  pair('跨架构编译', '为不同芯片架构编译',
    'CUDA 跨架构：-arch=sm_XX 与 PTX 兼容性？',
    'Ascend 跨芯片：soc_version 切换（910A→910B）注意点？'),
  pair('算子兼容', '算子在不同芯片上的行为差异',
    '同一 CUDA kernel 在不同 GPU 架构上行为差异？',
    '同一 Ascend C 算子在不同 soc_version 上行为差异？'),
  pair('模型重编译', '换芯片后模型需重新编译',
    'TensorRT engine 换 GPU 架构要重建吗？',
    'OM 模型换 Ascend 芯片要重新 ATC 吗？'),
  pair('性能对齐', '跨芯片性能基线对比',
    'A100 vs H100 性能对比方法论？',
    '910A vs 910B 性能对比方法论？')
]);
setPairs('novice', 'Z', [
  pair('并行模型', '理解平台并行抽象',
    'CUDA thread/block/grid/warp 层级关系？SPMD 模型？',
    'Ascend C SPMD、Vector/Cube 核、UB/GM 层级关系？'),
  pair('内存层级', '理解各级存储及访问',
    'CUDA register/shared/global memory 特点与访问？',
    'Ascend UB/L1/GM 特点与访问？与 CUDA 如何类比？'),
  pair('编程范式', '从 CUDA 思维迁移到 Ascend',
    'CUDA kernel 写法哪些可直接映射到 Ascend C？哪些不能？',
    'Copy-In/Compute/Copy-Out 流水线 vs CUDA kernel 差异？'),
  pair('生态对照', '两套生态组件心智对照',
    'cuBLAS↔?、TensorRT↔?、NCCL↔? 昇腾侧各对应什么？',
    '给出 CUDA 生态到 CANN 生态的组件对照表？')
]);

const ROLE_LENS = {
  app: { cuda: '', cann: '' },
  migrate: { cuda: '【迁移源】', cann: '【迁移目标】' },
  op: { cuda: '【算子】', cann: '【算子】' },
  sys: { cuda: '【调优】', cann: '【调优】' }
};

function adaptForRole(pairs, role) {
  if (role === 'app') return pairs.map(p => ({ ...p }));
  const lens = ROLE_LENS[role];
  return pairs.map(p => ({
    ...p,
    cuda: lens.cuda + p.cuda,
    cann: lens.cann + p.cann
  }));
}

function getPairedQuestions(role, taskKey) {
  const base = PAIRED_BASE.novice[taskKey];
  if (!base) return [];
  return adaptForRole(base, role).slice(0, 1);
}
