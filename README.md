# CUDA vs CANN · 大模型算子 Workflow 对照研究

同场景配对对照 CUDA 与 CANN，度量大模型在算子开发 workflow 中的**回答质量**与**执行效率**。

## 在线访问（GitHub Pages）

https://schihhsin.github.io/model-compare/

| 页面 | 文件 | 说明 |
|------|------|------|
| 一 | `cuda-cann-journey-map.html` | 旅程地图 · 26 任务配对提问 |
| 二 | `operator-novice-workflow.html` | 新手 vector add 完整 workflow |
| 三 | `llm-metrics-framework.html` | 10 项聚焦度量指标框架 |
| 四 | `llm-metrics-results.html` | 环节 1 首对问题跑分结果 |
| — | `llm-simulation.html` | 单次任务执行时间线（静态记录） |

## 本地开发（含 API 试跑）

```bash
./start-server.sh
# → http://127.0.0.1:8765/operator-novice-workflow.html
```

API：

- `POST /api/run/<id>` — 执行任务并写记录
- `GET /api/record/<id>` — 读取 simulation 记录
- `GET /api/metrics/pair/s1-q1` — 配对计分

计分：

```bash
python3 -m runner.metrics
```

## 目录

- `runner/` — 任务执行、记录、计分、HTTP 服务
- `simulation-records/` — 18 个任务的 JSON 执行记录
- `metrics-results/` — 预生成配对计分结果
