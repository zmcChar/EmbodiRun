# EmbodiRun

[English](README.md) | **简体中文**

### Embodied AI, Ready to Run.

**面向具身智能的高效部署与执行 Runtime。**

**Deploy Models. Accelerate Inference. Run Robots.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-embodirun.readthedocs.io-informational.svg)](https://embodirun.readthedocs.io/)

EmbodiRun 是面向具身智能的部署与执行 Runtime，把模型推理、服务部署、跨节点通信和机器人执行组织成一套可复现的运行系统：配置好模型、算力节点与机器人（或仿真器），即可启动服务并运行任务。

高性能推理由 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 提供。它保持独立引擎形态，也可以单独使用。

> **Build embodied applications, not integration code.**

---

## 为什么用 EmbodiRun

- **计算与控制分离。** 贴近机器人的 Control 进程负责传感器采集与动作执行，推理可部署在 Jetson、工作站或远端 GPU 上，两侧独立配置与维护。
- **复用稳定的执行链路。** 观测 → 推理请求 → 动作映射 → 有界执行 → 反馈只实现一次，并内置动作校验、控制仲裁、人工接管和软件停止。
- **可接入 Agent、可替换后端。** 无额外依赖的公共客户端（`observe` / `propose` / `execute` / `inspect` / `cancel` / `stop`）让 Agent 框架保留自己的规划循环；推理后端通过 provider 注册，可替换。
- **部署可复现。** 一份 YAML 描述节点、环境、服务、机器人、相机与绑定，Host CLI 负责校验、准备、启动与停止。

EmbodiRun **不**负责 checkpoint、模型框架、prompt 构造和模型输出解析。它通过带版本号的 HTTP（或可选 WirelessComm）API 与推理服务通信。

---

## 快速开始

EmbodiRun 使用 [uv](https://docs.astral.sh/uv/) 从源码安装。公共 lock 文件不依赖任何私有仓库。

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen            # 核心 CLI + 开发工具
uv run pytest -q            # 在本机（仅 CPU）验证代码
```

每个进程只安装自己需要的能力组：

```bash
uv sync --frozen --no-dev --group host            # SSH 编排 / Host CLI
uv sync --frozen --no-dev --group robot-so101     # SO-101 控制
uv sync --frozen --no-dev --group robot-fr3       # Franka FR3 控制
uv sync --frozen --no-dev --group robot-arx5      # ARX5 控制
uv sync --frozen --no-dev --group sim-libero      # LIBERO 仿真
```

不连接任何节点即可校验部署文件：

```bash
uv run embodirun --config configs/http-wireless-inference/http.yaml validate
```

### 无需真机

下面的示例用内存中的模拟关节和假相机启动一个只有机器人的 Control 服务。它不打开任何硬件，暴露真实的 Host JSON API，是了解 observe/execute 链路的最佳起点：

```bash
uv run embodirun --config examples/shared-device-fake.yaml validate
uv run embodirun --config examples/shared-device-fake.yaml init
uv run embodirun --config examples/shared-device-fake.yaml up
uv run embodirun --config examples/shared-device-fake.yaml describe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config examples/shared-device-fake.yaml down
```

完整流程（观测、媒体、执行、录制）见 [`examples/README.md`](examples/README.md)。

### 真机部署

1. 复制一份示例配置，替换其中的占位符（设备路径、标定、checkpoint、节点地址）。
2. 执行生命周期命令：

```bash
uv run embodirun --config my-deployment.yaml validate   # 静态校验
uv run embodirun --config my-deployment.yaml probe      # 连通性与工具探测
uv run embodirun --config my-deployment.yaml init       # 准备环境与源码
uv run embodirun --config my-deployment.yaml up         # 启动服务
uv run embodirun --config my-deployment.yaml run \
  --runtime so101-1-runtime \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10
uv run embodirun --config my-deployment.yaml down
```

`run` 会驱动真实机器人。默认只提交一个有界动作块；`--max-steps`、`--chunk-steps`、`--control-hz` 用于限制执行规模。返回的每一行动作仍受配置中的关节与夹爪步长限制约束。限幅是逐行的速率限制，不是避障。

> **使用真机时：** 服务启动阶段不会连接或移动机械臂，但 `run` 会。请先阅读 [`docs/control.md`](docs/control.md) 与 [`docs/safety.md`](docs/safety.md)，全程保持操作员在场，并把物理急停放在伸手可及处。

更多内容见 [`docs/installation.md`](docs/installation.md) 与 [`docs/quickstart.md`](docs/quickstart.md)。

---

## 推理性能

EmbodiRun 把三层性能分开测量，让每个数字都有明确范围：模型推理、Runtime 链路、应用任务。推理由 EmbodiInfer 执行。

一组已记录的 π0.5 离线结果：

| 条件 | 结果 |
|---|---|
| 硬件 / 配置 | RTX 4090，B=1，BF16，10 次去噪 |
| 数据 | LIBERO-10，1,600 条离线观测 |
| 平均端到端推理延迟 | **74.33 → 38.89 ms** |
| 吞吐 | **13.45 → 25.71 observations/s** |

计时范围从已解码 CPU 输入到 CPU 动作输出，不包含相机采集、网络通信、机器人执行、模型加载与首次预热。这是相对原 EmbodiInfer/VVLA 路径的历史结果，不代表所有设备或任务。完整条件见 [`EmbodiInfer/benchmarks/pi05-benchmark`](https://github.com/BUAA-CI-LAB/EmbodiInfer)。

---

## 支持状态

支持范围按**完整组合**记录，而不是按单个组件。某个模型、机器人或平台可用，不代表它们可以任意组合。

- **Tested** —— 在标注层级（软件接口 / 仿真 / 离线模型）有本仓库测试覆盖。真机验证记录由部署负责人单独维护。
- **Experimental** —— 维护者提供的试验性路径，存在已知限制。
- **Planned** —— 尚无实现。

| 组合 | 实现 | 本仓库验证范围 | 公开配置 |
|---|---|---|---|
| π0.5 + SO-101 / Bi-SO-101 | 控制、绑定、部署配置 | Tested —— 软件（CPU 测试）；已有离线动作核验记录 | `configs/pi05/bi-so101-vvla.yaml` |
| LIBERO + π0.5 | 仿真适配、SGLang/VVLA 配置 | Tested —— 软件；闭环需要 GPU 与 checkpoint | `configs/simulation/libero-pi05-*.yaml` |
| Habitat + StreamVLN | 仿真适配与配置 | Experimental | `configs/simulation/habitat-streamvln-vvla.yaml` |
| VLABench + π0.5 | 仿真适配与配置 | Experimental | `configs/simulation/vlabench-pi05-*.yaml` |
| Isaac Sim + StreamVLN | 仿真适配与配置 | Experimental（需接受 NVIDIA Isaac Sim EULA） | `configs/simulation/isaac-streamvln-vvla.yaml` |
| Unitree Go2 + StreamVLN | 机器人 agent、绑定、SSH 部署 | Experimental | `src/embodirun/robots/unitree/go2` |
| SGLang HTTP 后端 | provider、客户端、适配器 | Tested —— 软件（无 sglang 时跳过） | `[sglang]` extra |
| 多控制节点共享推理 | 配置、benchmark、控制仲裁 | Tested —— 软件 benchmark | `configs/http-wireless-inference/` |
| RPent Agent 适配 | `agents/rpent` 公共客户端会话 | Experimental，仅软件 | `agents/rpent/README.md` |
| Astra + π0.5 评审循环 | `agents/astra_pi05` 示例 | Experimental，模拟 reviewer | `agents/astra_pi05/README.md` |
| XLeRobot external owner | `integrations/xlerobot_owner` | Experimental，独立安装 | `integrations/xlerobot_owner/README.md` |

“Tested —— 软件”表示该路径有本仓库自动化测试覆盖，**不**表示真机已完成任务。

---

## 架构

```text
应用 / RPent / 自定义 Agent
        │  公共客户端：observe / propose / execute / inspect / cancel / stop
        ▼
EmbodiRun
├─ Deployment runtime  —— 配置、环境、节点、服务生命周期
├─ Application runtime —— job、proposal、执行协调、默认模型循环
├─ Device runtime      —— 连接所有权、共享观测、执行仲裁
├─ Model services      —— 带版本的推理契约与 provider 注册
└─ 机器人 / 仿真器适配器 + 策略绑定
        │  HTTP 或 WirelessComm（带版本的 policy API）
        ▼
EmbodiInfer（或 SGLang 等外部后端）
```

进程边界清晰：**Host** 运行在操作员机器上，**Control** 运行在机器人旁并独占硬件，**Inference** 负责模型计算。三者可以同机，也可以按配置跨节点部署。节点目前由配置指定，自动放置属于后续工作。

```text
src/embodirun/
├── client/          # 公共 Agent HTTP 客户端
├── deployment/      # 配置、计划、状态、SSH/进程生命周期
├── application/     # job、proposal、执行协调、运行循环
├── devices/         # 连接所有权、共享观测、执行仲裁
├── model_services/  # 推理契约、协议、provider
├── services/        # CLI/HTTP 入口与兼容导入
├── robots/          # 机器人适配器
├── bindings/        # 策略到机器人的映射与安全边界
└── simulators/      # 仿真器适配器
```

---

## Agent 接入

Agent 保留自己的规划循环，通过公共客户端调用 Control 服务：

```python
from embodirun.client import ControlClient

client = ControlClient("http://127.0.0.1:8100")
observation = client.observe(runtime="so101-1-runtime", session_id="task-1")
proposal = client.propose(runtime="so101-1-runtime", session_id="task-1",
                          prompt="Pick up the cube.")
job = client.execute(runtime="so101-1-runtime", session_id="task-1",
                     proposal_id=proposal.proposal_id, chunk_steps=10)
result = client.inspect(job.job_id)
```

客户端无额外依赖，说明见 [`agents/CLIENT.md`](agents/CLIENT.md)。`observe` 与 `propose` 不会让机器人运动；`execute` 与 CLI 走同一套授权、校验与安全检查。请求被接受不等于任务成功——请读取返回的执行证据与下一次观测。

参考适配器位于 [`agents/`](agents/README.md)：[`agents/rpent`](agents/rpent/README.md) 对应 RPent 风格 Agent，[`agents/astra_pi05`](agents/astra_pi05/README.md) 对应评审循环。二者都是软件示例，不属于安装后的核心包。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/installation.md`](docs/installation.md) | 安装、extra、可选机器人/仿真 SDK |
| [`docs/quickstart.md`](docs/quickstart.md) | 第一次部署演练 |
| [`docs/configuration.md`](docs/configuration.md) | 部署 YAML 参考 |
| [`docs/control.md`](docs/control.md) | 人工控制、仲裁与软件停止 |
| [`docs/http_api.md`](docs/http_api.md) | 推理服务 API v1 |
| [`docs/pi05-bi-so101.md`](docs/pi05-bi-so101.md) | 双臂 SO-101 π0.5 部署指南 |
| [`docs/safety.md`](docs/safety.md) | 安全限制与操作员清单 |
| [`docs/architecture.md`](docs/architecture.md) | 运行时域与进程边界 |
| [`docs/support-matrix.md`](docs/support-matrix.md) | 完整支持状态 |

---

## 许可证

Apache License 2.0，见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE) 与 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。模型 checkpoint、数据集、机器人 SDK 与仿真器各自保留其许可证，本仓库不再分发。

---

**EmbodiRun — Embodied AI, Ready to Run.**

**Deploy Models. Accelerate Inference. Run Robots.**

**让具身智能用得起、部署快、跑得好。**
