<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

[English](README.md) | **简体中文**

**Deploy Models. Accelerate Inference. Run Robots.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Documentation Status](https://readthedocs.org/projects/embodirun/badge/?version=latest)](https://embodirun.readthedocs.io/en/latest/?badge=latest)
[![Contributing](https://img.shields.io/badge/contributing-guide-brightgreen.svg)](CONTRIBUTING.md)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

EmbodiRun 是具身智能的部署与执行运行时。只需指定一个模型、一个计算节点，以及一台机器人或仿真器，它就能通过一个配置文件跑通整个闭环——观测、推理、动作映射、有界执行、反馈。

模型推理由 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 完成，这是一个独立引擎，也可以单独使用。EmbodiRun 不负责检查点、提示词或模型框架；它通过带版本号的 HTTP 或 WirelessComm API 与推理服务通信。

> **构建具身应用，而不是集成代码。**

## 为什么选择 EmbodiRun

- **控制留在机器人身边，算力放在 GPU 所在之处。** 面向机器人的 Control 进程与推理服务独立部署，可以同机运行，也可以跨节点拆分。
- **执行路径只需编写一次。** 动作校验、控制仲裁、有界执行、人工接管和软件停机由所有机器人和所有智能体共用。
- **智能体保留自己的规划循环。** 一个无依赖客户端通过 `observe` / `propose` / `execute` / `inspect` / `cancel` / `stop` 暴露运行时，而不限定规划器。
- **一次部署就是一个文件。** 节点、环境、服务、机器人、传感器和策略绑定都写在一份 YAML 中，由 CLI 完成校验、准备、启动和停止。

## 快速开始

使用 [uv](https://docs.astral.sh/uv/) 从源码安装：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run pytest -q          # CPU-only check of the checkout
```

### 在没有机器人的情况下运行

一个仅面向机器人的 Control 服务，由仿真关节和虚拟摄像头支撑。它不打开任何硬件，但提供真实的 API：

```bash
uv run embodirun --config examples/shared-device-fake.yaml validate
uv run embodirun --config examples/shared-device-fake.yaml init
uv run embodirun --config examples/shared-device-fake.yaml up
uv run embodirun --config examples/shared-device-fake.yaml describe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config examples/shared-device-fake.yaml down
```

[`examples/README.md`](examples/README.md) 完整演示了观测、媒体、执行和录制。

[MicroDuck VLN 示例](docs/microduck-vln.md) 在 MuJoCo 中调用真实策略推理，提供一键入口、可选 Slurm 申请、视频和逐集指标；运行前需另行准备仿真资产与检查点。

### 在机器人上运行

从 [`configs/`](configs) 复制一份配置，替换设备路径、标定参数、检查点和节点地址，然后执行：

```bash
uv run embodirun --config my-deployment.yaml validate   # static checks
uv run embodirun --config my-deployment.yaml probe      # connectivity and tools
uv run embodirun --config my-deployment.yaml init       # environments and sources
uv run embodirun --config my-deployment.yaml up         # start services
uv run embodirun --config my-deployment.yaml run \
  --runtime so101-1-runtime \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10
uv run embodirun --config my-deployment.yaml down
```

启动服务并不会连接或移动机器人；`run` 才会。`--max-steps`、`--chunk-steps` 和 `--control-hz` 限定实际执行的内容，并且每一行返回结果仍然受配置中的关节和夹爪步数限制约束。裁剪是逐行的速率限制，而不是碰撞规避。

> ⚠️ **在首次实机运行之前**，请阅读 [Control](docs/control.md) 和 [Safety](docs/safety.md)，确保操作员在场，并让硬件急停按钮触手可及。

## 工作原理

```text
Application / agent
        │  public client: observe · propose · execute · inspect · cancel · stop
        ▼
EmbodiRun
├─ Deployment runtime — configuration, environments, nodes, lifecycle
├─ Application runtime — jobs, proposals, execution coordination
├─ Device runtime — connection ownership, shared observations, arbitration
├─ Model services — versioned inference contracts and provider registry
└─ Robot / simulator adapters + policy bindings
        │  HTTP or WirelessComm (versioned policy API)
        ▼
EmbodiInfer, or an external backend such as SGLang
```

三个进程刻意分离：**Host** 运行在操作员机器上，**Control** 运行在机器人旁边并掌管硬件，**Inference** 负责模型计算。它们可以共用一台机器，也可以跨节点拆分；部署位置由配置决定，自动放置属于未来工作。

只有 `execute` 会移动任何东西。`observe`、`propose`、`media` 和 `inspect` 都是只读的，并且被接受的请求绝不会被报告为任务成功——请读取返回的执行证据和下一次观测。

## 支持状态

支持情况按**完整组合**记录。某个模型、机器人或平台受支持，并不意味着任意组合都能正常工作。

| 组合 | 状态 |
|---|---|
| π0.5 + SO-101 / Bi-SO-101 | 已测试——软件（CPU 测试套件、离线动作检查） |
| LIBERO + π0.5 | 已测试——软件；闭环需要 GPU 和检查点 |
| VLABench + π0.5、Habitat 或 Isaac Sim + StreamVLN | 实验性 |
| MicroDuck MuJoCo + ActiveVLN | 实验性；含 CPU 回归测试，完整运行需 GPU 与外部资产 |
| 多节点共享推理 | 已测试——软件基准 |
| RPent 和 Astra 智能体适配器、XLeRobot owner、LightNav-0 | 实验性，仅限软件 |

**已测试——软件**表示该路径已由本仓库中的自动化测试覆盖。它并不意味着实体机器人完成了任务。真机记录由部署方自行保存。

→ [完整支持矩阵](docs/support-matrix.md)

## 性能

在 RTX 4090 上记录到的一个 π0.5 离线结果——B=1、BF16、10 步去噪、LIBERO-10、1,600 次观测：

| 条件 | 结果 |
|---|---|
| 平均端到端推理延迟 | **74.33 → 38.89 ms** |
| 吞吐量 | **13.45 → 25.71 observations/s** |

计时范围从解码后的 CPU 输入到 CPU 动作输出，不包含摄像头采集、网络通信、机器人执行、模型加载和预热。这是针对该配置的一个有范围的历史结果，并不代表所有设备或任务；完整条件记录在 [EmbodiInfer benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer) 中。

## 接入

智能体保留自己的规划循环，并通过一个无依赖客户端调用运行时：

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

- [公共 Agent 客户端](agents/CLIENT.md)——完整的请求与响应契约。
- [Inference API v1](docs/http_api.md)——带版本号的策略 API，可通过 HTTP 或 WirelessComm 使用。
- [RPent 集成](docs/rpent-integration.md)——一个智能体框架接入示例，包含针对真实 π0.5 服务的可复现软件链路。
- 参考适配器位于 [`agents/`](agents/README.md)；它们是软件示例，不在已安装的包内。

## 文档

| | |
|---|---|
| 入门 | [安装](docs/installation.md) · [快速开始](docs/quickstart.md) · [配置](docs/configuration.md) |
| 运维 | [Control](docs/control.md) · [Safety](docs/safety.md) · [支持矩阵](docs/support-matrix.md) |
| 接入 | [Inference API v1](docs/http_api.md) · [RPent 集成](docs/rpent-integration.md) · [π0.5 与两个 SO-101](docs/pi05-bi-so101.md) |
| 了解原理 | [架构](docs/architecture.md) · [实验](docs/experiments.md) |
| 项目 | [参与贡献](CONTRIBUTING.md) · [行为准则](CODE_OF_CONDUCT.md) · [安全](SECURITY.md) · [许可证与再许可](docs/license.md) |

完整文档：**https://embodirun.readthedocs.io/**

## 参与贡献

欢迎贡献。[`CONTRIBUTING.md`](CONTRIBUTING.md) 介绍了开发环境搭建以及 pull request 需要满足的要求；[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) 说明了社区行为期望。请按照 [`SECURITY.md`](SECURITY.md) 私下报告漏洞，切勿在公开 issue 中披露。

## 许可证

Apache License 2.0——参见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE) 和 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。模型检查点、数据集、机器人 SDK 和仿真器各自保留其许可证，不在此处分发。
