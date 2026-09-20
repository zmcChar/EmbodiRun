<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

<p align="center"><strong>面向具身智能的部署与执行运行时</strong></p>
<p align="center">
  <a href="https://embodirun.readthedocs.io/">文档</a> ·
  <a href="docs/zh/quickstart.md">快速开始</a> ·
  <a href="#演示">演示</a> ·
  <a href="docs/support-matrix.md">支持矩阵</a> ·
  <a href="README.md">English</a>
</p>

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Documentation](https://readthedocs.org/projects/embodirun/badge/?version=latest)](https://embodirun.readthedocs.io/)

## 概览

EmbodiRun 将机器人和仿真器连接到模型推理服务，管理服务部署、观测、动作映射和有界执行，让应用在本机或跨机器运行观测—推理—动作循环。

你可以部署策略与机器人、接入仿真器，或通过公共客户端为 Agent 提供观测和执行接口。
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 提供第一方推理引擎；其他推理服务通过 provider 集成接入。

## 功能

- **配置驱动部署**：用 YAML 描述节点、环境、设备、模型与 runtime，通过 Host CLI 校验、准备、启动和停止服务。
- **控制与算力分离**：Control 靠近设备，推理运行于 GPU 节点，通过 HTTP 或可选 WirelessComm 通信。
- **共享设备访问**：由服务统一管理观测、录制和执行仲裁。
- **Agent 接入**：读取观测、请求策略建议、提交有界动作、查询或取消作业。
- **显式策略绑定**：硬件适配器与模型输出到机器人指令的转换分别维护。

详见[架构](docs/architecture.md)与 [Agent 执行流程](docs/agent-workflow.md)。

## 演示

| 共享推理服务 | 推理引擎对比 |
|---|---|
| [![三台 SO-101 的录制](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg)](https://embodirun.readthedocs.io/en/latest/demos/multi-robot-serving/) | [![SO-101 引擎对比](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/engine_e2e_contrast.jpg)](https://embodirun.readthedocs.io/en/latest/demos/engine-e2e-contrast/) |
| 三台 SO-101 共享一个 π0.5 推理服务，每台设备运行独立的 rollout 进程。 | Jetson AGX Thor 与 SO-101 上的 EmbodiInfer HTTP/WirelessComm、SGLang 和原生 LeRobot 对比。 |

演示页面介绍硬件配置和测量结果，分别展示推理延迟与完整动作 chunk 周期。

## 快速开始

使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/) 0.12.x 从源码安装：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run python examples/run_shared_device_fake.py
```

示例启动本地 Control 服务，以模拟关节和虚拟相机演示观测、执行、录制与取消，然后关闭服务。
不需要机器人、模型权重或 GPU。

继续阅读[快速开始](docs/zh/quickstart.md)了解部署 CLI。使用硬件前，先从[支持矩阵](docs/support-matrix.md)选择组合，
配置设备与标定，并阅读[安全说明](docs/safety.md)。

## 集成

| 用途 | 入口 |
|---|---|
| π0.5 与双 SO-101 | [部署指南](docs/pi05-bi-so101.md) |
| 仿真器 | [示例配置](configs/simulation/)与[支持矩阵](docs/support-matrix.md) |
| 外部推理服务 | [配置](docs/configuration.md)与[推理协议](docs/http_api.md) |
| 自带规划循环的 Agent | [公共客户端](agents/CLIENT.md)与[执行流程](docs/agent-workflow.md) |
| RPent | [集成说明](docs/rpent-integration.md) |
| 可选硬件与模型包 | [Integrations](integrations/README.md) |

支持状态按模型、设备和后端的完整组合记录。软件测试、离线模型验证与真机演示的证明范围不同。
自动算力放置和跨模型 GPU 调度尚未实现。

## 性能

部署耗时包括观测采集、推理、传输和动作播放。参见[引擎对比](docs/demos/engine-e2e-contrast.md)
和[传输实验](docs/inference-transport.md)的分段计时。

模型离线性能见 [EmbodiInfer benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/benchmark.md)。
部署实验分别测量推理、通信和动作执行耗时，方便定位整个控制循环的瓶颈。

## 文档

[安装](docs/installation.md) · [配置](docs/configuration.md) · [人工控制](docs/control.md) ·
[安全](docs/safety.md) · [Python API](docs/api.md) · [实验](docs/experiments.md)

目前中文站点覆盖首页和快速开始，其余链接指向英文文档。

## 参与贡献

开发与检查流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。问题与建议请提交
[GitHub issue](https://github.com/BUAA-CI-LAB/EmbodiRun/issues)，漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。
社区遵循[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

Apache-2.0。参见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。
模型权重、数据集、机器人 SDK 和仿真器保留各自的许可证。
