<p align="center">
  <img src="https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/logo.png" alt="EmbodiRun" width="440">
</p>

<h3 align="center">从模型预测，到机器人行动。</h3>
<p align="center">
  <a href="https://embodirun.readthedocs.io/">文档</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#演示">演示</a> ·
  <a href="#性能">性能</a> ·
  <a href="docs/support-matrix.md">支持矩阵</a> ·
  <a href="README.md">English</a>
</p>

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Documentation](https://readthedocs.org/projects/embodirun/badge/?version=latest)](https://embodirun.readthedocs.io/)

**EmbodiRun 是面向具身智能的部署与执行运行时。** 用 YAML 描述设备、推理服务和计算节点，
由统一运行时管理观测—推理—动作循环。控制进程可以运行在机器人旁，推理运行在 GPU 主机上，也可以部署在同一台机器。

你可以使用 [EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer) 推理引擎，接入外部模型服务，
或让拥有独立规划循环的 Agent 调用机器人。

## 演示

| 三台机器人，共享一个推理服务 | SO-101 上的推理引擎对比 |
|---|---|
| [![三台 SO-101 的录制](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/multi_robot_serving.jpg)](https://embodirun.readthedocs.io/en/latest/demos/multi-robot-serving/) | [![SO-101 引擎对比](https://raw.githubusercontent.com/BUAA-CI-LAB/misc/main/embodirun/v0.1/engine_e2e_contrast.jpg)](https://embodirun.readthedocs.io/en/latest/demos/engine-e2e-contrast/) |
| 三台 SO-101 共享一个 π0.5 推理服务，每台设备运行独立的 rollout 进程。 | Jetson AGX Thor 与 SO-101 上的 EmbodiInfer HTTP/WirelessComm、SGLang 和原生 LeRobot 对比。 |

点击预览图观看视频，了解配置与测量结果。

## 为什么使用 EmbodiRun？

- **一份配置管理部署。** 集中描述节点、环境、设备与模型绑定，由 Host CLI 准备环境、启动服务、查看状态并关闭部署。
- **多个设备共享推理服务。** 各设备运行独立控制循环，通过 HTTP 或 WirelessComm 连接模型端点。
- **为 Agent 提供机器人接口。** 通过无额外依赖的 Python 客户端观测、请求策略建议、执行、查询和取消；Agent 保留自己的规划器，运行时负责设备所有权与执行。
- **复用执行能力。** 观测采集、录制、动作校验和人工接管由运行时统一处理，硬件差异交给机器人适配器和策略绑定。

## 工作原理

![Host 部署 Control 与推理服务，Control 连接应用、机器人及模型服务](docs/assets/runtime-overview.svg)

**Host** 准备环境并启动部署；**Control** 管理机器人连接、观测和动作执行，**Simulation** 提供仿真环境服务；
**Inference** 将观测转为模型预测。这些服务可以分别运行在不同机器上。完整设计见[架构文档](docs/architecture.md)。

## 性能

### SO-101 真机上的 π0.5

在 Jetson AGX Thor 的录制对比中，推理延迟中位数从 **1,061 ms 降至 162 ms**，
完整控制循环 chunk 从 **3,592 ms 缩短至 2,660 ms**。推理加速缩短了循环，而每个 chunk 的动作播放仍约为 2.45 秒。

| 引擎 | 传输 | 推理延迟 | 完整 chunk 耗时 |
|---|---|---:|---:|
| **EmbodiInfer** | WirelessComm | **162 ms** | **2,660 ms** |
| EmbodiInfer | HTTP | 170 ms | 2,666 ms |
| SGLang | HTTP | 194 ms | 2,713 ms |
| 原生 LeRobot | HTTP | 1,061 ms | 3,592 ms |

每次运行统计 15 个 chunk 的中位数，使用相同 SO-101 权重、10 步去噪、两个相机，以及按 20 Hz 播放的 50 步动作块。
EmbodiInfer 使用优化路径，SGLang 使用上游默认配置，LeRobot 使用 eager 执行。
[演示报告](docs/demos/engine-e2e-contrast.md)列出硬件、引擎配置与分段计时。

传输性能见 [HTTP/WirelessComm 实验](docs/inference-transport.md)，模型离线性能见
[EmbodiInfer](https://github.com/BUAA-CI-LAB/EmbodiInfer#performance)。

## 快速开始

### 在本机体验运行时

使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/) 0.12.x 从源码安装：

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run python examples/run_shared_device_fake.py
```

示例启动本地 Control 服务，以模拟关节和虚拟相机演示观测、执行、录制与取消，然后关闭服务。
不需要机器人、模型权重或 GPU。

### 接入机器人或仿真器

继续阅读[快速开始](docs/zh/quickstart.md)了解部署 CLI。使用硬件前，先从[支持矩阵](docs/support-matrix.md)选择组合，
配置设备与标定，并阅读[安全说明](docs/safety.md)。

## 支持的集成

### 机器人与仿真器

| 设备或环境 | 策略 | 推理后端 | 集成情况 |
|---|---|---|---|
| SO-101 | π0.5 | EmbodiInfer | [部署配置](configs/http-wireless-inference/http.yaml)、软件测试、[真机演示](#演示) |
| Bi-SO-101 | π0.5 | EmbodiInfer | [双臂部署](docs/pi05-bi-so101.md)、软件测试 |
| Franka FR3 | π0.5 | 策略服务 API | 适配器与绑定、软件测试 |
| ARX5 | DM0.5 | EmbodiInfer | 实验性适配器与绑定 |
| Unitree Go2 | StreamVLN | EmbodiInfer | 实验性机器人 Agent 与绑定 |
| LIBERO | π0.5 | EmbodiInfer / SGLang | [部署配置](configs/simulation/)、软件测试 |
| VLABench | π0.5 | EmbodiInfer | 实验性[部署配置](configs/simulation/vlabench-pi05-vvla.yaml) |
| Habitat | StreamVLN | EmbodiInfer | 实验性[部署配置](configs/simulation/habitat-streamvln-vvla.yaml) |
| Isaac Sim | StreamVLN | EmbodiInfer | 实验性[部署配置](configs/simulation/isaac-streamvln-vvla.yaml) |

[完整支持矩阵](docs/support-matrix.md)列出各组合的硬件要求、可选依赖与测试覆盖。
SO-101 共享推理使用独立单臂客户端；Bi-SO-101 使用协同控制的双臂策略。

### Agent 与外部服务

| 集成 | 接入方式 | 指南 |
|---|---|---|
| 自有 Agent 或规划器 | Python 客户端：观测、策略建议、执行、查询、取消 | [Agent 执行流程](docs/agent-workflow.md) |
| RPent | 实验性 Agent 适配器 | [RPent 集成](docs/rpent-integration.md) |
| 外部推理服务 | 版本化策略 API 与 provider 配置 | [推理协议](docs/http_api.md)、[配置](docs/configuration.md) |
| XLeRobot | 实验性、单独安装的硬件所有者包 | [Owner 集成](integrations/xlerobot_owner/README.md) |

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
