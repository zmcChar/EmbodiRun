# XLeRobot 薯片递送 Demo

让 XLeRobot 从交付点出发，到桌前用 VLA 抓取一包薯片，携物返回，再伸展右臂递给人。这个 Demo 将路线执行、视觉抓取和人机交付组织成一个完整的移动操作场景。

<!-- Demo 视频或 GIF 放在这里；素材由演示作者提供。随视频标注路线来源、模型、计算设备和播放倍速。性能数字附对应测量记录。 -->

## Demo 场景

场景由交付点、放置薯片的取物桌和两者之间的通行路线组成。小车到达桌前并停稳后调用抓取模型，确认持物后返回，使用本机标定的右臂姿态完成交付。

## 系统组成与 Model / Agent 分工

![XLeRobot 薯片递送 Demo 的硬件、运行时、Agent 与模型分工](assets/system-overview.svg)

这套组合包含 **XLeRobot 双轮差速底盘、两条 SO-101 机械臂、前视与双腕三路相机，以及 AGX Orin 车载计算设备**。模型计算可按部署配置安排在本机或远端。

| 组件 | 在这个 Demo 中负责什么 |
| --- | --- |
| **RPent** | 理解任务，选择或规划路线，编排去程、抓取、返程与交付工具，并根据反馈判断任务结果。 |
| **π0.5 / VLA** | 根据相机图像、关节状态和抓取指令生成候选动作。 |
| **Astra（可选）** | 在 RPent 侧只读判断场景，给出 VLA 指令与短段执行上限。 |
| **EmbodiRun** | 管理服务部署、共享观测、模型请求、动作校验、执行、停止与反馈。 |
| **EmbodiInfer** | 承载 π0.5 的模型加载、推理与优化。 |
| **XLeRobot owner** | 统一访问真实电机与相机，提供关节和轮速反馈、限位与停止确认。 |

抓取时，观测经 EmbodiRun 送入模型服务，候选动作返回上层；选定的动作再经 EmbodiRun 和 owner 执行。Astra 属于可选的上层评审模型，RPent 是 Agent 框架；伸臂交付使用本机标定姿态。接口与调用顺序见 [架构说明](architecture.md)。

## Recipe 是什么

Recipe 是一个场景的可复现运行配方，把任务步骤、硬件与模型配置、路线输入和启动入口放在一起。本例位于 `recipes/xlerobot/snack_delivery/`，由 [run.py](run.py) 组织任务步骤；路线理解与规划交给 RPent。

## 快速开始

在仓库根目录执行，先安装环境、检查配置，再进行不访问硬件的流程演练：

```bash
scripts/recipes/xlerobot_snack.sh setup
scripts/recipes/xlerobot_snack.sh plan
scripts/recipes/xlerobot_snack.sh dry-run
```

装配与标定完成，并按 [完整运行指南](guide.md#bring-up-and-calibration) 启动和检查 owner、底盘与机械臂 Control 服务后，使用自己的部署配置启动场景：

```bash
XLR_SNACK_CONFIG=/path/to/deployment-snack.json \
  scripts/recipes/xlerobot_snack.sh hardware
```

该命令启动 recipe；硬件服务与模型服务按运行指南准备。设备、模型、路线与交付姿态的配置入口见 [config.example.json](config.example.json)。

## Route 来源

- **录制路线**：在自己的场地遥操作小车，分别记录去程与返程，并导出 recipe 使用的路线文件。
- **路线示意图**：向 RPent 提供标有交付点、取物点与障碍物的示意图，由 RPent 选择本地已录制、已标注的路线片段；无比例尺图片不直接变成电机指令。

两种方式使用同一任务流程。路线输入与文件格式见 [路线配置说明](guide.md#route-input-choices)。

## 详细文档

| 想了解什么 | 从这里开始 |
| --- | --- |
| 买什么硬件、预算多少、如何搭车与标定 | [硬件和软件清单](hardware.md) |
| 如何配置环境、服务、路线和交付动作 | [完整运行指南](guide.md) |
| Agent、模型、Runtime 与硬件如何配合 | [架构与接口说明](architecture.md) |
| 场景参数与代码入口 | [配置示例](config.example.json) · [run.py](run.py) |
| 安装与启动脚本 | [场景启动脚本](../../../scripts/recipes/xlerobot_snack.sh) · [环境安装脚本](../../../scripts/recipes/xlerobot_snack_setup.sh) |
| 硬件服务的安装与生命周期 | [XLeRobot owner](../../../integrations/xlerobot_owner/README.md) |

## References

- [XLeRobot](https://github.com/Vector-Wangel/XLeRobot)：机器人平台、装配与硬件设计。
- [LeRobot](https://github.com/huggingface/lerobot)：SO-101 与机器人软件生态。
- [RPent](https://github.com/RLinf/RPent)：Agent 框架与任务编排。
- [RPent 适配说明](../../../agents/rpent/README.md) 与 [Astra / π0.5 协作接口](../../../agents/astra_pi05/README.md)。
- 更多装配教程、软件文档和采购参考见 [hardware.md](hardware.md#official-references)。
