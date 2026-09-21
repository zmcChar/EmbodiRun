# XLeRobot 薯片递送 Demo

这个 Demo 展示一个具身 Agent 如何驱动 XLeRobot 到取物点，用视觉语言动作（VLA）模型抓取一包薯片，携物返回，再伸展右臂把薯片递给人。本文是面向公开仓库的集成 recipe：说明接口边界和需要准备的硬件，把路线和录屏留给实际部署者维护。

> **Demo 视频 / GIF 占位**
>
> 视频或 GIF 准备好后放在这里。配文应说明路线来源、模型、计算设备和播放倍速。

![XLeRobot 薯片递送 Demo 的系统组成与 Model / Agent 分工](assets/xlerobot-snack-delivery.svg)

这份 SVG 保留为可编辑文字，后续可以替换标签、测量结果或最终媒体链接，不需要重新绘制系统图。

## 场景说明

场景包含一个交付点、一个放置薯片的取物桌，以及两者之间可通行的路线：

1. RPent 接收“帮我拿一包薯片”之类的任务，并选择去程路线。
2. EmbodiRun 驱动 XLeRobot 到取物点，从 owner 服务取得新鲜观测。
3. EmbodiInfer 提供 π0.5 风格的 VLA 推理。模型读取相机图像、机器人状态和抓取指令，返回候选动作。
4. EmbodiRun 校验短动作前缀，通过 XLeRobot owner 执行，并等待反馈确认已经持物。
5. RPent 选择返程路线。机器人到达交付点后，使用本机标定的右臂姿态伸臂，把薯片递给人。

伸臂姿态、夹爪开度、路线几何和动作限幅都与具体机器人有关，必须在使用者自己的机器上标定。

## Recipe 是什么

Recipe 是一个场景的公开、可复现运行配方：把任务阶段、硬件角色、模型接口、路线输入和启动检查组织在一起。它不是私有录制轨迹的副本，也不包含轨迹、访问 token、检查点、串口路径或场地地图。

本例支持两种路线输入：

- **使用者录制路线。** 在目标场地遥操作小车，保存去程和返程路线文件，再在自己的部署配置中引用。
- **交给 RPent 的路线示意图。** 在简单图中标出交付点、取物点和障碍物，让 RPent 选择或组合本地已经命名的路线片段。示意图是规划输入，不会直接变成电机指令。

两种输入都会进入同一套 recipe 阶段。路线理解和规划留在 RPent 中，EmbodiRun 不再实现第二套 planner。

## 系统组成与 Model / Agent 分工

| 组件 | 在这个 Demo 中负责什么 |
| --- | --- |
| **RPent** | 理解任务，选择或规划路线，调用工具，编排取物、返程和交付阶段，并解释反馈。 |
| **π0.5 / VLA** | 根据当前图像、机器人状态和抓取指令生成候选抓取动作。 |
| **Astra（可选）** | 只读评审新鲜观测与 VLA 候选动作，可以建议 hold 或 approve，但不能绕过运行时。 |
| **EmbodiRun** | 部署服务、共享观测、调用模型 API、校验动作、执行有界前缀、停止并报告反馈。 |
| **EmbodiInfer** | 加载并提供 VLA 检查点，执行推理或优化。 |
| **XLeRobot owner** | 统一访问电机和相机，提供关节 / 轮速反馈，执行限位并确认停止状态。 |

控制边界是 EmbodiRun 的公共 API。Agent 只调用 `observe`、`propose`、`execute`、`inspect`、`cancel` 和 `stop`，不直接打开串口或摄像头。请求和执行规则见 [Agent 执行流程](https://embodirun.readthedocs.io/en/latest/agent-workflow/)、[Control](https://embodirun.readthedocs.io/en/latest/control/) 和 [Safety](https://embodirun.readthedocs.io/en/latest/safety/)。

## 硬件清单

下面是规划清单，不是购买报价。价格是从链接的上游页面整理出的近似值，购买前需要按地区、税费、运费和供货情况重新核对。

| 项目 | 本场景是否需要 | 规划参考 |
| --- | --- | --- |
| XLeRobot 双轮移动底盘与两条 SO-101 follower 臂 | 是 | XLeRobot 上游 README 给出的基础自购版本起价约 **660 美元**，不含 3D 打印、工具、运费和税费。见 [XLeRobot](https://github.com/Vector-Wangel/XLeRobot) 及[双轮装配指南](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble_2wheel.html)。 |
| 前视相机、左腕相机、右腕相机 | 是 | 使用稳定的 owner 相机角色和时间戳。上游项目提供相机升级示例；本 recipe 不绑定品牌。见 [LeRobot 文档](https://huggingface.co/docs/lerobot/)。 |
| 电池、充电器、电机线束、安装件和物理急停 | 是 | 按装配后的底盘和机械臂选型。标定和测试期间必须能触达急停。 |
| Linux 计算主机 | 是 | AGX Orin 64 GB 级别设备可以作为本机主机。NVIDIA 在 [Jetson FAQ](https://developer.nvidia.com/embedded/faq) 中列出的 AGX Orin 64 GB Developer Kit 价格为 **3,499 美元**。如果部署网络和 API 契约已配置，也可以把推理放在远端。 |
| 操作工作站与网络链路 | 是 | 用于本仓库、配置、日志，以及访问 owner 和 Control 服务的 SSH 或本地链路。 |
| 遥操作控制器 | 仅录制路线模式需要 | 使用 owner 集成支持的控制器，或 owner 提供的键盘路径，在目标场地录制路线。 |

总预算取决于已有设备。XLeRobot 平台的估算不含计算主机、相机、电池、工具和本地加工。购买前请重新查看 [XLeRobot README](https://github.com/Vector-Wangel/XLeRobot)、[SO-101 装配指南](https://huggingface.co/docs/lerobot/main/assemble_so101) 和 [NVIDIA Jetson FAQ](https://developer.nvidia.com/embedded/faq)。

## 软件与配置

一次实机运行需要由部署者分别准备下列部分：

| 层 | 使用者需要准备什么 |
| --- | --- |
| Recipe 主机 | Python 3.10+、本仓库和公共 EmbodiRun 客户端。 |
| XLeRobot owner | owner 集成、唯一串口 owner、稳定设备身份、电机 ID、机械臂标定、相机角色和反馈时间戳。先以只读模式启动，见 [XLeRobot owner](https://github.com/BUAA-CI-LAB/EmbodiRun/tree/main/integrations/xlerobot_owner)。 |
| EmbodiRun Control | 一份部署 YAML，写明 Control runtime、设备 binding、动作限幅、观测新鲜度检查和服务地址。见[配置](https://embodirun.readthedocs.io/en/latest/configuration/)和 [Control](https://embodirun.readthedocs.io/en/latest/control/)。 |
| VLA 服务 | 由 manipulation runtime ID 选择的 π0.5 / VLA 服务。检查点和模型服务凭据留在部署配置中，不提交到仓库。见 [π0.5 与两个 SO-101](https://embodirun.readthedocs.io/en/latest/pi05-bi-so101/) 和 [Inference API v1](https://embodirun.readthedocs.io/en/latest/http_api/)。 |
| RPent | 只通过 EmbodiRun 公共边界调用运行时的任务与路线规划器。见 [RPent 集成](https://embodirun.readthedocs.io/en/latest/rpent-integration/)。 |
| 路线文件 | 使用者录制的去程 / 返程片段，或 RPent 根据示意图生成的路线文件。私有地图和录制文件放在公开仓库之外。 |

公开 README 和本文不会要求用户粘贴 token。认证信息、私有地址、模型检查点和场地路径应放在部署环境或带认证的网关中。

## 配置、启动与复现路径

开始这个场景前，先按仓库的一般安装与安全流程操作：

1. 依据上游硬件指南装配底盘和两条 SO-101 臂，在通电前标记总线和电机。
2. 在目标机器人上配置电机 ID 并分别标定两条机械臂。读取回传的单位和限位，不要直接复制另一台机器的标定文件。
3. 安装前视和腕部相机，按稳定设备身份绑定，确认 owner 能报告新鲜图像、状态和停止状态。
4. 安装 owner 与 Control 服务。先以只读模式检查串口归属、反馈、相机角色和急停，再允许运动。
5. 选择路线来源，用 plan 或 dry run 校验路线文件。不要把录制路线作为本例的一部分公开。
6. 针对交付表面和接收者，重新标定右臂交付姿态与夹爪开度。
7. 以低速、有人监管的方式开始运行。每个有界动作都要检查，并在进入下一阶段前等待新鲜反馈。

[快速开始](quickstart.md)、[Agent 执行流程](https://embodirun.readthedocs.io/en/latest/agent-workflow/)、[Control](https://embodirun.readthedocs.io/en/latest/control/) 和 [Safety](https://embodirun.readthedocs.io/en/latest/safety/) 介绍 CLI 与运行时语义。请求被接受不等于机器人已经完成实体动作；请保留 owner 反馈和最终交付证据。

> **性能材料占位**
>
> 结果公开后，在这里补充端到端延迟、路线时长、抓取成功率、交付成功率、硬件版本、模型版本和测试次数。每个数字都要绑定到明确配置。

> **公开运行材料占位**
>
> 结果准备好后，在这里补充脱敏部署示例、路线示意图或短运行日志。不要放入凭据、私有地图、原始路线轨迹或个人数据。

## 参考链接

- [EmbodiRun 安装](https://embodirun.readthedocs.io/en/latest/installation/)
- [EmbodiRun 架构](https://embodirun.readthedocs.io/en/latest/architecture/)
- [EmbodiRun Inference API](https://embodirun.readthedocs.io/en/latest/http_api/)
- [EmbodiRun RPent 集成](https://embodirun.readthedocs.io/en/latest/rpent-integration/)
- [XLeRobot 源码与成本说明](https://github.com/Vector-Wangel/XLeRobot)
- [XLeRobot 双轮装配](https://xlerobot.readthedocs.io/en/latest/hardware/getting_started/assemble_2wheel.html)
- [LeRobot SO-101 装配与标定](https://huggingface.co/docs/lerobot/main/assemble_so101)
- [LeRobot 文档](https://huggingface.co/docs/lerobot/)
- [NVIDIA Jetson FAQ](https://developer.nvidia.com/embedded/faq)
