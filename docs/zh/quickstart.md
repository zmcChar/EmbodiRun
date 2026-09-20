# 快速开始

本文走一遍第一次运行。先是一个不需要硬件的例子，然后是真实部署的形态。

!!! note "翻译进度"
    本页是[英文版](https://embodirun.readthedocs.io/en/latest/quickstart/)的译文。
    英文版始终是最新的；如有出入，以英文版为准。

## 0. 安装

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiRun.git
cd EmbodiRun
uv sync --frozen
uv run pytest -q
```

能力分组与 optional extras 见[安装（英文）](https://embodirun.readthedocs.io/en/latest/installation/)。

## 1. 不需要机器人

`examples/shared-device-fake.yaml` 会启动一个只有机器人、没有模型和推理端点的 Control
服务。它使用内存中的模拟关节和一台假相机，不打开任何硬件，但暴露真实的 Host JSON API。

```bash
CONFIG=examples/shared-device-fake.yaml

uv run embodirun --config "$CONFIG" validate
uv run embodirun --config "$CONFIG" init
uv run embodirun --config "$CONFIG" up

uv run embodirun --config "$CONFIG" describe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config "$CONFIG" observe \
  --runtime fake-device --caller-id example-agent --session-id example-session --json
uv run embodirun --config "$CONFIG" down
```

`describe` 报告能力，`observe` 返回一条共享观测，`media` 按观测 ID 取帧数据。
执行走 `execute`，只有在配置了模型或动作来源时才有意义。包含录制与取消的完整自足演练见
[`examples/README.md`](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md)。

## 2. 校验一份真实配置

复制一份示例，并在接触硬件之前替换掉每一个占位符：

```bash
cp configs/pi05/bi-so101-vvla.yaml my-deployment.yaml
$EDITOR my-deployment.yaml

uv run embodirun --config my-deployment.yaml validate
```

`validate` 是静态的：它检查引用、端口、binding 和必填字段，不会联系任何节点。
见[配置（英文）](https://embodirun.readthedocs.io/en/latest/configuration/)。

## 3. 准备并启动

```bash
uv run embodirun --config my-deployment.yaml probe   # 连通性与工具
uv run embodirun --config my-deployment.yaml init    # 环境与源码
uv run embodirun --config my-deployment.yaml up      # 启动服务
```

`init` 会在本地记录成功状态；如果配置在 `init` 之后被改动过，`up` 会拒绝启动。
启动 Control 会加载静态配置并检查模型健康和相机，但不会连接机械臂，也不会让它运动。

## 4. 跑一个任务

```bash
uv run embodirun --config my-deployment.yaml run \
  --runtime so101-1-runtime \
  --prompt "Pick up the cube and put it into the bowl." \
  --chunk-steps 10 \
  --max-steps 1
```

| 参数 | 含义 |
|---|---|
| `--runtime` | 要使用的已配置 runtime ID。 |
| `--prompt` | 指令覆盖。仿真器可以自带指令。 |
| `--task`、`--seed` | 仿真器任务与 episode 种子。 |
| `--chunk-steps` | 每个推理 chunk 执行多少步动作（受 binding 上限约束）。 |
| `--max-steps` | 最多执行多少个推理/动作 chunk（默认 1）。 |
| `--control-hz` | 动作回放频率（默认 5）。 |
| `--request-timeout` | 单次推理请求的超时（默认 60 s）。 |

π0.5 的响应里没有任务完成信号，所以 chunk 数量上限永远是停机条件。
请求的 chunk 长度超过 binding 上限时，会在连接机器人之前就被拒绝。

## 5. 停止

```bash
uv run embodirun --config my-deployment.yaml down
```

`down` 按进程身份停止服务，并且在配置变更之后仍然可用。
`--target control` 或 `--target model` 可以把它限制在某一类服务上。

## 疑难排查

- **`uv` 版本报错** —— 安装 uv 0.12.x，仓库锁定了这个版本区间。
- **`validate` 报某个路径失败** —— 替换掉每一个 `REPLACE_*` 占位符；
  缺失的相机或标定路径会在执行任何动作之前就被报出来。
- **`up` 拒绝运行** —— 先跑 `init`，或者 YAML 改过之后重新跑 `init`。
- **服务不健康** —— 查看 `up` 打印的每个服务的日志路径；
  模型健康并不保证一次推理请求一定能成功。
- **`sync` 要求先停止 Control** —— 已经在运行的进程早已导入了旧模块。

人工控制与软件急停见 [Control（英文）](https://embodirun.readthedocs.io/en/latest/control/)
和 [Safety（英文）](https://embodirun.readthedocs.io/en/latest/safety/)。
