# HTTP / WirelessComm Inference Benchmark

比较同一台 Thor 上的 PI0.5 服务，经 HTTP 和 WirelessComm 为两台 Orin 返回动作的端到端性能。
每个 Orin 各运行一个客户端，最多一个未完成请求；两端预热完成后，Host 同时放行。
默认每端预热 20 次、正式测量 200 次，共 400 次正式请求。模型服务当前按 B=1 处理请求；双客户端测试包含争用和排队。

## 拓扑与配置

| 节点 | SSH | 角色 |
|---|---|---|
| Thor | `operator@192.0.2.10` | PI0.5，`/home/user/models/pi05_so101` |
| AGX Orin | `operator@192.0.2.10` | `so101-1-runtime`，串口尾号 `5C4C125563`，两台相机 |
| Orin NX | `jetson@192.168.2.148` | `so101-2-runtime`，串口尾号 `5C4C125310`，两台相机 |

- HTTP：[http.yaml](../../configs/http-wireless-inference/http.yaml)。Thor 监听 `8000`。
- WirelessComm：[wireless.yaml](../../configs/http-wireless-inference/wireless.yaml)。Thor 和两个客户端分别在各自节点监听 `9300`。

两份配置使用相同 checkpoint、BF16、10 步去噪、完整循环 CUDA Graph 和 50×6 动作输出。
AGX Orin 沿用已存在的 `thor_so101_follower` 标定文件名，文件实际位于 `/home/user/`。
WirelessComm 未额外设置带宽限速；这是同一 LAN 上的协议对比，不是 Wi-Fi 与有线网络的对比。
`models.<id>.server_args` 将额外模型参数传给对应 VVLA 服务入口。

## 输入和计时口径

输入为 JSONL，每行一条 SO101 观测。图片路径相对 JSONL 所在目录：

```json
{"instruction":"Pick up the cube.","source":"recorded-so101","state":{"joint_positions_deg":[0,0,0,0,0],"gripper_position":0},"images":{"observation.images.front":"front.jpg","observation.images.wrist":"wrist.jpg"}}
```

使用录制的相机图像、关节状态和指令。两种协议及两个客户端都重放同一 manifest，按文件顺序循环取样。
Host 预加载图片并传给两个客户端；SHA-256 覆盖指令、状态、相机名称、编码图片字节和样本顺序。
这里没有公开数据集适配或任务成功率评测，不能把合成输入的结果称为真实任务性能。

| 指标 | 定义 |
|---|---|
| E2E mean / P50 / P95 / P99（ms） | Orin 上构造请求开始，至收到完整响应并通过 SO101 动作映射和 50 行校验结束 |
| RPC 延迟（ms） | `client.step`，包含请求序列化、传输、服务排队、预处理、模型计算、后处理及响应解析 |
| 每端 calls/s | 该端成功请求数 / 该端正式测量总耗时 |
| aggregate calls/s | 两端成功请求总数 / Host 发出开始信号至收到两端完成信号的耗时 |
| 失败 | 保留失败尝试及耗时；首个失败后停止该端，不重试，整轮标记失败 |

E2E 排除文件读取、SSH 输入分发、预热、建连/建会话、传感器采集及机械臂运动。
两端各自使用单调时钟，延迟不依赖设备时钟同步；aggregate 的分母包含少量 SSH 开始/完成信令开销。
返回的 `server_timing_ms` 原样保留，仅作辅助分析，不把 RPC 与模型耗时之差当成纯网络延迟。
每次请求使用独立 ID，预热与正式请求不会命中幂等结果缓存。
两端完成测量并分别关闭服务器会话后，Host 才同时允许传输连接退出；会话关闭失败保留测量数据，并将整轮标记失败。
`benchmark.py` 只使用推理客户端和值映射，不打开机械臂串口或摄像头，也不执行动作。

## 真实观测采集

不需要公开数据集或仿真器。先在两台 Orin 上各采集前视/腕部相机图像和对应关节、夹爪状态，再重放同一份输入。
先在 Host 初始化、生成配置并停止控制服务（见下方运行步骤），再同步只读适配器：

```bash
uv run rlinf-deploy --config configs/http-wireless-inference/http.yaml sync --target deploy
```

`sync` 只同步 Deploy 包；将 `capture.py` 另行复制到各 Orin，例如 `~/capture.py`。
在对应 Orin 的 SO101 Python 环境中运行，并指定更新后的源码路径，例如 AGX Orin：

```bash
PYTHONPATH="$HOME/.local/share/rlinf-deploy/thor-so101-http/overlays/deploy/current/src" \
  "$HOME/.local/share/rlinf-deploy/thor-so101-http/sources/deploy/.venv-robot-so101/bin/python" \
  ~/capture.py \
  --config ~/.local/share/rlinf-deploy/thor-so101-http/generated/control-so101-1-runtime.control.json \
  --output ~/so101-observations \
  --prompt 'Pick up the cube and place it in the bowl.' \
  --count 24 --interval 0.5
```

Orin NX 使用自己的 `control-so101-2-runtime.control.json`。配置文件由 Host `up` 生成；采集前停止控制服务，释放串口和相机。
脚本和 Deploy 源码都应使用包含本次修改的版本。输出目录必须不存在；采集失败可能留下部分文件，只有命令成功退出的录制可作为完整输入。

`capture.py` 使用 `SO101Adapter(..., read_only=True)`：读取并核对电机标定，读取位置；连接和断开都不写电机寄存器，也不改变力矩状态。该连接拒绝执行动作和保持姿态命令。
相机使用 640×480、30 FPS、JPEG quality=90，预热 3 帧。每条记录顺序采集两路图像、读取状态，记录采集时刻和耗时；不宣称硬件同步。
默认各采 24 组、间隔 0.5 秒。把两台设备的录制目录复制到 Host，合并 manifest 时将图片路径加上各自目录前缀，即可获得 48 组共用输入。
正式测量前人工检查图像清晰度、照明、相机位置与关节状态；黑暗图像只能用于链路诊断，其 JPEG 字节数不能代表正常场景。

这一步验证真实硬件的观测获取和推理返回。机械臂实际执行及闭环任务成功率需要另外验证，不包含在本 benchmark 的计时中。

## 运行

在当前 Deploy checkout 的 Host 环境执行。示例的 SSH 管理连接通过 Host 上的 SOCKS5 `127.0.0.1:1080`，需要本机 `nc`；Host 可直连实验室网段时，删除三处 `connection.proxy_command` 即可。
三台设备之间仍直接使用配置中的 LAN 地址交换推理请求。
依赖、标定和模型资源通过既有 `init` 检查；先停止旧部署，避免其服务占用端口。两种协议逐轮运行，共用一台 GPU，禁止同时启动两个推理服务进行对比。

```bash
uv sync --frozen

# 没有录制输入时，可生成固定 640×480 RGB PNG，仅用于链路 smoke test。
# 随机 PNG 数据量较大，正式结果应换成真实相机编码输入。
uv run python benchmarks/http-wireless-inference-benchmark/benchmark.py fixture \
  --output artifacts/so101-input

uv run rlinf-deploy --config configs/http-wireless-inference/http.yaml validate
uv run rlinf-deploy --config configs/http-wireless-inference/http.yaml init
uv run rlinf-deploy --config configs/http-wireless-inference/http.yaml up --wait-timeout 600
# benchmark 替代控制客户端，保留模型服务；释放 WirelessComm 所需的客户端端口。
uv run rlinf-deploy --config configs/http-wireless-inference/http.yaml down --target control
uv run python benchmarks/http-wireless-inference-benchmark/benchmark.py run \
  --config configs/http-wireless-inference/http.yaml \
  --observations artifacts/so101-input/observations.jsonl \
  --output artifacts/so101-http.json
uv run rlinf-deploy --config configs/http-wireless-inference/http.yaml down

uv run rlinf-deploy --config configs/http-wireless-inference/wireless.yaml init
uv run rlinf-deploy --config configs/http-wireless-inference/wireless.yaml up --wait-timeout 600
uv run rlinf-deploy --config configs/http-wireless-inference/wireless.yaml down --target control
uv run python benchmarks/http-wireless-inference-benchmark/benchmark.py run \
  --config configs/http-wireless-inference/wireless.yaml \
  --observations artifacts/so101-input/observations.jsonl \
  --output artifacts/so101-wireless.json
uv run rlinf-deploy --config configs/http-wireless-inference/wireless.yaml down

uv run python benchmarks/http-wireless-inference-benchmark/benchmark.py compare \
  --http artifacts/so101-http.json --wireless artifacts/so101-wireless.json
```

`run` 自动使用 `init` 状态中两个 Orin 的 Python 环境，通过 SSH 分发当前 worker 代码与输入，无需在设备上手动复制脚本或安装额外 benchmark 包。
如果 Host 使用了自定义 `--state-dir`，`run` 也需要传入同一个目录。
benchmark 自动复用 `connection.proxy_command`。额外的 OpenSSH 选项可用 `--ssh-option` 传入。

`--runtime so101-1-runtime` 或 `--runtime so101-2-runtime` 可测单客户端。建议分别完成 AGX Orin 单端、Orin NX 单端和双端并发三组 HTTP/Wireless 对照。
正式实验建议交替协议顺序各重复至少 3 轮，使用不同输出文件，并记录设备功耗模式、温度及其他负载。保持两轮 checkpoint 内容不变；报告记录路径和代码版本，不重新散列全部模型权重。

## 产物与结果

JSON 包含每次请求的延迟、样本序号、动作行数、模型计时和版本，以及每端摘要、输入哈希、参数、并发规模和整体吞吐。
同目录的 `*.stderr.log` 保留 SSH/客户端诊断。已有报告不会覆盖。
`compare` 拒绝失败轮次，以及数据、客户端、模型参数、脚本或部署/推理版本不一致的结果。
产物默认写入已忽略的 `artifacts/`，不提交图片、日志或性能报告。

### 2026-09-07 真机采集与链路诊断

两台 SO101 各录制 24 组真实观测，共 48 组、96 张 JPEG；六个电机的标定寄存器均与各自文件匹配。
指令为 `Pick up the object and place it in the container.`。
四路相机画面几乎全黑：超过 99.5% 的像素灰度低于 16/255，前视画面仅能看到少量指示灯。
AGX Orin / Orin NX 的每组两张 JPEG 平均合计仅 11,106 / 12,164 字节。
因此下面结果只验证这组黑暗观测的真实设备链路，不能作为正常场景的最终性能或抓取成功率。

- 三台设备功耗模式均为 MAXN；保留已有系统服务，未改频率或风扇设置。
- Thor：Python 3.12.3、Torch 2.10.0+cu130、LeRobot 0.5.1、Transformers 5.3.0。
- 两台 Orin：Python 3.12.13、Torch 2.11.0、LeRobot 0.6.1；两端不加载 PI0.5，只运行 Deploy 推理客户端。
- 三端 WirelessComm 0.1.0、OpenCV headless 4.13.0.92。
- 使用独立轻量虚拟环境复用设备已有平台依赖；加载当前 Deploy 源码及固定 Inference 源码，不重新安装 GPU 依赖。
- Deploy 基线 `ac74a0f84783d313d757ac87b5a617b40f2ab955` 加本次工作区修改；Inference `06e82ca80f6385079ef7390e32466180088b4d80`。
- 输入 SHA-256：`64ec3ce094b335b9edce0aa8f7931a6d3ea7cf6ee738ca1a65b69b48f056919f`。产物目录 `artifacts/so101-real-20260907/` 保存环境清单、源码哈希和修改补丁。

两种协议各完成 400 次正式请求，全部返回 50×6 动作，客户端会话清理成功。以下为使用同一版本脚本、同一输入、每端预热 20 次后的单轮结果；不据此判断正常场景下哪种协议更快。

| 协议 | 客户端 | 成功请求 | E2E mean（ms） | P50（ms） | P95（ms） | P99（ms） |
|---|---|---:|---:|---:|---:|---:|
| http | AGX Orin | 200 | 434.81 | 435.05 | 443.28 | 444.04 |
| http | Orin NX | 200 | 433.74 | 435.88 | 442.60 | 444.54 |
| wireless | AGX Orin | 200 | 419.31 | 421.62 | 425.42 | 426.87 |
| wireless | Orin NX | 200 | 420.34 | 421.63 | 425.32 | 426.92 |

总吞吐：HTTP **4.595 calls/s**；WirelessComm **4.755 calls/s**。

原始报告为 [http-dark-v2.json](reports/baseline/http-dark-v2.json)、[wireless-dark-v2.json](reports/baseline/wireless-dark-v2.json)，比较工具已通过输入、参数和版本一致性检查。前一轮失败报告单独保留，不纳入上表。

对两台机械臂的首组观测分别检查了真实 PI0.5 输出，均返回 50×6 有限数值。
AGX Orin / Orin NX 的首个预测目标与当前姿态最大关节差约为 113.0° / 56.2°，夹爪差约为 21.5 / 15.5，均超过配置的单步限制。
没有向电机下发模型动作。正常照明和可见工作空间确认后，才可另行进行限幅动作与闭环任务验证。

固定 Inference 版本还存在 WirelessComm 退出问题：`WirelessPolicyServer._serve_peer` 的 `recv` 抛出客户端断开异常后，整个服务退出，影响另一客户端的后续 RPC。
本地真实双客户端测试已复现，并验证了候选上游修复（断开后另一端继续服务、原端重新连接，4 项测试通过）；补丁保存在实验产物中，尚未应用。
benchmark 的结束屏障确保两端先关闭会话，再断开传输连接；实测中该版本模型服务在两端退出后再次异常结束。下一轮需要重新启动模型服务，持续部署需要修复上游。

### 推理侧分段 profile

`profile_inference.py analyze` 从逐请求报告汇总分段时间；差值先按请求计算再统计，保留负值，不把 P95 相减当作某阶段的 P95。

```bash
uv run python benchmarks/http-wireless-inference-benchmark/profile_inference.py analyze \
  --reports artifacts/so101-real-20260907/http-dark-v2.json \
            artifacts/so101-real-20260907/wireless-dark-v2.json \
  --output artifacts/so101-profile-summary.json
```

原报告可以区分客户端动作映射、`policy_ms` 和 `RPC - policy_ms`。后者仍包含网络、协议编解码、服务路由等开销，不是纯网络延迟。
固定 Inference 中，`PolicyService.step` 覆盖了适配器的 `policy_ms`，这一计时包含等待 PI0.5 共享锁、预处理、模型计算和后处理；引擎原有的 CUDA 分段计时未被传回。

`profile_inference.py serve` 是独立的诊断入口，只加载一次 PI0.5，通过 stdin 接收 `{"action":"start","transport":"http"}`、`{"action":"stop"}`、`{"action":"start","transport":"wireless"}` 和 `{"action":"exit"}`。
每次只启动一种协议，HTTP 和 WirelessComm 复用同一个已加载的模型、处理器和 CUDA Graph；每轮建立新的协议服务及会话。
启动参数包含 `--checkpoint`、`--adapter-config`、`--comm-config`；后两者分别使用 Host 生成的 PI0.5 适配器 JSON 和模型 WirelessComm JSON。
所有模型依赖仅在 Thor 的推理环境中导入。该入口用方法包装和线程局部记录采样，不修改固定的 Inference 文件，也不额外插入 CUDA 同步。

| 新增计时 | 口径 |
|---|---|
| `profile_lock_wait_ms` | 等待 PI0.5 共享推理锁的墙钟时间 |
| `profile_state_ms` / `profile_images_ms` | 状态转换 / JPEG 解码及图像 Tensor 构造的调用时间 |
| `profile_prepare_ms` | 模型输入处理器的调用时间 |
| `profile_engine_wall_ms` | 整个 `EngineCore.execute` 调用，包括其同步和动作搬回 CPU |
| `profile_core_prefill_ms` / `profile_core_decode_ms` | 引擎已有 CUDA event 的流时间，分别覆盖前缀阶段和去噪阶段；不是纯 kernel 执行时间之和 |
| `profile_core_e2e_ms` | 引擎已有墙钟计时，在动作打包前结束 |
| `profile_restore_ms` | 动作反归一化等后处理的调用时间 |
| `profile_adapter_other_ms` | 适配器剩余工作，包括返回动作检查、转换为列表和计时包装开销 |

CUDA 分段是 engine wall 内部的辅助指标，不能再加到端到端分解中。独占墙钟阶段可以相加；不同进程的绝对时钟不用于计算单向网络延迟。
诊断响应增加计时字段，profile 结果与原始 benchmark 分开保存。

### 2026-09-08 推理 profile 实测

本次在 Thor 上只加载一次模型（PID `458351`），复用上述 checkpoint、输入和优化配置。
两个 Orin 到 Thor 的路由均走 `eno1` 有线 LAN。只重放已有真实观测，没有重新采集，也没有打开串口或执行动作。
每轮每端预热 20 次；先测 AGX Orin 单客户端 HTTP / WirelessComm，各 50 次正式请求；
再按 HTTP → WirelessComm → WirelessComm → HTTP 顺序测双客户端，每轮每端 100 次。
共 900 次正式请求全部成功，均返回 50×6 动作。双客户端每种协议汇总 400 次；吞吐按总请求数除以各轮测量时间之和计算。

| 场景 | HTTP E2E mean | WirelessComm E2E mean | E2E 降幅 | HTTP / WirelessComm 总吞吐 | 吞吐增幅 |
|---|---:|---:|---:|---:|---:|
| AGX Orin 单客户端 | 217.25 ms | 207.13 ms | 4.65% | 4.586 / 4.811 calls/s | 4.91% |
| AGX Orin + Orin NX 双客户端，合并两轮 | 437.03 ms | 418.12 ms | 4.33% | 4.560 / 4.765 calls/s | 4.49% |

双客户端分轮 E2E mean：HTTP `437.82 / 436.23 ms`，WirelessComm `416.47 / 419.76 ms`。
合并后的 E2E P95 为 `442.37 / 421.92 ms`，P99 为 `443.85 / 423.12 ms`（HTTP / WirelessComm）。
单客户端只有一轮，双客户端每种协议只有两轮；表中的百分比是本次观测值，不是跨设备或网络条件的稳定保证。

双客户端逐请求独占阶段的均值如下，可相加得到 E2E。正差值表示 WirelessComm 在该阶段更短。

| 阶段 | HTTP（ms） | WirelessComm（ms） | HTTP − WirelessComm（ms） |
|---|---:|---:|---:|
| 客户端请求 / 动作映射 | 0.63 | 0.64 | −0.01 |
| RPC 外围：网络、协议编解码、服务路由等 | 7.89 | 6.17 | 1.72 |
| 等待共享推理锁 | 208.41 | 200.92 | 7.49 |
| 输入预处理：状态、图像、processor | 13.32 | 10.11 | 3.21 |
| 引擎调用，包括同步与动作搬回 CPU | 203.43 | 197.58 | 5.85 |
| 动作后处理、适配器及服务计时区间剩余工作 | 3.34 | 2.69 | 0.65 |
| **E2E** | **437.03** | **418.12** | **18.91** |

输入预处理内部，图像解码 / Tensor 构造为 `10.41 / 7.53 ms`。
引擎内已有 CUDA event 分段：prefill `92.77 / 89.91 ms`，10 步去噪 `109.91 / 107.40 ms`。
这些 CUDA 流时间已经包含在引擎调用中，不另行累加。

本次可支持的判断：

- 当前实现的 WirelessComm 在这组负载下带来约 **4.5% 总吞吐提升**。此前独立启动模型的结果为 3.47%，两次测量都属于几个百分点的收益。
- 通信及 RPC 外围只减少 **1.72 ms**，不能把全部 **18.91 ms** 的 E2E 差值称为网络提速。旧报告的 14.45 ms 差值中，该外围区间也只减少约 2.02 ms；加上客户端映射后，共减少约 1.99 ms。
- 双客户端时，引擎调用加锁等待占 E2E 的 **94%–95%**；单客户端锁等待不足 0.001 ms。增加到两个客户端后，吞吐仍约 4.6–4.8 calls/s，而各端请求延迟接近翻倍，说明当前 `serialized_b1` 服务的主要限制是模型处理能力及其排队。
- 相同模型在两个入口中的预处理和 CUDA 流时间仍有差异，表明独立加载不是唯一干扰。当前 HTTP 客户端使用 `urllib.request.urlopen`，每个请求携带 `Connection: close`（[Python 3.12 文档](https://docs.python.org/3.12/library/urllib.request.html#urllib.request.urlopen)）；服务端是每连接一个线程的 `ThreadingHTTPServer`。WirelessComm 复用连接，通过 `asyncio.to_thread` 使用线程池。线程复用、调度与设备频率可能影响服务内部时间；本次未做线程复用消融，也未逐请求记录频率，因此不能确定这些差值各自的根因，更不能称为 WirelessComm 加速了模型算子。

下一步若要比较协议本身，优先增加 HTTP 连接复用的对照，使线程复用条件更接近，再重复上述分段测量。
本次每请求两路黑暗 JPEG 平均合计约 11.6 kB，不代表正常照明图像、大数据传输、Wi-Fi 或权重同步场景；训练侧收益需要独立评测。

最终报告副本随 snapshot 保存在本目录已忽略的 `reports/` 下；原始实验目录仍为 `artifacts/so101-profile-20260908/`，原始 JSON 内容未改写。
单端原始报告：[HTTP](reports/profile/http-agx.json)、[WirelessComm](reports/profile/wireless-agx.json)。
双端原始报告：[HTTP 第一轮](reports/profile/http-dual-1.json)、[第二轮](reports/profile/http-dual-2.json)，
[WirelessComm 第一轮](reports/profile/wireless-dual-1.json)、[第二轮](reports/profile/wireless-dual-2.json)。
[逐轮汇总](reports/profile/summary.json)、[合并汇总](reports/profile/pooled-summary.json)保留各阶段分布。
[metadata.json](reports/profile/metadata.json) 记录 profile 脚本 SHA-256，服务日志记录各轮相同的模型 PID。

![单端与双端推理耗时分解](reports/profile/inference-breakdown.png)

```bash
uv run python benchmarks/http-wireless-inference-benchmark/profile_inference.py analyze \
  --reports benchmarks/http-wireless-inference-benchmark/reports/profile/http-agx.json \
            benchmarks/http-wireless-inference-benchmark/reports/profile/wireless-agx.json \
            benchmarks/http-wireless-inference-benchmark/reports/profile/http-dual-1.json \
            benchmarks/http-wireless-inference-benchmark/reports/profile/wireless-dual-1.json \
            benchmarks/http-wireless-inference-benchmark/reports/profile/wireless-dual-2.json \
            benchmarks/http-wireless-inference-benchmark/reports/profile/http-dual-2.json \
  --output artifacts/so101-profile-recomputed.json
```

原版 WirelessComm 服务仍在客户端退出后触发前述 `ConnectionClosedError`；本次三轮均记录在 `lifecycle.json`，没有修改上游实现。
诊断入口在每轮结束后回收该协议任务，再启动下一轮服务，因此可以保留同一个模型；这不等于修复持续部署的断连问题。
实验结束后已退出诊断进程，并确认 Thor 的 `8000 / 9300` 端口释放。

### 实验 snapshot

本次实验整理为 [20260908011603.tar.gz](../snapshots/20260908011603.tar.gz)，校验文件为 [SHA256](../snapshots/20260908011603.tar.gz.sha256)。
包内按仓库相对路径保存本目录的三个必要脚本、README、两份部署配置、依赖文件、最终报告、分段图和环境 / 源码版本记录。
原始 JSON 保持原样；包内 `SNAPSHOT.json` 为每个文件记录 SHA-256。脚本内容与实测版本一致，本次整理目录、文档和测试，实验脚本与运行时代码未改变，没有重跑实验。

实验 snapshot 记录的是基线 commit 加工作区修改，不能只靠基线 commit 复现；包内 `reports/provenance/deploy-runtime.patch` 保存相对基线的运行时代码修改，`dataset-and-source.json` 保存实测源码哈希。
包根目录的 `SNAPSHOT.md` 说明如何恢复基线并叠加补丁。完整模型框架、checkpoint、原始相机图片和虚拟环境不包含在包中，需在原运行环境中准备。
已排除失败试跑、临时状态文件和未应用的上游候选补丁。`reports/` 和 `benchmarks/snapshots/` 均由 Git 忽略，源码提交不包含这些产物。
