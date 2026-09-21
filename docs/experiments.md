# Experiments

These experiments explore inference transport, camera capture, and robot
integration. Each page describes the setup, measurements, and findings.

For installation and operation, start with [Quick start](quickstart.md).
The [support matrix](support-matrix.md) records verification by combination;
the experiment pages provide measurement details. Some runs terminate actions
in memory and do not exercise physical hardware.

## What each note covers

| Note | Question | Current conclusion |
|---|---|---|
| [Inference transport](inference-transport.md) | Does HTTP or WirelessComm change end-to-end inference latency, and where does a request's time go? | WirelessComm gave roughly a 4.5% throughput gain on the tested LAN. The communication band accounted for only 1.72 ms of an 18.91 ms E2E difference; the limit is model queueing. Payloads were small and dark, so this is a link result, not a normal-scene one. |
| [Camera-only capture](camera-only-experiments.md) | Can cameras be captured, recorded, and published with no motion path at all? | Works as a read-only publisher; the motion API does not exist by construction. Paired transport measurements and TCP-fault recovery were taken under the recorded load, not as general claims. |
| [Transport candidates](transport-experiments.md) | Do shared memory, raw frames, GStreamer, Zenoh, or NIXL/UCX reduce latency? | Implemented and locally tested as opt-in paths. None has yet demonstrated a speedup or better recovery on the tested hardware. |
| [LightNav-0 with XLeRobot](lightnav0_xlerobot.md) | Can decoded local waypoints drive an XLeRobot base? | The binding and its CPU tests exist. Real-checkpoint evaluation, navigation success, and physical-robot validation are outstanding. |

## Comparing configurations

Use the hardware, software versions, and workload listed on each page as the
starting point for your own measurements. Transport comparisons also record
input fingerprints to check that both paths receive the same data.
