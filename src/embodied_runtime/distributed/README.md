# Group 2 boundary

This domain will own all-to-all registration and communication among cloud,
edge, and robot runtimes:

- `registry/`: leases, heartbeats, capability advertisement;
- `communication/`: direct and brokered transports;
- `routing.py`: endpoint placement;
- `failover.py`: asynchronous result authority, staleness, and recovery.

Preemption here operates on results from distributed runtimes, not on
single-device kernels. The first implementation keeps an edge endpoint hot,
allows a fresh cloud result to take output authority, and invalidates that
authority on disconnect or connection-epoch change.

It may import only `embodied_runtime.contracts`. It must not know π0.5 internals or
vendor SDK APIs.
