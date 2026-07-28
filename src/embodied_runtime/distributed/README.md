# Group 2 boundary

This domain will own all-to-all registration and communication among cloud,
edge, and robot runtimes:

- `registry/`: leases, heartbeats, capability advertisement;
- `communication/`: direct and brokered transports;
- `routing.py`: placement and failover policy;
- future preemption and recovery operate on distributed jobs, not on
  single-device kernels.

It may import only `embodied_runtime.contracts`. It must not know π0.5 internals or
vendor SDK APIs.
