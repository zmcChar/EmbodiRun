# Group 5 boundary

This domain owns the physical last mile:

- sensor schema and time synchronization;
- model-to-robot observation mapping;
- action scaling, frame conversion, and calibration;
- control-loop timing, watchdog, safety limits, and emergency stop.

It does not load models or call hardware acceleration APIs. A composition root
connects a `RobotAdapter`, an `ObservationMapper`, an execution engine, and an
`ActionMapper`.
