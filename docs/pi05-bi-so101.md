# Pi0.5 with two SO-101 followers

`lerobot.bi_so101` composes two existing calibrated SO-101 adapters. Each arm
has its own serial port and calibration identity; state and action order is
left six values followed by right six values, with grippers in the native
`range_0_100` convention. The binding matches action features by name and
rejects malformed or over-sized commands before the first bus write.

Use [configs/pi05/bi-so101-vvla.yaml](../configs/pi05/bi-so101-vvla.yaml) as a
validation template. Replace serial paths, camera paths, calibration IDs, and
the inference checkpoint. `validate` and `build_plan` do not contact devices.
The dual adapter belongs to the same Deploy Control resource owner as the
single-arm adapter; do not run it concurrently with the legacy AGX teleop
service on those serial ports.

The implementation keeps each bus sequential. If the second command fails,
both arms receive a measured hold attempt and the original failure remains
visible. Disconnect torque handling, step limits, passive connection, and
explicit preparation are inherited from the single-arm adapter. This software
change has no physical-robot validation evidence.
