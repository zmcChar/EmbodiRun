# Python API

The curated public surface of the `embodirun` package: exactly the names exported from
the package root. Anything not listed here is internal and may change without notice.

A deployment normally drives the runtime through its command-line entry points and its
configuration files, covered in [Configuration](configuration.md) and
[Control](control.md). This page is for callers that embed the runtime in Python — a
custom simulator, a hardware adapter, or a test harness.

## Inference clients

::: embodirun
    options:
      members:
        - PolicyClient
        - SglangHttpClient
        - VvlaHttpClient
        - VvlaWirelessClient

## Inference contracts

::: embodirun
    options:
      members:
        - PolicyObservation
        - PolicyResult
        - PolicyAction
        - ImagePayload
        - Session

## Robot adapters

::: embodirun
    options:
      members:
        - RobotAdapter
        - RobotObservation
        - RobotAction
        - RobotProfile
