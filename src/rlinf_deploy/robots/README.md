# Physical device boundary

This domain owns vendor SDK adapters, sensor and control clients, and the small
services deployed beside physical hardware. Shared devices such as cameras stay
independent of a particular embodiment and can be composed with different
robots.

It does not load models, translate policy-native outputs, or coordinate policy
sessions. Those responsibilities belong to `bindings`, `inference`, and the
top-level deployment runtime.
