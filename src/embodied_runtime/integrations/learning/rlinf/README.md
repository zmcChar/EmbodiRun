# RLinf integration seam

RLinf is an initial upstream learning framework, not the owner of the local
runtime. This adapter will translate:

- RLinf policy/checkpoint output into model-package build requests;
- robot trajectories and runtime metrics into RLinf-consumable records;
- policy versions into deployment lifecycle events.

Core model, engine, backend, distributed, and robot packages remain usable
without RLinf.
