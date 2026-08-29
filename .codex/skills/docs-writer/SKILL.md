---
name: docs-writer
description: Create or update developer documentation, API explanations, docstrings, configuration guidance, and executable examples so they accurately reflect the code.
---

# Documentation Writer

Write documentation from the current implementation and verified project
contracts rather than assumptions.

## Approach

- Identify the intended reader and the task they need to complete.
- Inspect the relevant code, configuration, tests, and existing terminology.
- Preserve established vocabulary and avoid duplicating information without a
  clear reason.
- Prefer concise explanations followed by runnable examples where examples add
  value.
- Include prerequisites, defaults, constraints, and failure behavior that affect
  successful use.
- Keep commands and code examples consistent with current entry points and
  dependencies.
- Do not claim that unverified hardware, services, or integrations were tested.

When code changes alter a documented public interface, update only the affected
documentation and examples. Check links, paths, commands, and snippets when
practical, and report anything that could not be verified.
