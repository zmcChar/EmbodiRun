---
name: code-reviewer
description: Review code, diffs, or pull requests for concrete defects, regressions, security problems, and missing tests. Use for review requests; do not use merely because code is being implemented.
---

# Code Reviewer

Review the requested change without modifying it unless the user separately asks
for fixes.

## Review priorities

Look first for:

1. Incorrect behavior and violated contracts.
2. Safety or security problems.
3. Regressions and compatibility breaks.
4. Race conditions, resource leaks, and error-path failures.
5. Missing tests for behavior that could realistically regress.

Trace important behavior into its callers and tests when necessary. Distinguish
defects introduced by the reviewed change from unrelated pre-existing issues.
Do not report personal style preferences unless they materially affect
correctness or maintainability.

## Output

List findings first, ordered by severity. For each finding:

- state the concrete failure or risk;
- cite the file and exact line when available;
- explain the conditions under which it occurs;
- suggest the smallest reasonable correction.

Keep summaries secondary. If there are no findings, say so explicitly and note
any important validation gap, such as tests or hardware that could not be run.
