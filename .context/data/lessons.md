# Lessons Learned

<!-- Append-only. Record what the team learned the hard way. -->
<!-- Format:
## YYYY-MM-DD — [Brief title]
**What happened**: What went wrong or what was discovered
**Root cause**: Why it happened
**Fix**: What was done about it
**Rule going forward**: What to do (or avoid) in the future
-->

## 2026-03-02 — Dead code accumulation during refactors
**What happened**: Large component changes left hundreds of lines of orphaned code from deprecated or deleted components
**Root cause**: Refactors focused only on building the new thing without cleaning up what the old thing left behind
**Fix**: Manual cleanup after discovering the bloat
**Rule going forward**: Every refactor must include a dead code sweep. This is a first-class concern, not an afterthought.

## 2026-03-02 — Tunnel vision on implementation details
**What happened**: Going deep on implementation rabbit holes produced outputs that weren't actually useful for the educational goal
**Root cause**: Losing sight of the "is this useful for teaching?" question while focused on technical correctness
**Fix**: Stepped back and re-evaluated against the educational mission
**Rule going forward**: Sanity check every significant change: (1) Does this help someone understand transformers? (2) Is this accurate enough for correct intuition? (3) Am I in a rabbit hole?
