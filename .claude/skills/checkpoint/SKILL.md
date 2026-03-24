---
name: checkpoint
disable-model-invocation: false
description: >
  Run a memory checkpoint — proposes a sessions.md entry, checks for pending
  decision/lesson updates, and runs cross-file consolidation if needed.
  Use after substantive work, when decisions are made, when mistakes are caught,
  or at session end.
---

# Memory Checkpoint

Run a full memory maintenance pass for this project's ContextKit memory system.

## Steps

### 1. Read current state

Read these files to understand what's already recorded:
- `.context/data/sessions.md` — recent session entries
- `.context/data/decisions.md` — existing decision records
- `.context/data/lessons.md` — existing lessons
- The project routing file (CLAUDE.md, AGENTS.md, or tool equivalent) — Critical Rules and Module Map

### 2. Propose a sessions.md entry

Append a new entry using this format:

```markdown
## [today's date] — [Brief title]
**Area**: [What part of the project was touched]
**Work done**: [Concise summary of meaningful work]
**Decisions made**: [Any decisions, or "None"]
**Memory created**: [Any new modules/data entries, or "None"]
**Open threads**: [Unfinished work, unanswered questions, or "None"]
```

Use neutral reporting — state what changed, what failed, and what's uncertain. Avoid framing routine work as accomplishments.

### 3. Check for pending decision records

Review the work done this session. If any decisions were made (tool choices, architecture changes, pattern adoption, scope changes), propose a new entry in `decisions.md`:

```markdown
## [Decision title]
**Date**: [today's date]
**Context**: What prompted this decision
**Options considered**: What alternatives were evaluated
**Decision**: What was chosen
**Reasoning**: Why
**Revisit if**: Conditions that would warrant reconsidering
```

### 4. Check for pending lessons

If any mistakes were caught, surprising behaviors discovered, or approaches abandoned, propose a new entry in `lessons.md`:

```markdown
## [today's date] — [Brief title]
**What happened**: What went wrong or what was discovered
**Root cause**: Why it happened
**Fix**: What was done about it
**Rule going forward**: What to do (or avoid) in the future
**What was ruled out**: Approaches considered and rejected, and why
```

### 5. Cross-file consolidation check

If `sessions.md` has ~30+ entries, propose consolidation: summarize the oldest 20 into a single dated summary block, preserving key decisions and open threads.

Also check:
- Do the routing file's Critical Rules still reflect accumulated decisions and lessons?
- Have 3+ decisions accumulated in one domain that warrants a new module?
- Do any lessons contradict existing decisions? Flag the conflict.

### 6. Present changes

Present ALL proposed changes as diffs. State which files would change and what each change is. Wait for user approval before writing anything. Never update memory silently.
