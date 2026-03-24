---
name: lessons
disable-model-invocation: false
description: >
  Review lessons.md for stale, contradictory, or outdated entries and propose
  corrections. Use when hitting patterns that contradict existing lessons,
  or periodically to keep lessons fresh.
---

# Lessons Review

Review the project's lessons learned and propose corrections for anything stale or wrong.

## Steps

### 1. Read current state

Read these files:
- `.context/data/lessons.md` — all lesson entries
- `.context/data/decisions.md` — decision records (for cross-referencing)
- `.context/data/sessions.md` — session log (for timeline cross-referencing)
- The project routing file — Critical Rules (which may encode lessons)

### 2. Audit each lesson

For each entry in `lessons.md`, check:
- **Still valid?** Has the codebase, tooling, or approach changed in a way that makes this lesson obsolete?
- **Rule followed?** Is the "Rule going forward" being followed in current code, or has it been silently abandoned?
- **Contradicted?** Does any decision in `decisions.md` contradict or supersede this lesson?
- **Duplicated?** Is this lesson substantially the same as another entry?

### 3. Propose corrections

For stale or wrong entries, propose an inline update:

```markdown
[YYYY-MM-DD UPDATE]: [Correction or superseding note explaining what changed]
```

Place the update directly beneath the original entry, preserving the original text.

### 4. Propose consolidation if needed

If `lessons.md` has ~30+ entries, or entries have grown contradictory or redundant, propose consolidation:
- Merge duplicates into a single entry
- Resolve contradictions by keeping the most current lesson and updating the older one
- Remove lessons that are now encoded in Critical Rules or modules (with a note explaining where the knowledge moved)

### 5. Check Critical Rules alignment

If accumulated lessons suggest a rule important enough to be in the routing file's Critical Rules section, flag it as a promotion candidate.

### 6. Present changes

Present ALL proposed changes as diffs. Never update lessons silently — wait for user approval before writing anything.
