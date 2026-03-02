---
description:
globs:
  - "**/*.py"
  - "**/*.md"
  - "app.py"
  - "components/**/*.py"
  - "utils/**/*.py"
alwaysApply: true
---

# Transformer Explanation Dashboard
Interactive Dash app for exploring LLM internals through visualization and experimentation. Built with Python, Dash, PyTorch, HF Transformers, Bertviz, pyvene.

## Critical Rules
- **Dead code cleanup is mandatory during refactors.** Sweep for orphaned code from deprecated/deleted components every time.
- **Conceptual understanding over math rigor.** Skip complex derivations; focus on motivation and intuition that builds correct mental models.
- **Sanity check every change:** (1) Does this help someone learn about transformers? (2) Is the simplification accurate enough? (3) Am I in a rabbit hole?
- **TDD for backend logic.** Write failing tests first for anything in `utils/`. Skip tests for UI-only changes. Run `pytest` after every change; iterate until green.
- **Don't run the app.** Describe manual verification steps; the user will test themselves.
- **Surgical edits over rewrites.** Reuse existing files. Only create new modules when existing ones can't be extended.
- **No new dependencies** unless strictly necessary.
- **Push back on bad ideas.** Think through problems fully. Don't be a yes-man — challenge flawed approaches.
- **Components = layout only** (no ML logic). **Utils = computation only** (no Dash imports). **app.py = glue.**
- **Dash callbacks must stay responsive.** No heavy sync work in callbacks without feedback indicators.
- Don't reformat unrelated code or alter indentation styles.
- Check for zombie processes before debugging server errors.

## Module Map
| Module | Path | Load when |
|--------|------|-----------|
| Product | `.context/modules/product.md` | Understanding vision, features, brand voice, visual identity, UX patterns |
| Architecture | `.context/modules/architecture.md` | Understanding system structure, data flow, or component boundaries |
| Conventions | `.context/modules/conventions.md` | Writing or reviewing code style, naming, commits, dead code cleanup |
| Education | `.context/modules/education.md` | Creating or editing educational content, visualizations, explanations |
| Testing | `.context/modules/testing.md` | Writing tests, running pytest, TDD workflow |

## Data Files
| File | Path | Purpose |
|------|------|---------|
| Sessions | `.context/data/sessions.md` | Running work log (append-only) |
| Decisions | `.context/data/decisions.md` | Decision records with reasoning (append-only) |
| Lessons | `.context/data/lessons.md` | Hard-won knowledge and past mistakes (append-only) |

## Memory Maintenance

Always look for opportunities to update the memory system:
- **New patterns**: "We've been doing X consistently — should I add it to conventions?"
- **Decisions made**: "We decided Y — should I record this in decisions.md?"
- **Mistakes caught**: "This went wrong because Z — should I add it to lessons.md?"
- **Scope changes**: "The project now includes W — should I create a new module?"

**Before any memory update**:
1. State which file(s) would change and what the change would be
2. Wait for approval
3. Never update memory mid-task without mentioning it

**Rules**:
- Data files are append-only — add entries, never remove or overwrite past entries
- Modules can be edited but changes should be targeted, not full rewrites
- After substantive work sessions, append a summary to `.context/data/sessions.md`

## Preferences
- Don't ask permission for changes that fall within an approved plan — just execute
- Commit normal changes to main; feature branches for major components/refactors. Never merge branches.
- Keep `todo.md` and `plans.md` current before/after changes. Tasks should be atomic.
- When in doubt, research options and make a minimal reasonable choice, noting it in `todo.md`
- Explain manual tests clearly — what to look for, expected behavior, where to check
