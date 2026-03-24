---
name: spec
description: >
  Generate a SPEC-*.md for the current task before implementation.
  Creates a structured spec with problem, options, decision, implementation plan,
  exit criteria, and out-of-scope sections.
disable-model-invocation: true
---

# Spec Generation

Generate a specification document before implementing a non-trivial task.

## Usage

```
/spec [task description or feature name]
```

## Steps

### 1. Understand the task

Read `$ARGUMENTS` as the task description. If the description is vague, ask clarifying questions before generating the spec.

Review relevant `.context/modules/` files for conventions, architecture, and patterns that apply.

### 2. Generate the spec

Create the spec file at `docs/SPEC-[feature-name].md`. If the `docs/` directory does not exist, create it first.

Use this format:

```markdown
# SPEC: [Feature Name]

## Problem
What are we solving and why. Include the user impact or technical motivation.

## Options Considered
At least 2-3 approaches with tradeoffs:

### Option A: [Name]
- **Approach**: How it works
- **Pros**: What's good about it
- **Cons**: What's bad about it

### Option B: [Name]
- **Approach**: How it works
- **Pros**: What's good about it
- **Cons**: What's bad about it

## Decision
What we're doing and why the alternatives were rejected.

## Implementation Plan
Step-by-step with file-level specifics:
1. [Step] — `path/to/file.ext`
2. [Step] — `path/to/file.ext`

## Exit Criteria
Exactly what must be true when this is done:
- [ ] [Test that must pass or behavior to verify]
- [ ] [Test that must pass or behavior to verify]

This is the task contract — implementation is not complete until all criteria are met.

## Out of Scope
What we explicitly are NOT doing, and why.
```

### 3. Wait for approval

Present the spec to the user. Do not begin implementation until the spec is explicitly approved. The user may edit, request changes, or reject the spec entirely.
