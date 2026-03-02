# Conventions

## Python Style

Google Python Style Guide conventions:
- `snake_case` for functions, methods, variables, modules
- `PascalCase` for classes
- `ALL_CAPS` for constants
- 4-space indentation, 80-char line length
- Type hints on public APIs
- Docstrings on public functions/classes (`Args:`, `Returns:`, `Raises:`)
- f-strings for formatting
- Group imports: stdlib → third-party → local
- Run `pylint` to catch bugs and style issues
- No mutable default arguments (`[]`, `{}`)
- Use implicit false (`if not my_list:`) and `if foo is None:` for None checks

## Code Hygiene

- **Dead code cleanup is mandatory during refactors.** Every refactor must include a sweep for orphaned code from deprecated or deleted components. This is a recurring problem — treat it as a first-class concern.
- Prefer small, surgical edits over broad rewrites.
- Reuse existing files before creating new modules.
- Remove or simplify unnecessary code only when it reduces complexity.
- Add concise comments explaining intent only where the change is non-obvious.
- Do not reformat unrelated code or alter indentation styles.

## Naming & Organization

- Components go in `components/` — UI layout only, no ML logic
- Utilities go in `utils/` — computation only, no Dash imports
- Tests go in `tests/` with `test_` prefix matching the module they test
- RAG knowledge goes in `rag_docs/` as markdown files

## Commit Messages

Format: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(ablation): Add multi-layer head selection`
- `fix(pipeline): Correct embedding stage token count`
- `refactor(utils): Remove unused activation caching code`

## Error Handling

- Use built-in exception classes
- No bare `except:` clauses
- User-facing errors should be clear and non-technical

## Dependencies

- Avoid adding new dependencies unless strictly needed
- Document any new dependency in requirements.txt with minimum version

## Dash-Specific

- Callbacks must remain responsive — avoid heavy synchronous work without feedback indicators
- Use `dcc.Store` for session state; no server-side persistence

## Quality Gates

Before marking any task complete:
- All tests pass
- Code coverage >80% for new code
- Follows style guide
- Public functions/methods have docstrings
- Type hints on public APIs
- No linting errors
- No security vulnerabilities (no hardcoded secrets, input validation present)
