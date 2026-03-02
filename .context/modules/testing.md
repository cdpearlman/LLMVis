# Testing

## Approach

Test-Driven Development (TDD) for all backend/utils logic:
1. Write failing tests that define expected behavior
2. Implement minimum code to pass
3. Refactor with confidence

## What to Test

- All `utils/` modules — these contain the core logic
- Model configuration and pattern matching
- Ablation metrics and scoring
- Head detection and categorization
- Beam search behavior
- Token attribution computations
- OpenRouter client (mock external calls)

## What NOT to Test

- UI/frontend layout changes (components/ files)
- Trivial additions and documentation
- CSS and JavaScript assets

## Framework & Conventions

- **Framework**: pytest
- **Location**: `tests/` directory
- **Naming**: `test_<module_name>.py` matching the module in `utils/`
- **Fixtures**: Defined in `conftest.py` for shared test state
- **Mocking**: Mock external dependencies (API calls, model loading when appropriate)
- **Coverage target**: >80% for new code

## Running Tests

```bash
pytest                           # Run all tests
pytest tests/test_<name>.py      # Run specific test file
pytest --cov=utils --cov-report=html  # With coverage
```

## When Tests Are Required

- New utility functions or modules
- Bug fixes (write a test that reproduces the bug first)
- Changes to computation logic
- Refactors that touch testable behavior

## When Tests Are Optional

- Pure UI/layout changes
- Documentation updates
- Configuration changes

## After Every Change

- Run `pytest` to verify all tests still pass
- If tests fail, iterate on debugging until fixed — don't move on with broken tests
