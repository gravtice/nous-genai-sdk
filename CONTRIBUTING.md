# Contributing to nous-genai-sdk

Thank you for your interest in contributing to nous-genai-sdk!

## Development Setup

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Getting Started

```bash
# Clone the repository
git clone https://github.com/gravtice/nous-genai-sdk.git
cd nous-genai-sdk

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_client_timeout.py -v

# Run with coverage
uv run pytest tests/ --cov=nous --cov-report=term-missing
```

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check linting
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Making Changes

### Branch Naming

- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `docs/<name>` - Documentation updates
- `refactor/<name>` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Refactoring
- `chore`: Build/tooling

Examples:
```
feat(client): add timeout configuration
fix(gemini): handle empty response
docs: update README examples
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure all tests pass: `uv run pytest tests/ -v`
5. Ensure code style checks pass: `uv run ruff check .`
6. Submit a pull request

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code follows project style
- [ ] Documentation updated (if applicable)
```

## Adding a New Provider

1. Create adapter in `nous/genai/providers/<provider>.py`
2. Implement the adapter interface (see existing adapters for reference)
3. Register in `nous/genai/providers/__init__.py`
4. Add model catalog data in `nous/genai/reference/model_catalog_data/<provider>.py`
5. Update `nous/genai/client.py` to initialize the adapter
6. Add tests in `tests/`
7. Update documentation

## Reporting Issues

When reporting issues, please include:

- Python version
- nous-genai-sdk version
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback (if applicable)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
