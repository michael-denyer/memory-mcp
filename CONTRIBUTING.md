# Contributing to Memory MCP

Thank you for your interest in contributing to Memory MCP! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+ (3.11+ recommended)
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Getting Started

```bash
# Clone the repository
git clone https://github.com/michael-denyer/memory-mcp.git
cd memory-mcp

# Install dependencies (including dev tools)
uv sync

# Run tests to verify setup
uv run pytest -v
```

### Optional: Apple Silicon Acceleration

If you're on an M-series Mac, install with MLX extras for faster embeddings:

```bash
uv sync --extra mlx
```

## Project Structure

```
src/memory_mcp/
├── server/             # MCP server package
│   ├── app.py          # FastMCP setup, resources, lifespan
│   └── tools/          # Tool implementations by domain
│       ├── cold_storage.py   # remember, recall, forget
│       ├── hot_cache.py      # promote, demote, pin, unpin
│       ├── mining.py         # log_output, run_mining
│       └── ...               # 12 tool modules total
├── storage/            # Storage package
│   ├── core.py         # Storage class, transactions, schema
│   ├── search.py       # Vector search, scoring
│   ├── hot_cache.py    # Promotion, demotion, eviction
│   └── ...             # 16 mixin modules total
├── mining.py           # Pattern extraction from outputs
├── config.py           # Settings and configuration
├── cli.py              # CLI commands for hooks and administration
├── embeddings.py       # Embedding providers (sentence-transformers, MLX)
├── responses.py        # Pydantic response models for MCP tools
├── models.py           # Enums and dataclasses (domain models)
├── migrations.py       # Database schema and version migrations
├── helpers.py          # Helper functions
├── text_parsing.py     # Content chunking for seeding
├── logging.py          # Structured logging configuration
└── metrics.py          # Metrics collection and observability
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

Pre-commit hooks run automatically on commit. To set them up:

```bash
uv run pre-commit install
```

### Style Guidelines

- Line length: 100 characters
- Use type hints for all function signatures
- Use Google-style docstrings
- Prefer composition over inheritance
- Keep functions focused and small

## Testing

```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_storage.py -v

# Run tests matching a pattern
uv run pytest -k "hot_cache" -v

# Run with coverage
uv run pytest --cov=memory_mcp --cov-report=term-missing
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names: `test_<what>_<when>_<expected>`
- Use fixtures for common setup (see `tests/conftest.py`)

## Making Changes

### Before You Start

1. Check existing [issues](https://github.com/michael-denyer/memory-mcp/issues) for related work
2. For significant changes, open an issue first to discuss the approach
3. Keep changes focused - one feature/fix per PR

### Development Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the code style guidelines

3. Add or update tests as needed

4. Ensure all tests pass:
   ```bash
   uv run pytest -v
   ```

5. Commit with a descriptive message:
   ```bash
   git commit -m "feat: add support for custom embedding models"
   ```

### Commit Message Format

Use [conventional commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|--------|---------|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation changes |
| `refactor:` | Code changes that neither fix bugs nor add features |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance tasks |

### Pull Request Process

1. Push your branch and open a PR against `main`
2. Fill out the PR template with a clear description
3. Ensure CI passes (tests, linting)
4. Request review from maintainers
5. Address any feedback
6. Once approved, a maintainer will merge

## Architecture Principles

Memory MCP is built around a **two-tier memory architecture**. When contributing, keep these principles in mind:

1. **Hot cache is the differentiator** - Instant recall via MCP resource injection
2. **Automatic over manual** - Features should work without user intervention
3. **Local-first** - All data stays in SQLite, no cloud dependencies
4. **Simple over complex** - Avoid over-engineering; question new configuration options

### What We're Looking For

- Bug fixes and performance improvements
- Better test coverage
- Documentation improvements
- Usability enhancements
- Integration examples

### What to Avoid

- Features requiring extensive manual configuration
- External service dependencies
- Breaking changes to the storage format without migration
- Features that don't serve the two-tier memory model

## Getting Help

- Open a [GitHub issue](https://github.com/michael-denyer/memory-mcp/issues) for bugs or feature requests
- Start a [discussion](https://github.com/michael-denyer/memory-mcp/discussions) for questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
