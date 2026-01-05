# Development Guide

This guide describes the procedures for checking compilation, running tests, executing benchmarks, and adhering to coding standards.

## Prerequisites

Ensure you have the following tools installed:
- Rust (cargo)
- Python 3.10+
- `uv` (for Python package management)
- `maturin` (for building the Rust extension)

Install development dependencies:
```bash
cd riichienv
uv sync --dev
```

## Pre-commit

```bash
❯ cd riichienv
❯ uv run pre-commit run --config ../.pre-commit-config.yaml
rustfmt..................................................................Passed
clippy...................................................................Passed
ruff-check...............................................................Passed
ty-check.................................................................Passed
pytest...................................................................Passed
ruff-format..............................................................Passed
```

## Rust Development

### Setup Rust Environment

```bash
❯ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

### Compilation Check
To check if the Rust code compiles:
```bash
cd riichienv
cargo check
```

### Formatting
We use `rustfmt`. To format Rust code:
```bash
cd riichienv
cargo fmt
```

### Linting
We use `clippy`. To run Rust linters:
```bash
cd riichienv
cargo clippy
```

### Unit Tests
To run Rust unit tests:
```bash
cd riichienv
cargo test
```

### Build
To build the Python extension (install into `.venv`):
```bash
cd riichienv
uv run maturin develop
# For release build (optimized):
uv run maturin develop --release
```

## Python Development

### Unit Tests
Run the Python test suite using `pytest`:

```bash
cd riichienv
uv run pytest
```

### Formatting
We use `ruff` for formatting.
```bash
cd riichienv
uv run ruff format .
```

### Linting
We use `ruff` for linting and `ty` for type checking.

Run Linter:
```bash
cd riichienv
uv run ruff check .
# To automatically fix fixable errors:
uv run ruff check --fix .
```

Run Type Checker:
```bash
cd riichienv
uv run ty check
```

## Benchmarks

To run the Agari calculation benchmark (performance verification):

```bash
# Build riichienv in release mode first
cd riichienv
uv run maturin develop --release

# Run benchmark from benchmark project
cd ../benchmark
uv sync  # Install dependencies (riichienv, mahjong)
uv run agari
```


## Commit Messages

We follow the **Conventional Commits** specification.
Format: `<type>(<scope>): <subject>`

### Common Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation

### Examples
- `feat(env): add Kyoku.events serialization`
- `fix(score): correct ura dora calculation`
- `docs(readme): update installation instructions`
