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
uv sync --dev
```

## Pre-commit

```bash
❯ uv run pre-commit run --config .pre-commit-config.yaml
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
cargo check
```

### Formatting
We use `rustfmt`. To format Rust code:
```bash
cargo fmt
```

### Linting
We use `clippy`. To run Rust linters:
```bash
cargo clippy
```

### Unit Tests
To run Rust unit tests:
```bash
cargo test
```

### Build
To build the Python extension (install into `.venv`):
```bash
uv run maturin develop
# For release build (optimized):
uv run maturin develop --release
```

## Python Development

### Unit Tests
Run the Python test suite using `pytest`:

```bash
uv run pytest
```

### Formatting
We use `ruff` for formatting.
```bash
uv run ruff format .
```

### Linting
We use `ruff` for linting and `ty` for type checking.

Run Linter:
```bash
uv run ruff check .
# To automatically fix fixable errors:
uv run ruff check --fix .
```

Run Type Checker:
```bash
uv run ty check
```

## Benchmarks

To run the Agari calculation benchmark (performance verification):

```bash
# Build riichienv in release mode first
uv run maturin develop --release

# Run benchmark from benchmark project
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

## Release Process

This project uses an automated GitHub Actions workflow for releases.

### 1. Prerequisites
- You need a PyPI account.
- Generate a **Trusted Publisher** token or an API token on PyPI.

### 2. Configure GitHub Secrets
1. Go to your repository on GitHub.
2. Navigate to **Settings** > **Secrets and variables** > **Actions**.
3. Create a new repository secret named `PYPI_API_TOKEN`.
4. Paste your PyPI API token as the value.

### 3. Creating a Release
To publish a new version:

1. Update the version number in `pyproject.toml` (e.g., `0.1.0` -> `0.1.1`).
2. Update the version in `Cargo.toml` (under `[package]` in `native/Cargo.toml`) if necessary, though `maturin` often handles the mismatch or you should keep them in sync.
3. Commit the changes:
   ```bash
   git add pyproject.toml native/Cargo.toml
   git commit -m "chore: bump version to 0.1.1"
   git push
   ```
4. Create and push a tag starting with `v`:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

The GitHub Actions workflow will automatically:
- Build wheels for Linux, Windows, and macOS.
- Create a GitHub Release with the changelog.
- Upload the binary artifacts to the GitHub Release.
- Publish the package to PyPI.
