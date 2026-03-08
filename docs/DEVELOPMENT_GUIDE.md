# Development Guide

This guide describes the procedures for checking compilation, running tests, executing benchmarks, and adhering to coding standards.

## Prerequisites

Ensure you have the following tools installed:
- Rust (cargo)
- Python 3.10+
- `uv` (for Python package management)
- `maturin` (for building the Rust extension)
- Node.js (v18+) and `npm` (for UI development)
- `wasm-pack` (for WASM builds, https://rustwasm.github.io/wasm-pack/)

Install development dependencies:
```bash
uv sync --dev
```

## Workspace Structure

The project uses a Cargo workspace with multiple crates and a TypeScript UI package:

| Crate / Package | Role |
|---|---|
| `riichienv-core` | Pure Rust library (rlib). No Python dependency by default. |
| `riichienv-python` | PyO3 wrapper (cdylib). Depends on `riichienv-core` with `python` feature. |
| `riichienv-wasm` | WASM wrapper (cdylib). Depends on `riichienv-core` with `wasm` feature. |
| `riichienv-ui` | TypeScript UI (replay viewer + live viewer). Depends on `riichienv-wasm`. |

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

## Feature Flags

The `riichienv-core` crate uses feature flags to control optional dependencies and bindings:

| Feature | Description |
|---|---|
| `default` | Enables `flate2` (gzip) — standard desktop/server builds |
| `python` | Enables PyO3 bindings (`#[pyclass]`, `#[pymethods]`, etc.) + `flate2` |
| `wasm` | Marker feature for WASM builds (no additional deps) |
| *(no default features)* | Minimal pure Rust library — no `flate2`/PyO3 |

`default = ["dep:flate2"]` — standard builds include gzip support.

The `riichienv-python` crate depends on `riichienv-core` with the `python` feature enabled, and adds the `extension-module` feature for maturin builds (configured in `pyproject.toml` under `[tool.maturin]`).

The `riichienv-wasm` crate depends on `riichienv-core` with `default-features = false, features = ["wasm"]` to disable optional gzip/Python bindings.

## Rust Development

### Setup Rust Environment

```bash
❯ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

### Compilation Check

To check if the Rust core compiles (pure Rust, no Python dependency):
```bash
cargo check -p riichienv-core
```

To check with Python bindings enabled:
```bash
cargo check -p riichienv-core --features python
```

To check the Python wrapper crate:
```bash
cargo check -p riichienv-python
```

To check the WASM crate (requires `wasm32-unknown-unknown` target):
```bash
rustup target add wasm32-unknown-unknown
cargo check -p riichienv-wasm --target wasm32-unknown-unknown
```

### Formatting
We use `rustfmt`. To format Rust code:
```bash
cargo fmt
```

### Linting
We use `clippy`. To run Rust linters:
```bash
# Pure Rust mode
cargo clippy -p riichienv-core

# With all features (including Python bindings)
cargo clippy --all-targets --all-features
```

### Unit Tests
To run Rust unit tests:
```bash
# Pure Rust tests (agari, score, yaku, hand_evaluator, mjai_event, etc.)
cargo test -p riichienv-core
```

### Build
To build the Python extension (install into `.venv`):
```bash
uv run maturin develop
# For release build (optimized):
uv run maturin develop --release
```

### Conditional Compilation Patterns

When adding new code, follow these patterns for `#[cfg(feature = "python")]`:

**Struct definitions** — use `cfg_attr` on the struct, not on fields:
```rust
// For structs where all fields should be readable from Python:
#[cfg_attr(feature = "python", pyclass(get_all))]
pub struct Foo {
    pub field_a: u32,
    pub field_b: String,
}

// For structs with mixed access, use manual #[getter]/#[setter] in pymethods:
#[cfg_attr(feature = "python", pyclass)]
pub struct Bar {
    pub field_a: u32,       // will have manual getter
    pub(crate) internal: i32, // not exposed to Python
}

#[cfg(feature = "python")]
#[pymethods]
impl Bar {
    #[getter]
    fn get_field_a(&self) -> u32 { self.field_a }
}
```

> **Note:** `#[cfg_attr(feature = "python", pyo3(get))]` on struct fields does NOT work. The `pyo3(get)` attribute is consumed by the `pyclass` proc macro, and when `pyclass` is applied via `cfg_attr`, the compiler cannot resolve `pyo3` as a known attribute. Use `get_all`/`set_all` on the `pyclass(...)` attribute or manual `#[getter]` methods instead.

**Pure Rust logic** — keep it in a regular `impl` block. Python wrappers go in a separate `#[cfg(feature = "python")] #[pymethods]` block with a `_py` suffix:
```rust
impl Foo {
    pub fn compute(&self) -> RiichiResult<u32> { /* ... */ }
}

#[cfg(feature = "python")]
#[pymethods]
impl Foo {
    #[pyo3(name = "compute")]
    pub fn compute_py(&self) -> PyResult<u32> {
        self.compute().map_err(Into::into)
    }
}
```

**Error handling** — use `RiichiError` / `RiichiResult<T>` (defined in `errors.rs`) for pure Rust code. `RiichiError` is an enum with variants: `Parse`, `InvalidAction`, `InvalidState`, `Serialization`. The `From<RiichiError> for PyErr` conversion is provided when the `python` feature is enabled, so `?` works seamlessly in Python wrappers.

## WASM Development

### Prerequisites

- `wasm-pack` ([https://rustwasm.github.io/wasm-pack/](https://rustwasm.github.io/wasm-pack/))
- `rustup target add wasm32-unknown-unknown`

### Build

```bash
wasm-pack build riichienv-wasm --target web
```

### Notes

- `.cargo/config.toml` contains `getrandom_backend = "wasm_js"` rustflag for `wasm32-unknown-unknown` target.
- `riichienv-core` is used with `default-features = false, features = ["wasm"]` to disable optional `flate2`.
- The `riichienv-wasm` crate exposes functions via `wasm-bindgen`: `calc_waits`, `calc_score`, `mjai_to_tile_id`, `tile_id_to_mjai`.

## UI Development

### Prerequisites

- Node.js (v18+) and `npm`
- `wasm-pack` (for WASM builds)

### Build (full: WASM + UI)

```bash
cd riichienv-ui
npm install
npm run build
```

This runs the full build pipeline: `build:wasm` → `build:tiles` → `build:bundle` → `build:compress` → `build:copy`.

### Build (UI only, skip WASM rebuild)

```bash
cd riichienv-ui
npm run build:no-wasm
```

### Development

```bash
cd riichienv-ui
npm run dev    # Local dev server with hot-reload
npm run watch  # File watcher with auto-rebuild
```

### Build Chain

```
riichienv-core → riichienv-wasm (wasm-pack) → riichienv-ui (esbuild) → src/riichienv/visualizer/assets/viewer.js.gz
```

The esbuild step inlines the WASM binary into the JavaScript bundle (`--loader:.wasm=binary`), producing a single self-contained `viewer.js` file. This is then gzip-compressed and copied to the Python package assets directory.

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

The `riichienv-core` crate includes [Criterion](https://bheisler.github.io/criterion.rs/book/)-based benchmarks for agari detection, hand evaluation, and score calculation.

### Running Benchmarks

```bash
cd riichienv-core
cargo bench --bench agari_bench
```

Results are printed to stdout and HTML reports are generated under `target/criterion/report/index.html`.

### Benchmark Groups

| Group | What it measures |
|---|---|
| `is_agari/positive` | Agari detection on winning hands (816 cases) |
| `is_agari/negative` | Agari detection on non-winning hands (200 cases) |
| `is_tenpai` | Tenpai check on 13-tile hands |
| `find_divisions` | Mentsu decomposition of winning hands |
| `hand_evaluator/calc_4p` | Full 4P pipeline: agari, yaku, fu, score (816 cases) |
| `hand_evaluator/calc_3p` | Full 3P pipeline (402 cases) |
| `calculate_score` | Pure score calculation across all tiers (28 combos) |

### Comparing Across Versions

Criterion saves baseline data in `target/criterion/`. Use named baselines to compare performance between branches:

```bash
# Save baseline on the current branch
cargo bench --bench agari_bench -- --save-baseline main

# Switch to another branch and compare
git checkout feature-branch
cargo bench --bench agari_bench -- --baseline main
```

### Correctness Tests

Integration tests verify that all benchmark fixture data produces the expected han, fu, yaku, and score values:

```bash
cd riichienv-core
cargo test --test agari_correctness
```

### Fixture Data

Benchmark cases are stored in `riichienv-core/benches/data/`:

| File | Contents |
|---|---|
| `agari_4p.json` | 816 winning hands (4P, extracted from MjSoul MJAI logs + synthetic yakuman) |
| `agari_3p.json` | 402 winning hands (3P) |
| `hands_negative.json` | 200 non-winning hands |

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
- **Trusted Publisher Setup** (Recommended, no token needed):
  1. Go to your [PyPI Publishing settings](https://pypi.org/manage/account/publishing/).
  2. If the project doesn't exist on PyPI yet, select **Add a new pending publisher**.
  3. **Project Name**: `riichienv`
  4. **Owner**: Your GitHub username or organization name.
  5. **Repository name**: `RiichiEnv`
  6. **Workflow name**: `release.yml`
  7. **Environment name**: `pypi`
  8. Click **Add**.

### 2. Configure GitHub Settings
1. **Environments**:
   - Go to your repository **Settings** > **Environments**.
   - Create a new environment named `pypi`.
   - (Optional) Configure "Required reviewers" to require manual approval before publishing.
   - **Note**: You do *not* need to set `PYPI_API_TOKEN` secret if using Trusted Publisher.

### 3. Creating a Release
To publish a new version:

1. Update the version number in `riichienv-core/Cargo.toml`, `riichienv-python/Cargo.toml`, `riichienv-wasm/Cargo.toml`, and `pyproject.toml`.
2. Commit and push the changes:
   ```bash
   git add riichienv-core/Cargo.toml riichienv-python/Cargo.toml riichienv-wasm/Cargo.toml pyproject.toml
   git commit -m "chore: bump version to X.Y.Z"
   git push
   ```
3. **Draft a Release on GitHub**:
   - Go to the **Releases** page on GitHub.
   - Click **Draft a new release**.
   - **Choose a tag**: Create a new tag (e.g., `vX.Y.Z`) on the target branch.
   - **Release title**: `vX.Y.Z` (or your preferred title).
   - Write your release notes.
   - Click **Publish release**.

The GitHub Actions workflow will automatically:
- Trigger when the release is published.
- Build wheels for Linux, Windows, and macOS.
- **Upload the binary artifacts** to your existing release.
- Publish the package to PyPI.

### 4. Publishing to crates.io
To publish the Rust core library to crates.io:

```bash
# Dry-run first
cargo publish -p riichienv-core --dry-run

# Publish
cargo publish -p riichienv-core
```
