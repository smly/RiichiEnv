# Development Guide

This document describes how to build and develop the RiichiEnv UI (`riichienv-ui`).

## Prerequisites

- Node.js (v20 or later recommended)
- `npm`
- Rust toolchain (for WASM builds)
- `wasm-pack` ([https://rustwasm.github.io/wasm-pack/](https://rustwasm.github.io/wasm-pack/))
- `rustup target add wasm32-unknown-unknown`

## Setup

1. **Install Dependencies**
   ```bash
   cd riichienv-ui
   npm install
   ```

## Asset Generation

`npm run build:tiles` embeds the committed `assets/tile-art/tiles.png` atlas and
`tiles.json` positions in `src/tiles.ts`. Both viewer builds and tests run this step
automatically. Edit the atlas when changing tile artwork.

The 2D viewer also uses the committed `src/char_assets.ts` wind-character sprite.
Normal builds do not require a font download or character-sprite generation.
To change that sprite, place `ZenAntiqueSoft-Regular.ttf` in `assets/` and run
`node scripts/gen_sprite.js`.

## Build

To compile the TypeScript code, build WASM, and bundle the viewer:

```bash
npm run build
```

This command will:
1. Build `riichienv-wasm` via `wasm-pack` (`npm run build:wasm`).
2. Generate tile assets (`scripts/gen_tiles.js`).
3. Bundle and minify the code using `esbuild` into two formats:
   - `dist/viewer.js` (IIFE) — for `<script>` tag usage, registers globals on `window`.
   - `dist/viewer.esm.js` (ESM) — for `import` usage.
   Both formats inline the WASM binary (`--loader:.wasm=binary`).
4. Compress the output (`viewer.js.gz`) and copy it to the Python package directory.

To skip the WASM rebuild when only modifying TypeScript code:

```bash
npm run build:no-wasm
```

## Development

To run a local development server with esbuild's built-in serve mode:

```bash
npm run dev
```

This starts a server at `http://localhost:8000/` and serves `index.html` as a demo page. The demo loads `example_before_injection.jsonl` and displays it with the 3D viewer.

To simply watch for changes and rebuild:

```bash
npm run watch
```

> [!NOTE]
> `npm run dev` and `npm run watch` do not rebuild WASM. If you modify `riichienv-wasm` or `riichienv-core`, run `npm run build:wasm` first, then restart the dev server.

## Linting and Formatting

The project uses [Biome](https://biomejs.dev/) for linting and formatting.

```bash
npm run lint          # Check for lint issues
npm run lint:fix      # Auto-fix lint issues
npm run format        # Format source files
npm run format:check  # Check formatting without modifying
```

## Testing

Tests use [Vitest](https://vitest.dev/).

```bash
npm run test          # Run tests once
npm run test:watch    # Run tests in watch mode
```
