# RiichiEnv UI

A standalone, web-based UI for RiichiEnv featuring a replay viewer and live game viewer.

## Structure

- `src/`: TypeScript source code.
  - `src/wasm/`: WASM integration layer (`loader.ts`, `bridge.ts`).
  - `src/live_viewer.ts`: Live game viewer (incremental event processing).
  - `src/live_controller.ts`: Keyboard/UI controller for live mode.
  - `src/game_state.ts`: Game state management with WASM-accelerated wait calculation.
- `riichienv-mahjong-tiles-regular/`: SVG tile assets.
- `scripts/gen_tiles.js`: Script to bundle SVG tiles into `src/tiles.ts`.
- `scripts/build-wasm.sh`: Script to build WASM module via `wasm-pack`.
- `dist/viewer.js`: The bundled and minified JavaScript file.
- `dist/viewer.js.gz`: The Gzip-compressed version of the bundled JavaScript.

## Prerequisites

- Node.js (v18+) and `npm`
- Rust toolchain (for WASM builds)
- `wasm-pack` ([https://rustwasm.github.io/wasm-pack/](https://rustwasm.github.io/wasm-pack/))
- `rustup target add wasm32-unknown-unknown`

## Build Instructions

1.  Navigate to this directory:
    ```bash
    cd riichienv-ui
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Build the viewer (full build including WASM):
    ```bash
    npm run build
    ```
    This will:
    - Build `riichienv-wasm` via `wasm-pack` (`npm run build:wasm`).
    - Generate `src/tiles.ts` from SVG assets (`npm run build:tiles`).
    - Bundle and minify everything into `dist/viewer.js` using `esbuild` with WASM binary inlined (`--loader:.wasm=binary`).
    - Compress the bundle into `dist/viewer.js.gz`.
    - Copy `dist/viewer.js.gz` to `../src/riichienv/visualizer/assets/viewer.js.gz`.

To skip the WASM rebuild (e.g., when only changing TypeScript code):
```bash
npm run build:no-wasm
```

## Release Procedure

When you want to update the viewer used by the `riichienv` package:

1.  Follow the **Build Instructions** above (`npm run build`).
2.  The build pipeline automatically copies `dist/viewer.js.gz` to the Python package assets directory (`src/riichienv/visualizer/assets/viewer.js.gz`).
3.  Commit the updated assets.

Note: `src/tiles.ts`, `src/wasm/pkg/`, and `dist/` are excluded from the repository. The visualizer package transparently handles the Gzipped asset.

## LiveViewer

The `LiveViewer` class supports real-time game visualization by accepting events incrementally:

```typescript
import { LiveViewer } from './live_viewer';

const viewer = new LiveViewer(container, { viewpoint: 0 });
viewer.pushEvent({ type: 'start_kyoku', ... });
viewer.pushEvent({ type: 'tsumo', ... });
```

When WASM is loaded, wait tiles are automatically calculated in the browser for hands without pre-computed `meta.waits`.
