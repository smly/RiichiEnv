# RiichiEnv UI

A standalone, web-based UI for RiichiEnv featuring 2D/3D replay viewers and a live game viewer.

## Structure

- `src/`: TypeScript source code.
  - `src/index.ts`: Entry point. Exports `Viewer`, `Viewer3D`, `LiveViewer`, `RiichiViewer`.
  - `src/riichi_viewer.ts`: High-level API (`RiichiViewer.mount()`) with event system.
  - `src/base_viewer.ts`: Abstract base class shared by `Viewer` (2D) and `Viewer3D`.
  - `src/viewer.ts`: 2D replay viewer.
  - `src/viewer_3d.ts`: 3D replay viewer.
  - `src/live_viewer.ts`: Live game viewer (incremental event processing).
  - `src/live_controller.ts`: Keyboard/UI controller for live mode.
  - `src/game_state.ts`: Game state management with WASM-accelerated wait calculation.
  - `src/config.ts`: Layout configuration for 3P/4P games.
  - `src/renderers/`: Renderer implementations.
    - `renderer_2d.ts`: 2D renderer.
    - `renderer_3d.ts`: 3D renderer (CSS 3D transforms).
    - `tile_renderer.ts`, `hand_renderer.ts`, `river_renderer.ts`, `center_renderer.ts`, `info_renderer.ts`, `result_renderer.ts`: Sub-renderers.
  - `src/wasm/`: WASM integration layer (`loader.ts`, `bridge.ts`).
  - `src/styles.ts`, `src/styles_3d.ts`: CSS styles for 2D/3D modes.
- `riichienv-mahjong-tiles-regular/`: SVG tile assets.
- `scripts/`: Build scripts (`gen_tiles.js`, `gen_sprite.js`, `build-wasm.sh`, `compress.js`).
- `dist/viewer.js`: Bundled IIFE format (registers globals on `window`).
- `dist/viewer.esm.js`: Bundled ESM format.
- `dist/viewer.js.gz`: Gzip-compressed version for the Python package.

## Prerequisites

- Node.js (v20+) and `npm`
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
    - Bundle and minify into `dist/viewer.js` (IIFE) and `dist/viewer.esm.js` (ESM) using `esbuild` with WASM binary inlined.
    - Compress the bundle into `dist/viewer.js.gz`.
    - Copy `dist/viewer.js.gz` to `../src/riichienv/visualizer/assets/viewer.js.gz`.

To skip the WASM rebuild (e.g., when only changing TypeScript code):
```bash
npm run build:no-wasm
```

## Usage

### RiichiViewer (recommended API)

```typescript
import { RiichiViewer } from 'riichienv-ui';

const viewer = RiichiViewer.mount('container-id', {
    log: events,           // MjaiEvent[]
    renderer: '3d',        // '2d' or '3d' (default: '3d')
    perspective: 0,        // player viewpoint (0-3)
    freeze: false,         // disable controls
    initialPosition: { kyoku: 0 },
});

viewer.on('positionChange', ({ kyokuIndex, step }) => { ... });
viewer.on('kyokuChange', ({ kyokuIndex, round, honba }) => { ... });
viewer.on('viewpointChange', ({ viewpoint }) => { ... });
viewer.destroy();
```

### Direct constructors (via script tag)

When loaded via `<script src="dist/viewer.js">`, the following globals are available:

- `window.RiichiViewer` - High-level API
- `window.RiichiEnvViewer` - 2D viewer (`Viewer`)
- `window.RiichiEnv3DViewer` - 3D viewer (`Viewer3D`)
- `window.RiichiEnvLiveViewer` - Live viewer (`LiveViewer`)

```javascript
// 3D viewer
new RiichiEnv3DViewer('container-id', events);

// 2D viewer
new RiichiEnvViewer('container-id', events);
```

### LiveViewer

The `LiveViewer` class supports real-time game visualization by accepting events incrementally:

```typescript
import { LiveViewer } from 'riichienv-ui';

const viewer = new LiveViewer(container, { viewpoint: 0 });
viewer.pushEvent({ type: 'start_kyoku', ... });
viewer.pushEvent({ type: 'tsumo', ... });
```

When WASM is loaded, wait tiles are automatically calculated in the browser for hands without pre-computed `meta.waits`.

## Release Procedure

1.  Follow the **Build Instructions** above (`npm run build`).
2.  The build pipeline automatically copies `dist/viewer.js.gz` to the Python package assets directory (`src/riichienv/visualizer/assets/viewer.js.gz`).
3.  Commit the updated assets.

Note: `src/tiles.ts`, `src/wasm/pkg/`, and `dist/` are excluded from the repository. The visualizer package transparently handles the Gzipped asset.
