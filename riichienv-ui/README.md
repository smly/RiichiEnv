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

### Replay controls (3D)

The CSS 3D table uses a bundled PNG tile atlas and needs no WebGL renderer,
external fonts, or texture downloads. Dora indicators and counters sit in the
upper left; the center shows the round, remaining wall tiles, and scores. Player
panels show a fixed avatar frame with the name below: click one to change viewpoint.
A yellow bar beside the center display marks the active player's direction and blinks
on a 1.2-second cycle. It follows viewpoint changes; player-name panels do not
repeat the turn indicator. Reduced-motion preferences keep the bar steady.
Scores share its warm yellow; round text, wall count, and replay controls use
a consistent blue accent.

The compact replay panel floats over the lower-right space above the hand. When
the table would scale below 72%, the panel moves below the table at its normal
size. Both fit the viewport height, accounting for content above the viewer.
Drag the timeline to seek, choose 0.5× / 1× / 2× / 4× playback speed,
or use the round selector to jump to a hand. Seeking or manual navigation pauses
playback. Arrow keys step through events (left/right) or turns (up/down); Space
plays or pauses while the table has focus. Input fields retain their native key
behavior. The round selector supports Tab navigation and Escape to close.

The round selector summarizes winners, ron/tsumo, hand limits, riichi declarations,
and each player's starting/ending scores and net movement. Net movement includes
riichi deposits. Hand-limit badges appear alongside the winning player, and draws
are marked in the round column. The compact table omits introductory text and filters.
On narrow screens, the round column stays visible
while the player columns scroll horizontally. The summary scans raw events without
replaying the game or moving its cursor. It uses score metadata when present and
otherwise infers limits from payments; unavailable results remain explicitly unknown.

`freeze: true` hides the controls and disables viewpoint changes. The 2D viewer
retains its existing layout. Reduced-motion preferences disable 3D animations.

Tile faces share one embedded PNG sprite (about 78 KiB); the tile sides,
depth, and shadows remain independent of the artwork. Normal and red fives use
separate cells, with `0m`/`0p`/`0s` aliases for the red variants. Preview all faces
in `demos/tile_gallery.html`; see `assets/tile-art/README.md` for the atlas format.

Table tiles use a cached PNG for the rounded white resin body and thin colored
back cap, plus one CSS plane for the printed face. This replaces the previous
42–44 CSS surfaces per tile with two surfaces. The small body PNG is generated
with Canvas 2D on a cache miss; there is no continuous canvas rendering or WebGL
dependency. The cache holds at most 384 body images and 32 geometry templates.
Size, pose, perspective, and highlight colors select the image. Transform reads
are batched before images are attached to avoid layout reads between writes.
The bitmap leaves the printed-face opening transparent so it cannot obscure
the live artwork. Its supporting plane sits in front of the body and is
reprojected to keep the silhouette in place, preventing the lower rim from
intersecting the tabletop. Opponent melds instead project their body images onto
the artwork plane and paint whole tiles from far to near, including added-kan
pairs and separate meld groups. This prevents adjacent bodies and faces from
interleaving while retaining two surfaces per tile.
The orange back occupies 20% of the depth so a band
remains visible above the rounded lower edge. Its color (`#d47a05`) is sampled
from the atlas's upper lip and shared by concealed-tile previews.
The flat printed area is cropped from the same atlas
so its illustrated frame is not duplicated on 3D faces. Depth follows the reference
16.5/20.5 thickness-to-width ratio, including concealed hands and melds. The own
hand uses the complete images without stacked extrusion shadows. The hand and
melds share one bottom-aligned row: concealed tiles start at the left, melds at
the right. The projected lower rim is accounted for so their visible bases align. Open
`demos/tile_geometry.html` to inspect concealed tiles, rotated discards, and all
kan/chi poses from each seat.

Opponent rows span a fixed part of each table edge: hands start at that player's
left, and melds accumulate from the right toward the left in call order. Their
anchors stay in place as the hand size changes, in both three- and four-player games.

For performance comparisons, use the same browser, viewport, fixture, and seat.
Measure inside `Renderer3D.render`, not just `BaseViewer.update` (which only
schedules a frame). Compare both cold renders and repeated updates after cache
warmup, as well as DOM/surface counts. Time to subsequent animation frames also
includes browser scheduling and must not be reported as GPU rendering time.

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

### Display language

The replay viewer supports Japanese (`ja`), English (`en`), Simplified Chinese
(`zh-Hans`), and Traditional Chinese (`zh-Hant`), defaulting to Japanese.
Change **設定 → 表示言語** to switch the
current viewer immediately, or use the embedding API:

```js
const viewer = RiichiViewer.mount('viewer', { log, language: 'en' });
viewer.setLanguage('ja');
console.log(viewer.getLanguage()); // 'ja'
```

Language is stored per viewer instance. Switching preserves the replay position,
viewpoint, and settings form inputs. It does not translate player names, tile
artwork, or raw replay/debug data. The other settings controls remain UI previews.

To add a language, add a complete message catalog matching `Messages` in
`src/i18n/ja.ts`, supply the yaku ID catalog, and register its BCP 47 locale ID and
native display name in `src/i18n/index.ts`. The settings options and exported
`Locale` type derive from that registry. Translation completeness and placeholder
compatibility are tested. The four registered languages appear in a compact language selector; scripts
remain independent so their terminology and regional wording can evolve separately.
