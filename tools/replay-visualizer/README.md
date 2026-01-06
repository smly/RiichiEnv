# RiichiEnv Replay Visualizer

A standalone, web-based replay viewer for RiichiEnv.

## Structure

- `src/`: TypeScript source code.
- `riichienv-mahjong-tiles-regular/`: SVG tile assets.
- `scripts/gen_tiles.js`: Script to bundle SVG tiles into `src/tiles.ts`.
- `dist/viewer.js`: The bundled and minified JavaScript file.
- `dist/viewer.js.gz`: The Gzip-compressed version of the bundled JavaScript.
- `scripts/compress.js`: Script to compress the bundle.

## Build Instructions

1.  Navigate to this directory:
    ```bash
    cd tools/replay-visualizer
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Build the viewer:
    ```bash
    npm run build
    ```
    This will:
    - Generate `src/tiles.ts` from SVG assets (optimized using `svgo`).
    - Bundle and minify everything into `dist/viewer.js` using `esbuild`.
    - Compress the bundle into `dist/viewer.js.gz` using `zlib`.

## Release Procedure

When you want to update the viewer used by the `riichienv` package:

1.  Follow the **Build Instructions** above.
2.  Copy the compressed `dist/viewer.js.gz` to the package assets directory:
    ```bash
    cp dist/viewer.js.gz ../../src/riichienv/visualizer/assets/viewer.js.gz
    ```
3.  (Optional) Release the uncompressed `dist/viewer.js` as well if needed.
4.  Commit the updated assets.

Note: `src/tiles.ts` and `dist/` are excluded from the repository to minimize size. The visualizer package transparently handles the Gzipped asset.
