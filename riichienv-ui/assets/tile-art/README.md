# Tile face atlas

`tiles.png` is the final artwork used by both HTML renderers. `tiles.json`
defines the tile IDs in its 10 × 4 grid, with 120 × 192 pixels per cell.
The first three rows contain 1–9 of each suit followed by its red five.
The last row contains the winds, dragons, back and blank; its last cell is unused.
Artwork and auxiliary labels are baked into the image, so no font or external
image service is required at runtime.

From `riichienv-ui`, run `npm run build:tiles` to embed the PNG and cell positions
in the generated `src/tiles.ts`. `npm run build` also builds WASM, the JavaScript
bundles and the packaged Python asset. With WASM already built,
`npm run build:no-wasm` rebuilds those assets without rebuilding WASM.

To update artwork, replace the relevant complete cell while preserving the grid
dimensions. Serve the UI directory and open `demos/tile_gallery.html` to inspect
the final tiles. Historical artwork, comparison images and authoring inputs are
not required by the build.

## Artwork attribution

The manzu and 東南西北發中 glyphs are adapted from the lower two rows (行書體)
of **Kanji var mj.svg** by **Cangjie6**:

- Source: https://commons.wikimedia.org/wiki/File:Kanji_var_mj.svg
- Original SVG: https://upload.wikimedia.org/wikipedia/commons/2/24/Kanji_var_mj.svg
- License: **CC BY-SA 4.0**, https://creativecommons.org/licenses/by-sa/4.0/

Changes include extracting, recoloring, scaling and positioning the glyphs,
then adjusting their contours toward the previous tile artwork to create an
intermediate design. The modified glyph artwork and resulting atlas are
distributed under CC BY-SA 4.0. This asset license does not change the license
of the application code.
