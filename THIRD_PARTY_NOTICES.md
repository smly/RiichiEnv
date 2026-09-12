# Third-party notices

The following notices apply to the identified third-party material, including
copies embedded in generated JavaScript, WASM, Python packages, and notebook
outputs. RiichiEnv's application code remains licensed under Apache-2.0; that
license does not replace the licenses below.

## Tile artwork — CC BY-SA 4.0

The kanji in `riichienv-ui/assets/tile-art/tiles.png` are derived from the lower
two rows of **Kanji var mj.svg** by **Cangjie6**. The upstream file history also
credits revisions by **Wj654cj86** and **Smasongarrison**.

Source: https://commons.wikimedia.org/wiki/File:Kanji_var_mj.svg

License: Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/

Changes by RiichiEnv: extracted the glyph paths, recolored, translated and
scaled them, and rasterized the result into the tile atlas. These notices retain the attribution
and modification history recorded during the artwork's creation.

The modified glyph artwork and resulting tile atlas are distributed under
CC BY-SA 4.0. The same attribution applies when that artwork is embedded in
viewer bundles or reproduced in screenshots and comparison images. This does
not change the license of the application code. The artwork is provided without
warranties; see Section 5 of the license:
https://creativecommons.org/licenses/by-sa/4.0/legalcode.en#s5

## Heroicons — MIT

Material: SVG icon paths in `riichienv-ui/src/icons.ts` and their copies in
generated viewers and saved notebook outputs.

Source: https://github.com/tailwindlabs/heroicons

MIT License

Copyright (c) Tailwind Labs, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Nyanten — MIT

Material: `riichienv-core/src/data/nyanten_*.bin` and Nyanten-derived lookup
tables in `riichienv-core/src/shanten.rs`, including compiled copies in native
and WASM binaries. The five binary key tables contain the arrays from
`nyanten/standard/keys.hpp` converted to bytes.

Source: https://github.com/Cryolite/nyanten
Reference revision: 581aa72ce53e0ffeb0ef8d1ed14df2521ca5fb9b

The same MIT notice is included separately in `riichienv-core/LICENSE.nyanten`
for distribution with the standalone Rust crate.

MIT License

Copyright (c) 2024 Cryolite
Copyright (c) 2025 Cryolite. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next
paragraph) shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
