const fs = require('fs');
const path = require('path');

const notices = fs.readFileSync(path.join(__dirname, '..', '..', 'THIRD_PARTY_NOTICES.md'), 'utf8');
const assetDir = path.join(__dirname, '..', 'assets', 'tile-art');
const atlas = fs.readFileSync(path.join(assetDir, 'tiles.png'));
const unlabeledAtlas = fs.readFileSync(path.join(assetDir, 'tiles-unlabeled.png'));
const { tileWidth, tileHeight, rows } = JSON.parse(
    fs.readFileSync(path.join(assetDir, 'tiles.json'), 'utf8'),
);
const columns = rows[0].length;
if (rows.some(row => row.length !== columns)
    || [atlas, unlabeledAtlas].some(image => image.readUInt32BE(16) !== columns * tileWidth
        || image.readUInt32BE(20) !== rows.length * tileHeight)) {
    throw new Error('Tile atlas dimensions do not match tiles.json');
}
const positions = rows.flatMap((row, y) => row.flatMap((id, x) => id ? [[id, [x, y]]] : []));
for (const suit of ['m', 'p', 's']) {
    positions.push([`0${suit}`, positions.find(([id]) => id === `5${suit}r`)[1]]);
}
const output = `/*!\n${notices}*/
export const TILE_SPRITE_URL = 'data:image/png;base64,${atlas.toString('base64')}';
export const TILE_SPRITE_UNLABELED_URL = 'data:image/png;base64,${unlabeledAtlas.toString('base64')}';
export const TILE_COLUMNS = ${columns};
export const TILE_ROWS = ${rows.length};
export const TILE_POSITIONS = new Map<string, readonly [number, number]>(${JSON.stringify(positions)});
`;
fs.writeFileSync(path.join(__dirname, '..', 'src', 'tiles.ts'), output);
console.log(`Embedded ${positions.length - 3} tiles (+3 red-five aliases) in labeled and unlabeled PNG atlases (${((atlas.length + unlabeledAtlas.length) / 1024).toFixed(1)} KiB total).`);
