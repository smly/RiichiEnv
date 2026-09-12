const fs = require('fs');
const path = require('path');

const assetDir = path.join(__dirname, '..', 'assets', 'tile-art');
const atlas = fs.readFileSync(path.join(assetDir, 'tiles.png'));
const { tileWidth, tileHeight, rows } = JSON.parse(
    fs.readFileSync(path.join(assetDir, 'tiles.json'), 'utf8'),
);
const columns = rows[0].length;
if (rows.some(row => row.length !== columns)
    || atlas.readUInt32BE(16) !== columns * tileWidth
    || atlas.readUInt32BE(20) !== rows.length * tileHeight) {
    throw new Error('Tile atlas dimensions do not match tiles.json');
}
const positions = rows.flatMap((row, y) => row.flatMap((id, x) => id ? [[id, [x, y]]] : []));
for (const suit of ['m', 'p', 's']) {
    positions.push([`0${suit}`, positions.find(([id]) => id === `5${suit}r`)[1]]);
}
const output = `export const TILE_SPRITE_URL = 'data:image/png;base64,${atlas.toString('base64')}';
export const TILE_COLUMNS = ${columns};
export const TILE_ROWS = ${rows.length};
export const TILE_POSITIONS = new Map<string, readonly [number, number]>(${JSON.stringify(positions)});
`;
fs.writeFileSync(path.join(__dirname, '..', 'src', 'tiles.ts'), output);
console.log(`Embedded ${positions.length - 3} tiles (+3 red-five aliases) in one ${(atlas.length / 1024).toFixed(1)} KiB PNG atlas.`);
