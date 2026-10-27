import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { TileRenderer } from '../renderers/tile_renderer';
import { TILE_COLUMNS, TILE_POSITIONS, TILE_ROWS, TILE_SPRITE_UNLABELED_URL, TILE_SPRITE_URL } from '../tiles';

describe('tile atlas', () => {
    it('keeps both atlases aligned and preserves artwork, red-five dots and transparency outside corner labels', async () => {
        const decode = (url: string) =>
            sharp(Uint8Array.from(atob(url.split(',')[1]), (char) => char.charCodeAt(0)))
                .raw()
                .toBuffer({ resolveWithObject: true });
        const [labeled, unlabeled] = await Promise.all([decode(TILE_SPRITE_URL), decode(TILE_SPRITE_UNLABELED_URL)]);
        expect(unlabeled.info).toEqual(labeled.info);
        const { width, height, channels } = labeled.info;
        expect(width).toBe(TILE_COLUMNS * 120);
        expect(height).toBe(TILE_ROWS * 192);
        for (const column of [2, 3, 5]) {
            // The space above 西・北・發 must be a clean tile face, including
            // the left edge of the former W/N/G labels and their white outlines.
            let residue = 0;
            for (let y = 22; y < 52; y++) {
                for (let x = 70; x < 112; x++) {
                    const offset = ((3 * 192 + y) * width + column * 120 + x) * channels;
                    if ([0, 1, 2].some((channel) => unlabeled.data[offset + channel] !== 228)) residue++;
                }
            }
            expect(residue, `corner label residue in honor column ${column}`).toBe(0);
        }
        const restored = new Uint8Array(unlabeled.data);
        const changedTiles = new Set<string>();
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const column = Math.floor(x / 120);
                const row = Math.floor(y / 192);
                const labelLeft = row === 3 ? [88, 88, 74, 82, 54, 84, 88][column] : 88;
                if (row === 3 && column >= 7) continue;
                if (x % 120 < labelLeft || x % 120 >= 116 || y % 192 < 22 || y % 192 >= 58) continue;
                const offset = (y * width + x) * channels;
                for (let c = 0; c < 3; c++) {
                    if (restored[offset + c] !== labeled.data[offset + c]) changedTiles.add(`${column},${row}`);
                    restored[offset + c] = labeled.data[offset + c];
                }
            }
        }
        // Only RGB inside the label regions may differ; alpha must always match.
        expect(restored.every((value, index) => value === labeled.data[index])).toBe(true);
        expect(changedTiles.size).toBe(37);
    });

    it('covers all 34 normal tiles and three distinct red fives without out-of-bounds cells', () => {
        const ids = ['m', 'p', 's'].flatMap((suit) => Array.from({ length: 9 }, (_, i) => `${i + 1}${suit}`));
        ids.push('E', 'S', 'W', 'N', 'P', 'F', 'C', '5mr', '5pr', '5sr');
        const cells = ids.map((id) => {
            const cell = TILE_POSITIONS.get(id);
            expect(cell, id).toBeDefined();
            const [x, y] = cell!;
            expect(x).toBeGreaterThanOrEqual(0);
            expect(x).toBeLessThan(TILE_COLUMNS);
            expect(y).toBeGreaterThanOrEqual(0);
            expect(y).toBeLessThan(TILE_ROWS);
            return `${x},${y}`;
        });
        expect(new Set(cells).size).toBe(37);
    });

    it.each(['m', 'p', 's'])('renders 0%s as a red five, separately from the normal five', (suit) => {
        const red = TileRenderer.getTileHtml(`5${suit}r`);
        expect(TileRenderer.getTileHtml(`0${suit}`)).toBe(red);
        expect(TileRenderer.getTileHtml(`5${suit}`)).not.toBe(red);
    });

    it('renders concealed tiles and unknown IDs without interpolating input into HTML', () => {
        expect(TileRenderer.getTileHtml('?')).toBe(TileRenderer.getTileHtml('back'));
        const blank = TileRenderer.getTileHtml('blank');
        expect(TileRenderer.getTileHtml('<img src=x onerror=alert(1)>')).toBe(blank);
        expect(TileRenderer.getTileHtml('constructor')).toBe(blank);
    });
});
