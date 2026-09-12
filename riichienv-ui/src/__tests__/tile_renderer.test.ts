import { describe, expect, it } from 'vitest';
import { TileRenderer } from '../renderers/tile_renderer';
import { TILE_COLUMNS, TILE_POSITIONS, TILE_ROWS } from '../tiles';

describe('tile atlas', () => {
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
