import { TILE_COLUMNS, TILE_POSITIONS, TILE_ROWS } from '../tiles';

export class TileRenderer {
    /** Pre-parsed DOM cache: parse once per tile type, cloneNode thereafter. */
    private static _elementCache: Map<string, HTMLElement> = new Map();

    static getTileHtml(tileStr: string): string {
        const [column, row] = TILE_POSITIONS.get(tileStr === '?' ? 'back' : tileStr) || TILE_POSITIONS.get('blank')!;
        const x = (column / (TILE_COLUMNS - 1)) * 100;
        const y = (row / (TILE_ROWS - 1)) * 100;
        // The shared stylesheet embeds the PNG once; each face selects one cell.
        // Keep tile-layer/tile-bg so the CSS 3D box and meld shadows stay intact.
        const backClass = tileStr === 'back' || tileStr === '?' ? ' tile-back' : '';
        return `<div class="tile-layer${backClass}"><div class="tile-bg"><div class="tile-sprite" style="background-position:${x}% ${y}%"></div></div></div>`;
    }

    /**
     * Return a cloned DOM element for a tile.
     * The tile HTML is parsed only once per tile type and cached;
     * subsequent calls return a deep clone (no HTML parsing).
     */
    static getTileElement(tileStr: string): HTMLElement {
        let template = TileRenderer._elementCache.get(tileStr);
        if (!template) {
            const container = document.createElement('div');
            container.innerHTML = TileRenderer.getTileHtml(tileStr);
            template = container.firstElementChild as HTMLElement;
            TileRenderer._elementCache.set(tileStr, template);
        }
        return template.cloneNode(true) as HTMLElement;
    }
}
