import { TILES } from '../tiles';

export class TileRenderer {
    /** Pre-parsed DOM cache: parse SVG once, cloneNode thereafter. */
    private static _elementCache: Map<string, HTMLElement> = new Map();

    static getTileHtml(tileStr: string): string {
        if (tileStr === 'back') {
            const svg = TILES['back'] || TILES['blank'];
            return `<div class="tile-layer"><div class="tile-bg">${svg}</div></div>`;
        }

        const frontSvg = TILES['front'] || '';
        let fgSvg = TILES[tileStr];
        if (!fgSvg) {
            fgSvg = TILES['blank'] || '';
        }

        return `
            <div class="tile-layer">
                <div class="tile-bg">${frontSvg}</div>
                <div class="tile-fg">${fgSvg}</div>
            </div>
        `;
    }

    /**
     * Return a cloned DOM element for a tile.
     * The SVG HTML is parsed only once per tile type and cached;
     * subsequent calls return a deep clone (no HTML parsing).
     */
    static getTileElement(tileStr: string): HTMLElement {
        let template = this._elementCache.get(tileStr);
        if (!template) {
            const container = document.createElement('div');
            container.innerHTML = this.getTileHtml(tileStr);
            template = container.firstElementChild as HTMLElement;
            this._elementCache.set(tileStr, template);
        }
        return template.cloneNode(true) as HTMLElement;
    }
}
