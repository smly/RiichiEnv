import { TileRenderer } from './tile_renderer';

export class HandRenderer {
    static renderHand(hand: string[], melds: any[], playerIndex: number, highlightTiles?: Set<string>, hasDraw?: boolean, dahaiAnim?: { insertIdx: number, tsumogiri: boolean }, shouldAnimate: boolean = true): HTMLElement {
        // Hand & Melds Area
        const handArea = document.createElement('div');
        Object.assign(handArea.style, {
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-end',
            width: '580px',
            height: '56px',
            position: 'absolute',
            bottom: '0px',
            left: '50%',
            transform: 'translateX(-50%)'
        });

        // Closed Hand - Anchor Left
        const tilesDiv = document.createElement('div');
        Object.assign(tilesDiv.style, {
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'flex-start',
            flexGrow: 1 // Let it take available space but align start
        });

        const totalTiles = hand.length + melds.length * 3;
        // Use passed hasDraw flag, default to false if undefined
        const isSeparated = hasDraw || false;

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        hand.forEach((t, idx) => {
            const tDiv = document.createElement('div');
            tDiv.style.width = '40px'; tDiv.style.height = '56px';
            tDiv.style.position = 'relative'; // For absolute overlay
            tDiv.innerHTML = TileRenderer.getTileHtml(t);
            // Only separate if isSeparated is true AND it's the very last tile of the hand
            if (isSeparated && idx === hand.length - 1) {
                tDiv.style.marginLeft = '12px';
                if (shouldAnimate) {
                    tDiv.classList.add('tsumo-anim');
                }
            }

            // Sort Animation (for Te-dashi)
            if (dahaiAnim && !dahaiAnim.tsumogiri && idx === dahaiAnim.insertIdx) {
                tDiv.classList.add('sort-anim');
                // Calculate distance from Tsumo slot (theoretical idx 13 + gap)
                // Source X: 13 * 40 + 12
                // Target X: idx * 40
                const sortDx = (hand.length - idx) * 40 + 12;
                // After discard, hand has 13 tiles. 
                // The tile moved FROM the 14th slot (index 13, plus margin).
                // So (13 - idx) * 40 + 12 should be correct.

                // Use variable
                tDiv.style.setProperty('--sort-dx', `${sortDx}px`);
            }

            // Check Highlight
            if (highlightTiles) {
                const normT = normalize(t);
                if (highlightTiles.has(normT)) {
                    // Create Overlay
                    const overlay = document.createElement('div');
                    Object.assign(overlay.style, {
                        position: 'absolute',
                        top: '0', left: '0',
                        width: '100%', height: '100%',
                        backgroundColor: 'rgba(255, 0, 0, 0.4)', // Slightly stronger red for visibility
                        zIndex: '10',
                        pointerEvents: 'none',
                        borderRadius: '4px' // Match tile radius roughly
                    });
                    tDiv.appendChild(overlay);
                }
            }

            tilesDiv.appendChild(tDiv);
        });
        handArea.appendChild(tilesDiv);

        // Melds (Furo)
        const meldsDiv = document.createElement('div');
        Object.assign(meldsDiv.style, {
            display: 'flex',
            flexDirection: 'row-reverse',
            gap: '2px',
            alignItems: 'flex-end'
        });

        if (melds.length > 0) {
            melds.forEach(m => {
                this.renderMeld(meldsDiv, m, playerIndex);
            });
        }
        handArea.appendChild(meldsDiv);
        return handArea;
    }

    public static renderMeld(container: HTMLElement, m: { type: string, tiles: string[], from: number }, actor: number) {
        const mGroup = document.createElement('div');
        Object.assign(mGroup.style, {
            display: 'flex',
            alignItems: 'flex-end',
            marginLeft: '5px',
            gap: '0px' // Reduce gap between tiles within meld to 0 (borders provide separation)
        });

        // Determine relative position of target: (target - actor + 4) % 4
        // 1: Right, 2: Front, 3: Left
        const rel = (m.from - actor + 4) % 4;

        const tiles = [...m.tiles]; // 3 for Pon/Chi, 4 for Kan




        // Define Column Structure
        interface MeldColumn {
            tiles: string[];
            rotate: boolean;
        }
        let columns: MeldColumn[] = [];

        if (m.type === 'ankan') {
            // Ankan: [Back, Tile, Tile, Back]
            tiles.forEach((t, i) => {
                const tileId = (i === 0 || i === 3) ? 'back' : t;
                columns.push({ tiles: [tileId], rotate: false });
            });
        } else if (m.type === 'kakan') {
            const added = tiles.pop()!;
            const ponTiles = tiles; // 3 remaining

            // Pon Logic
            const stolen = ponTiles.pop()!;
            const consumed = ponTiles; // 2 remaining

            // Reconstruct Pon cols
            if (rel === 1) { // Right
                // [c1, c2, stolen(Rot)]
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                columns.push({ tiles: [stolen, added], rotate: true });
            } else if (rel === 2) { // Front
                // [c1, stolen(Rot), c2]
                if (consumed.length >= 2) {
                    columns.push({ tiles: [consumed[0]], rotate: false });
                    columns.push({ tiles: [stolen, added], rotate: true });
                    columns.push({ tiles: [consumed[1]], rotate: false });
                } else {
                    // Fallback
                    consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                    columns.push({ tiles: [stolen, added], rotate: true });
                }
            } else if (rel === 3) { // Left
                // [stolen(Rot), c1, c2]
                columns.push({ tiles: [stolen, added], rotate: true });
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
            } else {
                // Self (Shouldn't happen)
                [...consumed, stolen, added].forEach(t => columns.push({ tiles: [t], rotate: false }));
            }
        } else if (m.type === 'daiminkan') {
            // Open Kan
            const stolen = tiles.pop()!;
            const consumed = tiles; // 3 remaining

            if (rel === 1) { // Right
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                columns.push({ tiles: [stolen], rotate: true });
            } else if (rel === 2) { // Front
                if (consumed.length >= 3) {
                    columns.push({ tiles: [consumed[0]], rotate: false });
                    columns.push({ tiles: [consumed[1]], rotate: false });
                    columns.push({ tiles: [stolen], rotate: true });
                    columns.push({ tiles: [consumed[2]], rotate: false });
                } else {
                    columns.push({ tiles: [consumed[0]], rotate: false });
                    columns.push({ tiles: [consumed[1]], rotate: false });
                    columns.push({ tiles: [stolen], rotate: true }); // Fallback
                }
            } else if (rel === 3) { // Left
                columns.push({ tiles: [stolen], rotate: true });
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
            } else {
                [...consumed, stolen].forEach(t => columns.push({ tiles: [t], rotate: false }));
            }
        } else {
            // Pon / Chi
            const stolen = tiles.pop()!;
            const consumed = tiles; // 2 remaining

            if (rel === 1) { // Right
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                columns.push({ tiles: [stolen], rotate: true });
            } else if (rel === 2) { // Front
                if (consumed.length >= 2) {
                    columns.push({ tiles: [consumed[0]], rotate: false });
                    columns.push({ tiles: [stolen], rotate: true });
                    columns.push({ tiles: [consumed[1]], rotate: false });
                } else {
                    consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                    columns.push({ tiles: [stolen], rotate: true });
                }
            } else if (rel === 3) { // Left
                columns.push({ tiles: [stolen], rotate: true });
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
            } else {
                [...consumed, stolen].forEach(t => columns.push({ tiles: [t], rotate: false }));
            }
        }

        // Render Columns
        columns.forEach(col => {
            const div = document.createElement('div');
            if (col.rotate) {
                // Rotated Column
                Object.assign(div.style, {
                    width: '42px', // Rotated height becomes width (42px)
                    height: '42px', // Match upright height
                    position: 'relative',
                    marginLeft: '0px',
                    marginRight: '0px'
                });

                // Wrapper to rotate
                const rotator = document.createElement('div');
                Object.assign(rotator.style, {
                    transform: 'rotate(90deg)',
                    transformOrigin: 'center center',
                    width: '100%',
                    height: '100%',
                    display: 'flex', // Use flex to stack
                    gap: '0px',
                    justifyContent: 'center',
                    alignItems: 'center',
                    position: 'relative',
                    top: '6px' // Push down to align visual bottom with baseline (42-30)/2 = 6px gap to close
                });

                col.tiles.forEach((t, idx) => {
                    const inner = document.createElement('div');
                    inner.innerHTML = TileRenderer.getTileHtml(t);
                    Object.assign(inner.style, {
                        width: '30px',
                        height: '42px',
                        display: 'block' // Ensure block
                    });

                    rotator.appendChild(inner);
                });
                div.appendChild(rotator);
            } else {
                // Upright
                Object.assign(div.style, {
                    width: '30px',  // Reduced from 40px to match meld scale
                    height: '42px', // Reduced from 56px to match meld scale
                    display: 'flex',
                    alignItems: 'flex-end',
                    justifyContent: 'center',
                    marginLeft: '0px',
                    marginRight: '0px'
                });
                if (col.tiles.length > 0) {
                    div.innerHTML = TileRenderer.getTileHtml(col.tiles[0]);
                }
            }
            mGroup.appendChild(div);
        });

        container.appendChild(mGroup);
    }
}
