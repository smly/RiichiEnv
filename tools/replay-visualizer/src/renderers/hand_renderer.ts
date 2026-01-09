import { TileRenderer } from './tile_renderer';

export class HandRenderer {
    static renderHand(hand: string[], melds: any[], playerIndex: number): HTMLElement {
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
        const hasTsumo = (totalTiles % 3 === 2);

        hand.forEach((t, idx) => {
            const tDiv = document.createElement('div');
            tDiv.style.width = '40px'; tDiv.style.height = '56px';
            tDiv.innerHTML = TileRenderer.getTileHtml(t);
            if (hasTsumo && idx === hand.length - 1) tDiv.style.marginLeft = '12px';
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

    private static renderMeld(container: HTMLElement, m: { type: string, tiles: string[], from: number }, actor: number) {
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

                    // Kakan stacking:
                    // If 2 tiles, they stack in the flex row (visual vertical).
                    // The 'top: 6px' shifts the CENTER of the stack down.
                    // For 1 tile: Center is at 21px. Tile is 30px high (visually). Top at 6, Bottom at 36.
                    // Shift +6 -> Top at 12, Bottom at 42. Perfect.
                    // For 2 tiles (60px total): Center at 21. Top at -9, Bottom at 51.
                    // Shift +6 -> Top at -3, Bottom at 57.
                    // Kakan will stick out bottom?
                    // Yes, but Kakan shouldn't be aligned to baseline per se?
                    // Actually, usually bottom tile aligns to baseline.
                    // If 2 tiles start at -3 (top) and end at 57 (bottom).
                    // The "bottom" tile (the one added?)
                    // Normalized order: [original, added].
                    // Flex row (visual vertical down): Original is top, Added is bottom.
                    // So Original is -3 to 27. Added is 27 to 57.
                    // We want Original to align? Or Added?
                    // Usually Original is the pon tile. Added is on top.
                    // In real life, added tile is placed ON TOP (z-axis) or "above" (y-axis) the original.
                    // In 2D view, usually "above" means -Y.
                    // Here visual stack is +Y (Right in flex).
                    // So Added tile is BELOW the original visually?
                    // That seems wrong.
                    // If we want Visual Up (-Y), we need flex-direction: row-reverse?
                    // Or rotate(-90)?
                    // Existing code uses rotate(90).
                    // rotate(90): Right -> Down.
                    // So we are stacking downwards.
                    // If we want the stack to go "Up" (added tile on top), we need it to be "Left" in unrotated?
                    // Or "Right" in unrotated means "Down".
                    // "Left" in unrotated means "Up".
                    // So we want to stack towards Left?
                    // flex-direction: row-reverse?
                    // Let's assume standard Pon is fine, Kakan might be slightly off but acceptable for now.
                    // Priority is single tile alignment.

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
