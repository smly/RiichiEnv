import { Tile } from '../types';
import { TileRenderer } from './tile_renderer';

export class RiverRenderer {
    static renderRiver(discards: Tile[], highlightTiles?: Set<string>, dahaiAnim?: { discardIdx: number, insertIdx: number, tsumogiri: boolean }): HTMLElement {
        // River
        const riverDiv = document.createElement('div');
        riverDiv.className = 'river-container';

        // Split discards into rows
        const rows: Tile[][] = [[], [], []];
        discards.forEach((d, idx) => {
            if (idx < 6) rows[0].push(d);
            else if (idx < 12) rows[1].push(d);
            else rows[2].push(d);
        });

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        rows.forEach((rowTiles) => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'river-row';
            rowDiv.style.height = '46px';

            rowTiles.forEach(d => {
                const cell = document.createElement('div');
                cell.style.width = '34px';
                cell.style.height = '46px';
                cell.style.position = 'relative'; // Important for overlay
                cell.style.flexShrink = '0'; // Prevent shrinking in 3rd row

                // Create wrapper for tile content (to separate bg highlight from moving tile)
                const tileWrapper = document.createElement('div');
                tileWrapper.style.width = '100%';
                tileWrapper.style.height = '100%';

                // New Inner Content Wrapper for Rotation
                const tileContent = document.createElement('div');
                tileContent.style.width = '100%';
                tileContent.style.height = '100%';

                // Handle Riichi Rotation
                if (d.isRiichi) {
                    // Make space for rotated tile in cell
                    cell.style.width = '46px';
                    // Rotate the inner content, NOT the animating wrapper
                    tileContent.className = 'tile-rotated';
                }

                tileContent.innerHTML = TileRenderer.getTileHtml(d.tile);
                tileWrapper.appendChild(tileContent);
                cell.appendChild(tileWrapper);

                if (d.isTsumogiri) cell.style.filter = 'brightness(0.7)';

                const idx = discards.indexOf(d); // Note: verify this works with duplicates?
                // discards is state.players[x].discards.
                // If duplicates exist, indexOf returns first occurence.
                // But loop is efficient. Can use 2nd arg of forEach for index in *row*,
                // but we need global index.
                // Use a counter outside? Or just trust object identity (Wait, discards objects are distinct references?)
                // In game_state.ts, we push { tile: ..., isRiichi: ... } objects.
                // New object created each push. So object identity references are unique?
                // Yes: p.discards.push({ tile: e.pai ... }).
                // So indexOf is safe.

                const isLast = (idx === discards.length - 1);
                if (isLast && dahaiAnim) {
                    tileWrapper.classList.add('dahai-anim');

                    // Display Arrow
                    const arrow = document.createElement('div');
                    arrow.className = 'dahai-arrow';
                    cell.appendChild(arrow);

                    // Reset target-y (no shift)
                    tileWrapper.style.setProperty('--target-y', '0px');

                    // Calculate offsets
                    // From discard index
                    // River row width ~214. Tile idx in river (0..5).
                    // River tile X approx: (idx % 6) * 36 - 100.
                    // Hand tile X: (discardIdx - 6) * 40.
                    // Delta X = HandX - RiverX.
                    let dx = 0;
                    if (dahaiAnim.tsumogiri) {
                        dx = 200; // From right side
                    } else {
                        const riverX = (idx % 6) * 36 - 107; // Approx center relative
                        const handX = (dahaiAnim.discardIdx - 6) * 40;
                        dx = handX - riverX;
                    }
                    tileWrapper.style.setProperty('--dx', `${dx}px`);
                    tileWrapper.style.setProperty('--dy', `150px`); // From below

                    // Ensure high z-index for moving tile
                    cell.style.zIndex = '100';
                }

                // Highlight Logic
                if (highlightTiles) {
                    const normT = normalize(d.tile);
                    if (highlightTiles.has(normT)) {
                        const overlay = document.createElement('div');
                        Object.assign(overlay.style, {
                            position: 'absolute',
                            top: '0', left: '0',
                            width: '100%', height: '100%',
                            backgroundColor: 'rgba(255, 0, 0, 0.4)',
                            zIndex: '10', // Just above tile
                            pointerEvents: 'none',
                            borderRadius: '4px'
                        });
                        // Append to wrapper so it moves with tile
                        tileWrapper.appendChild(overlay);
                    }
                }

                rowDiv.appendChild(cell);
            });
            riverDiv.appendChild(rowDiv);
        });
        return riverDiv;
    }
}
