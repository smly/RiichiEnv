import { I18n } from '../i18n/index';
import { ICON_ARROW_LEFT, ICON_ARROW_RIGHT } from '../icons';
import type { BoardState } from '../types';
import { HandRenderer } from './hand_renderer';
import { TileRenderer } from './tile_renderer';

export class ResultRenderer {
    // We keep track of the current page index internally, or we can just render the modal and let it handle its own state.
    // Since Renderer.render creates new elements every time, we should create a self-contained component.

    static renderModal(results: any[], state: BoardState, i18n = new I18n()): HTMLElement {
        const modal = document.createElement('div');
        modal.className = 're-modal-overlay';
        // Prevent click propagation to close needed? The overlay itself might be the close target...
        // But usually we want click outside to close? The previous code didn't really support close properly.
        // It appended modal to board.
        // Let's implement click-to-close on the overlay background.

        const content = document.createElement('div');
        content.className = 're-modal-content re-result-dialog';
        // Stop propagation so clicking content doesn't close
        content.onclick = (e) => e.stopPropagation();

        modal.appendChild(content);

        // State for pagination
        let pageIndex = 0;
        const totalPages = results.length;

        const renderPage = (idx: number) => {
            content.innerHTML = ''; // Clear content

            const res = results[idx];
            const score = res.score;
            const actor = res.actor;
            const target = res.target;
            const isTsumo = actor === target;

            // Header (Result Title)
            const header = document.createElement('div');
            header.className = 're-modal-title';
            const actorName = state.playerNames?.[actor] || `Player${actor}`;
            const targetName = state.playerNames?.[target] || `Player${target}`;
            const titleText = i18n.text(isTsumo ? '{actor} Tsumo' : '{actor} Ron from {target}', {
                actor: actorName,
                target: targetName,
            });
            header.textContent = totalPages > 1 ? `${titleText} (${idx + 1}/${totalPages})` : titleText;
            content.appendChild(header);

            // Body
            const body = document.createElement('div');
            body.className = 're-result-body';

            // (Old Result Title removed from here)

            // 1. Hand & Melds
            const player = state.players[actor];
            if (player) {
                const handContainer = document.createElement('div');
                handContainer.className = 're-result-hand';

                // Closed Hand
                const closedDiv = document.createElement('div');
                Object.assign(closedDiv.style, {
                    display: 'flex',
                    alignItems: 'flex-end',
                });

                // If winningTile is known (from enriched meta), we exclude it from the main hand logic if it was a Tsumo?
                // Actually, logic:
                // Tsumo: Tile is in `player.hand`.
                // Ron: Tile is NOT in `player.hand`.
                // We want to display `[Hand] [WinningTile]`.
                // If Tsumo, we should remove the winning tile from the `hand` array we display in the first block,
                // and show it in the second block.
                // If Ron, `hand` is already correct (waiting state), and we show `winningTile` in second block.

                // Clone hand to avoid mutating state
                const displayHand = [...player.hand];
                const winningTile = res.winningTile;

                /* 
                   Correction: In our GameState.ts, for Tsumo, we pushed the tile to hand. 
                   So for Tsumo, the winning tile IS in displayHand.
                   For Ron, it is NOT. 
                   So if Tsumo, we pop the winning tile (or remove the specific instance).
                   Problem: exact instance matching. `winningTile` string (e.g. '1m').
                   If multiple '1m', removing one is fine.
                */
                const tileToShow = winningTile;

                if (isTsumo && winningTile) {
                    // Remove one instance of winningTile from displayHand
                    const idx = displayHand.indexOf(winningTile);
                    if (idx >= 0) {
                        displayHand.splice(idx, 1);
                    }
                }

                // 1a. Hand Body
                displayHand.forEach((t) => {
                    const tDiv = document.createElement('div');
                    tDiv.style.width = '40px';
                    tDiv.style.height = '56px';
                    tDiv.innerHTML = TileRenderer.getTileHtml(t);
                    closedDiv.appendChild(tDiv);
                });
                handContainer.appendChild(closedDiv);

                // 1b. Winning Tile (Agari Hai)
                if (tileToShow) {
                    const winDiv = document.createElement('div');
                    Object.assign(winDiv.style, {
                        display: 'flex',
                        alignItems: 'flex-end',
                        marginLeft: '10px', // Exact 5px gap as requested
                    });

                    const tDiv = document.createElement('div');
                    tDiv.style.width = '40px';
                    tDiv.style.height = '56px';
                    // Add a glow or specific style? User didn't ask, but good for visibility.
                    tDiv.innerHTML = TileRenderer.getTileHtml(tileToShow);
                    winDiv.appendChild(tDiv);

                    handContainer.appendChild(winDiv);
                }

                // 1c. Melds
                if (player.melds.length > 0) {
                    const meldsDiv = document.createElement('div');
                    Object.assign(meldsDiv.style, {
                        display: 'flex',
                        alignItems: 'flex-end',
                        gap: '5px',
                        marginLeft: '30px', // Separated from winning tile
                    });

                    // User said: "winning tile is to the right of hand. If melds, winning tile further right is melds."
                    // Layout: [Closed] -- [Win] -- [Melds]
                    // My code order is correct.

                    player.melds.forEach((m) => {
                        HandRenderer.renderMeld(meldsDiv, m, actor);
                    });
                    handContainer.appendChild(meldsDiv);
                }

                body.appendChild(handContainer);
            }

            // 3. Dora & Ura Dora
            const doraContainer = document.createElement('div');
            doraContainer.className = 're-result-dora';

            // Dora
            const createIndicatorRow = (label: string, tiles: string[]) => {
                const row = document.createElement('div');
                Object.assign(row.style, {
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                });
                const lbl = document.createElement('span');
                lbl.style.fontWeight = 'bold';
                lbl.textContent = label;
                row.appendChild(lbl);

                const tilesRow = document.createElement('div');
                tilesRow.style.display = 'flex';
                tilesRow.style.gap = '2px';

                tiles.forEach((t) => {
                    const tDiv = document.createElement('div');
                    tDiv.style.width = '30px';
                    tDiv.style.height = '42px'; // Smaller for indicators
                    tDiv.innerHTML = TileRenderer.getTileHtml(t);
                    tilesRow.appendChild(tDiv);
                });
                row.appendChild(tilesRow);
                return row;
            };

            if (state.doraMarkers && state.doraMarkers.length > 0) {
                doraContainer.appendChild(createIndicatorRow(i18n.text('Dora:'), state.doraMarkers));
            }

            // Ura Dora
            // Now accessed via res.uraMarkers (enriched in GameState)
            if (res.uraMarkers && Array.isArray(res.uraMarkers) && res.uraMarkers.length > 0) {
                doraContainer.appendChild(createIndicatorRow(i18n.text('Ura Dora:'), res.uraMarkers));
            }

            body.appendChild(doraContainer);

            // Guard: if score is unavailable (WASM failed and no meta), show minimal info
            if (!score) {
                const noScoreMsg = document.createElement('div');
                Object.assign(noScoreMsg.style, {
                    marginTop: '15px',
                    padding: '10px',
                    textAlign: 'center',
                    color: '#aaa',
                    fontStyle: 'italic',
                });
                noScoreMsg.textContent = i18n.text('Score data unavailable');
                body.appendChild(noScoreMsg);
                content.appendChild(body);
            } else {
                // 4. Yaku List - Updated Style
                if (score.yaku && score.yaku.length > 0) {
                    // Sort by ID ascending (copy to avoid mutation)
                    const sortedYaku = [...score.yaku].sort((a: number, b: number) => a - b);

                    const yakuList = document.createElement('ul');
                    yakuList.className = 're-yaku-list';

                    sortedYaku.forEach((yId: number) => {
                        const li = document.createElement('li');
                        li.textContent = i18n.yaku(yId);
                        yakuList.appendChild(li);
                    });
                    body.appendChild(yakuList);
                }

                // Calculate Limit / Rank
                const han = score.han;
                const fu = score.fu;
                // Check for Yakuman IDs (>= 35)
                const isYakuman = score.yaku?.some((y: number) => y >= 35) ?? false;
                const isKazoe = !isYakuman && han >= 13;

                // Apply special styling to content if Yakuman
                if (isYakuman || isKazoe) {
                    content.classList.add('is-yakuman'); // Helper class for specific borders
                } else {
                    content.classList.remove('is-yakuman');
                }

                // Limit Banner
                const limitInfo = ResultRenderer.getLimitInfo(han, fu, isYakuman, isKazoe);
                if (limitInfo) {
                    const limitBanner = document.createElement('div');
                    limitBanner.className = `limit-banner ${limitInfo.css}`;
                    limitBanner.textContent = i18n.text(limitInfo.text);
                    body.appendChild(limitBanner);
                }

                // 5. Han / Fu
                // Yakuman hands are not scored by han/fu. Keep this line for kazoe yakuman.
                if (!isYakuman || isKazoe) {
                    const statsRow = document.createElement('div');
                    statsRow.className = 're-result-stats';
                    for (const [key, count] of [
                        ['{count} Han', score.han],
                        ['{count} Fu', score.fu],
                    ] as const) {
                        const value = document.createElement('span');
                        value.textContent = i18n.text(key, { count });
                        statsRow.appendChild(value);
                    }
                    body.appendChild(statsRow);
                }

                content.appendChild(body);

                // 6. Points
                const scoreDisplay = document.createElement('div');
                scoreDisplay.className = 're-score-display';
                scoreDisplay.textContent = i18n.text('{count} Points', { count: score.points });
                content.appendChild(scoreDisplay);
            }

            // Pagination Controls
            if (totalPages > 1) {
                const controls = document.createElement('div');
                Object.assign(controls.style, {
                    display: 'flex',
                    justifyContent: 'center',
                    gap: '20px',
                    marginTop: '15px',
                    alignItems: 'center',
                });

                const prevBtn = ResultRenderer.createNavBtn(ICON_ARROW_LEFT, idx > 0);
                prevBtn.onclick = () => {
                    if (pageIndex > 0) {
                        pageIndex--;
                        renderPage(pageIndex);
                    }
                };
                controls.appendChild(prevBtn);

                const indicator = document.createElement('span');
                indicator.textContent = `${idx + 1} / ${totalPages}`;
                controls.appendChild(indicator);

                const nextBtn = ResultRenderer.createNavBtn(ICON_ARROW_RIGHT, idx < totalPages - 1);
                nextBtn.onclick = () => {
                    if (pageIndex < totalPages - 1) {
                        pageIndex++;
                        renderPage(pageIndex);
                    }
                };
                controls.appendChild(nextBtn);

                content.appendChild(controls);
            }
        };

        // Initial Render
        renderPage(0);

        return modal;
    }

    static renderRyukyokuModal(
        details: { reason?: string; deltas?: number[]; scores?: number[] },
        state: BoardState,
        i18n = new I18n(),
    ): HTMLElement {
        const modal = document.createElement('div');
        modal.className = 're-modal-overlay';

        const content = document.createElement('div');
        content.className = 're-modal-content re-result-dialog';
        content.onclick = (e) => e.stopPropagation();

        modal.appendChild(content);

        // Header
        const header = document.createElement('div');
        header.className = 're-modal-title';
        header.textContent = i18n.text('Ryukyoku');
        content.appendChild(header);

        // Body
        const body = document.createElement('div');
        body.className = 're-result-body';
        Object.assign(body.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: '20px',
        });

        // Reason
        if (details.reason) {
            const reasonDiv = document.createElement('div');
            reasonDiv.style.marginBottom = '20px';
            reasonDiv.style.fontSize = '1.5em';
            reasonDiv.style.fontWeight = 'bold';

            reasonDiv.textContent = i18n.text('Reason: {reason}', { reason: i18n.reason(details.reason) });
            if (details.reason === 'Error') reasonDiv.style.color = '#ff4757';
            body.appendChild(reasonDiv);
        }

        // Score info (Deltas)
        if (details.deltas && details.deltas.length >= 3) {
            const scoreContainer = document.createElement('div');
            Object.assign(scoreContainer.style, {
                display: 'flex',
                gap: '20px',
                marginTop: '20px',
                width: '100%',
                justifyContent: 'center',
            });

            details.deltas.forEach((delta: number, i: number) => {
                const pDiv = document.createElement('div');
                Object.assign(pDiv.style, {
                    background: 'rgba(255,255,255,0.1)',
                    padding: '15px',
                    borderRadius: '8px',
                    textAlign: 'center',
                    minWidth: '80px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '5px',
                });

                const name = document.createElement('div');
                name.textContent = state.playerNames?.[i] || `Player${i}`;
                name.style.fontSize = '0.9em';
                name.style.color = '#ccc';

                const val = document.createElement('div');
                val.textContent = delta > 0 ? `+${delta}` : `${delta}`;
                val.style.fontWeight = 'bold';
                val.style.fontSize = '1.3em';
                if (delta > 0) val.style.color = '#7bed9f';
                else if (delta < 0) val.style.color = '#ff6b6b';
                else val.style.color = '#fff';

                pDiv.appendChild(name);
                pDiv.appendChild(val);
                scoreContainer.appendChild(pDiv);
            });
            body.appendChild(scoreContainer);
        }

        content.appendChild(body);
        return modal;
    }

    private static createNavBtn(svgContent: string, enabled: boolean): HTMLElement {
        const btn = document.createElement('div');
        Object.assign(btn.style, {
            width: '32px',
            height: '32px',
            cursor: enabled ? 'pointer' : 'default',
            opacity: enabled ? '1' : '0.3',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
        });

        if (!enabled) return btn;

        btn.className = 'icon-btn'; // Use existing style logic for hover effects

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '1.5');
        svg.style.width = '24px';
        svg.style.height = '24px';
        svg.innerHTML = svgContent;
        btn.appendChild(svg);

        return btn;
    }

    private static getLimitInfo(
        han: number,
        fu: number,
        isYakuman: boolean,
        isKazoe: boolean,
    ): { text: string; css: string } | null {
        if (isYakuman) return { text: 'Yakuman', css: 'limit-yakuman' };
        if (isKazoe) return { text: 'Kazoe Yakuman', css: 'limit-yakuman' }; // Re-use yakuman style for Kazoe
        if (han >= 11) return { text: 'Sanbaiman', css: 'limit-sanbaiman' };
        if (han >= 8) return { text: 'Baiman', css: 'limit-baiman' };
        if (han >= 6) return { text: 'Haneman', css: 'limit-haneman' };

        // Mangan Check
        if (han >= 5) return { text: 'Mangan', css: 'limit-mangan' };
        // Kiriage Mangan: 4 han 30 fu is 7700 (not mangan), 4 han 40 fu is Mangan (8000)
        // 3 Han 70 fu is Mangan (8000)
        if (han === 4 && fu >= 40) return { text: 'Mangan', css: 'limit-mangan' };
        if (han === 3 && fu >= 70) return { text: 'Mangan', css: 'limit-mangan' };

        return null;
    }
}
