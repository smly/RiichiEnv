import { BoardState, PlayerState } from './types';
import { TILES } from './tiles';
import { VIEWER_CSS } from './styles';
import { TileRenderer } from './renderers/tile_renderer';
import { RiverRenderer } from './renderers/river_renderer';
import { HandRenderer } from './renderers/hand_renderer';
import { InfoRenderer } from './renderers/info_renderer';
import { CenterRenderer } from './renderers/center_renderer';
import { ResultRenderer } from './renderers/result_renderer';



export class Renderer {
    container: HTMLElement;
    private boardElement: HTMLElement | null = null;
    private styleElement: HTMLStyleElement | null = null;
    viewpoint: number = 0;

    constructor(container: HTMLElement) {
        this.container = container;

        let style = document.getElementById('riichienv-viewer-style') as HTMLStyleElement;
        if (!style) {
            style = document.createElement('style');
            style.id = 'riichienv-viewer-style';
            style.textContent = `
                ${VIEWER_CSS}
            `;
            document.head.appendChild(style);
        }
        this.styleElement = style;
    }

    onViewpointChange: ((pIdx: number) => void) | null = null;
    onCenterClick: (() => void) | null = null;
    private readonly BASE_SIZE = 800;

    resize(width: number) {
        if (!this.boardElement) return;
        const scale = width / this.BASE_SIZE;
        this.boardElement.style.transform = `translate(-50%, -50%) scale(${scale})`;
    }

    render(state: BoardState, debugPanel?: HTMLElement) {
        // Reuse board element to prevent flickering
        if (!this.boardElement) {
            this.boardElement = document.createElement('div');
            this.boardElement.className = 'mahjong-board';
            Object.assign(this.boardElement.style, {
                width: `${this.BASE_SIZE}px`,
                height: `${this.BASE_SIZE}px`,
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)', // Initial transform, will be overridden by resize
                transformOrigin: 'center center'
            });
            this.container.appendChild(this.boardElement);
        }
        const board = this.boardElement;

        console.log("[Renderer] render() called. State:", {
            round: state.round,
            players: state.players.length,
            doraMarkers: state.doraMarkers,
            eventIndex: state.eventIndex
        });

        board.innerHTML = '';

        // Clear any existing modals from container (since we append them to container now)
        const oldModals = this.container.querySelectorAll('.re-modal-overlay');
        oldModals.forEach(el => el.remove());

        // Center Info
        const center = CenterRenderer.renderCenter(state, this.onCenterClick, this.viewpoint);
        board.appendChild(center);

        const angles = [0, -90, 180, 90];

        state.players.forEach((p, i) => {
            const relIndex = (i - this.viewpoint + 4) % 4;

            const wrapper = document.createElement('div');
            Object.assign(wrapper.style, {
                position: 'absolute',
                top: '50%',
                left: '50%',
                width: '0',
                height: '0',
                display: 'flex',
                justifyContent: 'center',
                transform: `rotate(${angles[relIndex]}deg)`
            });

            const pDiv = document.createElement('div');
            Object.assign(pDiv.style, {
                width: '600px',
                height: '250px', // Adjusted height to prevent cutoff
                display: 'block',
                transform: 'translateY(120px)', // Lifted up
                transition: 'background-color 0.3s',
                position: 'relative'
            });

            // Active player highlighting - Removed old highlight
            // New logic adds bar to infoBox below
            pDiv.style.padding = '10px';

            // Call Overlay Logic
            let showOverlay = false;
            let label = '';

            // Standard Checks (Actor-based)
            if (state.lastEvent && state.lastEvent.actor === i) {
                const type = state.lastEvent.type;
                if (['chi', 'pon', 'kan', 'ankan', 'daiminkan', 'kakan', 'reach'].includes(type)) { // Added reach
                    label = type.charAt(0).toUpperCase() + type.slice(1);
                    if (type === 'daiminkan') label = 'Kan';
                    if (type === 'reach') label = 'Reach'; // Ensure capitalization
                    showOverlay = true;
                } else if (type === 'hora') {
                    label = (state.lastEvent.target === state.lastEvent.actor) ? 'Tsumo' : 'Ron';
                    showOverlay = true;
                }
            }

            // Ryukyoku Check (For Viewpoint Player Only)
            if (state.lastEvent && state.lastEvent.type === 'ryukyoku' && i === this.viewpoint) {
                label = 'Ryukyoku';
                showOverlay = true;
            }

            if (showOverlay && label) {
                const overlay = document.createElement('div');
                overlay.className = 'call-overlay';
                overlay.textContent = label;
                pDiv.appendChild(overlay);
            }

            // Riichi Stick rendering moved to CenterRenderer

            // Wait Indicator Logic (Persistent)
            if (p.waits && p.waits.length > 0) {
                const wDiv = document.createElement('div');
                Object.assign(wDiv.style, {
                    position: 'absolute',
                    top: '130px', left: '50%', transform: 'translateX(140px)', // Aligned with InfoBox
                    background: 'rgba(0,0,0,0.8)', color: '#fff', padding: '5px 10px',
                    borderRadius: '4px', fontSize: '14px', zIndex: '50',
                    display: 'flex', gap: '4px', alignItems: 'center', pointerEvents: 'none'
                });
                wDiv.innerHTML = '<span style="margin-right:4px;">Wait:</span>';
                p.waits.forEach((w: string) => {
                    wDiv.innerHTML += `<div style="width:24px; height:34px;">${TileRenderer.getTileHtml(w)}</div>`;
                });
                pDiv.appendChild(wDiv);
            }

            // --- River + Info Container ---
            // ABSOLUTE POSITIONED RIVER
            // We use riverRow as the container for the river tiles, absolutely positioned.
            const riverRow = document.createElement('div');
            Object.assign(riverRow.style, {
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'center',
                position: 'absolute',
                top: '10px',
                left: '50%',
                transform: 'translateX(-50%)',
                zIndex: '5'
            });

            // Collect active waits (global) to highlight in ALL hands AND rivers
            const activeWaits = new Set<string>();
            const normalize = (t: string) => t.replace('0', '5').replace('r', '');

            state.players.forEach(pl => {
                if (pl.waits && pl.waits.length > 0) {
                    pl.waits.forEach(w => activeWaits.add(normalize(w)));
                }
            });

            // Use independent RiverRenderer
            let riverDAnim = undefined;
            if (state.dahaiAnim && state.currentActor === i) {
                riverDAnim = state.dahaiAnim;
            }
            const riverDiv = RiverRenderer.renderRiver(p.discards, activeWaits, riverDAnim);
            riverRow.appendChild(riverDiv);
            pDiv.appendChild(riverRow);

            // Info Box (New Overlay) - Anchored to pDiv 
            const infoBox = InfoRenderer.renderPlayerInfo(p, i, this.viewpoint, state.currentActor, this.onViewpointChange || (() => { }));
            pDiv.appendChild(infoBox);

            // Render Hand
            // Check if this player has just drawn a tile (Tsumo position)
            // Conditions: 
            // 1. It is this player's turn (active actor)
            // 2. The last event was 'tsumo' (they drew) OR 'start_kyoku' (Oya's first 14th tile)
            // Note: After pon/chi/kan, it is their turn but they did NOT draw from wall (they took from river).
            // So we do NOT separate.
            let hasDraw = false;
            let shouldAnimate = false;

            if (state.currentActor === i && state.lastEvent) {
                const type = state.lastEvent.type;
                if (type === 'tsumo' && state.lastEvent.actor === i) {
                    hasDraw = true;
                    shouldAnimate = true;
                } else if (type === 'start_kyoku' && state.lastEvent.oya === i) {
                    hasDraw = true;
                    shouldAnimate = false;
                } else if (type === 'reach' && state.lastEvent.actor === i) {
                    // During Reach declaration step, keep tile separated but no fly-in animation
                    hasDraw = true;
                    shouldAnimate = false;
                }
            }

            const playerState = state.players[i];

            // Only pass dahaiAnim if it's THIS player's action
            let dAnim = undefined;
            if (state.dahaiAnim && state.currentActor === i) {
                dAnim = state.dahaiAnim;
            }

            const hand = HandRenderer.renderHand(playerState.hand, playerState.melds, i, activeWaits, hasDraw, dAnim, shouldAnimate);

            // HIDE HANDS LOGIC REMOVED

            // Assuming positionElement and handsTarget are part of the class or a refactor
            // For now, append directly to pDiv as per original structure,
            // but keep the new hand variable and hide logic.
            pDiv.appendChild(hand);

            wrapper.appendChild(pDiv);
            board.appendChild(wrapper);
        });

        // End Kyoku Modal
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta && state.lastEvent.meta.results) {
            const results = state.lastEvent.meta.results;
            // Use new ResultRenderer
            // We assume YAKU_MAP is handled internally or passed if needed, but ResultRenderer imports it.
            const modal = ResultRenderer.renderModal(results, state);

            // Allow close by clicking overlay (handled by caller? No, ResultRenderer creates overlay).
            // We need to support 'removing' it from board.
            // The renderer.ts recreates content every time, so simply appending it works.
            // But if user wants to close it?
            // If the event is still 'end_kyoku', re-render will bring it back.
            // The modal should perhaps NOT be rendered if the user has dismissed it for this specific event instance?
            // But usually 'end_kyoku' is a state. You move next to start next kyoku.
            // So closing it might just hide it. 
            // Let's add simple click-to-hide logic to the overlay in ResultRenderer or here.
            // ResultRenderer didn't implement creating a close handler that removes it from DOM.
            // But since board.innerHTML is cleared on render, 'close' just means removing from DOM *until next render*.
            // Next render happens on next/prev step.
            // So clicking background to remove() is fine.
            modal.onclick = (e) => {
                if (e.target === modal) {
                    modal.remove();
                }
            };

            // Append to container to cover the whole board area, not just the inner mahjong-board
            this.container.appendChild(modal);
        }

        if (debugPanel) {
            const lastEvStr = state.lastEvent ? JSON.stringify(state.lastEvent, null, 2) : 'null';
            const text = `Event: ${state.eventIndex} / ${state.totalEvents}\nLast Event:\n${lastEvStr}`;
            if (debugPanel.textContent !== text) {
                debugPanel.textContent = text;
            }
        }
    }
}
