import { BoardState, PlayerState } from '../types';
import { TILES } from '../tiles';
import { VIEWER_CSS } from '../styles';
import { LayoutConfig, createLayoutConfig4P } from '../config';
import { IRenderer } from './renderer_interface';
import { TileRenderer } from './tile_renderer';
import { RiverRenderer } from './river_renderer';
import { HandRenderer } from './hand_renderer';
import { InfoRenderer } from './info_renderer';
import { CenterRenderer } from './center_renderer';
import { ResultRenderer } from './result_renderer';



export class Renderer2D implements IRenderer {
    container: HTMLElement;
    private boardElement: HTMLElement | null = null;
    private styleElement: HTMLStyleElement | null = null;
    private layout: LayoutConfig;
    viewpoint: number = 0;

    constructor(container: HTMLElement, layout?: LayoutConfig) {
        this.container = container;
        this.layout = layout ?? createLayoutConfig4P();

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

    resize(width: number) {
        if (!this.boardElement) return;
        const scale = width / this.layout.boardSize;
        this.boardElement.style.transform = `translate(-50%, -50%) scale(${scale})`;
    }

    render(state: BoardState, debugPanel?: HTMLElement) {
        const pc = state.playerCount;

        // Reuse board element to prevent flickering
        if (!this.boardElement) {
            this.boardElement = document.createElement('div');
            this.boardElement.className = 'mahjong-board';
            Object.assign(this.boardElement.style, {
                width: `${this.layout.boardSize}px`,
                height: `${this.layout.boardSize}px`,
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)', // Initial transform, will be overridden by resize
                transformOrigin: 'center center'
            });
            this.container.appendChild(this.boardElement);
        }
        const board = this.boardElement;

        console.log("[Renderer2D] render() called. State:", {
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

        const angles = this.layout.playerAngles;

        state.players.forEach((p, i) => {
            const relIndex = (i - this.viewpoint + pc) % pc;

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

            // Active player highlighting adds bar to infoBox below
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
            // Relative positioned river container
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
            let hasDraw = false;
            let shouldAnimate = false;

            if (state.currentActor === i && state.lastEvent) {
                const type = state.lastEvent.type;
                if (type === 'tsumo' && state.lastEvent.actor === i) {
                    hasDraw = true;
                    shouldAnimate = true;
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

            const hand = HandRenderer.renderHand(playerState.hand, playerState.melds, i, activeWaits, hasDraw, dAnim, shouldAnimate, pc);

            pDiv.appendChild(hand);

            wrapper.appendChild(pDiv);
            board.appendChild(wrapper);
        });

        // End Kyoku Modal
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta) {
            let modal: HTMLElement | null = null;

            if (state.lastEvent.meta.ryukyoku) {
                // Ryukyoku / Error Result
                modal = ResultRenderer.renderRyukyokuModal(state.lastEvent.meta.ryukyoku, state);
            } else if (state.lastEvent.meta.results) {
                // Hora Result
                const results = state.lastEvent.meta.results;
                modal = ResultRenderer.renderModal(results, state);
            }

            if (modal) {
                // Click background to close
                modal.onclick = (e) => {
                    if (e.target === modal) {
                        modal!.remove();
                    }
                };
                this.container.appendChild(modal);
            }
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
