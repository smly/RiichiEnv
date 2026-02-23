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

    // Persistent DOM slots to avoid full destruction/rebuild
    private centerSlot: HTMLElement | null = null;
    private playerWrappers: HTMLElement[] = [];
    private playerDivs: HTMLElement[] = [];
    private hadModal: boolean = false;

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

    /** Ensure persistent board structure exists. Creates slots on first call. */
    private ensureBoardStructure(pc: number): void {
        if (this.boardElement && this.centerSlot && this.playerWrappers.length === pc) return;

        if (!this.boardElement) {
            this.boardElement = document.createElement('div');
            this.boardElement.className = 'mahjong-board';
            Object.assign(this.boardElement.style, {
                width: `${this.layout.boardSize}px`,
                height: `${this.layout.boardSize}px`,
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                transformOrigin: 'center center'
            });
            this.container.appendChild(this.boardElement);
        }

        // Create center slot
        this.centerSlot = document.createElement('div');
        this.centerSlot.className = 'center-slot';
        this.boardElement.appendChild(this.centerSlot);

        // Create player wrapper slots
        this.playerWrappers = [];
        this.playerDivs = [];
        for (let i = 0; i < pc; i++) {
            const wrapper = document.createElement('div');
            Object.assign(wrapper.style, {
                position: 'absolute',
                top: '50%',
                left: '50%',
                width: '0',
                height: '0',
                display: 'flex',
                justifyContent: 'center',
            });

            const pDiv = document.createElement('div');
            Object.assign(pDiv.style, {
                width: '600px',
                height: '250px',
                display: 'block',
                transform: 'translateY(120px)',
                position: 'relative',
                padding: '10px',
                contain: 'layout style',
            });

            wrapper.appendChild(pDiv);
            this.boardElement.appendChild(wrapper);
            this.playerWrappers.push(wrapper);
            this.playerDivs.push(pDiv);
        }
    }

    render(state: BoardState, debugPanel?: HTMLElement) {
        const pc = state.playerCount;

        // Ensure persistent DOM structure exists
        this.ensureBoardStructure(pc);
        const board = this.boardElement!;

        // Suppress style recalculation during DOM batch updates.
        // content-visibility: hidden tells the browser to skip layout/paint/style
        // for all descendants while we mutate the DOM. On restore, only a single
        // style recalc pass is performed instead of incremental per-mutation recalcs.
        board.style.contentVisibility = 'hidden';

        // Clear modals only if we had one previously
        if (this.hadModal) {
            const oldModals = this.container.querySelectorAll('.re-modal-overlay');
            oldModals.forEach(el => el.remove());
            this.hadModal = false;
        }

        // Update center slot content via DocumentFragment (avoids incremental reflows)
        const center = CenterRenderer.renderCenter(state, this.onCenterClick, this.viewpoint);
        const centerFrag = document.createDocumentFragment();
        centerFrag.appendChild(center);
        this.centerSlot!.replaceChildren(centerFrag);

        const angles = this.layout.playerAngles;

        // Collect active waits once (shared across all players)
        const activeWaits = new Set<string>();
        const normalize = (t: string) => t.replace('0', '5').replace('r', '');
        state.players.forEach(pl => {
            if (pl.waits && pl.waits.length > 0) {
                pl.waits.forEach(w => activeWaits.add(normalize(w)));
            }
        });

        state.players.forEach((p, i) => {
            const relIndex = (i - this.viewpoint + pc) % pc;

            // Update wrapper transform (may change with viewpoint)
            this.playerWrappers[i].style.transform = `rotate(${angles[relIndex]}deg)`;

            // Build new content for the player div
            const children: HTMLElement[] = [];

            // Call Overlay Logic
            let showOverlay = false;
            let label = '';

            // Standard Checks (Actor-based)
            if (state.lastEvent && state.lastEvent.actor === i) {
                const type = state.lastEvent.type;
                if (['chi', 'pon', 'kan', 'ankan', 'daiminkan', 'kakan', 'reach'].includes(type)) {
                    label = type.charAt(0).toUpperCase() + type.slice(1);
                    if (type === 'daiminkan') label = 'Kan';
                    if (type === 'reach') label = 'Reach';
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
                children.push(overlay);
            }

            // Wait Indicator Logic
            if (p.waits && p.waits.length > 0) {
                const wDiv = document.createElement('div');
                Object.assign(wDiv.style, {
                    position: 'absolute',
                    top: '130px', left: '50%', transform: 'translateX(140px)',
                    background: 'rgba(0,0,0,0.8)', color: '#fff', padding: '5px 10px',
                    borderRadius: '4px', fontSize: '14px', zIndex: '50',
                    display: 'flex', gap: '4px', alignItems: 'center', pointerEvents: 'none'
                });
                const waitLabel = document.createElement('span');
                waitLabel.style.marginRight = '4px';
                waitLabel.textContent = 'Wait:';
                wDiv.appendChild(waitLabel);
                p.waits.forEach((w: string) => {
                    const d = document.createElement('div');
                    d.style.width = '24px';
                    d.style.height = '34px';
                    d.appendChild(TileRenderer.getTileElement(w));
                    wDiv.appendChild(d);
                });
                children.push(wDiv);
            }

            // River
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

            let riverDAnim = undefined;
            if (state.dahaiAnim && state.currentActor === i) {
                riverDAnim = state.dahaiAnim;
            }
            const riverDiv = RiverRenderer.renderRiver(p.discards, activeWaits, riverDAnim);
            riverRow.appendChild(riverDiv);
            children.push(riverRow);

            // Info Box
            const infoBox = InfoRenderer.renderPlayerInfo(p, i, this.viewpoint, state.currentActor, this.onViewpointChange || (() => { }));
            children.push(infoBox);

            // Hand
            let hasDraw = false;
            let shouldAnimate = false;

            if (state.currentActor === i && state.lastEvent) {
                const type = state.lastEvent.type;
                if (type === 'tsumo' && state.lastEvent.actor === i) {
                    hasDraw = true;
                    shouldAnimate = true;
                } else if (type === 'reach' && state.lastEvent.actor === i) {
                    hasDraw = true;
                    shouldAnimate = false;
                }
            }

            const playerState = state.players[i];
            let dAnim = undefined;
            if (state.dahaiAnim && state.currentActor === i) {
                dAnim = state.dahaiAnim;
            }

            const hand = HandRenderer.renderHand(playerState.hand, playerState.melds, i, activeWaits, hasDraw, dAnim, shouldAnimate, pc);
            children.push(hand);

            // Replace player div content via DocumentFragment (single reflow)
            const frag = document.createDocumentFragment();
            for (const child of children) frag.appendChild(child);
            this.playerDivs[i].replaceChildren(frag);
        });

        // Restore rendering — triggers a single style recalc for the entire board
        board.style.contentVisibility = '';

        // End Kyoku Modal
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta) {
            let modal: HTMLElement | null = null;

            if (state.lastEvent.meta.ryukyoku) {
                modal = ResultRenderer.renderRyukyokuModal(state.lastEvent.meta.ryukyoku, state);
            } else if (state.lastEvent.meta.results) {
                const results = state.lastEvent.meta.results;
                modal = ResultRenderer.renderModal(results, state);
            }

            if (modal) {
                modal.onclick = (e) => {
                    if (e.target === modal) {
                        modal!.remove();
                    }
                };
                this.container.appendChild(modal);
                this.hadModal = true;
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
