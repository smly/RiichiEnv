import { BoardState, PlayerState, Tile } from '../types';
import { VIEWER_CSS } from '../styles';
import { VIEWER_3D_CSS } from '../styles_3d';
import { LayoutConfig3D, createLayout3DConfig4P } from '../config';
import { IRenderer } from './renderer_interface';
import { TileRenderer } from './tile_renderer';
import { ResultRenderer } from './result_renderer';
import { COLORS, CALL_TYPES } from '../constants';
import { CHAR_SPRITE_BASE64, CHAR_MAP } from '../char_assets';
import { AVATAR_PLACEHOLDER } from '../icons';

export class Renderer3D implements IRenderer {
    container: HTMLElement;
    viewpoint: number = 0;
    onViewpointChange: ((pIdx: number) => void) | null = null;
    onCenterClick: (() => void) | null = null;

    private sceneEl: HTMLElement | null = null;
    private layout: LayoutConfig3D;
    private _hadModal: boolean = false;

    constructor(container: HTMLElement, layout?: LayoutConfig3D) {
        this.container = container;
        this.layout = layout ?? createLayout3DConfig4P();
        this.injectStyles();
    }

    private injectStyles(): void {
        // Inject shared 2D styles (tile-layer, modals, buttons, etc.)
        if (!document.getElementById('riichienv-viewer-style')) {
            const s = document.createElement('style');
            s.id = 'riichienv-viewer-style';
            s.textContent = VIEWER_CSS;
            document.head.appendChild(s);
        }
        // Inject 3D-specific styles
        if (!document.getElementById('riichienv-viewer-3d-style')) {
            const s = document.createElement('style');
            s.id = 'riichienv-viewer-3d-style';
            s.textContent = VIEWER_3D_CSS;
            document.head.appendChild(s);
        }
    }

    resize(_width: number): void {
        // Handled by Viewer3D's ResizeObserver
    }

    /**
     * Set 3D tile content on a table-surface element.
     * Creates a CSS 3D box with top face and specified side faces.
     * Returns the top-face element for appending overlays (e.g. highlights).
     *
     * @param faces Which side faces to render (default: front + right).
     *   - relIndex 0 (self):     { front: true }
     *   - relIndex 1 (right):    { back: true, left: true }
     *   - relIndex 2 (opposite): { back: true }
     *   - relIndex 3 (left):     { right: true, back: true }
     */
    private setTile3D(
        el: HTMLElement, tileId: string, depth: number,
        faces: { front?: boolean; back?: boolean; left?: boolean; right?: boolean } = { front: true, right: true },
    ): HTMLElement {
        el.style.transformStyle = 'preserve-3d';
        // Side faces are 1px taller/wider than depth to overlap with the top face,
        // preventing sub-pixel rendering gaps at the seams.
        const d1 = depth + 1;

        // Build via DOM API + cloneNode (avoids innerHTML SVG parsing per tile)
        const topFace = document.createElement('div');
        topFace.className = 'tile-3d-top';
        topFace.style.transform = `translateZ(${depth}px)`;
        topFace.appendChild(TileRenderer.getTileElement(tileId));
        el.appendChild(topFace);

        if (faces.front) {
            const f = document.createElement('div');
            f.className = 'tile-3d-front';
            f.style.height = `${d1}px`;
            el.appendChild(f);
        }
        if (faces.back) {
            const b = document.createElement('div');
            b.className = 'tile-3d-back';
            b.style.height = `${d1}px`;
            el.appendChild(b);
        }
        if (faces.right) {
            const r = document.createElement('div');
            r.className = 'tile-3d-right';
            r.style.width = `${d1}px`;
            el.appendChild(r);
        }
        if (faces.left) {
            const l = document.createElement('div');
            l.className = 'tile-3d-left';
            l.style.width = `${d1}px`;
            el.appendChild(l);
        }
        return topFace;
    }

    render(state: BoardState, debugPanel?: HTMLElement): void {
        const pc = state.playerCount;

        // 1. Create/reuse scene container
        if (!this.sceneEl) {
            this.sceneEl = document.createElement('div');
            this.sceneEl.className = 'scene-3d';
            this.container.appendChild(this.sceneEl);
        }

        // Build entire scene off-DOM into a DocumentFragment to avoid
        // triggering style recalculation during construction.
        // Only a single replaceChildren() at the end touches the live DOM.
        const sceneFrag = document.createDocumentFragment();

        // 2. Clear old modals
        if (this._hadModal) {
            const oldModals = this.container.querySelectorAll('.re-modal-overlay');
            oldModals.forEach(el => el.remove());
            this._hadModal = false;
        }

        // 3. Build Layer 1: 3D Table Scene
        const perspectiveEl = document.createElement('div');
        perspectiveEl.className = 'table-perspective';
        Object.assign(perspectiveEl.style, {
            perspective: `${this.layout.perspective}px`,
            perspectiveOrigin: '50% 40%',
        });

        const tableSurface = document.createElement('div');
        tableSurface.className = 'table-surface';
        // Position: flatter tilt → move table up to balance with hand layer
        const tableTop = this.layout.tiltAngle <= 40 ? '40%' : '42%';
        const frameWidth = 76;
        const surfaceSize = this.layout.tableSize + frameWidth * 2;
        Object.assign(tableSurface.style, {
            width: `${surfaceSize}px`,
            height: `${surfaceSize}px`,
            top: tableTop,
            transform: `translate(-50%, -50%) rotateX(${this.layout.tiltAngle}deg)`,
        });

        // Table inner border (contains all game content)
        const tableInner = document.createElement('div');
        tableInner.className = 'table-inner';
        tableSurface.appendChild(tableInner);

        // Center info
        const center = this.renderCenter3D(state);
        tableInner.appendChild(center);

        // Riichi sticks on table
        this.renderRiichiSticks(tableInner, state, pc);

        // Floating score labels on table (above riichi sticks)
        this.renderFloatingScores(tableInner, state, pc);

        // Collect active waits for highlighting
        const activeWaits = new Set<string>();
        const normalize = (t: string) => t.replace('0', '5').replace('r', '');
        state.players.forEach(pl => {
            if (pl.waits && pl.waits.length > 0) {
                pl.waits.forEach(w => activeWaits.add(normalize(w)));
            }
        });

        // Per-player table elements
        state.players.forEach((p, i) => {
            const relIndex = (i - this.viewpoint + pc) % pc;

            // River (discards)
            const river = this.renderRiver3D(p.discards, relIndex, state, i, activeWaits);
            tableInner.appendChild(river);

            // Opponent hand + melds on table (skip viewpoint player)
            if (relIndex !== 0) {
                const oppHand = this.renderOpponentHandArea(p, i, relIndex, pc, activeWaits);
                tableInner.appendChild(oppHand);
            }
        });

        perspectiveEl.appendChild(tableSurface);
        sceneFrag.appendChild(perspectiveEl);

        // 4. Build Layer 2: Hand Layer (flat, bottom)
        const handLayer = document.createElement('div');
        handLayer.className = 'hand-layer-3d';
        Object.assign(handLayer.style, {
            height: `${this.layout.handLayerHeight}px`,
        });

        const vpPlayer = state.players[this.viewpoint];
        if (vpPlayer) {
            const handEl = this.renderOwnHand(vpPlayer, this.viewpoint, state, pc, activeWaits);
            handLayer.appendChild(handEl);
        }
        sceneFrag.appendChild(handLayer);

        // 5. Build Layer 3: UI Overlay
        const uiOverlay = document.createElement('div');
        uiOverlay.className = 'ui-overlay-3d';

        // Score panels (at viewport edges)
        state.players.forEach((p, i) => {
            const relIndex = (i - this.viewpoint + pc) % pc;
            const panel = this.renderPlayerPanel(p, i, relIndex, state);
            uiOverlay.appendChild(panel);
        });

        // Center click zone (2D overlay for reliable click on 3D center panel)
        if (this.onCenterClick) {
            const centerClick = document.createElement('div');
            centerClick.className = 'center-click-zone';
            centerClick.onclick = (e) => {
                e.stopPropagation();
                if (this.onCenterClick) this.onCenterClick();
            };
            // Sync hover state to 3D center panel
            centerClick.addEventListener('mouseenter', () => {
                center.classList.add('hover');
            });
            centerClick.addEventListener('mouseleave', () => {
                center.classList.remove('hover');
            });
            uiOverlay.appendChild(centerClick);
        }

        // Call overlay
        this.renderCallOverlay(uiOverlay, state);

        // Wait indicators for all players (UI overlay)
        state.players.forEach((p, i) => {
            if (p.waits && p.waits.length > 0) {
                const relIndex = (i - this.viewpoint + pc) % pc;
                const waitEl = this.renderWaitIndicator(p.waits, relIndex);
                uiOverlay.appendChild(waitEl);
            }
        });

        sceneFrag.appendChild(uiOverlay);

        // Single DOM swap — replaces all scene content in one operation
        this.sceneEl.replaceChildren(sceneFrag);

        // 6. Result modals
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta) {
            let modal: HTMLElement | null = null;
            if (state.lastEvent.meta.ryukyoku) {
                modal = ResultRenderer.renderRyukyokuModal(state.lastEvent.meta.ryukyoku, state);
            } else if (state.lastEvent.meta.results) {
                modal = ResultRenderer.renderModal(state.lastEvent.meta.results, state);
            }
            if (modal) {
                modal.onclick = (e) => {
                    if (e.target === modal) modal!.remove();
                };
                this.container.appendChild(modal);
                this._hadModal = true;
            }
        }

        // 7. Debug panel
        if (debugPanel) {
            const lastEvStr = state.lastEvent ? JSON.stringify(state.lastEvent, null, 2) : 'null';
            const text = `Event: ${state.eventIndex} / ${state.totalEvents}\nLast Event:\n${lastEvStr}`;
            if (debugPanel.textContent !== text) {
                debugPanel.textContent = text;
            }
        }
    }

    // =========================================================================
    // Center Info (on table)
    // =========================================================================
    private renderCenter3D(state: BoardState): HTMLElement {
        const center = document.createElement('div');
        center.className = 'center-info-3d';

        const pc = state.playerCount || 4;

        // Wind labels at corners
        const windMap = ['東_red', '南', '西', '北'].slice(0, pc);
        state.players.forEach((p: PlayerState, i: number) => {
            const relPos = (i - this.viewpoint + pc) % pc;
            const windIdx = p.wind;
            if (windIdx < 0 || windIdx >= pc) return;

            const key = windMap[windIdx];
            const asset = CHAR_MAP[key];
            if (!asset) return;

            const icon = document.createElement('div');
            Object.assign(icon.style, {
                position: 'absolute',
                width: `${asset.w}px`,
                height: `${asset.h}px`,
                pointerEvents: 'none',
                backgroundImage: `url(${CHAR_SPRITE_BASE64})`,
                backgroundPosition: `-${asset.x}px -${asset.y}px`,
                backgroundRepeat: 'no-repeat',
                transformOrigin: 'center center',
            });

            const targetSize = 34;
            const maxDim = Math.max(asset.w, asset.h);
            const scale = targetSize / maxDim;

            let rotation = '0deg';
            if (relPos === 1) rotation = '-90deg';
            else if (relPos === 2) rotation = '180deg';
            else if (relPos === 3) rotation = '90deg';

            icon.style.transform = `rotate(${rotation}) scale(${scale})`;

            if (relPos === 0) { icon.style.bottom = '6px'; icon.style.left = '6px'; }
            else if (relPos === 1) { icon.style.right = '6px'; icon.style.bottom = '6px'; }
            else if (relPos === 2) { icon.style.top = '6px'; icon.style.right = '6px'; }
            else if (relPos === 3) { icon.style.left = '6px'; icon.style.top = '6px'; }

            center.appendChild(icon);
        });

        // Content container
        const contentDiv = document.createElement('div');
        Object.assign(contentDiv.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '2px',
        });

        // Row 1: Round info text
        const roundWindNames = ['東', '南', '西', '北'];
        const rWindIdx = Math.floor(state.round / pc);
        const rNumIdx = state.round % pc;
        const roundText = `${roundWindNames[rWindIdx] || '東'}${rNumIdx + 1}局`;

        const row1 = document.createElement('div');
        Object.assign(row1.style, {
            fontSize: '26px', fontWeight: 'bold', color: 'white',
            fontFamily: 'sans-serif', marginBottom: '2px',
        });
        row1.textContent = roundText;
        contentDiv.appendChild(row1);

        // Row 2: Honba / Kyotaku
        const row2 = document.createElement('div');
        row2.textContent = `Depo: ${state.kyotaku}, Honba: ${state.honba}`;
        Object.assign(row2.style, {
            fontSize: '16px', fontWeight: 'bold', color: 'white',
            fontFamily: 'sans-serif', marginBottom: '6px',
        });
        contentDiv.appendChild(row2);

        // Row 3: Dora markers
        const row3 = document.createElement('div');
        Object.assign(row3.style, { display: 'flex', gap: '2px' });

        const doraTiles = [...state.doraMarkers];
        while (doraTiles.length < 5) doraTiles.push('back');

        doraTiles.forEach(t => {
            const d = document.createElement('div');
            d.className = 'dora-tile-3d';
            this.setTile3D(d, t, this.layout.tileSizes.doraTile[0]);
            row3.appendChild(d);
        });
        contentDiv.appendChild(row3);

        center.appendChild(contentDiv);
        return center;
    }

    // =========================================================================
    // Riichi Sticks on table
    // =========================================================================
    private renderRiichiSticks(table: HTMLElement, state: BoardState, pc: number): void {
        const ts = this.layout.tableSize;
        state.players.forEach((p, i) => {
            if (!p.riichi) return;
            const relPos = (i - this.viewpoint + pc) % pc;

            const stick = document.createElement('div');
            stick.className = 'riichi-stick-3d';
            const dot = document.createElement('div');
            dot.className = 'dot';
            stick.appendChild(dot);

            // Position just inside the edges of the center info panel (250x220, centered)
            const centerHalf = 125; // half of 250px center panel width
            const inset = 5;
            const nearEdge = Math.round(ts / 2 + centerHalf - inset);
            const farEdge = Math.round(ts / 2 - centerHalf + inset);
            if (relPos === 0) {
                Object.assign(stick.style, {
                    left: '50%', top: `${nearEdge - 15}px`,
                    transform: 'translateX(-50%)',
                });
            } else if (relPos === 1) {
                Object.assign(stick.style, {
                    left: `${nearEdge}px`, top: '50%',
                    transform: 'translate(-50%, -50%) rotate(90deg)',
                });
            } else if (relPos === 2) {
                Object.assign(stick.style, {
                    left: '50%', top: `${farEdge}px`,
                    transform: 'translateX(-50%)',
                });
            } else if (relPos === 3) {
                Object.assign(stick.style, {
                    left: `${farEdge}px`, top: '50%',
                    transform: 'translate(-50%, -50%) rotate(90deg)',
                });
            }
            table.appendChild(stick);
        });
    }

    // =========================================================================
    // Floating score labels on table (positioned above riichi sticks)
    // =========================================================================
    private renderFloatingScores(table: HTMLElement, state: BoardState, pc: number): void {
        const ts = this.layout.tableSize;
        const centerHalf = 125; // half of 250px center panel width
        const offset = 30; // px above riichi stick (toward center)

        state.players.forEach((p, i) => {
            const relPos = (i - this.viewpoint + pc) % pc;

            const el = document.createElement('div');
            el.className = 'floating-score-3d';
            el.textContent = p.score.toString();

            // Position above riichi stick, rotated to face the player
            // "above" = closer to center from the riichi stick position
            const nearEdge = Math.round(ts / 2 + centerHalf - 5);
            const farEdge = Math.round(ts / 2 - centerHalf + 5);

            if (relPos === 0) {
                // Bottom player: stick at nearEdge-15, score above (closer to center)
                Object.assign(el.style, {
                    left: '50%', top: `${nearEdge - 15 - offset - 10}px`,
                    transform: 'translateX(-50%)',
                });
            } else if (relPos === 1) {
                // Right player (下家): rotated 180° from before, slightly outward
                Object.assign(el.style, {
                    left: `${nearEdge - offset + 15}px`, top: '50%',
                    transform: 'translate(-50%, -50%) rotate(-90deg)',
                });
            } else if (relPos === 2) {
                // Top player: stick at farEdge, score below (closer to center)
                Object.assign(el.style, {
                    left: '50%', top: `${farEdge + offset - 10}px`,
                    transform: 'translateX(-50%) rotate(180deg)',
                });
            } else if (relPos === 3) {
                // Left player (上家): rotated 180° from before, slightly outward
                Object.assign(el.style, {
                    left: `${farEdge + offset - 15}px`, top: '50%',
                    transform: 'translate(-50%, -50%) rotate(90deg)',
                });
            }

            // Click to change viewpoint
            el.onclick = (e) => {
                e.stopPropagation();
                if (this.onViewpointChange) this.onViewpointChange(i);
            };

            table.appendChild(el);
        });
    }

    // =========================================================================
    // River (discards) on table
    // =========================================================================
    private renderRiver3D(
        discards: Tile[], relIndex: number, state: BoardState,
        playerIdx: number, activeWaits: Set<string>
    ): HTMLElement {
        const [tw, th] = this.layout.tileSizes.riverTile;
        const gap = 1;
        // Fixed river area size: 6 columns × 3 rows (+ extra width for one possible riichi rotated tile)
        const riverW = 6 * tw + 5 * gap + 10; // +10 for a rotated riichi tile being wider
        const riverH = 3 * th + 2 * gap;

        const wrapper = document.createElement('div');
        wrapper.className = 'river-3d';
        // Fix the wrapper size so tile positions don't shift as discards are added
        Object.assign(wrapper.style, {
            width: `${riverW}px`,
            height: `${riverH}px`,
        });

        // Position on table (proportional to table size)
        const ts = this.layout.tableSize;
        // All rivers scaled up for better visibility
        const riverScale = 1.35;
        // Left/right rivers shifted toward center by one tile height
        const positions: { [key: number]: { left: string; top: string; transform: string } } = {
            0: { left: '50%', top: `${Math.round(ts * 0.70)}px`, transform: `translate(-50%, -50%) scale(${riverScale})` },
            1: { left: `${Math.round(ts * 0.73 - th)}px`, top: '50%', transform: `translate(-50%, -50%) rotate(-90deg) scale(${riverScale})` },
            2: { left: '50%', top: `${Math.round(ts * 0.30)}px`, transform: `translate(-50%, -50%) rotate(180deg) scale(${riverScale})` },
            3: { left: `${Math.round(ts * 0.27 + th)}px`, top: '50%', transform: `translate(-50%, -50%) rotate(90deg) scale(${riverScale})` },
        };
        const pos = positions[relIndex] || positions[0];
        Object.assign(wrapper.style, pos);

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        // Determine which side faces to render based on viewing angle
        // relIndex 0 (self):     top + front
        // relIndex 1 (right):    top + back + left
        // relIndex 2 (opposite): top + back
        // relIndex 3 (left):     top + right + back
        const riverFaces: { [key: number]: { front?: boolean; back?: boolean; left?: boolean; right?: boolean } } = {
            0: { front: true },
            1: { back: true, left: true },
            2: { back: true },
            3: { right: true, back: true },
        };
        const faces = riverFaces[relIndex] || { front: true, right: true };

        // Split into 3 rows of 6
        const rows: Tile[][] = [[], [], []];
        discards.forEach((d, idx) => {
            if (idx < 6) rows[0].push(d);
            else if (idx < 12) rows[1].push(d);
            else rows[2].push(d);
        });

        rows.forEach(rowTiles => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'river-row-3d';

            rowTiles.forEach(d => {
                const isRiichi = d.isRiichi;
                const cell = document.createElement('div');
                cell.className = isRiichi ? 'table-tile-rotated' : 'table-tile';
                if (d.isTsumogiri) cell.classList.add('table-tile-tsumogiri');

                const tileDepth = tw;
                const topFace = this.setTile3D(cell, d.tile, tileDepth, faces);

                // Tsumogiri: darken with black overlay on top face only
                if (d.isTsumogiri) {
                    const ov = document.createElement('div');
                    Object.assign(ov.style, {
                        position: 'absolute', top: '0', left: '0',
                        width: '100%', height: '100%',
                        backgroundColor: 'rgba(0, 0, 0, 0.35)',
                        pointerEvents: 'none', borderRadius: '3px',
                        zIndex: '5',
                    });
                    topFace.appendChild(ov);
                }

                // Highlight (append to top face so it's at the correct Z level)
                if (activeWaits.size > 0) {
                    const normT = normalize(d.tile);
                    if (activeWaits.has(normT)) {
                        const overlay = document.createElement('div');
                        Object.assign(overlay.style, {
                            position: 'absolute', top: '0', left: '0',
                            width: '100%', height: '100%',
                            backgroundColor: 'rgba(255, 0, 0, 0.4)',
                            zIndex: '10', pointerEvents: 'none', borderRadius: '3px',
                        });
                        topFace.appendChild(overlay);
                    }
                }

                rowDiv.appendChild(cell);
            });
            wrapper.appendChild(rowDiv);
        });

        return wrapper;
    }

    // =========================================================================
    // Opponent hand + melds on table edge (combined on one line)
    // =========================================================================
    private renderOpponentHandArea(
        player: PlayerState, playerIdx: number, relIndex: number, pc: number,
        activeWaits: Set<string>
    ): HTMLElement {
        const [tw, th] = this.layout.tileSizes.opponentTile;

        const wrapper = document.createElement('div');
        wrapper.className = 'opp-hand-3d';

        // Compute position: place hand between river outer edge and table edge
        const ts = this.layout.tableSize;
        const [rtw, rth] = this.layout.tileSizes.riverTile;
        const riverH = 3 * rth + 2; // 3 rows + 2 gaps
        const riverScale = 1.35;
        const halfRiverExtent = riverH * riverScale / 2;

        // Perpendicular offset (distance from table edge)
        const positions: { [key: number]: { left: string; top: string; transform: string } } = {
            1: {
                left: `${Math.round((ts * 0.745 - rth + halfRiverExtent + ts) / 2)}px`,
                top: '50%',
                transform: 'translate(-50%, -50%) rotate(-90deg)',
            },
            2: {
                left: '50%',
                top: `${Math.round((ts * 0.28 - halfRiverExtent) / 2)}px`,
                transform: 'translate(-50%, -50%) rotate(180deg)',
            },
            3: {
                left: `${Math.round((ts * 0.255 + rth - halfRiverExtent) / 2)}px`,
                top: '50%',
                transform: 'translate(-50%, -50%) rotate(90deg)',
            },
        };
        const pos = positions[relIndex];
        if (pos) Object.assign(wrapper.style, pos);

        // Determine visible side faces based on relIndex
        const oppFaces: { [key: number]: { front?: boolean; back?: boolean; left?: boolean; right?: boolean } } = {
            1: { back: true, left: true },
            2: { back: true, right: true, left: true },
            3: { right: true, back: true },
        };
        const faces = oppFaces[relIndex] || { front: true, right: true };
        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        // Hand tiles (left side from player's perspective)
        const handDiv = document.createElement('div');
        handDiv.className = 'opp-tiles-inner';
        player.hand.forEach(t => {
            const tile = document.createElement('div');
            tile.className = 'opp-tile';
            const topFace = this.setTile3D(tile, t, tw, faces);
            if (activeWaits.size > 0 && activeWaits.has(normalize(t))) {
                const ov = document.createElement('div');
                Object.assign(ov.style, {
                    position: 'absolute', top: '0', left: '0',
                    width: '100%', height: '100%',
                    backgroundColor: 'rgba(255, 0, 0, 0.4)',
                    zIndex: '10', pointerEvents: 'none', borderRadius: '3px',
                });
                topFace.appendChild(ov);
            }
            handDiv.appendChild(tile);
        });
        wrapper.appendChild(handDiv);

        // Melds (right side from player's perspective)
        if (player.melds.length > 0) {
            const meldsDiv = document.createElement('div');
            meldsDiv.className = 'opp-melds-inner';

            player.melds.forEach(m => {
                const mGroup = document.createElement('div');
                mGroup.className = 'opp-meld-group';

                const rel = (m.from - playerIdx + pc) % pc;
                const tiles = [...m.tiles];

                const addWaitHighlight = (topFace: HTMLElement, t: string) => {
                    if (activeWaits.size > 0 && activeWaits.has(normalize(t))) {
                        const ov = document.createElement('div');
                        Object.assign(ov.style, {
                            position: 'absolute', top: '0', left: '0',
                            width: '100%', height: '100%',
                            backgroundColor: 'rgba(255, 0, 0, 0.4)',
                            zIndex: '10', pointerEvents: 'none', borderRadius: '3px',
                        });
                        topFace.appendChild(ov);
                    }
                };

                if (m.type === 'ankan') {
                    tiles.forEach((t, i) => {
                        const tileId = (i === 0 || i === 3) ? 'back' : t;
                        const d = document.createElement('div');
                        d.className = 'opp-tile';
                        const topFace = this.setTile3D(d, tileId, tw, faces);
                        addWaitHighlight(topFace, tileId);
                        mGroup.appendChild(d);
                    });
                } else {
                    const stolen = tiles.pop()!;
                    const consumed = tiles;

                    const addUpright = (t: string) => {
                        const d = document.createElement('div');
                        d.className = 'opp-tile';
                        const topFace = this.setTile3D(d, t, tw, faces);
                        addWaitHighlight(topFace, t);
                        mGroup.appendChild(d);
                    };
                    const addRotated = (t: string) => {
                        const d = document.createElement('div');
                        d.className = 'opp-tile-rotated';
                        const topFace = this.setTile3D(d, t, tw, faces);
                        addWaitHighlight(topFace, t);
                        mGroup.appendChild(d);
                    };

                    if (rel === 1) {
                        consumed.forEach(t => addUpright(t));
                        addRotated(stolen);
                    } else if (rel === 3) {
                        addRotated(stolen);
                        consumed.forEach(t => addUpright(t));
                    } else {
                        if (consumed.length >= 2) {
                            addUpright(consumed[0]);
                            addRotated(stolen);
                            addUpright(consumed[1]);
                        } else {
                            consumed.forEach(t => addUpright(t));
                            addRotated(stolen);
                        }
                    }
                }
                meldsDiv.appendChild(mGroup);
            });
            wrapper.appendChild(meldsDiv);
        }

        return wrapper;
    }

    // =========================================================================
    // Own hand (flat, bottom layer)
    // =========================================================================
    private renderOwnHand(
        player: PlayerState, vpIdx: number, state: BoardState,
        pc: number, activeWaits: Set<string>
    ): HTMLElement {
        const [tw, th] = this.layout.tileSizes.ownTile;
        const handArea = document.createElement('div');
        handArea.className = 'own-hand-area-3d';

        // Closed hand
        const tilesDiv = document.createElement('div');
        tilesDiv.className = 'own-tiles-3d';

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        // Check if player has drawn a tile
        let hasDraw = false;
        let shouldAnimate = false;
        if (state.currentActor === vpIdx && state.lastEvent) {
            const type = state.lastEvent.type;
            if (type === 'tsumo' && state.lastEvent.actor === vpIdx) {
                hasDraw = true;
                shouldAnimate = true;
            } else if (type === 'reach' && state.lastEvent.actor === vpIdx) {
                hasDraw = true;
                shouldAnimate = false;
            }
        }

        player.hand.forEach((t, idx) => {
            const tDiv = document.createElement('div');
            tDiv.className = 'own-tile-3d';
            tDiv.appendChild(TileRenderer.getTileElement(t));

            // Tsumo tile separation
            if (hasDraw && idx === player.hand.length - 1) {
                tDiv.style.marginLeft = '14px';
                if (shouldAnimate) tDiv.classList.add('tsumo-anim-3d');
            }

            // Highlight
            if (activeWaits.size > 0) {
                const normT = normalize(t);
                if (activeWaits.has(normT)) {
                    const overlay = document.createElement('div');
                    Object.assign(overlay.style, {
                        position: 'absolute', top: '0', left: '0',
                        width: '100%', height: '100%',
                        backgroundColor: 'rgba(255, 0, 0, 0.4)',
                        zIndex: '10', pointerEvents: 'none', borderRadius: '4px',
                    });
                    tDiv.appendChild(overlay);
                }
            }

            tilesDiv.appendChild(tDiv);
        });
        handArea.appendChild(tilesDiv);

        // Melds
        if (player.melds.length > 0) {
            const meldsDiv = document.createElement('div');
            meldsDiv.className = 'own-melds-3d';

            player.melds.forEach(m => {
                this.renderOwnMeld(meldsDiv, m, vpIdx, pc);
            });
            handArea.appendChild(meldsDiv);
        }

        return handArea;
    }

    private renderOwnMeld(
        container: HTMLElement,
        m: { type: string; tiles: string[]; from: number },
        actor: number, pc: number
    ): void {
        const mGroup = document.createElement('div');
        mGroup.className = 'own-meld-group-3d';

        const rel = (m.from - actor + pc) % pc;
        const tiles = [...m.tiles];

        const addUpright = (t: string) => {
            const d = document.createElement('div');
            d.className = 'meld-tile-own';
            d.appendChild(TileRenderer.getTileElement(t));
            mGroup.appendChild(d);
        };
        const addRotated = (t: string) => {
            const d = document.createElement('div');
            d.className = 'meld-tile-own-rotated';
            d.appendChild(TileRenderer.getTileElement(t));
            mGroup.appendChild(d);
        };

        if (m.type === 'ankan') {
            tiles.forEach((t, i) => {
                const tileId = (i === 0 || i === 3) ? 'back' : t;
                addUpright(tileId);
            });
        } else if (m.type === 'kakan') {
            const added = tiles.pop()!;
            const ponTiles = tiles;
            const stolen = ponTiles.pop()!;
            const consumed = ponTiles;

            // Kakan: stolen tile + added tile stacked rotated
            if (rel === 1) {
                consumed.forEach(t => addUpright(t));
                addRotated(stolen);
            } else if (rel === 3) {
                addRotated(stolen);
                consumed.forEach(t => addUpright(t));
            } else {
                if (consumed.length >= 2) {
                    addUpright(consumed[0]);
                    addRotated(stolen);
                    addUpright(consumed[1]);
                } else {
                    consumed.forEach(t => addUpright(t));
                    addRotated(stolen);
                }
            }
        } else {
            // Pon/Chi/Daiminkan
            const stolen = tiles.pop()!;
            const consumed = tiles;

            if (m.type === 'daiminkan') {
                if (rel === 1) {
                    consumed.forEach(t => addUpright(t));
                    addRotated(stolen);
                } else if (rel === 3) {
                    addRotated(stolen);
                    consumed.forEach(t => addUpright(t));
                } else {
                    if (consumed.length >= 3) {
                        addUpright(consumed[0]);
                        addUpright(consumed[1]);
                        addRotated(stolen);
                        addUpright(consumed[2]);
                    } else {
                        consumed.forEach(t => addUpright(t));
                        addRotated(stolen);
                    }
                }
            } else {
                // Pon / Chi
                if (rel === 1) {
                    consumed.forEach(t => addUpright(t));
                    addRotated(stolen);
                } else if (rel === 3) {
                    addRotated(stolen);
                    consumed.forEach(t => addUpright(t));
                } else if (rel === 2) {
                    if (consumed.length >= 2) {
                        addUpright(consumed[0]);
                        addRotated(stolen);
                        addUpright(consumed[1]);
                    } else {
                        consumed.forEach(t => addUpright(t));
                        addRotated(stolen);
                    }
                } else {
                    consumed.forEach(t => addUpright(t));
                    addRotated(stolen);
                }
            }
        }

        container.appendChild(mGroup);
    }

    // =========================================================================
    // Score panel (UI overlay)
    // =========================================================================
    private renderPlayerPanel(
        player: PlayerState, playerIdx: number,
        relIndex: number, state: BoardState
    ): HTMLElement {
        const panel = document.createElement('div');
        panel.className = 'player-panel-3d';
        if (playerIdx === this.viewpoint) panel.classList.add('active-vp');

        // Position — corners and edges
        const panelPositions: { [key: number]: { [k: string]: string } } = {
            0: { bottom: '130px', left: '25%', transform: 'translateX(-50%)' },
            1: { right: '50px', top: '45%', transform: 'translateY(-50%)' },
            2: { top: '100px', right: '380px' },
            3: { left: '100px', top: '120px' },
        };
        const pos = panelPositions[relIndex] || panelPositions[0];
        Object.assign(panel.style, pos);

        // Avatar (centered)
        const avatar = document.createElement('div');
        avatar.className = 'avatar-3d';
        const avatarImg = document.createElement('img');
        avatarImg.src = AVATAR_PLACEHOLDER;
        avatarImg.className = 'avatar-img';
        avatar.appendChild(avatarImg);
        panel.appendChild(avatar);

        // Player name
        const playerName = document.createElement('div');
        playerName.className = 'player-name';
        playerName.textContent = `P${playerIdx}`;
        panel.appendChild(playerName);

        // Active player bar
        if (playerIdx === state.currentActor) {
            const bar = document.createElement('div');
            bar.className = 'active-player-bar';
            Object.assign(bar.style, { marginTop: '3px' });
            panel.appendChild(bar);
        }

        // Click to change viewpoint
        panel.onclick = (e) => {
            e.stopPropagation();
            if (this.onViewpointChange) this.onViewpointChange(playerIdx);
        };

        return panel;
    }

    // =========================================================================
    // Call overlay (UI overlay)
    // =========================================================================
    private renderCallOverlay(overlay: HTMLElement, state: BoardState): void {
        if (!state.lastEvent) return;

        let label = '';
        let actorIdx: number | undefined;
        const evt = state.lastEvent;
        const pc = state.playerCount || 4;

        let callCssClass: string | undefined;

        if (evt.actor !== undefined) {
            const type = evt.type;
            const callDef = CALL_TYPES[type];
            if (callDef) {
                label = callDef.label;
                callCssClass = callDef.cssClass;
                actorIdx = evt.actor;
            } else if (type === 'hora') {
                label = (evt.target === evt.actor) ? 'Tsumo' : 'Ron';
                callCssClass = 'call-hora';
                actorIdx = evt.actor;
            }
        }

        if (evt.type === 'ryukyoku') {
            label = 'Ryukyoku';
        }

        if (label) {
            const el = document.createElement('div');
            el.className = 'call-overlay-3d';
            if (callCssClass) el.classList.add(callCssClass);
            el.textContent = label;

            if (actorIdx !== undefined) {
                const relIndex = (actorIdx - this.viewpoint + pc) % pc;
                // Position adjacent to each player's panel
                // Panel positions:
                //   0: bottom: 130px, left: 25%
                //   1: right: 50px, top: 45%
                //   2: top: 100px, right: 380px
                //   3: left: 100px, top: 120px
                const callPositions: { [key: number]: { [k: string]: string } } = {
                    0: { bottom: '180px', left: '25%', top: 'auto', right: 'auto', transform: 'translateX(-50%)' },
                    1: { right: '120px', top: '45%', bottom: 'auto', left: 'auto', transform: 'translateY(-50%)' },
                    2: { top: '95px', right: '470px', bottom: 'auto', left: 'auto', transform: 'none' },
                    3: { left: '170px', top: '115px', bottom: 'auto', right: 'auto', transform: 'none' },
                };
                const pos = callPositions[relIndex];
                if (pos) Object.assign(el.style, pos);
            }
            // Ryukyoku: keeps default CSS center position

            overlay.appendChild(el);
        }
    }

    // =========================================================================
    // Wait indicator
    // =========================================================================
    private renderWaitIndicator(waits: string[], relIndex: number): HTMLElement {
        const el = document.createElement('div');
        el.className = 'wait-indicator-3d';

        // Position near each player's panel on UI overlay
        // Panel positions: 0=bottom-left, 1=right, 2=top-right, 3=left
        const waitPositions: { [key: number]: { [k: string]: string } } = {
            0: { bottom: '110px', left: '40%' },
            1: { right: '50px', top: '55%' },
            2: { top: '55px', right: '380px' },
            3: { left: '70px', top: '30%' },
        };
        Object.assign(el.style, waitPositions[relIndex] || waitPositions[0]);

        const label = document.createElement('span');
        label.textContent = 'Wait:';
        label.style.marginRight = '4px';
        el.appendChild(label);

        waits.forEach(w => {
            const tile = document.createElement('div');
            tile.className = 'wait-tile-3d';
            tile.appendChild(TileRenderer.getTileElement(w));
            el.appendChild(tile);
        });

        return el;
    }
}
