import { BoardState, PlayerState, Tile } from '../types';
import { VIEWER_CSS } from '../styles';
import { VIEWER_3D_CSS } from '../styles_3d';
import { LayoutConfig3D, createLayout3DConfig4P } from '../config';
import { IRenderer } from './renderer_interface';
import { TileRenderer } from './tile_renderer';
import { ResultRenderer } from './result_renderer';
import { COLORS } from '../constants';
import { CHAR_SPRITE_BASE64, CHAR_MAP } from '../char_assets';

export class Renderer3D implements IRenderer {
    container: HTMLElement;
    viewpoint: number = 0;
    onViewpointChange: ((pIdx: number) => void) | null = null;
    onCenterClick: (() => void) | null = null;

    private sceneEl: HTMLElement | null = null;
    private layout: LayoutConfig3D;

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

    render(state: BoardState, debugPanel?: HTMLElement): void {
        const pc = state.playerCount;

        // 1. Create/reuse scene container
        if (!this.sceneEl) {
            this.sceneEl = document.createElement('div');
            this.sceneEl.className = 'scene-3d';
            this.container.appendChild(this.sceneEl);
        }
        this.sceneEl.innerHTML = '';

        // 2. Clear old modals
        const oldModals = this.container.querySelectorAll('.re-modal-overlay');
        oldModals.forEach(el => el.remove());

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
        Object.assign(tableSurface.style, {
            width: `${this.layout.tableSize}px`,
            height: `${this.layout.tableSize}px`,
            top: tableTop,
            transform: `translate(-50%, -50%) rotateX(${this.layout.tiltAngle}deg)`,
        });

        // Table inner border
        const tableInner = document.createElement('div');
        tableInner.className = 'table-inner';
        tableSurface.appendChild(tableInner);

        // Center info
        const center = this.renderCenter3D(state);
        tableSurface.appendChild(center);

        // Riichi sticks on table
        this.renderRiichiSticks(tableSurface, state, pc);

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
            tableSurface.appendChild(river);

            // Opponent hand + melds on table (skip viewpoint player)
            if (relIndex !== 0) {
                const oppHand = this.renderOpponentHand(p, relIndex, pc);
                tableSurface.appendChild(oppHand);

                if (p.melds.length > 0) {
                    const oppMelds = this.renderOpponentMelds(p.melds, i, relIndex, pc);
                    tableSurface.appendChild(oppMelds);
                }
            }
        });

        perspectiveEl.appendChild(tableSurface);
        this.sceneEl.appendChild(perspectiveEl);

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
        this.sceneEl.appendChild(handLayer);

        // 5. Build Layer 3: UI Overlay
        const uiOverlay = document.createElement('div');
        uiOverlay.className = 'ui-overlay-3d';

        // Score panels
        state.players.forEach((p, i) => {
            const relIndex = (i - this.viewpoint + pc) % pc;
            const panel = this.renderScorePanel(p, i, relIndex, state);
            uiOverlay.appendChild(panel);
        });

        // Call overlay
        this.renderCallOverlay(uiOverlay, state);

        // Wait indicators for viewpoint player
        if (vpPlayer && vpPlayer.waits && vpPlayer.waits.length > 0) {
            const waitEl = this.renderWaitIndicator(vpPlayer.waits);
            uiOverlay.appendChild(waitEl);
        }

        this.sceneEl.appendChild(uiOverlay);

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

        center.onclick = (e) => {
            e.stopPropagation();
            if (this.onCenterClick) this.onCenterClick();
        };

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

            const targetSize = 22;
            const maxDim = Math.max(asset.w, asset.h);
            const scale = Math.min(1, targetSize / maxDim);

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
            fontSize: '18px', fontWeight: 'bold', color: 'white',
            fontFamily: 'sans-serif', marginBottom: '2px',
        });
        row1.textContent = roundText;
        contentDiv.appendChild(row1);

        // Row 2: Honba / Kyotaku
        const row2 = document.createElement('div');
        row2.textContent = `${state.honba}, ${state.kyotaku}`;
        Object.assign(row2.style, {
            fontSize: '13px', fontWeight: 'bold', color: 'white',
            fontFamily: 'monospace', marginBottom: '6px',
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
            d.innerHTML = TileRenderer.getTileHtml(t);
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

            // Position between river and center (proportional to table size)
            const near = Math.round(ts * 0.6625);
            const far = Math.round(ts * 0.3375);
            if (relPos === 0) {
                Object.assign(stick.style, {
                    left: '50%', top: `${near}px`,
                    transform: 'translateX(-50%)',
                });
            } else if (relPos === 1) {
                Object.assign(stick.style, {
                    left: `${near}px`, top: '50%',
                    transform: 'translateY(-50%) rotate(90deg)',
                });
            } else if (relPos === 2) {
                Object.assign(stick.style, {
                    left: '50%', top: `${far}px`,
                    transform: 'translateX(-50%)',
                });
            } else if (relPos === 3) {
                Object.assign(stick.style, {
                    left: `${far}px`, top: '50%',
                    transform: 'translateY(-50%) rotate(90deg)',
                });
            }
            table.appendChild(stick);
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
        const wrapper = document.createElement('div');
        wrapper.className = 'river-3d';

        // Position on table (proportional to table size)
        const ts = this.layout.tableSize;
        const positions: { [key: number]: { left: string; top: string; transform: string } } = {
            0: { left: '50%', top: `${Math.round(ts * 0.75)}px`, transform: 'translate(-50%, -50%)' },
            1: { left: `${Math.round(ts * 0.775)}px`, top: '50%', transform: 'translate(-50%, -50%) rotate(-90deg)' },
            2: { left: '50%', top: `${Math.round(ts * 0.25)}px`, transform: 'translate(-50%, -50%) rotate(180deg)' },
            3: { left: `${Math.round(ts * 0.225)}px`, top: '50%', transform: 'translate(-50%, -50%) rotate(90deg)' },
        };
        const pos = positions[relIndex] || positions[0];
        Object.assign(wrapper.style, pos);

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

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

                cell.innerHTML = TileRenderer.getTileHtml(d.tile);

                // Highlight
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
                        cell.appendChild(overlay);
                    }
                }

                rowDiv.appendChild(cell);
            });
            wrapper.appendChild(rowDiv);
        });

        return wrapper;
    }

    // =========================================================================
    // Opponent hand on table edge
    // =========================================================================
    private renderOpponentHand(player: PlayerState, relIndex: number, pc: number): HTMLElement {
        const [tw, th] = this.layout.tileSizes.opponentTile;
        const wrapper = document.createElement('div');
        wrapper.className = 'opp-hand-3d';

        // Position at table edges
        const positions: { [key: number]: { [k: string]: string } } = {
            1: { right: '12px', top: '50%', transform: 'translateY(-50%) rotate(-90deg)' },
            2: { top: '12px', left: '50%', transform: 'translate(-50%, 0) rotate(180deg)' },
            3: { left: '12px', top: '50%', transform: 'translateY(-50%) rotate(90deg)' },
        };
        const pos = positions[relIndex];
        if (pos) Object.assign(wrapper.style, pos);

        player.hand.forEach(t => {
            const tile = document.createElement('div');
            tile.className = 'opp-tile';
            tile.innerHTML = TileRenderer.getTileHtml(t);
            wrapper.appendChild(tile);
        });

        return wrapper;
    }

    // =========================================================================
    // Opponent melds on table
    // =========================================================================
    private renderOpponentMelds(
        melds: { type: string; tiles: string[]; from: number }[],
        playerIdx: number, relIndex: number, pc: number
    ): HTMLElement {
        const wrapper = document.createElement('div');
        wrapper.className = 'opp-meld-3d';

        // Position near the corner, between hand and edge (proportional to table size)
        const ts = this.layout.tableSize;
        const meldOffset = `${Math.round(ts * 0.1)}px`;
        const positions: { [key: number]: { [k: string]: string } } = {
            1: { right: '12px', bottom: meldOffset, transform: 'rotate(-90deg)' },
            2: { right: meldOffset, top: '12px', transform: 'rotate(180deg)' },
            3: { left: '12px', top: meldOffset, transform: 'rotate(90deg)' },
        };
        const pos = positions[relIndex];
        if (pos) Object.assign(wrapper.style, pos);

        melds.forEach(m => {
            const mGroup = document.createElement('div');
            Object.assign(mGroup.style, {
                display: 'flex', alignItems: 'flex-end', marginLeft: '3px',
            });

            const rel = (m.from - playerIdx + pc) % pc;
            const tiles = [...m.tiles];

            if (m.type === 'ankan') {
                tiles.forEach((t, i) => {
                    const tileId = (i === 0 || i === 3) ? 'back' : t;
                    const d = document.createElement('div');
                    d.className = 'meld-tile-table';
                    d.innerHTML = TileRenderer.getTileHtml(tileId);
                    mGroup.appendChild(d);
                });
            } else {
                // Pon/Chi/Kan: last tile is stolen → rotated
                const stolen = tiles.pop()!;
                const consumed = tiles;

                const addUpright = (t: string) => {
                    const d = document.createElement('div');
                    d.className = 'meld-tile-table';
                    d.innerHTML = TileRenderer.getTileHtml(t);
                    mGroup.appendChild(d);
                };
                const addRotated = (t: string) => {
                    const d = document.createElement('div');
                    d.className = 'meld-tile-table-rotated';
                    d.innerHTML = TileRenderer.getTileHtml(t);
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
            wrapper.appendChild(mGroup);
        });

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
            tDiv.innerHTML = TileRenderer.getTileHtml(t);

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
            d.innerHTML = TileRenderer.getTileHtml(t);
            mGroup.appendChild(d);
        };
        const addRotated = (t: string) => {
            const d = document.createElement('div');
            d.className = 'meld-tile-own-rotated';
            d.innerHTML = TileRenderer.getTileHtml(t);
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
    private renderScorePanel(
        player: PlayerState, playerIdx: number,
        relIndex: number, state: BoardState
    ): HTMLElement {
        const panel = document.createElement('div');
        panel.className = 'score-panel-3d';
        if (playerIdx === this.viewpoint) panel.classList.add('active-vp');

        // Position
        const positions: { [key: number]: { [k: string]: string } } = {
            0: { bottom: '135px', left: '50%', transform: 'translateX(-50%)' },
            1: { right: '50px', top: '45%', transform: 'translateY(-50%)' },
            2: { top: '15px', left: '50%', transform: 'translateX(-50%)' },
            3: { left: '50px', top: '45%', transform: 'translateY(-50%)' },
        };
        const pos = positions[relIndex] || positions[0];
        Object.assign(panel.style, pos);

        // Wind label
        const winds = ['E', 'S', 'W', 'N'];
        const windLabel = document.createElement('div');
        windLabel.className = 'wind-label';
        windLabel.textContent = `${winds[player.wind] || '?'} · P${playerIdx}`;
        panel.appendChild(windLabel);

        // Score
        const scoreVal = document.createElement('div');
        scoreVal.className = 'score-value';
        scoreVal.textContent = player.score.toString();
        panel.appendChild(scoreVal);

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
        const evt = state.lastEvent;

        if (evt.actor !== undefined) {
            const type = evt.type;
            if (['chi', 'pon', 'kan', 'ankan', 'daiminkan', 'kakan', 'reach'].includes(type)) {
                label = type.charAt(0).toUpperCase() + type.slice(1);
                if (type === 'daiminkan') label = 'Kan';
                if (type === 'reach') label = 'Reach';
            } else if (type === 'hora') {
                label = (evt.target === evt.actor) ? 'Tsumo' : 'Ron';
            }
        }

        if (evt.type === 'ryukyoku') {
            label = 'Ryukyoku';
        }

        if (label) {
            const el = document.createElement('div');
            el.className = 'call-overlay-3d';
            el.textContent = label;
            overlay.appendChild(el);
        }
    }

    // =========================================================================
    // Wait indicator
    // =========================================================================
    private renderWaitIndicator(waits: string[]): HTMLElement {
        const el = document.createElement('div');
        el.className = 'wait-indicator-3d';
        Object.assign(el.style, {
            bottom: '135px', left: '50%',
            transform: 'translateX(-50%) translateY(-30px)',
        });

        const label = document.createElement('span');
        label.textContent = 'Wait:';
        label.style.marginRight = '4px';
        el.appendChild(label);

        waits.forEach(w => {
            const tile = document.createElement('div');
            tile.className = 'wait-tile-3d';
            tile.innerHTML = TileRenderer.getTileHtml(w);
            el.appendChild(tile);
        });

        return el;
    }
}
