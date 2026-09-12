import { createLayout3DConfig4P, type LayoutConfig3D } from '../config';
import { CALL_TYPES } from '../constants';
import { I18n } from '../i18n/index';
import { VIEWER_CSS } from '../styles';
import { VIEWER_3D_CSS } from '../styles_3d';
import type { BoardState, PlayerState, Tile } from '../types';
import { type DisplayOptions, presentBoard } from './board_presentation';
import { drawnTileIndex } from './hand_presentation';
import { playerAvatar } from './player_identity';
import type { IRenderer } from './renderer_interface';
import { ResultRenderer } from './result_renderer';
import { relativeSeat } from './seat_position';
import { paintTileBodies, type TileBodyRequest } from './tile_body_sprite';
import { TILE_BACK_COLOR, TILE_FACE_COLOR } from './tile_material';
import { TileRenderer } from './tile_renderer';

export class Renderer3D implements IRenderer {
    i18n = new I18n();
    container: HTMLElement;
    viewpoint: number = 0;
    onViewpointChange: ((pIdx: number) => void) | null = null;
    onCenterClick: (() => void) | null = null;
    onSettingsClick: (() => void) | null = null;

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

    private pendingBodies = new Map<HTMLElement, TileBodyRequest>();

    /** One raster body and one live art face retain the existing CSS pose. */
    private setTile3D(el: HTMLElement, tileId: string, tileWidth: number): void {
        const tileHeight =
            el.classList.contains('table-tile') || el.classList.contains('table-tile-rotated')
                ? this.layout.tileSizes.riverTile[1]
                : this.layout.tileSizes.opponentTile[1];
        const rotated = el.className.includes('-rotated');
        const w = rotated ? tileHeight : tileWidth;
        const h = rotated ? tileWidth : tileHeight;
        const depth = (tileWidth * 16.5) / 20.5;
        const radius = tileWidth * 0.14;
        const edge = tileWidth * 0.07;
        const flipped = tileId === 'back' || tileId === '?';
        el.style.transformStyle = 'preserve-3d';
        el.classList.toggle('tile-face-down', flipped);
        el.setAttribute('aria-label', flipped ? this.i18n.text('Hidden tile') : tileId);
        el.title = flipped ? this.i18n.text('Hidden tile') : tileId;
        this.pendingBodies.set(el, { element: el, width: w, height: h, depth, radius, edge, flipped, overlays: [] });
        // Overlap the bitmap rim slightly: two separately antialiased edges
        // otherwise expose a hairline of the blue tabletop between them.
        const faceInset = edge - 0.35;
        const top = document.createElement('div');
        top.className = 'tile-3d-surface tile-3d-top';
        Object.assign(top.style, {
            left: `${faceInset}px`,
            top: `${faceInset}px`,
            width: `${w - 2 * faceInset}px`,
            height: `${h - 2 * faceInset}px`,
            borderRadius: `${radius - faceInset}px`,
            transform: `translateZ(${depth + 0.02}px)`,
            background: flipped ? TILE_BACK_COLOR : TILE_FACE_COLOR,
        });
        if (!flipped) {
            const art = TileRenderer.getTileElement(tileId);
            if (rotated)
                Object.assign(art.style, {
                    width: `${tileWidth - 2 * faceInset}px`,
                    height: `${tileHeight - 2 * faceInset}px`,
                });
            top.appendChild(art);
        }
        el.appendChild(top);
    }

    /**
     * Add a colored overlay to all visible faces of a 3D tile element.
     */
    private addTile3DOverlay(el: HTMLElement, color: string, zIndex: string = '5'): void {
        // Tint the already composited body once, instead of adding an overlay
        // to every facet of the old mesh.
        this.pendingBodies.get(el)?.overlays.push(color);
        for (const face of el.querySelectorAll<HTMLElement>(':scope > .tile-3d-surface')) {
            const overlay = document.createElement('div');
            Object.assign(overlay.style, {
                position: 'absolute',
                inset: '0',
                backgroundColor: color,
                pointerEvents: 'none',
                borderRadius: 'inherit',
                zIndex,
            });
            face.appendChild(overlay);
        }
    }

    render(sourceState: BoardState, debugPanel?: HTMLElement, displayOptions?: Readonly<DisplayOptions>): void {
        const state = presentBoard(sourceState, this.viewpoint, displayOptions);
        this.pendingBodies.clear();
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
            oldModals.forEach((el) => el.remove());
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
        this.renderRiichiSticks(tableInner, state);

        // Floating score labels on table (above riichi sticks)
        this.renderFloatingScores(tableInner, state);

        // Collect waits and build per-player "danger waits" (exclude own waits from own hand highlight)
        const activeWaits = new Set<string>();
        const ownWaitsByPlayer = state.players.map(() => new Set<string>());
        const normalize = (t: string) => t.replace('0', '5').replace('r', '');
        state.players.forEach((pl, idx) => {
            if (pl.waits && pl.waits.length > 0) {
                pl.waits.forEach((w) => {
                    const normW = normalize(w);
                    activeWaits.add(normW);
                    ownWaitsByPlayer[idx].add(normW);
                });
            }
        });
        const dangerWaitsByPlayer = ownWaitsByPlayer.map((_, idx) => {
            const dangerWaits = new Set<string>();
            ownWaitsByPlayer.forEach((waits, otherIdx) => {
                if (otherIdx !== idx) {
                    waits.forEach((w) => dangerWaits.add(w));
                }
            });
            return dangerWaits;
        });

        // Per-player table elements
        state.players.forEach((p, i) => {
            const relIndex = relativeSeat(state, i, this.viewpoint);

            // River (discards)
            const river = this.renderRiver3D(p.discards, relIndex, activeWaits);
            tableInner.appendChild(river);

            // Opponent hand + melds on table (skip viewpoint player)
            if (relIndex !== 0) {
                const oppHand = this.renderOpponentHandArea(p, i, relIndex, dangerWaitsByPlayer[i], state);
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
            const handEl = this.renderOwnHand(vpPlayer, this.viewpoint, state, pc, dangerWaitsByPlayer[this.viewpoint]);
            handLayer.appendChild(handEl);
        }
        sceneFrag.appendChild(handLayer);

        // Share the table's exact projection, but paint own melds above the
        // hand gradient. Shallow clones retain border widths and layout only.
        if (vpPlayer && (vpPlayer.melds.length > 0 || (pc === 3 && vpPlayer.kitaCount > 0))) {
            const meldPerspective = perspectiveEl.cloneNode(false) as HTMLElement;
            meldPerspective.classList.add('own-meld-layer');
            const meldSurface = tableSurface.cloneNode(false) as HTMLElement;
            const meldInner = tableInner.cloneNode(false) as HTMLElement;
            meldInner.appendChild(this.renderOwnMelds(vpPlayer, this.viewpoint, state));
            if (pc === 3 && vpPlayer.kitaCount > 0) {
                const kita = this.renderKita(vpPlayer.kitaCount);
                kita.classList.add('own-kita-3d');
                meldInner.appendChild(kita);
            }
            meldSurface.appendChild(meldInner);
            meldPerspective.appendChild(meldSurface);
            sceneFrag.appendChild(meldPerspective);
        }

        // 5. Build Layer 3: UI Overlay
        const uiOverlay = document.createElement('div');
        uiOverlay.className = 'ui-overlay-3d';

        uiOverlay.appendChild(this.renderDoraPanel(state));

        const settingsButton = document.createElement('button');
        settingsButton.type = 'button';
        settingsButton.className = 'viewer-settings-button';
        settingsButton.setAttribute('aria-label', this.i18n.text('Settings'));
        settingsButton.setAttribute('aria-haspopup', 'dialog');
        settingsButton.title = this.i18n.text('Settings');
        settingsButton.innerHTML = `<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" aria-hidden="true"><path d="m9 3-.6 2.5-2 .9-2.2-.7-2 3.4L4 11v2l-1.8 1.9 2 3.4 2.2-.7 2 .9L9 21h6l.6-2.5 2-.9 2.2.7 2-3.4L20 13v-2l1.8-1.9-2-3.4-2.2.7-2-.9L15 3Z"/><circle cx="12" cy="12" r="3.2"/></svg>`;
        settingsButton.onclick = (event) => {
            event.stopPropagation();
            this.onSettingsClick?.();
        };
        uiOverlay.appendChild(settingsButton);

        // Player identity panels (at viewport edges)
        state.players.forEach((_, i) => {
            const relIndex = relativeSeat(state, i, this.viewpoint);
            const panel = this.renderPlayerPanel(i, relIndex, state);
            uiOverlay.appendChild(panel);
        });

        // Center click zone (2D overlay for reliable click on 3D center panel)
        if (this.onCenterClick) {
            const centerClick = document.createElement('button');
            centerClick.type = 'button';
            centerClick.setAttribute('aria-label', this.i18n.text('Jump to round'));
            centerClick.title = this.i18n.text('Jump to round');
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
                const relIndex = relativeSeat(state, i, this.viewpoint);
                const waitEl = this.renderWaitIndicator(p.waits, relIndex);
                uiOverlay.appendChild(waitEl);
            }
        });

        sceneFrag.appendChild(uiOverlay);

        // Single DOM swap — replaces all scene content in one operation
        const focusedViewpoint = this.sceneEl.contains(document.activeElement)
            ? (document.activeElement as HTMLElement)?.dataset.viewpoint
            : undefined;
        this.sceneEl.replaceChildren(sceneFrag);
        const bodies = [...this.pendingBodies.values()];
        for (const body of bodies) {
            body.facePlane = !!body.element.closest('.opp-hand-3d .opp-meld-group, .own-melds-3d .opp-meld-group');
        }
        paintTileBodies(bodies, this.sceneEl);
        if (focusedViewpoint !== undefined) {
            this.sceneEl
                .querySelector<HTMLButtonElement>(`[data-viewpoint="${focusedViewpoint}"]`)
                ?.focus({ preventScroll: true });
        }

        // 6. Result modals
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta) {
            let modal: HTMLElement | null = null;
            if (state.lastEvent.meta.ryukyoku) {
                modal = ResultRenderer.renderRyukyokuModal(state.lastEvent.meta.ryukyoku, sourceState, this.i18n);
            } else if (state.lastEvent.meta.results) {
                modal = ResultRenderer.renderModal(state.lastEvent.meta.results, sourceState, this.i18n);
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

        for (let side = 0; side < 4; side++) {
            const rail = document.createElement('span');
            rail.className = `center-rail center-rail-${side}`;
            rail.setAttribute('aria-hidden', 'true');
            center.appendChild(rail);
        }

        state.players.forEach((p, i) => {
            const wind = document.createElement('span');
            const seat = relativeSeat(state, i, this.viewpoint);
            wind.className = `center-wind center-wind-${seat}`;
            wind.classList.toggle('dealer', p.wind === 0);
            wind.textContent = this.i18n.wind(p.wind, true);
            center.appendChild(wind);
        });

        const display = document.createElement('div');
        display.className = 'center-display';
        const round = document.createElement('strong');
        round.textContent = this.i18n.round(state.round, state.playerCount);
        const remaining = document.createElement('div');
        remaining.className = 'wall-remaining';
        remaining.textContent = String(state.wallRemaining);
        remaining.setAttribute('aria-label', this.i18n.text('{count} tiles remaining', { count: state.wallRemaining }));
        remaining.title = this.i18n.text('{count} tiles remaining', { count: state.wallRemaining });
        display.append(round, remaining);
        center.appendChild(display);

        if (state.players[state.currentActor]) {
            const indicator = document.createElement('div');
            indicator.className = 'center-turn-indicator';
            indicator.dataset.player = String(state.currentActor);
            const seat = relativeSeat(state, state.currentActor, this.viewpoint);
            const name = state.playerNames[state.currentActor] || `Player${state.currentActor}`;
            indicator.setAttribute('role', 'img');
            indicator.setAttribute('aria-label', this.i18n.text('Turn: {name}', { name }));
            this.positionCenterEdge(indicator, seat, 59);
            // Keep the blink phase continuous when replay events rebuild the DOM.
            indicator.style.animationDelay = `-${performance.now() % 1200}ms`;
            center.appendChild(indicator);
        }
        return center;
    }

    private renderDoraPanel(state: BoardState): HTMLElement {
        const panel = document.createElement('div');
        panel.className = 'dora-panel-3d';
        panel.setAttribute('role', 'group');
        panel.setAttribute('aria-label', this.i18n.text('Dora indicators'));
        const tiles = document.createElement('div');
        tiles.className = 'dora-tiles';
        const markers = [...state.doraMarkers];
        while (markers.length < 5) markers.push('back');
        markers.forEach((marker) => {
            const tile = document.createElement('div');
            tile.className = `dora-marker${marker === 'back' ? ' tile-face-down' : ''}`;
            tile.title = marker === 'back' ? this.i18n.text('Unrevealed indicator') : marker;
            tile.style.width = `${this.layout.tileSizes.doraTile[0]}px`;
            tile.style.height = `${this.layout.tileSizes.doraTile[1]}px`;
            tile.appendChild(TileRenderer.getTileElement(marker));
            tiles.appendChild(tile);
        });
        const counters = document.createElement('div');
        counters.className = 'table-counters';
        counters.setAttribute('role', 'img');
        counters.setAttribute(
            'aria-label',
            this.i18n.text('{honba} honba / {kyotaku} riichi sticks', {
                honba: state.honba,
                kyotaku: state.kyotaku,
            }),
        );
        // 1,000-point stick (red dot), then 100-point stick (eight black dots).
        for (const [kind, count] of [
            ['riichi', state.kyotaku],
            ['honba', state.honba],
        ] as const) {
            const counter = document.createElement('span');
            counter.className = `table-counter table-counter-${kind}`;
            counter.setAttribute('aria-hidden', 'true');
            const dots =
                kind === 'riichi'
                    ? '<circle cx="17" cy="18" r="2.5" fill="#c82f38"/>'
                    : [9, 14, 19, 24]
                          .flatMap((x) => [16, 20].map((y) => `<circle cx="${x}" cy="${y}" r="1.25" fill="#30343a"/>`))
                          .join('');
            counter.innerHTML = `<svg viewBox="0 0 36 36" aria-hidden="true" focusable="false">
                <rect x="4" y="14" width="28" height="10" rx="1.5" fill="#061724" opacity=".6"/>
                <rect x="3" y="13" width="28" height="10" rx="1.5" fill="#f6f3eb" stroke="#a2a9ad"/>
                ${dots}
            </svg>`;
            const value = document.createElement('span');
            value.textContent = `×${count}`;
            counter.appendChild(value);
            counters.appendChild(counter);
        }
        panel.append(tiles, counters);
        return panel;
    }

    // =========================================================================
    // Riichi Sticks on table
    // =========================================================================
    private positionCenterEdge(element: HTMLElement, seat: number, distance: number): void {
        const [x, y] = [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ][seat];
        Object.assign(element.style, {
            left: `calc(50% + ${x * distance}px)`,
            top: `calc(50% + ${y * distance}px)`,
            transform: `translate(-50%, -50%) translateZ(2px) rotate(${-90 * seat}deg)`,
        });
    }

    private renderRiichiSticks(table: HTMLElement, state: BoardState): void {
        state.players.forEach((p, i) => {
            if (!p.riichi) return;
            const relPos = relativeSeat(state, i, this.viewpoint);

            const stick = document.createElement('div');
            stick.className = 'riichi-stick-3d';
            const dot = document.createElement('div');
            dot.className = 'dot';
            stick.appendChild(dot);

            // Center the stick in the 14px groove inside the panel's 3px border.
            this.positionCenterEdge(stick, relPos, 109);
            table.appendChild(stick);
        });
    }

    // =========================================================================
    // Floating score labels on table (positioned above riichi sticks)
    // =========================================================================
    private renderFloatingScores(table: HTMLElement, state: BoardState): void {
        state.players.forEach((p, i) => {
            const relPos = relativeSeat(state, i, this.viewpoint);

            const el = document.createElement('div');
            el.className = 'floating-score-3d';
            el.textContent = String(p.score);

            // Align the number's center, rather than its top edge, on every side.
            this.positionCenterEdge(el, relPos, relPos === 0 ? 86 : 76);

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
    private renderRiver3D(discards: Tile[], relIndex: number, activeWaits: Set<string>): HTMLElement {
        const [tw, th] = this.layout.tileSizes.riverTile;
        const gap = 1;
        // Fixed river area size: 6 columns × 3 rows (+ extra width for one possible riichi rotated tile)
        const riverW = 6 * tw + 5 * gap + (th - tw); // Extra width for the sideways riichi tile
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
        // Face-on rivers appear smaller after the table's perspective projection,
        // especially across the table. Enlarge these without changing side rivers.
        const baseScale = 1.35;
        const riverScale = relIndex === 0 || relIndex === 2 ? 1.5 : baseScale;
        // Expand outward to retain the clearance around the center display.
        const expansion = ((riverScale - baseScale) * riverH) / 2;
        // Leave clearance around the center display for the taller tile faces.
        // Left/right rivers are also shifted toward center by one tile height.
        const positions: { [key: number]: { left: string; top: string; transform: string } } = {
            0: {
                left: '50%',
                top: `${Math.round(ts * 0.712 + expansion + 4)}px`,
                transform: `translate(-50%, -50%) scale(${riverScale})`,
            },
            1: {
                left: `${Math.round(ts * 0.742 - th)}px`,
                top: '50%',
                transform: `translate(-50%, -50%) rotate(-90deg) scale(${riverScale})`,
            },
            2: {
                left: '50%',
                top: `${Math.round(ts * 0.288 - expansion)}px`,
                transform: `translate(-50%, -50%) rotate(180deg) scale(${riverScale})`,
            },
            3: {
                left: `${Math.round(ts * 0.258 + th)}px`,
                top: '50%',
                transform: `translate(-50%, -50%) rotate(90deg) scale(${riverScale})`,
            },
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

        rows.forEach((rowTiles) => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'river-row-3d';

            rowTiles.forEach((d) => {
                const isRiichi = d.isRiichi;
                const cell = document.createElement('div');
                cell.className = isRiichi ? 'table-tile-rotated' : 'table-tile';
                if (d.isTsumogiri) cell.classList.add('table-tile-tsumogiri');

                this.setTile3D(cell, d.tile, tw);

                // Tsumogiri: darken with overlay on all faces
                if (d.isTsumogiri) {
                    this.addTile3DOverlay(cell, 'rgba(0, 0, 0, 0.35)');
                }

                // Highlight dangerous tiles on all faces
                if (activeWaits.size > 0) {
                    const normT = normalize(d.tile);
                    if (activeWaits.has(normT)) {
                        this.addTile3DOverlay(cell, 'rgba(255, 0, 0, 0.4)', '10');
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
        player: PlayerState,
        playerIdx: number,
        relIndex: number,
        activeWaits: Set<string>,
        state: BoardState,
    ): HTMLElement {
        const [tw, th] = this.layout.tileSizes.opponentTile;

        const wrapper = document.createElement('div');
        wrapper.className = 'opp-hand-3d';

        // Compute position: place hand between river outer edge and table edge
        const ts = this.layout.tableSize;
        // Anchor both ends to a fixed stretch of this player's table edge.
        // A content-sized row recenters whenever tiles move from hand to melds.
        wrapper.style.width = `${ts * 0.7}px`;
        const [_rtw, rth] = this.layout.tileSizes.riverTile;
        const riverH = 3 * rth + 2; // 3 rows + 2 gaps
        const riverScale = 1.35;
        const halfRiverExtent = (riverH * riverScale) / 2;

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

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        // Hand tiles (left side from player's perspective)
        const handDiv = document.createElement('div');
        handDiv.className = 'opp-tiles-inner';
        // Move only the hand along its edge: right side up, opposite left, left side down.
        handDiv.style.marginLeft = `${4 * (tw + 1)}px`;
        const drawIndex = drawnTileIndex(player, playerIdx, state);
        const ownWaits = new Set((player.waits ?? []).map(normalize));
        player.hand.forEach((t, idx) => {
            const tile = document.createElement('div');
            tile.className = 'opp-tile';
            this.setTile3D(tile, t, tw);
            const isDrawnTile = idx === drawIndex;
            // Use the same proportional gap in this seat's local direction.
            if (isDrawnTile) tile.style.marginLeft = `${(tw * 28) / this.layout.tileSizes.ownTile[0]}px`;
            if (activeWaits.has(normalize(t)) || (isDrawnTile && ownWaits.has(normalize(t)))) {
                this.addTile3DOverlay(tile, 'rgba(255, 0, 0, 0.4)', '10');
            }
            handDiv.appendChild(tile);
        });
        if (state.playerCount === 3 && player.kitaCount > 0) {
            const kita = this.renderKita(player.kitaCount);
            // Anchor to the fixed table edge, not the hand's changing tile count.
            // Center-relative height also keeps the position stable when melds stack.
            kita.style.right = relIndex === 2 ? '0px' : `${4 * (tw + 1)}px`;
            kita.style.top = `calc(50% - ${1.5 * th + 45}px)`;
            kita.style.bottom = 'auto';
            wrapper.appendChild(kita);
        }
        wrapper.appendChild(handDiv);

        // Melds (right side from player's perspective)
        if (player.melds.length > 0) {
            const meldsDiv = document.createElement('div');
            meldsDiv.className = 'opp-melds-inner';

            player.melds.forEach((m) => {
                const mGroup = document.createElement('div');
                mGroup.className = 'opp-meld-group';

                const rel = relativeSeat(state, m.from, playerIdx);
                const tiles = [...m.tiles];

                const addWaitHighlight = (tileEl: HTMLElement, t: string) => {
                    if (activeWaits.size > 0 && activeWaits.has(normalize(t))) {
                        this.addTile3DOverlay(tileEl, 'rgba(255, 0, 0, 0.4)', '10');
                    }
                };

                const addUpright = (t: string) => {
                    const d = document.createElement('div');
                    d.className = 'opp-tile';
                    this.setTile3D(d, t, tw);
                    addWaitHighlight(d, t);
                    mGroup.appendChild(d);
                };
                const addRotated = (t: string, added?: string) => {
                    const d = document.createElement('div');
                    d.className = 'opp-tile-rotated';
                    this.setTile3D(d, t, tw);
                    addWaitHighlight(d, t);
                    if (added) {
                        const pair = document.createElement('div');
                        pair.className = 'kakan-pair-3d';
                        const extra = document.createElement('div');
                        extra.className = 'opp-tile-rotated';
                        this.setTile3D(extra, added, tw);
                        addWaitHighlight(extra, added);
                        pair.append(extra, d);
                        mGroup.appendChild(pair);
                    } else {
                        mGroup.appendChild(d);
                    }
                };

                if (m.type === 'ankan') {
                    tiles.forEach((t, i) => {
                        const tileId = i === 0 || i === 3 ? 'back' : t;
                        addUpright(tileId);
                    });
                } else if (m.type === 'kakan') {
                    // Kakan: tiles = [consumed0, consumed1, stolen, added]
                    const added = tiles.pop()!;
                    const stolen = tiles.pop()!;
                    const consumed = tiles;

                    if (rel === 1) {
                        consumed.forEach((t) => addUpright(t));
                        addRotated(stolen, added);
                    } else if (rel === 3) {
                        addRotated(stolen, added);
                        consumed.forEach((t) => addUpright(t));
                    } else {
                        if (consumed.length >= 2) {
                            addUpright(consumed[0]);
                            addRotated(stolen, added);
                            addUpright(consumed[1]);
                        } else {
                            consumed.forEach((t) => addUpright(t));
                            addRotated(stolen, added);
                        }
                    }
                } else {
                    // Pon / Chi / Daiminkan
                    const stolen = tiles.pop()!;
                    const consumed = tiles;

                    if (rel === 1) {
                        consumed.forEach((t) => addUpright(t));
                        addRotated(stolen);
                    } else if (rel === 3) {
                        addRotated(stolen);
                        consumed.forEach((t) => addUpright(t));
                    } else {
                        if (consumed.length >= 3) {
                            // daiminkan from front: [c0, stolen_rot, c1, c2]
                            addUpright(consumed[0]);
                            addRotated(stolen);
                            addUpright(consumed[1]);
                            addUpright(consumed[2]);
                        } else if (consumed.length >= 2) {
                            addUpright(consumed[0]);
                            addRotated(stolen);
                            addUpright(consumed[1]);
                        } else {
                            consumed.forEach((t) => addUpright(t));
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
    // Extracted North stays face up beside the hand, independently of hand visibility.
    // =========================================================================
    private renderKita(count: number): HTMLElement {
        const group = document.createElement('div');
        group.className = 'kita-display-3d';
        group.setAttribute('role', 'img');
        group.setAttribute('aria-label', `${this.i18n.text('Pei')} ×${count}`);
        if (count > 1) {
            const label = document.createElement('span');
            label.className = 'kita-count-3d';
            label.textContent = `x${count}`;
            group.appendChild(label);
        }
        const tile = document.createElement('div');
        tile.className = 'opp-tile kita-tile-3d';
        this.setTile3D(tile, 'N', this.layout.tileSizes.opponentTile[0]);
        group.appendChild(tile);
        return group;
    }

    // =========================================================================
    // Own melds lie on the near-right tabletop and share its camera.
    // =========================================================================
    private renderOwnMelds(player: PlayerState, playerIdx: number, state: BoardState): HTMLElement {
        const tw = this.layout.tileSizes.opponentTile[0];
        const wrapper = document.createElement('div');
        wrapper.className = 'own-melds-3d';

        const meldsDiv = document.createElement('div');
        meldsDiv.className = 'opp-melds-inner own-melds-inner';

        player.melds.forEach((m) => {
            const mGroup = document.createElement('div');
            mGroup.className = 'opp-meld-group';

            const rel = relativeSeat(state, m.from, playerIdx);
            const tiles = [...m.tiles];

            const addUpright = (t: string) => {
                const d = document.createElement('div');
                d.className = 'opp-tile';
                this.setTile3D(d, t, tw);
                mGroup.appendChild(d);
            };
            const addRotated = (t: string, added?: string) => {
                const d = document.createElement('div');
                d.className = 'opp-tile-rotated';
                this.setTile3D(d, t, tw);
                if (added) {
                    const pair = document.createElement('div');
                    pair.className = 'kakan-pair-3d';
                    const extra = document.createElement('div');
                    extra.className = 'opp-tile-rotated';
                    this.setTile3D(extra, added, tw);
                    pair.append(extra, d);
                    mGroup.appendChild(pair);
                } else {
                    mGroup.appendChild(d);
                }
            };

            if (m.type === 'ankan') {
                tiles.forEach((t, i) => {
                    const tileId = i === 0 || i === 3 ? 'back' : t;
                    addUpright(tileId);
                });
            } else if (m.type === 'kakan') {
                const added = tiles.pop()!;
                const stolen = tiles.pop()!;
                const consumed = tiles;

                if (rel === 1) {
                    consumed.forEach((t) => addUpright(t));
                    addRotated(stolen, added);
                } else if (rel === 3) {
                    addRotated(stolen, added);
                    consumed.forEach((t) => addUpright(t));
                } else {
                    if (consumed.length >= 2) {
                        addUpright(consumed[0]);
                        addRotated(stolen, added);
                        addUpright(consumed[1]);
                    } else {
                        consumed.forEach((t) => addUpright(t));
                        addRotated(stolen, added);
                    }
                }
            } else {
                const stolen = tiles.pop()!;
                const consumed = tiles;

                if (m.type === 'daiminkan') {
                    if (rel === 1) {
                        consumed.forEach((t) => addUpright(t));
                        addRotated(stolen);
                    } else if (rel === 3) {
                        addRotated(stolen);
                        consumed.forEach((t) => addUpright(t));
                    } else {
                        if (consumed.length >= 3) {
                            addUpright(consumed[0]);
                            addRotated(stolen);
                            addUpright(consumed[1]);
                            addUpright(consumed[2]);
                        } else {
                            consumed.forEach((t) => addUpright(t));
                            addRotated(stolen);
                        }
                    }
                } else {
                    if (rel === 1) {
                        consumed.forEach((t) => addUpright(t));
                        addRotated(stolen);
                    } else if (rel === 3) {
                        addRotated(stolen);
                        consumed.forEach((t) => addUpright(t));
                    } else if (rel === 2) {
                        if (consumed.length >= 2) {
                            addUpright(consumed[0]);
                            addRotated(stolen);
                            addUpright(consumed[1]);
                        } else {
                            consumed.forEach((t) => addUpright(t));
                            addRotated(stolen);
                        }
                    } else {
                        consumed.forEach((t) => addUpright(t));
                        addRotated(stolen);
                    }
                }
            }
            meldsDiv.appendChild(mGroup);
        });
        wrapper.appendChild(meldsDiv);

        return wrapper;
    }

    // =========================================================================
    // Own hand (flat, bottom layer)
    // =========================================================================
    private renderOwnHand(
        player: PlayerState,
        vpIdx: number,
        state: BoardState,
        _pc: number,
        activeWaits: Set<string>,
    ): HTMLElement {
        const [tw, th] = this.layout.tileSizes.ownTile;
        const handArea = document.createElement('div');
        handArea.className = 'own-hand-area-3d';

        // Closed hand
        const tilesDiv = document.createElement('div');
        tilesDiv.className = 'own-tiles-3d';
        // Start two tile pitches in, while preserving left alignment and the meld anchor.
        tilesDiv.style.marginLeft = `${2 * (tw + 2)}px`;

        const normalize = (t: string) => t.replace('0', '5').replace('r', '');

        const ownWaits = new Set((player.waits ?? []).map(normalize));

        const drawIndex = drawnTileIndex(player, vpIdx, state);
        const shouldAnimate = state.lastEvent?.type === 'tsumo';

        player.hand.forEach((t, idx) => {
            const tDiv = document.createElement('div');
            tDiv.className = 'own-tile-3d';
            tDiv.style.width = `${tw}px`;
            tDiv.style.height = `${th}px`;
            tDiv.title = t;
            tDiv.classList.toggle('tile-face-down', t === '?' || t === 'back');
            tDiv.appendChild(TileRenderer.getTileElement(t === '?' ? 'back' : t));

            // Tsumo tile separation
            const isDrawnTile = idx === drawIndex;
            if (isDrawnTile) {
                tDiv.style.marginLeft = '28px';
                if (shouldAnimate) tDiv.classList.add('tsumo-anim-3d');
            }

            // Highlight
            if (activeWaits.size > 0 || (isDrawnTile && ownWaits.size > 0)) {
                const normT = normalize(t);
                if (activeWaits.has(normT) || (isDrawnTile && ownWaits.has(normT))) {
                    const overlay = document.createElement('div');
                    Object.assign(overlay.style, {
                        position: 'absolute',
                        top: '0',
                        left: '0',
                        width: '100%',
                        height: '100%',
                        backgroundColor: 'rgba(255, 0, 0, 0.4)',
                        zIndex: '10',
                        pointerEvents: 'none',
                        borderRadius: '4px',
                    });
                    tDiv.appendChild(overlay);
                }
            }

            tilesDiv.appendChild(tDiv);
        });
        handArea.appendChild(tilesDiv);

        return handArea;
    }

    // =========================================================================
    // Player identity panel (UI overlay; scores live in the center display)
    // =========================================================================
    private renderPlayerPanel(playerIdx: number, relIndex: number, state: BoardState): HTMLElement {
        const panel = document.createElement('button');
        panel.type = 'button';
        panel.dataset.viewpoint = String(playerIdx);
        panel.className = 'player-panel-3d';
        panel.disabled = !this.onViewpointChange;
        panel.classList.toggle('is-viewpoint', playerIdx === this.viewpoint);
        panel.setAttribute('aria-pressed', String(playerIdx === this.viewpoint));
        const name = state.playerNames[playerIdx] || `Player${playerIdx}`;
        panel.setAttribute('aria-label', this.i18n.text('View from {name}', { name }));
        panel.title = this.i18n.text('View from {name}', { name });

        // Position — corners and edges
        const panelPositions: { [key: number]: { [k: string]: string } } = {
            0: { bottom: '120px', left: '260px' },
            1: { right: '40px', top: '156px' },
            2: { top: '24px', right: '300px' },
            3: { left: '32px', top: '172px' },
        };
        const pos = panelPositions[relIndex] || panelPositions[0];
        Object.assign(panel.style, pos);

        // Avatar (centered)
        const avatar = document.createElement('div');
        avatar.className = 'avatar-3d';
        const avatarImg = document.createElement('img');
        avatarImg.src = state.playerAvatars[playerIdx] || playerAvatar(name);
        avatarImg.className = 'avatar-img';
        avatarImg.alt = '';
        avatar.appendChild(avatarImg);
        panel.appendChild(avatar);

        // Player name
        const playerName = document.createElement('div');
        playerName.className = 'player-name';
        playerName.textContent = name;
        panel.appendChild(playerName);

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

        let callCssClass: string | undefined;

        if (evt.actor !== undefined) {
            const type = evt.type;
            const callDef = CALL_TYPES[type];
            if (callDef) {
                label = this.i18n.call(evt.type);
                callCssClass = callDef.cssClass;
                actorIdx = evt.actor;
            } else if (type === 'hora') {
                label = this.i18n.text(evt.target === evt.actor ? 'Tsumo' : 'Ron');
                callCssClass = 'call-hora';
                actorIdx = evt.actor;
            }
        }

        if (evt.type === 'ryukyoku') {
            label = this.i18n.text('Ryukyoku');
        }

        if (label) {
            const el = document.createElement('div');
            el.className = 'call-overlay-3d';
            if (callCssClass) el.classList.add(callCssClass);
            el.textContent = label;

            if (actorIdx !== undefined) {
                const relIndex = relativeSeat(state, actorIdx, this.viewpoint);
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
        const waitPositions: { [key: number]: { [k: string]: string } } = {
            0: { bottom: '110px', left: '384px' },
            1: { right: '40px', top: '254px' },
            2: { top: '122px', right: '380px' },
            3: { left: '32px', top: '270px' },
        };
        Object.assign(el.style, waitPositions[relIndex] || waitPositions[0]);

        const label = document.createElement('span');
        label.textContent = this.i18n.text('Wait:');
        label.style.marginRight = '4px';
        el.appendChild(label);

        waits.forEach((w) => {
            const tile = document.createElement('div');
            tile.className = 'wait-tile-3d';
            tile.appendChild(TileRenderer.getTileElement(w));
            el.appendChild(tile);
        });

        return el;
    }
}
