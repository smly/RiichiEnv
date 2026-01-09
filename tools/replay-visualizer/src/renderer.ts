import { BoardState, PlayerState } from './types';
import { TILES } from './tiles';

const YAKU_MAP: { [key: number]: string } = {
    1: "Menzen Tsumo", 2: "Riichi", 3: "Chankan", 4: "Rinshan Kaihou", 5: "Haitei Raoyue", 6: "Houtei Raoyui",
    7: "Haku", 8: "Hatsu", 9: "Chun", 10: "Jikaze (Seat Wind)", 11: "Bakaze (Prevalent Wind)",
    12: "Tanyao", 13: "Iipeiko", 14: "Pinfu", 15: "Chanta", 16: "Ittsu", 17: "Sanshoku Doujun",
    18: "Double Riichi", 19: "Sanshoku Doukou", 20: "Sankantsu", 21: "Toitoi", 22: "San Ankou",
    23: "Shousangen", 24: "Honroutou", 25: "Chiitoitsu", 26: "Junchan", 27: "Honitsu",
    28: "Ryanpeiko", 29: "Chinitsu", 30: "Ippatsu", 31: "Dora", 32: "Akadora", 33: "Ura Dora",
    35: "Tenhou", 36: "Chiihou", 37: "Daisangen", 38: "Suuankou", 39: "Tsuu Iisou",
    40: "Ryuu Iisou", 41: "Chinroutou", 42: "Kokushi Musou", 43: "Shousuushii", 44: "Suukantsu",
    45: "Chuuren Poutou", 47: "Junsei Chuuren Poutou", 48: "Suuankou Tanki", 49: "Kokushi Musou 13-wait",
    50: "Daisuushii"
};

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
                .mahjong-board {
                    position: relative;
                    width: 100%;
                    aspect-ratio: 1/1;
                    max-width: 900px;
                    margin: 0 auto;
                    background-color: #2d5a27;
                    border-radius: 12px;
                    overflow: hidden;
                    font-size: 14px;
                    color: white;
                    font-family: sans-serif;
                    box-sizing: border-box;
                }
                .mahjong-board svg { width: 100%; height: 100%; display: block; }
                .mahjong-board .center-info svg { width: auto; height: 100%; }
                
                .tile-layer {
                    position: relative;
                    width: 100%; 
                    height: 100%;
                }
                .tile-bg, .tile-fg {
                    position: absolute;
                    top: 0; 
                    left: 0;
                    width: 100%;
                    height: 100%;
                }
                .tile-bg { 
                    z-index: 1; 
                    border-radius: 4px;
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                }
                .tile-fg { 
                    z-index: 2; 
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .tile-fg svg {
                    width: 85% !important;
                    height: 85% !important;
                }
                
                @keyframes blink-yellow {
                    0% { opacity: 1; }
                    50% { opacity: 0.4; }
                    100% { opacity: 1; }
                }

                .active-player-bar {
                    height: 4px;
                    width: 100%;
                    background-color: #ffd700;
                    margin-top: 4px;
                    border-radius: 2px;
                    animation: blink-yellow 1s infinite;
                    box-shadow: 0 0 5px #ffd700;
                }
                
                .river-container {
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    width: 214px; /* Fixed width: 6 * 34px + 5 * 2px = 214px */
                    height: auto;
                    min-height: 142px; /* Fixed min-height: 3 * 46px + 2 * 2px = 142px */
                    justify-content: start;
                    align-content: start;
                }
                .river-row {
                    display: flex;
                    gap: 2px;
                }
                
                .tile-rotated {
                    transform: rotate(90deg) scale(0.9);
                    transform-origin: center center;
                }

                .center-info {
                    transition: background-color 0.2s;
                }
                .center-info:hover {
                    background-color: #2a4d25 !important;
                }

                .player-info-box {
                    background: rgba(0,0,0,0.6);
                    padding: 8px;
                    border-radius: 6px;
                    color: white;
                    text-align: center;
                    min-width: 80px;
                    z-index: 20;
                    margin-left: 20px;
                    transition: background-color 0.2s;
                    cursor: pointer;
                }
                .player-info-box:hover {
                    background-color: #444 !important;
                }
                .active-viewpoint {
                    border: 2px solid #aaa;
                    background: rgba(0,0,0,0.8);
                }

                .call-overlay {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 3em;
                    font-weight: bold;
                    color: white;
                    text-shadow: 0 0 5px #ff0000, 0 0 10px #000;
                    padding: 10px 30px;
                    background: rgba(0,0,0,0.6);
                    border-radius: 10px;
                    border: 2px solid white;
                    z-index: 100;
                    pointer-events: none;
                    animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                }

                .icon-btn {
                    width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: transparent;
                    color: white;
                    border: 1px solid #666;
                    border-radius: 8px;
                    cursor: pointer;
                    user-select: none;
                    transition: all 0.2s;
                    font-size: 20px;
                }
                .icon-btn:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: #999;
                }
                .icon-btn:active {
                    transform: translateY(1px);
                    background: rgba(255, 255, 255, 0.2);
                }
                .active-btn {
                    background: #2a4d25 !important;
                    border-color: #4caf50 !important;
                    box-shadow: 0 0 5px #4caf50;
                }

                .debug-panel {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 180px;
                    background: rgba(0, 0, 0, 0.85);
                    color: #0f0;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 13px;
                    padding: 15px;
                    overflow-y: auto;
                    z-index: 200;
                    display: none; /* Controlled by JS */
                    box-sizing: border-box;
                    border-bottom: 1px solid #333;
                }

                .re-modal-overlay {
                    position: absolute; /* Changed from fixed to absolute */
                    top: 0; left: 0;
                    width: 100%;
                    height: 100%;
                    box-sizing: border-box;
                    background: rgba(0,0,0,0.6);
                    z-index: 2000; /* Increased z-index */
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    animation: fadeIn 0.3s;
                }
                .re-modal-title { 
                    font-size: 1.5em; 
                    font-weight: bold; 
                    margin-bottom: 10px; 
                    border-bottom: 1px solid #777; 
                    padding-bottom: 5px; 
                }
                .re-modal-content {
                    background: #0d1f0d;
                    color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    max-width: 540px;
                    max-height: 80%;
                    min-width: 400px;
                    overflow-y: auto;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.8);
                    border: 1px solid #2a4d25;
                    text-align: left;
                }
                .re-yaku-list { margin: 10px 0; padding-left: 20px; columns: 2; }
                .re-score-display { font-size: 1.2em; text-align: center; margin-top: 15px; font-weight: bold; background: #333; padding: 5px; border-radius: 4px;}

                .re-kyoku-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                    background-color: transparent !important;
                    color: white !important;
                }
                .re-kyoku-table th, .re-kyoku-table td {
                    border: 1px solid #2a4d25;
                    padding: 8px;
                    text-align: center;
                    background-color: #0d1f0d !important;
                    color: white !important;
                }
                .re-kyoku-table th {
                    background-color: #1a3317 !important;
                    position: sticky;
                    top: 0;
                    color: white !important;
                }
                .re-kyoku-row:hover td {
                    background-color: #2a4d25 !important;
                    cursor: pointer;
                }
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

    getTileHtml(tileStr: string): string {
        if (tileStr === 'back') {
            const svg = TILES['back'] || TILES['blank'];
            return `<div class="tile-layer"><div class="tile-bg">${svg}</div></div>`;
        }

        const frontSvg = TILES['front'] || '';
        let fgSvg = TILES[tileStr];
        if (!fgSvg) {
            fgSvg = TILES['blank'] || '';
        }

        return `
            <div class="tile-layer">
                <div class="tile-bg">${frontSvg}</div>
                <div class="tile-fg">${fgSvg}</div>
            </div>
        `;
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

        // Center Info
        const center = document.createElement('div');
        center.className = 'center-info';
        Object.assign(center.style, {
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: '#1a3317',
            padding: '15px',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '10',
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            minWidth: '120px',
            cursor: 'pointer' // Added cursor pointer
        });

        center.onclick = (e) => {
            e.stopPropagation();
            if (this.onCenterClick) this.onCenterClick();
        };

        center.innerHTML = `
            <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">
                ${this.formatRound(state.round)} <span style="font-size:0.8em; opacity:0.8; margin-left:5px;">Honba: ${state.honba}</span>
            </div>
            <div style="margin-bottom: 8px;">Kyotaku: ${state.kyotaku}</div>
            <div style="display:flex; align-items: center; gap: 5px;">
                <span>Dora:</span>
                <div style="display:flex; gap:2px;">
                    ${state.doraMarkers.map(t =>
            `<div style="width:28px; height:38px;">${this.getTileHtml(t)}</div>`
        ).join('')}
                </div>
            </div>
        `;

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

            // ... (Wait Indicator Logic preserved)

            // ... (River, InfoBox preserved)

            // Riichi Stick (placed between river/info and center)
            if (p.riichi) {
                const stick = document.createElement('div');
                stick.className = 'riichi-stick';
                Object.assign(stick.style, {
                    position: 'absolute',
                    top: '-15px', // Between river (y=10) and Center info
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: '100px',
                    height: '8px',
                    backgroundColor: 'white',
                    borderRadius: '4px',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: '20'
                });

                // Red Dot
                const dot = document.createElement('div');
                Object.assign(dot.style, {
                    width: '6px',
                    height: '6px',
                    backgroundColor: '#d00', // Red
                    borderRadius: '50%'
                });
                stick.appendChild(dot);

                pDiv.appendChild(stick);
            }

            // Wait Indicator Logic (Persistent)
            if (p.waits && p.waits.length > 0) {
                const wDiv = document.createElement('div');
                Object.assign(wDiv.style, {
                    position: 'absolute',
                    bottom: '240px', left: '50%', transform: 'translateX(-50%)',
                    background: 'rgba(0,0,0,0.8)', color: '#fff', padding: '5px 10px',
                    borderRadius: '4px', fontSize: '14px', zIndex: '50',
                    display: 'flex', gap: '4px', alignItems: 'center', pointerEvents: 'none'
                });
                wDiv.innerHTML = '<span style="margin-right:4px;">Wait:</span>';
                p.waits.forEach((w: string) => {
                    wDiv.innerHTML += `<div style="width:24px; height:34px;">${this.getTileHtml(w)}</div>`;
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

            // River
            const riverDiv = document.createElement('div');
            riverDiv.className = 'river-container';

            // Split discards into rows
            const rows: any[][] = [[], [], []];
            p.discards.forEach((d, idx) => {
                if (idx < 6) rows[0].push(d);
                else if (idx < 12) rows[1].push(d);
                else rows[2].push(d);
            });

            rows.forEach((rowTiles) => {
                const rowDiv = document.createElement('div');
                rowDiv.className = 'river-row';
                rowDiv.style.height = '46px';

                rowTiles.forEach(d => {
                    const cell = document.createElement('div');
                    cell.style.width = '34px';
                    cell.style.height = '46px';
                    cell.style.position = 'relative';
                    cell.style.flexShrink = '0'; // Prevent shrinking in 3rd row

                    if (d.isRiichi) {
                        const inner = document.createElement('div');
                        inner.style.width = '100%'; inner.style.height = '100%';
                        inner.className = 'tile-rotated';
                        inner.innerHTML = this.getTileHtml(d.tile);
                        cell.appendChild(inner);
                    } else {
                        cell.innerHTML = this.getTileHtml(d.tile);
                    }
                    if (d.isTsumogiri) cell.style.filter = 'brightness(0.7)';
                    rowDiv.appendChild(cell);
                });
                riverDiv.appendChild(rowDiv);
            });
            riverRow.appendChild(riverDiv);
            pDiv.appendChild(riverRow);

            // Info Box (New Overlay) - Anchored to pDiv 
            const infoBox = document.createElement('div');
            infoBox.className = 'player-info-box';
            if (i === this.viewpoint) {
                infoBox.classList.add('active-viewpoint');
            }

            // Positioning: Absolute relative to pDiv
            Object.assign(infoBox.style, {
                position: 'absolute',
                top: '10px',
                left: '50%',
                transform: 'translateX(120px)',
                marginLeft: '0'
            });

            const winds = ['E', 'S', 'W', 'N'];
            const windLabel = winds[p.wind];
            const isOya = (p.wind === 0);

            infoBox.innerHTML = `
                <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 4px; color: ${isOya ? '#ff4d4d' : 'white'};">
                    ${windLabel} P${i}
                </div>
                <div style="font-family:monospace; font-size:1.1em;">${p.score}</div>
            `;
            if (p.riichi) {
                infoBox.innerHTML += '<div style="color:#ff6b6b; font-weight:bold; font-size:0.9em; margin-top:2px;">REACH</div>';
            }

            // Blinking Bar for Active Player
            if (i === state.currentActor) {
                const bar = document.createElement('div');
                bar.className = 'active-player-bar';
                infoBox.appendChild(bar);
            }

            infoBox.onclick = (e) => {
                e.stopPropagation(); // Prevent bubbling
                if (this.onViewpointChange) {
                    this.onViewpointChange(i);
                }
            };

            pDiv.appendChild(infoBox);

            // Riichi Stick (placed below river/info, above hand)


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

            const totalTiles = p.hand.length + p.melds.length * 3;
            const hasTsumo = (totalTiles % 3 === 2);

            p.hand.forEach((t, idx) => {
                const tDiv = document.createElement('div');
                tDiv.style.width = '40px'; tDiv.style.height = '56px';
                tDiv.innerHTML = this.getTileHtml(t);
                if (hasTsumo && idx === p.hand.length - 1) tDiv.style.marginLeft = '12px';
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

            if (p.melds.length > 0) {
                p.melds.forEach(m => {
                    this.renderMeld(meldsDiv, m, i);
                });
            }
            handArea.appendChild(meldsDiv);

            pDiv.appendChild(handArea);
            wrapper.appendChild(pDiv);
            board.appendChild(wrapper);
        });

        // End Kyoku Modal
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta && state.lastEvent.meta.results) {
            const results = state.lastEvent.meta.results;
            const modal = document.createElement('div');
            modal.className = 're-modal-overlay';
            Object.assign(modal.style, {
                maxHeight: '80vh',
                overflowY: 'auto',
                width: '80vh'
            });

            let combinedHtml = `<div class="re-modal-title">End Kyoku</div>`;
            results.forEach((res: any, idx: number) => {
                const score = res.score;
                const actor = res.actor;
                const target = res.target;
                const isTsumo = (actor === target);

                const yakuListHtml = score.yaku.map((yId: number) => {
                    const name = YAKU_MAP[yId] || `Yaku ${yId}`;
                    return `<li>${name}</li>`;
                }).join('');

                combinedHtml += `
                    <div class="re-result-item" style="margin-bottom: 20px; ${idx > 0 ? 'border-top: 1px solid #555; padding-top: 15px;' : ''}">
                        <div style="font-weight: bold; margin-bottom: 5px; color: #ffd700; font-size: 1.2em;">
                            P${actor} ${isTsumo ? 'Tsumo' : 'Ron from P' + target}
                        </div>
                        <div class="re-modal-content">
                            <ul class="re-yaku-list" style="columns: 2;">${yakuListHtml}</ul>
                            <div style="display:flex; justify-content:space-between; margin-top:10px; font-weight:bold;">
                                <span>${score.han} Han</span>
                                <span>${score.fu} Fu</span>
                            </div>
                        </div>
                        <div class="re-score-display">
                            ${score.points} Points
                        </div>
                    </div>
                `;
            });
            modal.innerHTML = combinedHtml;
            board.appendChild(modal);
        }

        if (debugPanel) {
            const lastEvStr = state.lastEvent ? JSON.stringify(state.lastEvent, null, 2) : 'null';
            const text = `Event: ${state.eventIndex} / ${state.totalEvents}\nLast Event:\n${lastEvStr}`;
            if (debugPanel.textContent !== text) {
                debugPanel.textContent = text;
            }
        }
    }

    renderMeld(container: HTMLElement, m: { type: string, tiles: string[], from: number }, actor: number) {
        const mGroup = document.createElement('div');
        Object.assign(mGroup.style, {
            display: 'flex',
            alignItems: 'flex-end',
            marginLeft: '5px'
        });

        // Determine relative position of target: (target - actor + 4) % 4
        // 1: Right, 2: Front, 3: Left
        const rel = (m.from - actor + 4) % 4;

        // Tiles data in GameState is [...consumed, stolen]
        // Kakan: [...consumed(3), added] but the "stolen" one from Pon is hidden inside consumed?
        // Actually, for Kakan, GameState usually gives all 4 tiles.
        // We need to treat it as: The "stolen" one from original Pon is the one rotated.
        // The "added" one is stacked on that.
        // Simplified Logic: 
        // 1. Identify which slot is rotated based on `rel`.
        // 2. Identify if it's Kakan -> Stack the last tile on the rotated slot.
        // 3. Identify if it's Ankan -> All upright.

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
            // 4 tiles. 
            // Layout depends on `rel` (direction of original Pon).
            // Usually Kakan preserves the shape of the Pon.
            // tiles[0..2] are the original Pon (roughly). tiles[3] is the Added tile.
            // CAUTION: GameState ordering might vary but assuming [...consumed, stolen/added].
            // Let's assume the standard Pon shape uses the first 3 tiles, and the 4th is added to the rotated one.

            const added = tiles.pop()!;
            const ponTiles = tiles; // 3 remaining

            // Build Pon columns first
            // Same logic as Pon/Daiminkan but using `ponTiles`
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
            // Open Kan (4 tiles). Logic similar to Pon but with 3 consumed + 1 stolen.
            // Or 4 tiles flat.
            // User Request for Toimen: 3rd tile rotated. `[t, t, rot, t]`
            // Deduce Right: `[t, t, t, rot]`?
            // Deduce Left: `[rot, t, t, t]`?
            const stolen = tiles.pop()!;
            const consumed = tiles; // 3 remaining

            if (rel === 1) { // Right
                // [c1, c2, c3, stolen(Rot)]
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                columns.push({ tiles: [stolen], rotate: true });
            } else if (rel === 2) { // Front (Toimen)
                // Request: 3rd from left is rotated.
                // columns: [c1, c2, stolen, c3]
                if (consumed.length >= 3) {
                    columns.push({ tiles: [consumed[0]], rotate: false });
                    columns.push({ tiles: [consumed[1]], rotate: false });
                    columns.push({ tiles: [stolen], rotate: true });
                    columns.push({ tiles: [consumed[2]], rotate: false });
                } else {
                    // Fallback
                    consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                    columns.push({ tiles: [stolen], rotate: true });
                }
            } else if (rel === 3) { // Left
                // [stolen(Rot), c1, c2, c3]
                columns.push({ tiles: [stolen], rotate: true });
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
            } else {
                [...consumed, stolen].forEach(t => columns.push({ tiles: [t], rotate: false }));
            }

        } else if (m.type === 'chi') {
            // Chi: From Left (3). Stolen on Right, Rotated.
            const stolen = tiles.pop()!;
            const consumed = tiles;
            consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
            columns.push({ tiles: [stolen], rotate: true });
        } else {
            // Pon
            const stolen = tiles.pop()!;
            const consumed = tiles;
            if (rel === 1) { // Right (Shimocha)
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                columns.push({ tiles: [stolen], rotate: true });
            } else if (rel === 2) { // Front (Toimen)
                if (consumed.length >= 2) {
                    columns.push({ tiles: [consumed[0]], rotate: false });
                    columns.push({ tiles: [stolen], rotate: true });
                    columns.push({ tiles: [consumed[1]], rotate: false });
                    for (let i = 2; i < consumed.length; i++) columns.push({ tiles: [consumed[i]], rotate: false });
                } else {
                    consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
                    columns.push({ tiles: [stolen], rotate: true });
                }
            } else if (rel === 3) { // Left (Kamicha)
                columns.push({ tiles: [stolen], rotate: true });
                consumed.forEach(t => columns.push({ tiles: [t], rotate: false }));
            } else {
                [...consumed, stolen].forEach(t => columns.push({ tiles: [t], rotate: false }));
            }
        }

        // Render Columns
        columns.forEach(col => {
            const div = document.createElement('div');

            // If rotated, it takes up 42px width. If upright, 30px width.
            // Height is usually 42px.

            div.style.position = 'relative';

            if (col.rotate) {
                // Rotated Column
                div.style.width = '42px';
                div.style.height = '42px';

                // Render tiles inside. 
                // Z-index or offset needed for stacking?
                // Standard Kakan: 2nd tile is placed "above" (closer to camera / top of screen).
                // In 2D: Shift Up (negative Y)?
                // Let's just create inner divs for each tile.
                // Rotation transform: rotate(-90deg).
                // If 2 tiles, where do they go?
                // Standard: They align side-by-side (left-right) inside the 42x42 box?
                // But 42x42 is square.
                // Total height of 2 stacked simple tiles (side view) is...
                // Hand tiles are 30x42.
                // Rotated: 42 wide, 30 high.
                // If we stack 2: Total height 60? Or do they overlap?
                // Usually spread slightly.

                // Implementation:
                // Base tile at centered/standard pos.
                // Added tile shifted slightly or just next to it.
                // Since this is 2D CSS, let's just stack them using absolute positioning.

                col.tiles.forEach((t, idx) => {
                    const inner = document.createElement('div');
                    inner.innerHTML = this.getTileHtml(t);
                    Object.assign(inner.style, {
                        width: '30px',
                        height: '42px',
                        transform: 'rotate(-90deg)',
                        transformOrigin: 'center center',
                        position: 'absolute',
                        top: '5px', // Center vertically (42-30)/2 roughly + adjust
                        left: '6px' // Center horizontally (42-30)/2
                    });

                    // If stacked (idx > 0), shift it?
                    // Kakan visualization: Usually the added tile is "above" the first one (closer to camera / top of screen).
                    // In 2D: Shift Up (negative Y)?
                    if (idx > 0) {
                        inner.style.top = '-25px'; // Shift up to show it "on top"
                        // Add z-index to ensure it renders on top
                        inner.style.zIndex = '10';
                    }

                    div.appendChild(inner);
                });

            } else {
                // Upright Column
                div.style.width = '30px';
                div.style.height = '42px';
                // Render just the first tile? 
                // We only support 1 tile per upright column essentially (Ankan/Pon/Chi/Daiminkan-non-rotated parts).
                // Kakan only stacks on rotated part.
                if (col.tiles.length > 0) {
                    div.innerHTML = this.getTileHtml(col.tiles[0]);
                }
            }
            mGroup.appendChild(div);
        });

        container.appendChild(mGroup);
    }

    formatRound(r: number) {
        const winds = ['East', 'South', 'West', 'North'];
        return `${winds[Math.floor(r / 4)]} ${r % 4 + 1}`;
    }
}
