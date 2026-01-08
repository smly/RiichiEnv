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
                    width: 90% !important;
                    height: 90% !important;
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

                .re-modal-overlay {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100%;
                    height: 100%;
                    box-sizing: border-box;
                    background: rgba(0,0,0,0.6);
                    z-index: 9999;
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
                    max-width: 90%;
                    max-height: 80%;
                    min-width: 400px;
                    overflow-y: auto;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.8);
                    border: 1px solid #2a4d25;
                    text-align: left;
                }
                .re-yaku-list { margin: 10px 0; padding-left: 20px; columns: 2; }
                .re-score-display { iconfont-size: 1.2em; text-align: center; margin-top: 15px; font-weight: bold; background: #333; padding: 5px; border-radius: 4px;}

                .re-kyoku-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                }
                .re-kyoku-table th, .re-kyoku-table td {
                    border: 1px solid #2a4d25;
                    padding: 8px;
                    text-align: center;
                }
                .re-kyoku-table th {
                    background-color: #1a3317;
                    position: sticky;
                    top: 0;
                }
                .re-kyoku-row:hover {
                    background-color: #2a4d25;
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
            if (state.lastEvent && state.lastEvent.actor === i) {
                let label = '';
                const type = state.lastEvent.type;
                if (['chi', 'pon', 'kan', 'ankan', 'daiminkan', 'kakan'].includes(type)) {
                    label = type.charAt(0).toUpperCase() + type.slice(1);
                    if (type === 'daiminkan') label = 'Kan';
                } else if (type === 'hora') {
                    label = (state.lastEvent.target === state.lastEvent.actor) ? 'Tsumo' : 'Ron';
                }

                if (label) {
                    const overlay = document.createElement('div');
                    overlay.className = 'call-overlay';
                    overlay.textContent = label;
                    pDiv.appendChild(overlay);
                }
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
            if (p.riichi) {
                const stick = document.createElement('div');
                stick.className = 'riichi-stick';
                Object.assign(stick.style, {
                    position: 'absolute',
                    top: '160px', // Adjusted position
                    left: '50%',
                    transform: 'translateX(-50%)',
                    marginBottom: '0'
                });
                pDiv.appendChild(stick);
            }

            // Hand & Melds Area
            const handArea = document.createElement('div');
            Object.assign(handArea.style, {
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-end',
                width: '580px',
                height: '56px',
                position: 'absolute',
                bottom: '10px',
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
                gap: '8px',
                alignItems: 'flex-end'
            });

            if (p.melds.length > 0) {
                p.melds.forEach(m => {
                    const mGroup = document.createElement('div');
                    mGroup.style.display = 'flex';
                    mGroup.style.marginLeft = '10px';
                    m.tiles.forEach(t => {
                        const mtDiv = document.createElement('div');
                        mtDiv.style.width = '30px'; mtDiv.style.height = '42px';
                        mtDiv.innerHTML = this.getTileHtml(t);
                        mGroup.appendChild(mtDiv);
                    });
                    meldsDiv.appendChild(mGroup);
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
                width: '450px'
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

    formatRound(r: number) {
        const winds = ['East', 'South', 'West', 'North'];
        return `${winds[Math.floor(r / 4)]} ${r % 4 + 1}`;
    }
}
