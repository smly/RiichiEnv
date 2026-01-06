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
    private styleElement: HTMLStyleElement;
    viewpoint: number = 0;

    constructor(container: HTMLElement) {
        this.container = container;

        let style = document.getElementById('riichienv-viewer-style') as HTMLStyleElement;
        if (!style) {
            style = document.createElement('style');
            style.id = 'riichienv-viewer-style';
            style.textContent = `
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
                    background-color: #fdfdfd; 
                    border-radius: 4px;
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                }
                .tile-fg { z-index: 2; }
                
                .active-player-highlight {
                    box-shadow: 0 0 20px 5px rgba(255, 230, 0, 0.4);
                    background-color: rgba(255, 230, 0, 0.05);
                    border-radius: 12px;
                }
                
                .river-grid {
                    display: grid;
                    grid-template-columns: repeat(6, 34px);
                    grid-template-rows: repeat(3, 46px);
                    gap: 2px;
                    width: 214px; 
                    height: 142px;
                    justify-content: start; 
                    align-content: start;
                }
                
                .tile-rotated {
                    transform: rotate(90deg) scale(0.9);
                    transform-origin: center center;
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

                .modal-overlay {
                    position: absolute;
                    top: 50%; left: 50%;
                    transform: translate(-50%, -50%);
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid #555;
                    z-index: 200;
                    text-align: center;
                    min-width: 400px;
                    box-shadow: 0 0 30px rgba(0,0,0,0.8);
                    animation: fadeIn 0.3s;
                }
                .modal-title { font-size: 1.5em; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #777; padding-bottom: 5px; }
                .modal-content { text-align: left; }
                .yaku-list { margin: 10px 0; padding-left: 20px; }
                .score-display { font-size: 1.2em; text-align: center; margin-top: 15px; font-weight: bold; background: #333; padding: 5px; border-radius: 4px;}

                .riichi-stick {
                    width: 100px;
                    height: 8px;
                    background: #eee;
                    border-radius: 4px;
                    border: 1px solid #999;
                    position: relative;
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                }
                .riichi-stick::after {
                    content: "";
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    width: 6px;
                    height: 6px;
                    background: red;
                    border-radius: 50%;
                    transform: translate(-50%, -50%);
                }

                .debug-panel {
                    position: fixed;
                    bottom: 10px;
                    right: 10px;
                    background: rgba(0,0,0,0.8);
                    color: #0f0;
                    font-family: monospace;
                    padding: 10px;
                    border-radius: 4px;
                    font-size: 11px;
                    max-width: 300px;
                    max-height: 200px;
                    overflow: auto;
                    z-index: 1000;
                    pointer-events: none;
                    white-space: pre-wrap;
                }

                @keyframes popIn {
                    from { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
                    to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
                }
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            `;
            document.head.appendChild(style);
        }
        this.styleElement = style;
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

    render(state: BoardState) {
        // Clear container
        this.container.innerHTML = '';

        // Main board layout
        const board = document.createElement('div');
        board.className = 'mahjong-board';
        Object.assign(board.style, {
            position: 'relative',
            width: '100%',
            aspectRatio: '1/1',
            maxWidth: '900px',
            margin: '0 auto',
            backgroundColor: '#2d5a27',
            borderRadius: '12px',
            overflow: 'hidden',
            fontSize: '14px',
            color: 'white',
            fontFamily: 'sans-serif',
            boxSizing: 'border-box'
        });

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
            minWidth: '120px'
        });

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
                height: 'auto',
                minHeight: '220px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                transform: 'translateY(140px)',
                transition: 'background-color 0.3s',
                position: 'relative'
            });

            if (i === state.currentActor) {
                pDiv.classList.add('active-player-highlight');
                pDiv.style.padding = '10px';
            } else {
                pDiv.style.padding = '10px';
            }

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

            // Info Layer
            const infoDiv = document.createElement('div');
            const winds = ['E', 'S', 'W', 'N'];
            const windLabel = winds[p.wind];
            const isOya = (p.wind === 0);

            let label = `P${i}`;
            infoDiv.innerHTML = `
                <span style="font-weight:bold; font-size:1.1em; margin-right:8px; color: ${isOya ? '#ff4d4d' : 'white'};">
                    ${windLabel}
                </span>
                <span style="font-weight:bold; font-size:1.1em; margin-right:8px;">${label}</span>
                <span style="font-family:monospace; font-size:1.3em;">${p.score}</span>
            `;

            if (p.riichi) {
                infoDiv.style.color = '#ff6b6b';
                infoDiv.innerHTML += ' <span style="font-weight:bold; border:2px solid currentColor; padding:0 4px; border-radius:4px; font-size:0.9em; margin-left:8px;">REACH</span>';
            }
            infoDiv.style.marginBottom = '8px';
            infoDiv.style.textShadow = '1px 1px 2px #000';

            // Riichi Stick
            if (p.riichi) {
                const stick = document.createElement('div');
                stick.className = 'riichi-stick';
                stick.style.marginBottom = '10px';
                pDiv.appendChild(stick);
            }

            // River
            const riverDiv = document.createElement('div');
            riverDiv.className = 'river-grid';
            p.discards.forEach(d => {
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
                riverDiv.appendChild(cell);
            });

            // Hand
            const handDiv = document.createElement('div');
            handDiv.style.display = 'flex';
            handDiv.style.alignItems = 'flex-end';
            handDiv.style.marginTop = '15px';
            handDiv.style.height = '56px';

            const tilesDiv = document.createElement('div');
            Object.assign(tilesDiv.style, {
                display: 'flex', width: '570px', height: '56px', justifyContent: 'flex-start'
            });

            const totalTiles = p.hand.length + p.melds.length * 3;
            const hasTsumo = (totalTiles % 3 === 2);

            p.hand.forEach((t, idx) => {
                const tDiv = document.createElement('div');
                tDiv.style.width = '40px'; tDiv.style.height = '56px';
                tDiv.innerHTML = this.getTileHtml(t);
                if (hasTsumo && idx === p.hand.length - 1) tDiv.style.marginLeft = '10px';
                tilesDiv.appendChild(tDiv);
            });
            handDiv.appendChild(tilesDiv);

            // Melds
            if (p.melds.length > 0) {
                const meldsDiv = document.createElement('div');
                meldsDiv.style.display = 'flex';
                meldsDiv.style.gap = '8px';
                meldsDiv.style.marginLeft = '10px';
                p.melds.forEach(m => {
                    const mGroup = document.createElement('div');
                    mGroup.style.display = 'flex';
                    m.tiles.forEach(t => {
                        const mtDiv = document.createElement('div');
                        mtDiv.style.width = '34px'; mtDiv.style.height = '46px';
                        mtDiv.innerHTML = this.getTileHtml(t);
                        mGroup.appendChild(mtDiv);
                    });
                    meldsDiv.appendChild(mGroup);
                });
                handDiv.appendChild(meldsDiv);
            }

            pDiv.appendChild(infoDiv);
            pDiv.appendChild(riverDiv);
            pDiv.appendChild(handDiv);
            wrapper.appendChild(pDiv);
            board.appendChild(wrapper);
        });

        // End Kyoku Modal
        if (state.lastEvent && state.lastEvent.type === 'end_kyoku' && state.lastEvent.meta && state.lastEvent.meta.results) {
            const results = state.lastEvent.meta.results;
            const modal = document.createElement('div');
            modal.className = 'modal-overlay';
            Object.assign(modal.style, {
                maxHeight: '80vh',
                overflowY: 'auto',
                width: '450px'
            });

            let combinedHtml = `<div class="modal-title">End Kyoku</div>`;
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
                    <div class="result-item" style="margin-bottom: 20px; ${idx > 0 ? 'border-top: 1px solid #555; padding-top: 15px;' : ''}">
                        <div style="font-weight: bold; margin-bottom: 5px; color: #ffd700; font-size: 1.2em;">
                            P${actor} ${isTsumo ? 'Tsumo' : 'Ron from P' + target}
                        </div>
                        <div class="modal-content">
                            <ul class="yaku-list" style="columns: 2;">${yakuListHtml}</ul>
                            <div style="display:flex; justify-content:space-between; margin-top:10px; font-weight:bold;">
                                <span>${score.han} Han</span>
                                <span>${score.fu} Fu</span>
                            </div>
                        </div>
                        <div class="score-display">
                            ${score.points} Points
                        </div>
                    </div>
                `;
            });
            modal.innerHTML = combinedHtml;
            board.appendChild(modal);
        }

        // Debug Panel
        const debug = document.createElement('div');
        debug.className = 'debug-panel';
        const lastEvStr = state.lastEvent ? JSON.stringify(state.lastEvent) : 'null';
        debug.textContent = `Event: ${state.eventIndex} / ${state.totalEvents}\nLast Event:\n${lastEvStr}`;
        board.appendChild(debug);

        this.container.appendChild(board);
    }

    formatRound(r: number) {
        const winds = ['East', 'South', 'West', 'North'];
        return `${winds[Math.floor(r / 4)]} ${r % 4 + 1}`;
    }
}
