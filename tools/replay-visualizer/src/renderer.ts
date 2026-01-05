import { BoardState, PlayerState } from './types';
import { TILES } from './tiles';

export class Renderer {
    container: HTMLElement;
    private styleElement: HTMLStyleElement; // Store reference to the style element

    constructor(container: HTMLElement) {
        this.container = container;

        // Inject styles if not present
        if (!document.getElementById('riichienv-viewer-style')) {
            const style = document.createElement('style');
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
                    background-color: #fdfdfd; /* Fallback if Font.svg missing/transparent */
                    border-radius: 3px/5px; /* Slight rounded corners */
                    box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                }
                .tile-fg { z-index: 2; }
            `;
            // Append to head to be shared or container? Helper to append to container is safer for encapsulation
            // But checking ID implies global. 
            // Let's append to container but class-scope check?
            // User might have multiple viewers. Styles should be global or scoped.
            // Using ID implies global.
            // Let's stick to container-scoped but robust.
            this.container.appendChild(style);
            this.styleElement = style;
        }
    }

    getTileHtml(tileStr: string): string {
        if (tileStr === 'back') {
            const svg = TILES['back'] || TILES['blank'];
            return `<div class="tile-layer"><div class="tile-bg">${svg}</div></div>`;
        }

        const frontSvg = TILES['front'] || ''; // Front SVG might be just the shading/face
        let fgSvg = TILES[tileStr];

        if (!fgSvg) {
            // Mapping check: 0m -> 5mr handled in gen_tiles
            console.warn(`Missing SVG for tile: ${tileStr}`);
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
        // Clear container (except style)
        const children = Array.from(this.container.children);
        children.forEach(c => {
            if (c.tagName !== 'STYLE') this.container.removeChild(c);
        });

        // Ensure style is there
        if (this.styleElement && !this.container.contains(this.styleElement)) {
            this.container.appendChild(this.styleElement);
        }

        // Main board layout
        const board = document.createElement('div');
        board.className = 'mahjong-board';
        Object.assign(board.style, {
            position: 'relative',
            width: '100%',
            aspectRatio: '1/1',
            maxWidth: '800px',
            margin: '0 auto',
            backgroundColor: '#2d5a27',
            borderRadius: '8px',
            overflow: 'hidden', // Clip river if it gets too wild?
            fontSize: '12px',
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
            padding: '10px',
            borderRadius: '5px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '10'
        });

        center.innerHTML = `
            <div>${this.formatRound(state.round)} - Honba: ${state.honba}</div>
            <div>Kyotaku: ${state.kyotaku}</div>
            <div style="margin-top:5px; display:flex; gap:2px;">
                Dora: ${state.doraMarkers.map(t =>
            `<div style="width:24px; height:34px;">${this.getTileHtml(t)}</div>`
        ).join('')}
            </div>
        `;

        board.appendChild(center);

        const angles = [0, -90, 180, 90]; // P0, P1, P2, P3

        state.players.forEach((p, i) => {
            // Wrapper at center, rotated
            const wrapper = document.createElement('div');
            Object.assign(wrapper.style, {
                position: 'absolute',
                top: '50%',
                left: '50%',
                width: '0',
                height: '0',
                display: 'flex',
                justifyContent: 'center', // Horizontal center
                transform: `rotate(${angles[i]}deg)`
            });

            // Player Content Div (Pushed out from center)
            const pDiv = document.createElement('div');
            Object.assign(pDiv.style, {
                width: '600px', // Fixed width to prevent jitter
                height: 'auto',
                minHeight: '200px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                transform: 'translateY(120px)' // Push away from center
                // Center is 120px away.
                // Info will be at top of pDiv (closest to center)
                // Hand will be at bottom of pDiv (furthest)
            });

            // Info (Score) - Closest to center
            const infoDiv = document.createElement('div');
            infoDiv.textContent = `P${i}: ${p.score}`;
            if (p.riichi) infoDiv.style.color = '#ff6b6b';
            infoDiv.style.fontWeight = 'bold';
            infoDiv.style.marginBottom = '5px';
            infoDiv.style.textShadow = '1px 1px 2px black';

            // River
            const riverDiv = document.createElement('div');
            Object.assign(riverDiv.style, {
                display: 'flex',
                flexWrap: 'wrap',
                width: '150px', // 6 tiles wide approx
                minHeight: '80px',
                marginBottom: '10px',
                justifyContent: 'center'
            });

            p.discards.forEach(t => {
                const dDiv = document.createElement('div');
                dDiv.style.width = '25px';
                dDiv.style.height = '35px';
                dDiv.innerHTML = this.getTileHtml(t);
                riverDiv.appendChild(dDiv);
            });

            // Hand
            const handDiv = document.createElement('div');
            handDiv.style.display = 'flex';
            handDiv.style.height = '42px';
            handDiv.style.justifyContent = 'center'; // Keep hand centered

            // Hand + Melds
            // We want Melds to be on the right side of the hand?
            // Usually [Hand] [Agari?] [Melds]
            // For simplicity: [Hand tiles] [Melds]
            const tilesDiv = document.createElement('div');
            tilesDiv.style.display = 'flex';
            p.hand.forEach(t => {
                const tDiv = document.createElement('div');
                tDiv.style.width = '30px';
                tDiv.style.height = '42px';
                tDiv.innerHTML = this.getTileHtml(t);
                tilesDiv.appendChild(tDiv);
            });

            handDiv.appendChild(tilesDiv);

            if (p.melds.length > 0) {
                const meldsDiv = document.createElement('div');
                meldsDiv.style.display = 'flex';
                meldsDiv.style.marginLeft = '10px';
                p.melds.forEach(m => {
                    const mGroup = document.createElement('div');
                    mGroup.style.display = 'flex';
                    mGroup.style.gap = '0';
                    m.tiles.forEach(t => {
                        const mtDiv = document.createElement('div');
                        mtDiv.style.width = '30px';
                        mtDiv.style.height = '42px';
                        mtDiv.innerHTML = this.getTileHtml(t);
                        mGroup.appendChild(mtDiv);
                    });
                    meldsDiv.appendChild(mGroup);
                });
                handDiv.appendChild(meldsDiv);
            }

            // Append in order (Center -> Edge)
            pDiv.appendChild(infoDiv);
            pDiv.appendChild(riverDiv);
            pDiv.appendChild(handDiv);

            wrapper.appendChild(pDiv);
            board.appendChild(wrapper);
        });

        this.container.appendChild(board);
    }

    formatRound(r: number) {
        const winds = ['East', 'South', 'West', 'North'];
        return `${winds[Math.floor(r / 4)]} ${r % 4 + 1}`;
    }
}

