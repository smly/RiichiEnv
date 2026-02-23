import { TileRenderer } from './tile_renderer';
import { COLORS } from '../constants';
import { CHAR_SPRITE_BASE64, CHAR_MAP } from '../char_assets';

export class CenterRenderer {
    static renderCenter(
        state: any,
        onCenterClick: (() => void) | null,
        viewpoint: number = 0
    ): HTMLElement {
        const center = document.createElement('div');
        center.className = 'center-info';
        Object.assign(center.style, {
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: COLORS.centerInfoBackground,
            padding: '15px',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: '10',
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            width: '200px',
            height: '200px',
            boxSizing: 'border-box',
            cursor: 'pointer' // Added cursor pointer
        });

        center.onclick = (e) => {
            e.stopPropagation();
            if (onCenterClick) onCenterClick();
        };

        // Helper to create sprite icon
        const makeImg = (key: string, size: number = 26) => {
            const asset = CHAR_MAP[key];
            if (!asset) return document.createElement('div');

            // Use 36 (base font size) as reference for 100% scale
            const scale = size / 36.0;
            // Use full asset width to ensure no clipping, margins are minimal (1px) now
            // Round UP + 1px safety buffer to ensure we don't cut off sub-pixels in layout
            const scaledW = Math.ceil(asset.w * scale) + 1;

            const d = document.createElement('div');
            Object.assign(d.style, {
                width: `${asset.w + 8}px`,
                height: `${asset.h}px`,
                backgroundImage: `url(${CHAR_SPRITE_BASE64})`,
                backgroundPosition: `-${asset.x}px -${asset.y}px`,
                backgroundRepeat: 'no-repeat',
                backgroundColor: 'transparent',
                transformOrigin: 'center center'
            });
            d.style.transform = `scale(${scale})`;

            // Wrapper fits the scaled content width (tight packing)
            const w = document.createElement('div');
            Object.assign(w.style, {
                width: `${scaledW + 8}px`,
                height: `${size}px`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'visible',
                marginRight: '-8px'
            });
            w.appendChild(d);
            return w;
        };

        // 1. Render Wind Labels (Corners)
        const pc = state.playerCount || 4;
        const windMap = ['東_red', '南', '西', '北'].slice(0, pc); // Keys in CHAR_MAP
        state.players.forEach((p: any, i: number) => {
            const relPos = (i - viewpoint + pc) % pc; // 0: Bottom, 1: Right, 2: Top, 3: Left
            const windIdx = p.wind; // 0: East, 1: South, ...
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
                transformOrigin: 'center center'
            });

            const targetSize = 26;
            const maxDim = Math.max(asset.w, asset.h);
            const scale = Math.min(1, targetSize / maxDim);

            let rotation = '0deg';
            if (relPos === 1) rotation = '-90deg';
            else if (relPos === 2) rotation = '180deg';
            else if (relPos === 3) rotation = '90deg';

            icon.style.transform = `rotate(${rotation}) scale(${scale})`;

            // Positioning Logic
            if (relPos === 0) { // Bottom
                icon.style.bottom = '8px';
                icon.style.left = '8px';
            } else if (relPos === 1) { // Right
                icon.style.right = '8px';
                icon.style.bottom = '8px';
            } else if (relPos === 2) { // Top
                icon.style.top = '8px';
                icon.style.right = '8px';
            } else if (relPos === 3) { // Left
                icon.style.left = '8px';
                icon.style.top = '8px';
            }
            center.appendChild(icon);
        });

        // Helper to render score row (Text version)
        const makeScoreRow = (score: number) => {
            const row = document.createElement('div');
            row.innerText = score.toString();
            Object.assign(row.style, {
                fontFamily: 'monospace',
                fontSize: '16px',
                fontWeight: 'bold',
                color: '#ffdd00', // Yellow text
                textAlign: 'center',
                whiteSpace: 'nowrap'
            });
            return row;
        };

        // Render Scores (Edges)
        state.players.forEach((p: any, i: number) => {
            const relPos = (i - viewpoint + pc) % pc;
            const scoreRow = makeScoreRow(p.score);

            Object.assign(scoreRow.style, {
                position: 'absolute',
                zIndex: '11'
            });

            if (relPos === 0) { // Bottom (Self)
                scoreRow.style.bottom = '20px';
                scoreRow.style.left = '50%';
                scoreRow.style.transform = 'translate(-50%, 0)';
            } else if (relPos === 1) { // Right (Shimocha)
                scoreRow.style.right = '26px';
                scoreRow.style.top = '50%';
                scoreRow.style.transform = 'translate(50%, -50%) rotate(-90deg)';
                scoreRow.style.transformOrigin = 'center center';
            } else if (relPos === 2) { // Top (Toimen)
                scoreRow.style.top = '20px';
                scoreRow.style.left = '50%';
                scoreRow.style.transform = 'translate(-50%, 0) rotate(180deg)';
            } else if (relPos === 3) { // Left (Kamicha)
                scoreRow.style.left = '26px';
                scoreRow.style.top = '50%';
                scoreRow.style.transform = 'translate(-50%, -50%) rotate(90deg)';
            }
            center.appendChild(scoreRow);

            // Riichi Stick
            if (p.riichi) {
                const stick = document.createElement('div');
                Object.assign(stick.style, {
                    position: 'absolute',
                    width: '100px',
                    height: '8px',
                    backgroundColor: 'white',
                    borderRadius: '4px',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: '12'
                });

                const dot = document.createElement('div');
                Object.assign(dot.style, {
                    width: '6px',
                    height: '6px',
                    backgroundColor: '#d00',
                    borderRadius: '50%'
                });
                stick.appendChild(dot);

                // Position relative to Center Info
                // We place it slightly outside the box, towards the player
                const offset = '10px'; // pushes it out by 10px

                if (relPos === 0) { // Bottom
                    stick.style.bottom = offset;
                    stick.style.left = '50%';
                    stick.style.transform = 'translate(-50%, 0)';
                } else if (relPos === 1) { // Right
                    stick.style.right = offset;
                    stick.style.top = '50%';
                    stick.style.transform = 'translate(50%, -50%) rotate(90deg)';
                } else if (relPos === 2) { // Top
                    stick.style.top = offset;
                    stick.style.left = '50%';
                    stick.style.transform = 'translate(-50%, 0)'; // No rotation needed for bar, it's symmetric. But if dot placement matters? Dot is centered.
                } else if (relPos === 3) { // Left
                    stick.style.left = offset;
                    stick.style.top = '50%';
                    stick.style.transform = 'translate(-50%, -50%) rotate(90deg)';
                }
                center.appendChild(stick);
            }
        });

        // 2. Center Content Container
        const contentContainer = document.createElement('div');
        Object.assign(contentContainer.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '2px'
        });

        // Row 1: [RoundWind] [RoundNum] [Kyoku] (Images)
        const row1 = document.createElement('div');
        Object.assign(row1.style, {
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0px',
            marginBottom: '4px'
        });

        // Round Wind (0-3 -> E S W N)
        const roundWindNames = ['東', '南', '西', '北'];
        const rWindIdx = Math.floor(state.round / pc);
        const rWindKey = roundWindNames[rWindIdx] || '東';

        // Round Number (0-3 -> 1 2 3 4)
        const rNumIdx = state.round % pc;
        const rNumKey = ['一', '二', '三', '四'][rNumIdx] || '一';

        row1.appendChild(makeImg(rWindKey, 26));
        row1.appendChild(makeImg(rNumKey, 26));
        row1.appendChild(makeImg('局', 26));

        contentContainer.appendChild(row1);

        // Row 2: Text "{honba}, {kyotaku}"
        const row2 = document.createElement('div');
        row2.innerText = `${state.honba}, ${state.kyotaku}`;
        Object.assign(row2.style, {
            fontSize: '1.2em',
            fontWeight: 'bold',
            color: 'white',
            marginBottom: '8px',
            fontFamily: 'monospace'
        });
        contentContainer.appendChild(row2);

        // Row 3: Dora Tiles
        const row3 = document.createElement('div');
        Object.assign(row3.style, {
            display: 'flex',
            gap: '2px'
        });

        const doraTiles = [...state.doraMarkers];
        while (doraTiles.length < 5) {
            doraTiles.push('back');
        }

        doraTiles.forEach((t: string) => {
            const d = document.createElement('div');
            d.style.width = '20px';
            d.style.height = '27px';
            d.appendChild(TileRenderer.getTileElement(t));
            row3.appendChild(d);
        });

        contentContainer.appendChild(row3);

        center.appendChild(contentContainer);

        return center;
    }
}
