import { CHAR_MAP, CHAR_SPRITE_BASE64 } from '../char_assets';
import { COLORS } from '../constants';
import { I18n } from '../i18n/index';
import { relativeSeat } from './seat_position';
import { TileRenderer } from './tile_renderer';

export class CenterRenderer {
    static renderCenter(
        state: any,
        onCenterClick: (() => void) | null,
        viewpoint: number = 0,
        i18n = new I18n(),
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
            cursor: 'pointer', // Added cursor pointer
        });

        center.onclick = (e) => {
            e.stopPropagation();
            if (onCenterClick) onCenterClick();
        };

        // 1. Render Wind Labels (Corners)
        const pc = state.playerCount || 4;
        const windMap = ['東_red', '南', '西', '北'].slice(0, pc); // Keys in CHAR_MAP
        state.players.forEach((p: any, i: number) => {
            const relPos = relativeSeat(state, i, viewpoint); // 0: Bottom, 1: Right, 2: Top, 3: Left
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
                transformOrigin: 'center center',
            });

            const targetSize = 26;
            const maxDim = Math.max(asset.w, asset.h);
            const scale = Math.min(1, targetSize / maxDim);

            let rotation: string;

            if (relPos === 1) rotation = '-90deg';
            else if (relPos === 2) rotation = '180deg';
            else if (relPos === 3) rotation = '90deg';
            else rotation = '0deg';

            icon.style.transform = `rotate(${rotation}) scale(${scale})`;
            if (i18n.locale !== 'ja') {
                icon.style.backgroundImage = 'none';
                icon.textContent = i18n.wind(windIdx, true);
                icon.style.fontSize = '30px';
                icon.style.fontWeight = 'bold';
                icon.style.color = windIdx === 0 ? '#ff6b6b' : 'white';
            }

            // Positioning Logic

            if (relPos === 0) {
                icon.style.bottom = '8px';
                icon.style.left = '8px';
            } else if (relPos === 1) {
                icon.style.right = '8px';
                icon.style.bottom = '8px';
            } else if (relPos === 2) {
                icon.style.top = '8px';
                icon.style.right = '8px';
            } else if (relPos === 3) {
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
                whiteSpace: 'nowrap',
            });
            return row;
        };

        // Render Scores (Edges)
        state.players.forEach((p: any, i: number) => {
            const relPos = relativeSeat(state, i, viewpoint);
            const scoreRow = makeScoreRow(p.score);

            Object.assign(scoreRow.style, {
                position: 'absolute',
                zIndex: '11',
            });

            if (relPos === 0) {
                scoreRow.style.bottom = '20px';
                scoreRow.style.left = '50%';
                scoreRow.style.transform = 'translate(-50%, 0)';
            } else if (relPos === 1) {
                scoreRow.style.right = '26px';
                scoreRow.style.top = '50%';
                scoreRow.style.transform = 'translate(50%, -50%) rotate(-90deg)';
                scoreRow.style.transformOrigin = 'center center';
            } else if (relPos === 2) {
                scoreRow.style.top = '20px';
                scoreRow.style.left = '50%';
                scoreRow.style.transform = 'translate(-50%, 0) rotate(180deg)';
            } else if (relPos === 3) {
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
                    zIndex: '12',
                });

                const dot = document.createElement('div');
                Object.assign(dot.style, {
                    width: '6px',
                    height: '6px',
                    backgroundColor: '#d00',
                    borderRadius: '50%',
                });
                stick.appendChild(dot);

                // Position relative to Center Info
                // We place it slightly outside the box, towards the player
                const offset = '10px'; // pushes it out by 10px

                if (relPos === 0) {
                    stick.style.bottom = offset;
                    stick.style.left = '50%';
                    stick.style.transform = 'translate(-50%, 0)';
                } else if (relPos === 1) {
                    stick.style.right = offset;
                    stick.style.top = '50%';
                    stick.style.transform = 'translate(50%, -50%) rotate(90deg)';
                } else if (relPos === 2) {
                    stick.style.top = offset;
                    stick.style.left = '50%';
                    stick.style.transform = 'translate(-50%, 0)';
                } else if (relPos === 3) {
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
            gap: '2px',
        });

        // Row 1: [RoundWind] [RoundNum] [Kyoku] (Images)
        const row1 = document.createElement('div');
        Object.assign(row1.style, {
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0px',
            marginBottom: '4px',
        });

        row1.textContent = i18n.round(state.round, pc);
        row1.style.fontSize = '24px';
        row1.style.fontWeight = 'bold';

        contentContainer.appendChild(row1);

        // Row 2: Text "{honba}, {kyotaku}"
        const row2 = document.createElement('div');
        row2.innerText = i18n.text('{honba} honba / {kyotaku} riichi sticks', {
            honba: state.honba,
            kyotaku: state.kyotaku,
        });
        Object.assign(row2.style, {
            fontSize: '1.2em',
            fontWeight: 'bold',
            color: 'white',
            marginBottom: '8px',
            fontFamily: 'monospace',
        });
        contentContainer.appendChild(row2);

        // Row 3: Dora Tiles
        const row3 = document.createElement('div');
        Object.assign(row3.style, {
            display: 'flex',
            gap: '2px',
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
