import { TileRenderer } from './tile_renderer';
import { COLORS } from '../constants';

export class CenterRenderer {
    static renderCenter(
        state: any,
        onCenterClick: (() => void) | null
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
            minWidth: '120px',
            cursor: 'pointer' // Added cursor pointer
        });

        center.onclick = (e) => {
            e.stopPropagation();
            if (onCenterClick) onCenterClick();
        };

        // Dora: Always 5 tiles. Fill missing with 'back'.
        const doraTiles = [...state.doraMarkers];
        while (doraTiles.length < 5) {
            doraTiles.push('back');
        }

        const doraHtml = doraTiles.map((t: string) =>
            `<div style="width:28px; height:38px;">${TileRenderer.getTileHtml(t)}</div>`
        ).join('');

        // Helper for formatting round
        const formatRound = (r: number) => {
            const winds = ['E', 'S', 'W', 'N'];
            const w = winds[Math.floor(r / 4)];
            const k = (r % 4) + 1;
            return `${w}${k}`;
        };

        center.innerHTML = `
            <div style="margin-bottom: 8px;">
                <span style="font-size: 1.2em; font-weight: bold;">${formatRound(state.round)}-${state.honba}</span>
                <span style="font-size: 0.9em; margin-left: 5px;">Depo: ${state.kyotaku}</span>
            </div>
            <div style="display:flex; align-items: center; gap: 5px;">
                <div style="display:flex; gap:2px;">
                    ${doraHtml}
                </div>
            </div>
        `;

        return center;
    }
}
