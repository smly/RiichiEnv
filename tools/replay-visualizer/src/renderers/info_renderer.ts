export class InfoRenderer {
    static renderPlayerInfo(
        player: any,
        index: number,
        viewpoint: number,
        currentActor: number,
        onViewpointChange: (idx: number) => void
    ): HTMLElement {
        const infoBox = document.createElement('div');
        infoBox.className = 'player-info-box';
        if (index === viewpoint) {
            infoBox.classList.add('active-viewpoint');
        }

        // Positioning: Absolute relative to pDiv
        Object.assign(infoBox.style, {
            position: 'absolute',
            top: '30px',
            left: '50%',
            transform: 'translateX(140px)',
            marginLeft: '0'
        });

        const winds = ['E', 'S', 'W', 'N'];
        const windLabel = winds[player.wind];
        const isOya = (player.wind === 0);

        infoBox.innerHTML = `
            <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 4px; color: ${isOya ? '#ff4d4d' : 'white'};">
                ${windLabel} P${index}
            </div>
            <div style="font-family:monospace; font-size:1.1em;">${player.score}</div>
        `;
        if (player.riichi) {
            infoBox.innerHTML += '<div style="color:#ff6b6b; font-weight:bold; font-size:0.9em; margin-top:2px;">REACH</div>';
        }

        // Blinking Bar for Active Player
        if (index === currentActor) {
            const bar = document.createElement('div');
            bar.className = 'active-player-bar';
            infoBox.appendChild(bar);
        }

        infoBox.onclick = (e) => {
            e.stopPropagation(); // Prevent bubbling
            if (onViewpointChange) {
                onViewpointChange(index);
            }
        };

        return infoBox;
    }
}
