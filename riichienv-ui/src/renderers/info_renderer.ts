export class InfoRenderer {
    static renderPlayerInfo(
        _player: any,
        index: number,
        viewpoint: number,
        currentActor: number,
        onViewpointChange: (idx: number) => void,
        playerName?: string,
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
            marginLeft: '0',
        });

        const nameDiv = document.createElement('div');
        Object.assign(nameDiv.style, {
            fontSize: '1.2em',
            fontWeight: 'bold',
            marginBottom: '4px',
            color: 'white',
        });
        nameDiv.textContent = playerName || `Player${index}`;
        infoBox.appendChild(nameDiv);

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
