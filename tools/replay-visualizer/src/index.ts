import { GameState } from './game_state';
import { Renderer } from './renderer';
import { MjaiEvent } from './types';

export class Viewer {
    gameState: GameState;
    renderer: Renderer;
    container: HTMLElement;
    log: MjaiEvent[];

    debugPanel!: HTMLElement;
    // slider!: HTMLInputElement; // Removed

    constructor(containerId: string, log: MjaiEvent[]) {
        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;
        this.log = log;

        // Setup DOM Structure
        this.container.innerHTML = '';
        Object.assign(this.container.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            width: '100%',
            maxWidth: '1000px',
            margin: '0 auto',
            backgroundColor: '#f8f8f8',
            border: '1px solid #ddd',
            borderRadius: '8px',
            paddingBottom: '20px'
        });

        const viewArea = document.createElement('div');
        viewArea.id = `${containerId}-board`;
        Object.assign(viewArea.style, {
            width: '100%',
            aspectRatio: '1/1',
            position: 'relative' // Needed for overlay positioning
        });
        this.container.appendChild(viewArea);

        this.debugPanel = document.createElement('div');
        this.debugPanel.className = 'debug-panel';
        viewArea.appendChild(this.debugPanel); // Append to board area for overlay

        // Toggle Button
        const toggleBtn = document.createElement('div');
        toggleBtn.className = 'log-toggle-btn';
        toggleBtn.textContent = 'Show Log';
        toggleBtn.onclick = () => {
            if (this.debugPanel.style.display === 'none' || !this.debugPanel.style.display) {
                this.debugPanel.style.display = 'block';
                toggleBtn.textContent = 'Hide Log';
            } else {
                this.debugPanel.style.display = 'none';
                toggleBtn.textContent = 'Show Log';
            }
        };
        viewArea.appendChild(toggleBtn);

        this.gameState = new GameState(log);
        this.renderer = new Renderer(viewArea);

        // Handle Viewpoint Change from Renderer (Click on Player Info)
        this.renderer.onViewpointChange = (pIdx: number) => {
            if (this.renderer.viewpoint !== pIdx) {
                this.renderer.viewpoint = pIdx;
                this.update();
            }
        };

        this.update();

        // Handle Center Click -> Show Round Selector
        this.renderer.onCenterClick = () => {
            this.showRoundSelector();
        };

        this.update();

        // Mouse Wheel Navigation
        viewArea.addEventListener('wheel', (e: WheelEvent) => {
            e.preventDefault();
            if (e.deltaY > 0) {
                if (this.gameState.stepForward()) this.update();
            } else {
                if (this.gameState.stepBackward()) this.update();
            }
        }, { passive: false });

        // Responsive Scaling
        if ('ResizeObserver' in window) {
            const ro = new ResizeObserver(entries => {
                for (const entry of entries) {
                    this.renderer.resize(entry.contentRect.width);
                }
            });
            ro.observe(viewArea);
        } else {
            // Fallback for very old browsers
            window.addEventListener('resize', () => {
                this.renderer.resize(viewArea.clientWidth);
            });
            setTimeout(() => this.renderer.resize(viewArea.clientWidth), 0);
        }
    }

    showRoundSelector() {
        // Create Modal Overlay
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.onclick = () => {
            overlay.remove();
        };

        const content = document.createElement('div');
        content.className = 'modal-content';
        content.onclick = (e) => e.stopPropagation();

        const title = document.createElement('h3');
        title.textContent = 'Jump to Round';
        title.style.marginTop = '0';
        content.appendChild(title);

        const table = document.createElement('table');
        table.className = 'kyoku-table';

        // Header
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Round</th>
                <th>Honba</th>
                <th>P0 Score</th>
                <th>P1 Score</th>
                <th>P2 Score</th>
                <th>P3 Score</th>
            </tr>
        `;
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        const checkpoints = this.gameState.getKyokuCheckpoints();

        checkpoints.forEach((cp) => {
            const tr = document.createElement('tr');
            tr.className = 'kyoku-row';
            tr.onclick = () => {
                this.gameState.jumpTo(cp.index);
                this.update();
                overlay.remove();
            };

            const scores = cp.scores || [0, 0, 0, 0];
            tr.innerHTML = `
                <td>${this.renderer.formatRound(cp.round)}</td>
                <td>${cp.honba}</td>
                <td>${scores[0]}</td>
                <td>${scores[1]}</td>
                <td>${scores[2]}</td>
                <td>${scores[3]}</td>
            `;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        content.appendChild(table);

        overlay.appendChild(content);

        // Append to viewArea so it sits over the board
        const viewArea = this.container.querySelector('#' + this.container.id + '-board') || this.container;
        viewArea.appendChild(overlay);
    }

    update() {
        this.renderer.render(this.gameState.current, this.debugPanel);
    }
}

// Global Export for usage in HTML
// @ts-ignore
window.RiichiEnvViewer = Viewer;
