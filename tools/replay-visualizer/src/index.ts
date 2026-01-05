import { GameState } from './game_state';
import { Renderer } from './renderer';
import { MjaiEvent } from './types';

export class Viewer {
    gameState: GameState;
    renderer: Renderer;
    container: HTMLElement;
    controlPanel: HTMLElement;
    log: MjaiEvent[];

    constructor(containerId: string, log: MjaiEvent[]) {
        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;
        this.log = log;

        // Setup DOM Structure
        this.container.innerHTML = '';
        const viewArea = document.createElement('div');
        viewArea.id = `${containerId}-board`;
        Object.assign(viewArea.style, {
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-start', // Align top so it doesn't jump vertically
            width: '100%',
            overflow: 'auto' // Allow scroll if too small
        });
        this.container.appendChild(viewArea);

        this.controlPanel = document.createElement('div');
        Object.assign(this.controlPanel.style, {
            display: 'flex',
            gap: '10px',
            justifyContent: 'center',
            marginTop: '10px',
            padding: '10px',
            backgroundColor: '#f0f0f0',
            borderRadius: '4px'
        });
        this.container.appendChild(this.controlPanel);

        this.gameState = new GameState(log);
        this.renderer = new Renderer(viewArea);

        this.initControls();
        this.update();
    }

    initControls() {
        const btnStyle = "padding: 5px 10px; cursor: pointer;";
        const createBtn = (lbl: string, cb: () => void) => {
            const b = document.createElement('button');
            b.textContent = lbl;
            b.style.cssText = btnStyle;
            b.onclick = cb;
            return b;
        };

        const prev = createBtn('< Prev', () => {
            if (this.gameState.stepBackward()) this.update();
        });

        const next = createBtn('Next >', () => {
            if (this.gameState.stepForward()) this.update();
        });

        const reset = createBtn('<< Start', () => {
            this.gameState.reset();
            this.update();
        });

        // Slider
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = String(this.log.length);
        slider.value = '0';
        slider.style.flexGrow = '1';
        slider.oninput = (e) => {
            const val = parseInt((e.target as HTMLInputElement).value);
            this.gameState.jumpTo(val);
            this.update();
        };
        this.slider = slider;

        this.controlPanel.appendChild(reset);
        this.controlPanel.appendChild(prev);
        this.controlPanel.appendChild(slider);
        this.controlPanel.appendChild(next);
    }

    slider!: HTMLInputElement;

    update() {
        this.renderer.render(this.gameState.current);
        this.slider.value = String(this.gameState.cursor);
    }
}

// Global Export for usage in HTML
// @ts-ignore
window.RiichiEnvViewer = Viewer;
