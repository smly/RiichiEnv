import { GameState } from './game_state';
import { Renderer } from './renderer';
import { MjaiEvent } from './types';

export class Viewer {
    gameState: GameState;
    renderer: Renderer;
    container: HTMLElement;
    controlPanel: HTMLElement;
    log: MjaiEvent[];

    kyokuSelect!: HTMLSelectElement;
    viewpointSelect!: HTMLSelectElement;
    // slider!: HTMLInputElement; // Removed

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
        this.controlPanel.innerHTML = '';
        this.controlPanel.style.flexDirection = 'column';
        this.controlPanel.style.alignItems = 'center';

        const rowStyle = "display: flex; gap: 10px; align-items: center; justify-content: center; margin-bottom: 8px; flex-wrap: wrap;";
        const btnStyle = "padding: 6px 12px; cursor: pointer; border: 1px solid #ccc; background: white; border-radius: 4px; font-weight: bold; font-family: sans-serif; user-select: none;";

        const createBtn = (lbl: string, cb: () => void, title?: string) => {
            const b = document.createElement('button');
            b.textContent = lbl;
            b.style.cssText = btnStyle;
            b.onclick = cb;
            if (title) b.title = title;
            return b;
        };

        // --- Row 1: Turn Navigation ---
        const navRow = document.createElement('div');
        navRow.style.cssText = rowStyle;

        const prevTurn = createBtn('<< Turn', () => {
            this.gameState.stepTurn(false, this.renderer.viewpoint);
            this.update();
        }, "Previous Turn (Viewpoint Player)");

        const prevStep = createBtn('< Step', () => {
            if (this.gameState.stepBackward()) this.update();
        });

        const nextStep = createBtn('Step >', () => {
            if (this.gameState.stepForward()) this.update();
        });

        const nextTurn = createBtn('Turn >>', () => {
            this.gameState.stepTurn(true, this.renderer.viewpoint);
            this.update();
        }, "Next Turn (Viewpoint Player)");

        navRow.appendChild(prevTurn);
        navRow.appendChild(prevStep);
        navRow.appendChild(nextStep);
        navRow.appendChild(nextTurn);

        // --- Row 2: Kyoku & Viewpoint ---
        const metaRow = document.createElement('div');
        metaRow.style.cssText = rowStyle;

        // Kyoku Select
        const kyokuSel = document.createElement('select');
        kyokuSel.style.padding = '5px';
        kyokuSel.style.borderRadius = '4px';
        const checkpoints = this.gameState.getKyokuCheckpoints();

        const startOpt = document.createElement('option');
        startOpt.value = '0';
        startOpt.text = 'Start Game';
        kyokuSel.appendChild(startOpt);

        checkpoints.forEach((cp) => {
            const opt = document.createElement('option');
            opt.value = cp.index.toString();
            opt.text = `${this.renderer.formatRound(cp.round)} - ${cp.honba} Honba`;
            kyokuSel.appendChild(opt);
        });
        kyokuSel.onchange = () => {
            this.gameState.jumpTo(parseInt(kyokuSel.value));
            this.update();
        };
        this.kyokuSelect = kyokuSel;

        // Viewpoint Select
        const viewSel = document.createElement('select');
        viewSel.style.padding = '5px';
        viewSel.style.borderRadius = '4px';
        ['Self (P0)', 'Right (P1)', 'Opp (P2)', 'Left (P3)'].forEach((lbl, i) => {
            const opt = document.createElement('option');
            opt.value = i.toString();
            opt.text = lbl;
            viewSel.appendChild(opt);
        });
        viewSel.value = '0'; // Default P0
        viewSel.onchange = () => {
            this.renderer.viewpoint = parseInt(viewSel.value);
            this.update();
        };
        this.viewpointSelect = viewSel;

        // Prev/Next Kyoku Buttons
        const prevKyoku = createBtn('Prev Kyoku', () => {
            let target = 0;
            for (const cp of checkpoints) {
                if (cp.index < this.gameState.cursor - 5) {
                    target = cp.index;
                } else {
                    break;
                }
            }
            this.gameState.jumpTo(target);
            this.update();
        });

        const nextKyoku = createBtn('Next Kyoku', () => {
            const nextCp = checkpoints.find(cp => cp.index > this.gameState.cursor);
            if (nextCp) {
                this.gameState.jumpTo(nextCp.index);
                this.update();
            }
        });

        metaRow.appendChild(prevKyoku);
        metaRow.appendChild(kyokuSel);
        metaRow.appendChild(nextKyoku);

        const spacer = document.createElement('span');
        spacer.style.margin = '0 10px';
        spacer.style.borderLeft = '1px solid #999';
        spacer.style.height = '20px';
        metaRow.appendChild(spacer);

        metaRow.appendChild(viewSel);

        this.controlPanel.appendChild(navRow);
        this.controlPanel.appendChild(metaRow);
    }

    update() {
        this.renderer.render(this.gameState.current);

        // Sync Kyoku Select
        const checkpoints = this.gameState.getKyokuCheckpoints();
        // Find checkpoint <= cursor
        let activeIndex = 0;
        for (const cp of checkpoints) {
            if (cp.index <= this.gameState.cursor) {
                activeIndex = cp.index;
            } else {
                break;
            }
        }
        if (this.kyokuSelect) {
            this.kyokuSelect.value = activeIndex.toString();
        }
    }
}

// Global Export for usage in HTML
// @ts-ignore
window.RiichiEnvViewer = Viewer;
