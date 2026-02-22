import { GameState } from './game_state';
import { GameConfig, LayoutConfig, createGameConfig4P, createLayoutConfig4P } from './config';
import { COLORS } from './constants';
import { Renderer2D } from './renderers/renderer_2d';
import { IRenderer } from './renderers/renderer_interface';
import { MjaiEvent } from './types';
import { ReplayController } from './controller';
import { initWasm } from './wasm/loader';
import {
    ICON_EYE, ICON_ARROW_LEFT, ICON_ARROW_RIGHT,
    ICON_CHEVRON_LEFT, ICON_CHEVRON_RIGHT, ICON_PLAY_PAUSE
} from './icons';

export class Viewer {
    gameState: GameState;
    renderer: IRenderer;
    container: HTMLElement;
    log: MjaiEvent[];
    controller!: ReplayController;

    isFrozen: boolean = false;

    debugPanel!: HTMLElement;

    constructor(
        containerId: string,
        log: MjaiEvent[],
        initialStep?: number,
        perspective?: number,
        freeze: boolean = false,
        config?: GameConfig,
        layout?: LayoutConfig
    ) {
        const gc = config ?? createGameConfig4P();
        const lc = layout ?? createLayoutConfig4P();

        this.isFrozen = freeze;
        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;
        this.log = log;

        // Start WASM initialization in background (non-blocking)
        initWasm().then(() => {
            console.log('[Viewer] WASM module loaded successfully');
        }).catch(() => {
            console.warn('[Viewer] WASM unavailable, falling back to metadata');
        });

        // ... (styles) ...
        this.container.innerHTML = '';
        Object.assign(this.container.style, {
            display: 'block',
            maxWidth: '100%',
            overflow: 'hidden',
            backgroundColor: '#000',
            margin: '0',
            padding: '0',
            border: 'none',
            boxSizing: 'border-box'
        });

        // 1. Scrollable/Centering Container
        const scrollContainer = document.createElement('div');
        Object.assign(scrollContainer.style, {
            flex: '1',
            width: '100%',
            height: '100%',
            overflow: 'hidden',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-start',
            backgroundColor: '#000'
        });
        this.container.appendChild(scrollContainer);

        // 2a. Scale Wrapper
        const scaleWrapper = document.createElement('div');
        Object.assign(scaleWrapper.style, {
            position: 'relative',
            overflow: 'hidden'
        });
        scrollContainer.appendChild(scaleWrapper);

        // 2b. Content Wrapper
        const contentWrapper = document.createElement('div');
        Object.assign(contentWrapper.style, {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'flex-start',
            position: 'absolute',
            top: '0',
            left: '0',
            width: `${lc.contentWidth}px`,
            height: `${lc.contentHeight}px`,
            flexShrink: '0',
            transformOrigin: 'top left'
        });
        scaleWrapper.appendChild(contentWrapper);

        // 3. View Area
        const viewArea = document.createElement('div');
        viewArea.id = `${containerId}-board`;
        Object.assign(viewArea.style, {
            width: `${lc.viewAreaSize}px`,
            height: `${lc.viewAreaSize}px`,
            position: 'relative',
            backgroundColor: COLORS.boardBackground,
            boxShadow: '0 0 20px rgba(0,0,0,0.5)',
            flexShrink: '0'
        });
        contentWrapper.appendChild(viewArea);

        // 4. Sidebar
        const rightSidebar = document.createElement('div');
        rightSidebar.id = 'controls';
        Object.assign(rightSidebar.style, {
            width: '40px',
            backgroundColor: '#000000ff',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
            padding: '10px 10px',
            marginTop: '20px',
            alignItems: 'center',
            flexShrink: '0',
            zIndex: '500',
            height: 'auto',
            borderRadius: '0 12px 12px 0',
            marginLeft: '0px'
        });
        contentWrapper.appendChild(rightSidebar);

        this.debugPanel = document.createElement('div');
        this.debugPanel.className = 'debug-panel';
        this.debugPanel.id = 'log-panel';
        Object.assign(this.debugPanel.style, {
            position: 'absolute',
            top: '0',
            left: '0',
            width: '100%',
            zIndex: '1000'
        });
        viewArea.appendChild(this.debugPanel);

        // Helper to create buttons with SVG
        const createBtn = (id: string, svgContent: string, tooltip: string): HTMLDivElement => {
            const btn = document.createElement('div');
            btn.id = id;
            btn.className = 'icon-btn';
            btn.title = tooltip;

            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('viewBox', '0 0 24 24');
            svg.setAttribute('fill', 'none');
            svg.setAttribute('stroke', 'currentColor');
            svg.setAttribute('stroke-width', '1.5');
            svg.style.width = '24px';
            svg.style.height = '24px';
            svg.innerHTML = svgContent;

            btn.appendChild(svg);
            return btn;
        };

        console.log("[Viewer] Initializing GameState with log length:", log.length);
        this.gameState = new GameState(log, gc);
        console.log("[Viewer] GameState initialized. Current event index:", this.gameState.current.eventIndex);

        console.log("[Viewer] Initializing Renderer2D");
        this.renderer = new Renderer2D(viewArea, lc);

        // Create Buttons
        if (typeof perspective === 'number') {
            this.renderer.viewpoint = perspective;
        }

        // Create Buttons
        if (!this.isFrozen) {
            const btnLog = createBtn('btn-log', ICON_EYE, "Debug");
            btnLog.onclick = () => this.controller.toggleLog(btnLog, this.debugPanel);
            rightSidebar.appendChild(btnLog);



            const btnPTurn = createBtn('btn-pturn', ICON_ARROW_LEFT, "Prev Round");
            btnPTurn.onclick = () => this.controller.prevTurn();
            rightSidebar.appendChild(btnPTurn);

            const btnNTurn = createBtn('btn-nturn', ICON_ARROW_RIGHT, "Next Round");
            btnNTurn.onclick = () => this.controller.nextTurn();
            rightSidebar.appendChild(btnNTurn);

            const btnPrev = createBtn('btn-prev', ICON_CHEVRON_LEFT, "Prev Step");
            btnPrev.onclick = () => this.controller.stepBackward();
            rightSidebar.appendChild(btnPrev);

            const btnNext = createBtn('btn-next', ICON_CHEVRON_RIGHT, "Next Step");
            btnNext.onclick = () => this.controller.stepForward();
            rightSidebar.appendChild(btnNext);

            const btnAuto = createBtn('btn-auto', ICON_PLAY_PAUSE, "Play/Pause");
            btnAuto.onclick = () => this.controller.toggleAutoPlay(btnAuto);
            rightSidebar.appendChild(btnAuto);

            // Hidden button for Round Selector
            const rBtn = document.createElement('div');
            rBtn.id = 'btn-round';
            rBtn.style.display = 'none';
            rightSidebar.appendChild(rBtn);

            // Initialize Controller
            this.controller = new ReplayController(this);
            this.controller.setupKeyboardControls(window);
            this.controller.setupWheelControls(viewArea);
        } else {
            rightSidebar.style.display = 'none';
        }

        // Initial Seek Logic
        // Priority: initialStep argument > permalink ?eventStep=N
        const urlParams = new URLSearchParams(window.location.search);
        const eventStepParam = urlParams.get('eventStep');

        let targetStep = -1;

        if (typeof initialStep === 'number') {
            targetStep = initialStep;
            console.log(`[Viewer] Initializing with explicit step: ${targetStep}`);
        } else if (eventStepParam) {
            const parsed = parseInt(eventStepParam, 10);
            if (!isNaN(parsed)) {
                targetStep = parsed;
                console.log(`[Viewer] Initializing with permalink step: ${targetStep}`);
            }
        }

        if (targetStep !== -1) {
            // Note: jumpTo internally clamps values
            this.gameState.jumpTo(targetStep);
            this.update(); // Initial update happens at end of constructor, but we set state here
        }

        // Resize Logic to scale the entire content (Board + Sidebar)
        const baseW = lc.contentWidth;
        const baseH = lc.contentHeight;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const availableW = entry.contentRect.width;
                const availableH = window.innerHeight;

                if (availableW === 0) continue;

                const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);

                contentWrapper.style.transform = `scale(${scale})`;

                const finalW = Math.floor(baseW * scale);
                const finalH = Math.floor(baseH * scale);

                scaleWrapper.style.width = `${finalW}px`;
                scaleWrapper.style.height = `${finalH}px`;
            }
        });

        resizeObserver.observe(this.container);


        // Handle window resize to update vertical scaling if needed
        window.addEventListener('resize', () => {
            const availableW = this.container.clientWidth;
            const availableH = window.innerHeight;
            const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);
            contentWrapper.style.transform = `scale(${scale})`;
            scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
            scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
        });

        // Handle Viewpoint Change from Renderer (Click on Player Info)
        if (!this.isFrozen) {
            this.renderer.onViewpointChange = (pIdx: number) => {
                if (this.renderer.viewpoint !== pIdx) {
                    this.renderer.viewpoint = pIdx;
                    this.update();
                }
            };
        }

        // Handle Center Click -> Show Round Selector
        if (!this.isFrozen) {
            this.renderer.onCenterClick = () => {
                this.showRoundSelector();
            };
        }

        console.log("[Viewer] Calling first update()");
        this.update();
    }

    showRoundSelector() {
        const pc = this.gameState.config.playerCount;

        // Create Modal Overlay
        const overlay = document.createElement('div');
        overlay.className = 're-modal-overlay';
        overlay.onclick = () => {
            overlay.remove();
        };

        const content = document.createElement('div');
        content.className = 're-modal-content';
        content.onclick = (e) => e.stopPropagation();

        const title = document.createElement('h3');
        title.textContent = 'Jump to Round';
        title.className = 're-modal-title';
        title.style.marginTop = '0';
        content.appendChild(title);

        const table = document.createElement('table');
        table.className = 're-kyoku-table';

        // Header - dynamic columns based on player count
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th>Round</th><th>Honba</th>';
        for (let i = 0; i < pc; i++) {
            headerRow.innerHTML += `<th>P${i} Score</th>`;
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        const kyokus = this.gameState.kyokus;

        kyokus.forEach((k, idx) => {
            const tr = document.createElement('tr');
            tr.onclick = () => {
                this.gameState.jumpToKyoku(idx);
                this.update();
                overlay.remove();
            };

            // Round Name
            const winds = this.gameState.config.winds;
            const w = winds[Math.floor(k.round / pc)] || winds[0];
            const rNum = (k.round % pc) + 1;
            const roundStr = `${w}${rNum}`;

            let scoresCells = '';
            for (let i = 0; i < pc; i++) {
                scoresCells += `<td>${k.scores[i] ?? '-'}</td>`;
            }

            tr.innerHTML = `
                <td>${roundStr}</td>
                <td>${k.honba}</td>
                ${scoresCells}
            `;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        content.appendChild(table);

        overlay.appendChild(content);
        this.container.appendChild(overlay);
    }

    update() {
        if (!this.gameState || !this.renderer) return;
        const state = this.gameState.getState();
        this.renderer.render(state, this.debugPanel);
        // Update URL/History?
    }
}
