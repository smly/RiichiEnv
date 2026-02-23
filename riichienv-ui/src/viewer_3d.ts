import { GameState } from './game_state';
import { GameConfig, LayoutConfig3D, createGameConfig4P, createLayout3DConfig4P } from './config';
import { COLORS } from './constants';
import { Renderer3D } from './renderers/renderer_3d';
import { IRenderer } from './renderers/renderer_interface';
import { MjaiEvent } from './types';
import { ReplayController } from './controller';
import { initWasm } from './wasm/loader';
import {
    ICON_EYE, ICON_ARROW_LEFT, ICON_ARROW_RIGHT,
    ICON_CHEVRON_LEFT, ICON_CHEVRON_RIGHT, ICON_PLAY_PAUSE
} from './icons';

export class Viewer3D {
    gameState: GameState;
    renderer: IRenderer;
    container: HTMLElement;
    log: MjaiEvent[];
    controller!: ReplayController;

    isFrozen: boolean = false;

    debugPanel!: HTMLElement;
    private viewArea!: HTMLElement;

    constructor(
        containerId: string,
        log: MjaiEvent[],
        initialStep?: number,
        perspective?: number,
        freeze: boolean = false,
        config?: GameConfig,
        layout?: LayoutConfig3D
    ) {
        const gc = config ?? createGameConfig4P();
        const lc = layout ?? createLayout3DConfig4P();

        this.isFrozen = freeze;
        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;
        this.log = log;

        // Start WASM initialization in background (non-blocking)
        initWasm().then(() => {
            console.log('[Viewer3D] WASM module loaded successfully');
        }).catch(() => {
            console.warn('[Viewer3D] WASM unavailable, falling back to metadata');
        });

        // Setup container
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
            width: '100%',
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

        // 2b. Content Wrapper (16:9 aspect ratio)
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

        // 3. View Area (full 16:9 — sidebar is overlay)
        const viewArea = document.createElement('div');
        viewArea.id = `${containerId}-board`;
        Object.assign(viewArea.style, {
            width: `${lc.viewAreaWidth}px`,
            height: `${lc.viewAreaHeight}px`,
            position: 'relative',
            backgroundColor: '#000',
            flexShrink: '0',
            overflow: 'hidden',
            outline: 'none'
        });
        viewArea.tabIndex = 0;
        contentWrapper.appendChild(viewArea);
        this.viewArea = viewArea;

        // 4. Controls (overlay, bottom-right inside viewArea, 2-column grid)
        const rightSidebar = document.createElement('div');
        rightSidebar.id = `${containerId}-controls`;
        Object.assign(rightSidebar.style, {
            position: 'absolute',
            bottom: '20%',
            right: '20%',
            backgroundColor: 'rgba(0,0,0,0.65)',
            display: 'grid',
            gridTemplateColumns: 'repeat(3, auto)',
            gap: '6px',
            padding: '8px',
            alignItems: 'center',
            justifyItems: 'center',
            flexShrink: '0',
            zIndex: '500',
            borderRadius: '10px',
            backdropFilter: 'blur(4px)'
        });
        viewArea.appendChild(rightSidebar);

        // Debug panel
        this.debugPanel = document.createElement('div');
        this.debugPanel.className = 'debug-panel';
        this.debugPanel.id = `${containerId}-log-panel`;
        Object.assign(this.debugPanel.style, {
            position: 'absolute',
            top: '0',
            left: '0',
            width: '100%',
            zIndex: '1000'
        });
        viewArea.appendChild(this.debugPanel);

        // Helper to create SVG icon buttons
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

        console.log("[Viewer3D] Initializing GameState with log length:", log.length);
        this.gameState = new GameState(log, gc);
        console.log("[Viewer3D] GameState initialized. Current event index:", this.gameState.current.eventIndex);

        console.log("[Viewer3D] Initializing Renderer3D");
        this.renderer = new Renderer3D(viewArea, lc);

        // Apply initial viewpoint
        if (typeof perspective === 'number') {
            this.renderer.viewpoint = perspective;
        }

        // Create control buttons
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
            rBtn.id = `${containerId}-btn-round`;
            rBtn.style.display = 'none';
            rightSidebar.appendChild(rBtn);

            // Initialize Controller
            this.controller = new ReplayController(this);
            this.controller.setupKeyboardControls(viewArea);
            this.controller.setupWheelControls(viewArea);

            // Auto-focus on interaction for keyboard controls
            viewArea.addEventListener('mouseenter', () => viewArea.focus());
            viewArea.addEventListener('click', () => viewArea.focus());
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
            console.log(`[Viewer3D] Initializing with explicit step: ${targetStep}`);
        } else if (eventStepParam) {
            const parsed = parseInt(eventStepParam, 10);
            if (!isNaN(parsed)) {
                targetStep = parsed;
                console.log(`[Viewer3D] Initializing with permalink step: ${targetStep}`);
            }
        }

        if (targetStep !== -1) {
            this.gameState.jumpTo(targetStep);
            this.update();
        }

        // Resize Logic — scale to fit 16:9 content
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

        // Handle window resize
        window.addEventListener('resize', () => {
            const availableW = this.container.clientWidth;
            const availableH = window.innerHeight;
            const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);
            contentWrapper.style.transform = `scale(${scale})`;
            scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
            scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
        });

        // Handle Viewpoint Change from Renderer (Click on Score Panel)
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

        console.log("[Viewer3D] Calling first update()");
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
        this.viewArea.appendChild(overlay);
    }

    update() {
        if (!this.gameState || !this.renderer) return;
        const state = this.gameState.getState();
        this.renderer.render(state, this.debugPanel);
    }
}
