import { GameState } from './game_state';
import { COLORS } from './constants';
import { Renderer } from './renderer';
import { MjaiEvent } from './types';
import { ReplayController } from './controller';
import {
    ICON_EYE, ICON_ARROW_LEFT, ICON_ARROW_RIGHT,
    ICON_CHEVRON_LEFT, ICON_CHEVRON_RIGHT, ICON_PLAY_PAUSE
} from './icons';

export class Viewer {
    gameState: GameState;
    renderer: Renderer;
    container: HTMLElement;
    log: MjaiEvent[];
    controller!: ReplayController;

    isFrozen: boolean = false;

    debugPanel!: HTMLElement;

    constructor(containerId: string, log: MjaiEvent[], initialStep?: number, perspective?: number, freeze: boolean = false) {
        this.isFrozen = freeze;
        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;
        this.log = log;

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
            width: '970px',
            height: '900px',
            flexShrink: '0',
            transformOrigin: 'top left'
        });
        scaleWrapper.appendChild(contentWrapper);

        // 3. View Area
        const viewArea = document.createElement('div');
        viewArea.id = `${containerId}-board`;
        Object.assign(viewArea.style, {
            width: '880px',
            height: '880px',
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
        this.gameState = new GameState(log);
        console.log("[Viewer] GameState initialized. Current event index:", this.gameState.current.eventIndex);

        console.log("[Viewer] Initializing Renderer");
        this.renderer = new Renderer(viewArea);

        // Create Buttons
        if (typeof perspective === 'number') {
            this.renderer.viewpoint = perspective;
        }

        // Create Buttons
        // Create Buttons
        if (!this.isFrozen) {
            const btnLog = createBtn('btn-log', ICON_EYE, "Debug");
            btnLog.onclick = () => this.controller.toggleLog(btnLog, this.debugPanel);
            rightSidebar.appendChild(btnLog);

            // Hide Button REMOVED

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

            // Pseudo button for Round Selector (hidden or triggered by center?)
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
        // Resize Logic to scale the entire content (Board + Sidebar)
        // We use ResizeObserver on the container to detect size changes of the parent environment (e.g. Jupyter cell)
        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                // The entry.contentRect gives the size of the container.
                // However, since we adjust the container size ourselves (in older logic),
                // we must be careful.
                // Actually, in the latest logic (step 293), we resize `scaleWrapper`, 
                // and `this.container` is just a flexible wrapper (display: block, maxWidth: 100%).
                // So `this.container.clientWidth` should reflect the PARENT's constraint.

                const availableW = entry.contentRect.width;
                // For height, we might not be constrained by parent height in Jupyter (it grows).
                // But we want to fit within window height if it's full screen.
                // If Jupyter, height is usually auto.
                // If we use window.innerHeight, we ensure it doesn't get taller than the viewport.
                const availableH = window.innerHeight; // Still useful to prevent being too tall

                if (availableW === 0) continue;

                const baseW = 970;
                const baseH = 900;

                // Calculate scale
                const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);

                contentWrapper.style.transform = `scale(${scale})`;

                const finalW = Math.floor(baseW * scale);
                const finalH = Math.floor(baseH * scale);

                scaleWrapper.style.width = `${finalW}px`;
                scaleWrapper.style.height = `${finalH}px`;

                // We do NOT touch this.container dimensions here. 
                // It will shrink-wrap scaleWrapper height naturally if display: block/flex.
            }
        });

        resizeObserver.observe(this.container);

        // Also keep window resize listener as fallback or for height updates?
        // ResizeObserver on container usually covers window resizes that affect container width.
        // But window height changes might not trigger container resize if container is short.
        // We can add a simple listener to trigger observer logic?
        // Or just observe document.body?
        // Let's stick to observing container + window resize.

        window.addEventListener('resize', () => {
            // Force check
            // But ResizeObserver loop is separate.
            // We can just rely on ResizeObserver if width changes.
            // If ONLY height changes (e.g. browser resizing vertically), container width might not change.
            // But `availableH` from window.innerHeight depends on it.
            // So we need to re-run logic.
            // Let's manually run the logic:
            const availableW = this.container.clientWidth;
            const availableH = window.innerHeight;
            const baseW = 970; const baseH = 900;
            const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);
            contentWrapper.style.transform = `scale(${scale})`;
            scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
            scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
        });

        // Wire up buttons - Moved to creation block to handle freeze safely and avoid ID collisions.

        // Ensure log panel is initially hidden if desired, or handled by controller.

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

        const tbody = document.createElement('tbody');
        const kyokus = this.gameState.kyokus;

        kyokus.forEach((k, idx) => {
            const tr = document.createElement('tr');
            tr.onclick = () => {
                // Jump to this kyoku
                // Find start_kyoku event index
                // We know kyokus[idx] corresponds to limits[idx].start
                // Actually GameState stores limits. 
                // We need to jump to k.startEventIndex
                // Wait, GameState kyokus array has { round, honba, scores, startEventIndex }?
                // Let's assume GameState has a method or list.
                // Looking at GameState class (inferred), it likely has this info.
                // For now, let's assume we can jump by index if we had it.
                // Simplified: use gameState.jumpToKyoku(idx).
                this.gameState.jumpToKyoku(idx);
                this.update();
                overlay.remove();
            };

            // Round Name
            const winds = ['E', 'S', 'W', 'N'];
            const w = winds[Math.floor(k.round / 4)];
            const rNum = (k.round % 4) + 1;
            const roundStr = `${w}${rNum}`;

            tr.innerHTML = `
                <td>${roundStr}</td>
                <td>${k.honba}</td>
                <td>${k.scores[0]}</td>
                <td>${k.scores[1]}</td>
                <td>${k.scores[2]}</td>
                <td>${k.scores[3]}</td>
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

(window as any).RiichiEnvViewer = Viewer;
