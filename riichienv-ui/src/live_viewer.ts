import { GameState } from './game_state';
import { GameConfig, LayoutConfig, createGameConfig4P, createLayoutConfig4P } from './config';
import { COLORS } from './constants';
import { Renderer2D } from './renderers/renderer_2d';
import { IRenderer } from './renderers/renderer_interface';
import { MjaiEvent } from './types';
import { LiveController } from './live_controller';
import { ICON_EYE } from './icons';
import { initWasm } from './wasm/loader';

/**
 * Live game viewer that receives MJAI events incrementally.
 *
 * Usage:
 *   const viewer = new LiveViewer('container-id', { perspective: 0 });
 *   viewer.pushEvent({ type: 'start_kyoku', ... });
 *   viewer.pushEvent({ type: 'tsumo', actor: 0, pai: '1m' });
 */
export class LiveViewer {
    gameState: GameState;
    renderer: IRenderer;
    container: HTMLElement;
    controller: LiveController;
    debugPanel: HTMLElement;

    constructor(containerId: string, options?: { perspective?: number, config?: GameConfig, layout?: LayoutConfig }) {
        const gc = options?.config ?? createGameConfig4P();
        const lc = options?.layout ?? createLayoutConfig4P();

        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;

        // Start WASM initialization in background
        initWasm().then(() => {
            console.log('[LiveViewer] WASM module loaded');
        }).catch(e => {
            console.warn('[LiveViewer] WASM load failed, continuing without:', e);
        });

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

        // Scrollable/Centering Container
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

        // Scale Wrapper
        const scaleWrapper = document.createElement('div');
        Object.assign(scaleWrapper.style, {
            position: 'relative',
            overflow: 'hidden'
        });
        scrollContainer.appendChild(scaleWrapper);

        // Content Wrapper
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

        // View Area
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

        // Sidebar
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

        // Debug Panel
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

        // Initialize with empty event list
        this.gameState = new GameState([], gc);
        this.renderer = new Renderer2D(viewArea, lc);

        if (typeof options?.perspective === 'number') {
            this.renderer.viewpoint = options.perspective;
        }

        // Log toggle button
        const btnLog = createBtn('btn-log', ICON_EYE, "Debug");
        btnLog.onclick = () => this.controller.toggleLog(btnLog, this.debugPanel);
        rightSidebar.appendChild(btnLog);

        // Initialize Controller
        this.controller = new LiveController(this);
        this.controller.setupKeyboardControls(window);

        // Viewpoint change handler
        this.renderer.onViewpointChange = (pIdx: number) => {
            if (this.renderer.viewpoint !== pIdx) {
                this.renderer.viewpoint = pIdx;
                this.update();
            }
        };

        // Resize logic
        const baseW = lc.contentWidth;
        const baseH = lc.contentHeight;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const availableW = entry.contentRect.width;
                const availableH = window.innerHeight;
                if (availableW === 0) continue;

                const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);

                contentWrapper.style.transform = `scale(${scale})`;
                scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
                scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
            }
        });
        resizeObserver.observe(this.container);

        window.addEventListener('resize', () => {
            const availableW = this.container.clientWidth;
            const availableH = window.innerHeight;
            const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);
            contentWrapper.style.transform = `scale(${scale})`;
            scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
            scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
        });

        this.update();
    }

    /**
     * Push a single MJAI event and update the display.
     */
    pushEvent(event: MjaiEvent): void {
        this.gameState.appendEvent(event);
        this.update();
    }

    /**
     * Push multiple MJAI events and update the display once.
     */
    pushEvents(events: MjaiEvent[]): void {
        for (const event of events) {
            this.gameState.appendEvent(event);
        }
        this.update();
    }

    update(): void {
        if (!this.gameState || !this.renderer) return;
        const state = this.gameState.getState();
        this.renderer.render(state, this.debugPanel);
    }
}
