import { createGameConfig3P, createGameConfig4P, detectPlayerCount, type GameConfig } from './config';
import { COLORS } from './constants';
import { ReplayController } from './controller';
import { GameState } from './game_state';
import {
    ICON_ARROW_LEFT,
    ICON_ARROW_RIGHT,
    ICON_CHEVRON_LEFT,
    ICON_CHEVRON_RIGHT,
    ICON_EYE,
    ICON_PLAY_PAUSE,
} from './icons';
import type { IRenderer } from './renderers/renderer_interface';
import type { MjaiEvent } from './types';
import { initWasm } from './wasm/loader';

export interface BaseViewerInit {
    container: HTMLElement;
    log: MjaiEvent[];
    initialStep?: number;
    perspective?: number;
    freeze?: boolean;
    config?: GameConfig;
}

/**
 * Abstract base class shared by Viewer (2D) and Viewer3D.
 * Subclasses implement getLayoutInfo() and createRenderer() to configure the specific renderer.
 */
export abstract class BaseViewer {
    gameState: GameState;
    renderer: IRenderer;
    container: HTMLElement;
    log: MjaiEvent[];
    controller!: ReplayController;
    isFrozen: boolean = false;
    debugPanel!: HTMLElement;

    protected viewArea!: HTMLElement;
    private _rafId: number = 0;
    private _resizeObserver: ResizeObserver | null = null;
    private _windowResizeHandler: (() => void) | null = null;
    private _destroyed = false;

    /** Callback invoked after any navigation action changes position. */
    onPositionChange: (() => void) | null = null;
    /** Callback invoked when viewpoint changes. */
    onViewpointChangeCallback: ((viewpoint: number) => void) | null = null;

    constructor(init: BaseViewerInit) {
        const container = init.container;
        this.isFrozen = init.freeze ?? false;
        this.container = container;
        this.log = init.log;

        initWasm().catch(() => {});

        const gc = this.resolveGameConfig(init);
        this.gameState = new GameState(init.log, gc);

        // Build DOM skeleton (without renderer-specific parts)
        container.innerHTML = '';
        Object.assign(container.style, {
            display: 'block',
            maxWidth: '100%',
            overflow: 'hidden',
            backgroundColor: '#000',
            margin: '0',
            padding: '0',
            border: 'none',
            boxSizing: 'border-box',
        });

        const scrollContainer = document.createElement('div');
        Object.assign(scrollContainer.style, {
            width: '100%',
            overflow: 'hidden',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-start',
            backgroundColor: '#000',
        });
        container.appendChild(scrollContainer);

        const scaleWrapper = document.createElement('div');
        Object.assign(scaleWrapper.style, {
            position: 'relative',
            overflow: 'hidden',
        });
        scrollContainer.appendChild(scaleWrapper);

        // Get layout info from subclass first (dimensions only, viewArea not yet created)
        const layoutInfo = this.getLayoutInfo(gc, init.log);

        const contentWrapper = document.createElement('div');
        Object.assign(contentWrapper.style, {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'flex-start',
            position: 'absolute',
            top: '0',
            left: '0',
            width: `${layoutInfo.contentWidth}px`,
            height: `${layoutInfo.contentHeight}px`,
            flexShrink: '0',
            transformOrigin: 'top left',
        });
        scaleWrapper.appendChild(contentWrapper);

        // Create view area
        const viewArea = document.createElement('div');
        Object.assign(viewArea.style, {
            width: `${layoutInfo.viewAreaWidth}px`,
            height: `${layoutInfo.viewAreaHeight}px`,
            position: 'relative',
            backgroundColor: layoutInfo.sidebarStyle === 'grid' ? '#000' : COLORS.boardBackground,
            flexShrink: '0',
            overflow: layoutInfo.sidebarStyle === 'grid' ? 'hidden' : undefined,
            outline: 'none',
        });
        if (layoutInfo.sidebarStyle === 'column') {
            viewArea.style.boxShadow = '0 0 20px rgba(0,0,0,0.5)';
        }
        viewArea.tabIndex = 0;
        contentWrapper.appendChild(viewArea);
        this.viewArea = viewArea;

        // Create sidebar
        const rightSidebar = document.createElement('div');
        if (layoutInfo.sidebarStyle === 'grid') {
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
                backdropFilter: 'blur(4px)',
            });
            viewArea.appendChild(rightSidebar);
        } else {
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
                marginLeft: '0px',
            });
            contentWrapper.appendChild(rightSidebar);
        }

        // Debug panel
        this.debugPanel = document.createElement('div');
        this.debugPanel.className = 'debug-panel';
        Object.assign(this.debugPanel.style, {
            position: 'absolute',
            top: '0',
            left: '0',
            width: '100%',
            zIndex: '1000',
        });
        viewArea.appendChild(this.debugPanel);

        // Now create the renderer with the real viewArea
        this.renderer = this.createRenderer(viewArea, gc, init.log);

        if (typeof init.perspective === 'number') {
            this.renderer.viewpoint = init.perspective;
        }

        this.setupControls(rightSidebar);
        this.setupInitialSeek(init.initialStep);
        this.setupResize(layoutInfo.contentWidth, layoutInfo.contentHeight, scaleWrapper, contentWrapper);
        this.setupRendererCallbacks();

        this.updateImmediate();
    }

    /** Return layout dimensions and sidebar style. Called before viewArea is created. */
    protected abstract getLayoutInfo(
        gc: GameConfig,
        log: MjaiEvent[],
    ): {
        contentWidth: number;
        contentHeight: number;
        viewAreaWidth: number;
        viewAreaHeight: number;
        sidebarStyle: 'column' | 'grid';
    };

    /** Create and return the renderer, attached to the given viewArea. */
    protected abstract createRenderer(viewArea: HTMLElement, gc: GameConfig, log: MjaiEvent[]): IRenderer;

    private resolveGameConfig(init: BaseViewerInit): GameConfig {
        if (init.config) return init.config;
        const pc = detectPlayerCount(init.log);
        return pc === 3 ? createGameConfig3P() : createGameConfig4P();
    }

    private createBtn(_id: string, svgContent: string, tooltip: string): HTMLDivElement {
        const btn = document.createElement('div');
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
    }

    private setupControls(rightSidebar: HTMLElement) {
        if (!this.isFrozen) {
            const btnLog = this.createBtn('btn-log', ICON_EYE, 'Debug');
            btnLog.onclick = () => this.controller.toggleLog(btnLog, this.debugPanel);
            rightSidebar.appendChild(btnLog);

            const btnPTurn = this.createBtn('btn-pturn', ICON_ARROW_LEFT, 'Prev Round');
            btnPTurn.onclick = () => this.controller.prevTurn();
            rightSidebar.appendChild(btnPTurn);

            const btnNTurn = this.createBtn('btn-nturn', ICON_ARROW_RIGHT, 'Next Round');
            btnNTurn.onclick = () => this.controller.nextTurn();
            rightSidebar.appendChild(btnNTurn);

            const btnPrev = this.createBtn('btn-prev', ICON_CHEVRON_LEFT, 'Prev Step');
            btnPrev.onclick = () => this.controller.stepBackward();
            rightSidebar.appendChild(btnPrev);

            const btnNext = this.createBtn('btn-next', ICON_CHEVRON_RIGHT, 'Next Step');
            btnNext.onclick = () => this.controller.stepForward();
            rightSidebar.appendChild(btnNext);

            const btnAuto = this.createBtn('btn-auto', ICON_PLAY_PAUSE, 'Play/Pause');
            btnAuto.onclick = () => this.controller.toggleAutoPlay(btnAuto);
            rightSidebar.appendChild(btnAuto);

            const rBtn = document.createElement('div');
            rBtn.style.display = 'none';
            rightSidebar.appendChild(rBtn);

            this.controller = new ReplayController(this);
            this.controller.setupKeyboardControls(this.viewArea);
            this.controller.setupWheelControls(this.viewArea);

            this.viewArea.addEventListener('mouseenter', () => this.viewArea.focus());
            this.viewArea.addEventListener('click', () => this.viewArea.focus());
        } else {
            rightSidebar.style.display = 'none';
        }
    }

    private setupInitialSeek(initialStep?: number) {
        if (typeof initialStep === 'number') {
            this.gameState.jumpTo(initialStep);
            this.updateImmediate();
        }
    }

    private setupResize(baseW: number, baseH: number, scaleWrapper: HTMLElement, contentWrapper: HTMLElement) {
        const doResize = (availableW: number) => {
            const availableH = window.innerHeight;
            if (availableW === 0) return;
            const scale = Math.min(availableW / baseW, availableH / baseH, 1.0);
            contentWrapper.style.transform = `scale(${scale})`;
            scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
            scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
        };

        this._resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                doResize(entry.contentRect.width);
            }
        });
        this._resizeObserver.observe(this.container);

        this._windowResizeHandler = () => doResize(this.container.clientWidth);
        window.addEventListener('resize', this._windowResizeHandler);
    }

    private setupRendererCallbacks() {
        if (!this.isFrozen) {
            this.renderer.onViewpointChange = (pIdx: number) => {
                if (this.renderer.viewpoint !== pIdx) {
                    this.renderer.viewpoint = pIdx;
                    this.update();
                    this.onViewpointChangeCallback?.(pIdx);
                }
            };
            this.renderer.onCenterClick = () => {
                this.showRoundSelector();
            };
        }
    }

    showRoundSelector() {
        const pc = this.gameState.config.playerCount;

        const overlay = document.createElement('div');
        overlay.className = 're-modal-overlay';
        overlay.onclick = () => overlay.remove();

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
        (this.viewArea || this.container).appendChild(overlay);
    }

    update() {
        if (this._destroyed || !this.gameState || !this.renderer) return;
        if (this._rafId) cancelAnimationFrame(this._rafId);
        this._rafId = requestAnimationFrame(() => {
            this._rafId = 0;
            this.updateImmediate();
        });
    }

    updateImmediate() {
        if (this._destroyed || !this.gameState || !this.renderer) return;
        const state = this.gameState.getState();
        this.renderer.render(state, this.debugPanel);
    }

    destroy() {
        this._destroyed = true;
        if (this.controller) {
            this.controller.stopAutoPlay();
        }
        if (this._rafId) {
            cancelAnimationFrame(this._rafId);
            this._rafId = 0;
        }
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this._windowResizeHandler) {
            window.removeEventListener('resize', this._windowResizeHandler);
            this._windowResizeHandler = null;
        }
        this.container.innerHTML = '';
    }
}
