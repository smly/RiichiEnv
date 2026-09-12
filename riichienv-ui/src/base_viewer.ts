import { computeKyokuSummaries } from './analyzer';
import { createGameConfig3P, createGameConfig4P, detectPlayerCount, type GameConfig } from './config';
import { COLORS } from './constants';
import { ReplayController } from './controller';
import { GameState } from './game_state';
import { I18n, isLocale, type Locale } from './i18n/index';
import {
    ICON_ARROW_LEFT,
    ICON_ARROW_RIGHT,
    ICON_CHEVRON_DOUBLE_LEFT,
    ICON_CHEVRON_DOUBLE_RIGHT,
    ICON_CHEVRON_LEFT,
    ICON_CHEVRON_RIGHT,
    ICON_EYE,
    ICON_PLAY_PAUSE,
} from './icons';
import { DEFAULT_DISPLAY_OPTIONS, type DisplayOptions } from './renderers/board_presentation';
import type { IRenderer } from './renderers/renderer_interface';
import { createRoundSelector } from './renderers/round_selector';
import { createSettingsModal } from './renderers/settings_modal';
import { loadReplayFile, loadReplayUrl } from './replay_loader';
import type { MjaiEvent, PlayerConfig } from './types';
import { initWasm } from './wasm/loader';

export interface BaseViewerInit {
    container: HTMLElement;
    log: MjaiEvent[];
    initialStep?: number;
    perspective?: number;
    freeze?: boolean;
    config?: GameConfig;
    players?: PlayerConfig[];
}

/**
 * Abstract base class shared by Viewer (2D) and Viewer3D.
 * Subclasses implement getLayoutInfo() and createRenderer() to configure the specific renderer.
 */
export abstract class BaseViewer {
    readonly i18n = new I18n();
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
    private settingsLoad: AbortController | null = null;
    private displayOptions: DisplayOptions = { ...DEFAULT_DISPLAY_OPTIONS };
    private replayDock: HTMLElement | null = null;
    private replaySeek: HTMLInputElement | null = null;
    private replayPosition: HTMLElement | null = null;
    private replayRound: HTMLButtonElement | null = null;
    private replayPrevious: HTMLButtonElement | null = null;
    private replayNext: HTMLButtonElement | null = null;

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
        this.gameState = new GameState(init.log, gc, init.players);

        // Build DOM skeleton (without renderer-specific parts)
        container.innerHTML = '';
        Object.assign(container.style, {
            display: 'block',
            position: 'relative',
            maxWidth: '100%',
            overflow: 'hidden',
            backgroundColor: '#000',
            margin: '0',
            padding: '0',
            border: 'none',
            boxSizing: 'border-box',
            userSelect: 'none',
            WebkitUserSelect: 'none',
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
        scaleWrapper.className = 'viewer-stage';
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
            rightSidebar.className = 'replay-dock';
            this.replayDock = rightSidebar;
            container.appendChild(rightSidebar);
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

    private createBtn(_id: string, svgContent: string, tooltip: string): HTMLButtonElement {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset.i18nLabel = tooltip;
        btn.setAttribute('aria-label', this.i18n.text(tooltip));
        btn.className = 'icon-btn';
        btn.title = this.i18n.text(tooltip);
        btn.dataset.control = _id;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '1.5');
        svg.style.width = '28px';
        svg.style.height = '28px';
        svg.innerHTML = svgContent;

        btn.appendChild(svg);
        return btn;
    }

    private createLabeledBtn(id: string, svgContent: string, label: string): HTMLDivElement {
        const wrapper = document.createElement('div');
        Object.assign(wrapper.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '2px',
            cursor: 'pointer',
        });

        const btn = this.createBtn(id, svgContent, label);
        wrapper.appendChild(btn);

        const lbl = document.createElement('div');
        Object.assign(lbl.style, {
            fontSize: '10px',
            color: '#aaa',
            textAlign: 'center',
            fontFamily: 'sans-serif',
            lineHeight: '1',
        });
        lbl.dataset.i18n = label;
        lbl.textContent = this.i18n.text(label);
        wrapper.appendChild(lbl);

        return wrapper;
    }

    private setupControls(rightSidebar: HTMLElement) {
        if (this.replayDock && !this.isFrozen) {
            this.setupReplayDock(rightSidebar);
            return;
        }
        if (!this.isFrozen) {
            // Left options box (Debug + Auto)
            const optionsBox = document.createElement('div');
            Object.assign(optionsBox.style, {
                position: 'absolute',
                left: '10px',
                bottom: '20%',
                backgroundColor: 'rgba(0,0,0,0.65)',
                display: 'flex',
                flexDirection: 'column',
                gap: '10px',
                padding: '10px',
                alignItems: 'center',
                zIndex: '500',
                borderRadius: '10px',
                backdropFilter: 'blur(4px)',
            });

            const logWrapper = this.createLabeledBtn('btn-log', ICON_EYE, 'Debug');
            // Re-wire click to use the actual btn for toggle state
            const logBtn = logWrapper.querySelector('.icon-btn') as HTMLElement;
            logWrapper.onclick = (e) => {
                e.stopPropagation();
                this.controller.toggleLog(logBtn, this.debugPanel);
            };
            optionsBox.appendChild(logWrapper);

            const autoWrapper = this.createLabeledBtn('btn-auto', ICON_PLAY_PAUSE, 'Auto');
            const autoBtn = autoWrapper.querySelector('.icon-btn') as HTMLElement;
            autoWrapper.onclick = (e) => {
                e.stopPropagation();
                this.controller.toggleAutoPlay(autoBtn);
            };
            optionsBox.appendChild(autoWrapper);

            this.viewArea.appendChild(optionsBox);

            // Right sidebar (navigation only)
            const createRowLabel = (text: string): HTMLElement => {
                const lbl = document.createElement('div');
                Object.assign(lbl.style, {
                    fontSize: '13px',
                    color: '#aaa',
                    fontFamily: 'sans-serif',
                    whiteSpace: 'nowrap',
                    paddingRight: '4px',
                });
                lbl.dataset.i18n = text;
                lbl.textContent = this.i18n.text(text);
                return lbl;
            };

            rightSidebar.appendChild(createRowLabel('Round'));
            const btnPKyoku = this.createBtn('btn-pkyoku', ICON_CHEVRON_DOUBLE_LEFT, 'Prev Kyoku');
            btnPKyoku.onclick = () => this.controller.prevKyoku();
            rightSidebar.appendChild(btnPKyoku);
            const btnNKyoku = this.createBtn('btn-nkyoku', ICON_CHEVRON_DOUBLE_RIGHT, 'Next Kyoku');
            btnNKyoku.onclick = () => this.controller.nextKyoku();
            rightSidebar.appendChild(btnNKyoku);

            rightSidebar.appendChild(createRowLabel('Turn'));
            const btnPTurn = this.createBtn('btn-pturn', ICON_ARROW_LEFT, 'Prev Turn');
            btnPTurn.onclick = () => this.controller.prevTurn();
            rightSidebar.appendChild(btnPTurn);
            const btnNTurn = this.createBtn('btn-nturn', ICON_ARROW_RIGHT, 'Next Turn');
            btnNTurn.onclick = () => this.controller.nextTurn();
            rightSidebar.appendChild(btnNTurn);

            rightSidebar.appendChild(createRowLabel('Step'));
            const btnPrev = this.createBtn('btn-prev', ICON_CHEVRON_LEFT, 'Prev Step');
            btnPrev.onclick = () => this.controller.stepBackward();
            rightSidebar.appendChild(btnPrev);
            const btnNext = this.createBtn('btn-next', ICON_CHEVRON_RIGHT, 'Next Step');
            btnNext.onclick = () => this.controller.stepForward();
            rightSidebar.appendChild(btnNext);

            this.controller = new ReplayController(this);
            this.controller.setupKeyboardControls(this.viewArea);
            this.controller.setupWheelControls(this.viewArea);

            this.viewArea.addEventListener('mouseenter', () => this.viewArea.focus());
            this.viewArea.addEventListener('click', () => this.viewArea.focus());
        } else {
            rightSidebar.style.display = 'none';
        }
    }

    /** Keep the same two-row replay controls on the table at every size. */
    private setupReplayDock(dock: HTMLElement) {
        this.controller = new ReplayController(this);
        dock.setAttribute('role', 'group');
        dock.dataset.i18nLabel = 'Replay controls';
        dock.setAttribute('aria-label', this.i18n.text('Replay controls'));
        const timeline = document.createElement('div');
        timeline.className = 'replay-timeline';
        const round = this.createBtn('round', ICON_CHEVRON_DOUBLE_RIGHT, 'Jump to round');
        round.classList.add('replay-round');
        round.style.width = 'auto';
        round.style.fontSize = '13px';
        round.onclick = () => this.showRoundSelector();
        this.replayRound = round;
        const seek = document.createElement('input');
        seek.type = 'range';
        seek.min = this.log.length ? '1' : '0';
        seek.step = '1';
        seek.dataset.i18nLabel = 'Replay position';
        seek.setAttribute('aria-label', this.i18n.text('Replay position'));
        seek.oninput = () => this.controller.seekTo(Number(seek.value));
        this.replaySeek = seek;
        const position = document.createElement('span');
        position.className = 'replay-position';
        this.replayPosition = position;
        timeline.append(round, seek, position);
        dock.appendChild(timeline);

        const actions = document.createElement('div');
        actions.className = 'replay-actions';
        const group = (label?: string) => {
            const el = document.createElement('div');
            el.className = 'replay-group';
            if (label) {
                const text = document.createElement('span');
                text.className = 'replay-group-label';
                text.dataset.i18n = label;
                text.textContent = this.i18n.text(label);
                el.appendChild(text);
            }
            actions.appendChild(el);
            return el;
        };
        const playback = group();
        playback.classList.add('replay-playback');
        const previous = this.createBtn('previous', ICON_CHEVRON_LEFT, 'Previous step (←)');
        previous.onclick = () => this.controller.stepBackward();
        this.replayPrevious = previous;
        const play = this.createBtn('play', ICON_PLAY_PAUSE, 'Play / pause (Space)');
        play.classList.add('replay-play');
        play.setAttribute('aria-pressed', 'false');
        play.onclick = () => this.controller.toggleAutoPlay(play);
        const next = this.createBtn('next', ICON_CHEVRON_RIGHT, 'Next step (→)');
        next.onclick = () => this.controller.stepForward();
        this.replayNext = next;
        playback.append(play, previous, next);

        const turns = group('Turn');
        const previousTurn = this.createBtn('previous-turn', ICON_ARROW_LEFT, 'Previous turn (↑)');
        previousTurn.onclick = () => this.controller.prevTurn();
        const nextTurn = this.createBtn('next-turn', ICON_ARROW_RIGHT, 'Next turn (↓)');
        nextTurn.onclick = () => this.controller.nextTurn();
        turns.append(previousTurn, nextTurn);
        const rounds = group('Round');
        const previousRound = this.createBtn('previous-round', ICON_CHEVRON_DOUBLE_LEFT, 'Previous round');
        previousRound.onclick = () => this.controller.prevKyoku();
        const nextRound = this.createBtn('next-round', ICON_CHEVRON_DOUBLE_RIGHT, 'Next round');
        nextRound.onclick = () => this.controller.nextKyoku();
        rounds.append(previousRound, nextRound);

        const options = group();
        options.classList.add('replay-options');
        const speed = document.createElement('select');
        speed.dataset.i18nLabel = 'Playback speed';
        speed.setAttribute('aria-label', this.i18n.text('Playback speed'));
        for (const value of [0.5, 1, 2, 4]) {
            const option = document.createElement('option');
            option.value = String(value);
            option.textContent = `${value}×`;
            option.selected = value === 1;
            speed.appendChild(option);
        }
        speed.onchange = () => {
            this.controller.playbackSpeed = Number(speed.value);
        };
        const debug = this.createBtn('debug', ICON_EYE, 'Show event details');
        debug.setAttribute('aria-pressed', 'false');
        debug.onclick = () => this.controller.toggleLog(debug, this.debugPanel);
        options.append(speed, debug);
        // Keep the same two rows, including keyboard order, at every viewer size.
        actions.append(turns, rounds, playback, options);
        dock.appendChild(actions);
        this.controller.setupKeyboardControls(this.container, play);
        this.controller.setupWheelControls(this.viewArea);
        this.viewArea.addEventListener('click', (e) => {
            if (e.target instanceof Element && !e.target.closest('button, input, select, .re-modal-overlay')) {
                this.viewArea.focus({ preventScroll: true });
            }
        });
    }

    private setupInitialSeek(initialStep?: number) {
        if (typeof initialStep === 'number') {
            this.gameState.jumpTo(initialStep);
            this.updateImmediate();
        }
    }

    private setupResize(baseW: number, baseH: number, scaleWrapper: HTMLElement, contentWrapper: HTMLElement) {
        const doResize = (availableW: number) => {
            if (availableW === 0) return;
            // Account for page content above the viewer, not just viewport height.
            const viewportH = this.replayDock
                ? Math.max(1, window.innerHeight - Math.max(0, this.container.getBoundingClientRect().top))
                : Math.max(180, window.innerHeight);
            const scale = Math.min(availableW / baseW, viewportH / baseH, 1.0);
            this.replayDock?.classList.add('is-floating');
            contentWrapper.style.transform = `scale(${scale})`;
            scaleWrapper.style.width = `${Math.floor(baseW * scale)}px`;
            scaleWrapper.style.height = `${Math.floor(baseH * scale)}px`;
            if (this.replayDock) {
                const dock = this.replayDock;
                const rightInset = 232;
                dock.style.left = `${(availableW - baseW * scale) / 2 + (baseW - rightInset - dock.offsetWidth) * scale}px`;
                dock.style.top = `${(baseH - 246 - dock.offsetHeight) * scale}px`;
                dock.style.transform = `scale(${scale})`;
            }
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
        this.renderer.i18n = this.i18n;
        this.container.lang = this.i18n.locale;
        this.renderer.onSettingsClick = () => this.showSettings();
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

    private showSettings() {
        if (this.container.querySelector('.re-settings-overlay')) return;
        this.controller?.stopAutoPlay();
        const overlay = createSettingsModal(
            () => {
                this.settingsLoad?.abort();
                overlay.remove();
                this.container
                    .querySelector<HTMLButtonElement>('.viewer-settings-button')
                    ?.focus({ preventScroll: true });
            },
            this.i18n,
            (locale) => this.setLanguage(locale),
            (url) => this.loadSettingsReplay(url),
            (file) => this.loadSettingsReplay(file),
            this.displayOptions,
            (options) => {
                this.displayOptions = options;
                this.updateImmediate();
            },
        );
        this.container.appendChild(overlay);
        overlay.querySelector<HTMLButtonElement>('.settings-close')?.focus({ preventScroll: true });
    }

    private async loadSettingsReplay(source: string | File) {
        this.settingsLoad?.abort();
        const request = new AbortController();
        this.settingsLoad = request;
        const timeout = window.setTimeout(() => request.abort(), 30000);
        try {
            const events = await (typeof source === 'string'
                ? loadReplayUrl(source, request.signal)
                : loadReplayFile(source, request.signal));
            if (request.signal.aborted || this._destroyed) return;
            this.replaceReplay(events);
        } finally {
            window.clearTimeout(timeout);
            if (this.settingsLoad === request) this.settingsLoad = null;
        }
    }

    private replaceReplay(events: MjaiEvent[]) {
        const pc = detectPlayerCount(events);
        const config = pc === 3 ? createGameConfig3P() : createGameConfig4P();
        // Construct before replacing the live replay so a failed load keeps it intact.
        const next = new GameState(events, config);
        this.controller?.stopAutoPlay();
        if (pc !== this.gameState.config.playerCount) {
            this.viewArea.querySelector('.scene-3d')?.remove();
            this.renderer = this.createRenderer(this.viewArea, config, events);
            this.setupRendererCallbacks();
        }
        this.viewArea.querySelectorAll('.re-modal-overlay').forEach((modal) => modal.remove());
        this.gameState = next;
        this.log = events;
        this.renderer.viewpoint = 0;
        this.updateImmediate();
        this.onPositionChange?.();
        this.onViewpointChangeCallback?.(0);
    }

    setLanguage(locale: Locale) {
        if (!isLocale(locale)) throw new Error(`Unsupported language: ${locale}`);
        this.i18n.locale = locale;
        this.container.lang = locale;
        this.i18n.apply(this.container);
        this.container.querySelectorAll<HTMLSelectElement>('.settings-language').forEach((select) => {
            select.value = locale;
        });
        const roundSelectorOpen = !!this.container.querySelector('.re-round-selector');
        if (roundSelectorOpen) this.container.querySelector('.re-round-selector')?.remove();
        this.updateImmediate();
        if (roundSelectorOpen) this.showRoundSelector();
    }

    showRoundSelector() {
        if (this.container.querySelector('.re-round-selector')) return;
        this.controller?.stopAutoPlay();
        const opener = document.activeElement as HTMLElement | null;
        const pc = this.gameState.config.playerCount;

        const overlay = document.createElement('div');
        overlay.className = 're-modal-overlay re-round-selector';
        const close = () => {
            overlay.remove();
            opener?.focus({ preventScroll: true });
        };
        overlay.onclick = close;
        // Let the table scroll without triggering the replay's wheel shortcuts.
        overlay.onwheel = (event) => event.stopPropagation();

        const currentIndex = this.gameState.kyokus.reduce(
            (current, k, i) => (k.index < this.gameState.cursor ? i : current),
            0,
        );
        const content = createRoundSelector({
            i18n: this.i18n,
            summaries: computeKyokuSummaries(this.gameState),
            names: this.gameState.getState().playerNames,
            playerCount: pc,
            currentIndex,
            onClose: close,
            onSelect: (index) => {
                this.gameState.jumpToKyoku(index);
                this.update();
                this.onPositionChange?.();
                close();
            },
        });
        content.onclick = (e) => e.stopPropagation();
        overlay.onkeydown = (event) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                close();
            }
            if (event.key === 'Tab') {
                const buttons = Array.from(content.querySelectorAll<HTMLButtonElement>('button')).filter(
                    (b) => b.getClientRects().length > 0,
                );
                const first = buttons[0],
                    last = buttons[buttons.length - 1];
                if (event.shiftKey && document.activeElement === first) {
                    event.preventDefault();
                    last?.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                    event.preventDefault();
                    first?.focus();
                }
            }
        };

        overlay.appendChild(content);
        (this.replayDock ? this.container : this.viewArea).appendChild(overlay);
        content.querySelector<HTMLButtonElement>('.round-browser-close')?.focus({ preventScroll: true });
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
        this.renderer.render(state, this.debugPanel, this.displayOptions);
        if (this.replaySeek) {
            this.replaySeek.max = String(state.totalEvents);
            this.replaySeek.value = String(state.eventIndex);
            this.replaySeek.setAttribute(
                'aria-valuetext',
                this.i18n.text('Event {step} of {total}', { step: state.eventIndex, total: state.totalEvents }),
            );
        }
        if (this.replayPosition) this.replayPosition.textContent = `${state.eventIndex} / ${state.totalEvents}`;
        if (this.replayRound) {
            const pc = state.playerCount;
            this.replayRound.textContent = `${this.i18n.round(state.round, pc)} ▾`;
        }
        if (this.replayPrevious) this.replayPrevious.disabled = state.eventIndex <= 1;
        if (this.replayNext) this.replayNext.disabled = state.eventIndex >= state.totalEvents;
    }

    destroy() {
        this._destroyed = true;
        this.settingsLoad?.abort();
        if (this.controller) {
            this.controller.destroy();
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
