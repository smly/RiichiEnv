import type { GameState } from './game_state';
import type { IRenderer } from './renderers/renderer_interface';

/** Common interface for Viewer and Viewer3D, consumed by ReplayController. */
export interface ViewerLike {
    gameState: GameState;
    renderer: IRenderer;
    debugPanel: HTMLElement;
    isFrozen: boolean;
    update(): void;
    onPositionChange?: (() => void) | null;
}

export class ReplayController {
    viewer: ViewerLike;
    autoPlayTimer: number | null = null;
    playbackSpeed = 1;
    private listeners = new AbortController();
    private autoBtn: HTMLElement | null = null;
    private lastActionTime: number = 0;
    private static readonly ACTION_THROTTLE_MS = 80;

    constructor(viewer: ViewerLike) {
        this.viewer = viewer;
    }

    private isThrottled(): boolean {
        const now = Date.now();
        if (now - this.lastActionTime < ReplayController.ACTION_THROTTLE_MS) return true;
        this.lastActionTime = now;
        return false;
    }

    setupKeyboardControls(target: HTMLElement | Window, playButton?: HTMLElement) {
        target.addEventListener(
            'keydown',
            (e: any) => {
                const element = e.target as HTMLElement;
                if (element?.closest?.('input, select, textarea, [contenteditable="true"], .re-modal-overlay')) return;
                const actions: Record<string, () => void> = {
                    ArrowRight: () => this.stepForward(),
                    ArrowLeft: () => this.stepBackward(),
                    ArrowUp: () => this.prevTurn(),
                    ArrowDown: () => this.nextTurn(),
                };
                if (actions[e.key]) {
                    e.preventDefault();
                    actions[e.key]();
                } else if (e.key === ' ' && playButton && !element?.closest?.('button')) {
                    e.preventDefault();
                    this.toggleAutoPlay(playButton);
                }
            },
            { signal: this.listeners.signal },
        );
    }

    seekTo(index: number) {
        this.stopAutoPlay();
        this.viewer.gameState.jumpTo(index);
        this.viewer.update();
        this.notifyPositionChange();
    }

    setupWheelControls(target: HTMLElement) {
        let lastWheelTime = 0;
        const WHEEL_THROTTLE_MS = 100;

        target.addEventListener(
            'wheel',
            (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();

                const now = Date.now();
                if (now - lastWheelTime < WHEEL_THROTTLE_MS) return;
                lastWheelTime = now;

                if (e.deltaY > 0) {
                    this.stepForward();
                } else {
                    this.stepBackward();
                }
            },
            { passive: false, signal: this.listeners.signal },
        );
    }

    private notifyPositionChange() {
        this.viewer.onPositionChange?.();
    }

    stepForward() {
        this.stopAutoPlay();
        if (this.isThrottled()) return;
        if (this.viewer.gameState.stepForward()) {
            this.viewer.update();
            this.notifyPositionChange();
        }
    }

    stepBackward() {
        this.stopAutoPlay();
        if (this.isThrottled()) return;
        if (this.viewer.gameState.stepBackward()) {
            this.viewer.update();
            this.notifyPositionChange();
        }
    }

    nextTurn() {
        this.stopAutoPlay();
        if (this.isThrottled()) return;
        const vp = this.viewer.renderer.viewpoint;
        if (this.viewer.gameState.jumpToNextTurn(vp)) {
            this.viewer.update();
            this.notifyPositionChange();
        }
    }

    prevTurn() {
        this.stopAutoPlay();
        if (this.isThrottled()) return;
        const vp = this.viewer.renderer.viewpoint;
        if (this.viewer.gameState.jumpToPrevTurn(vp)) {
            this.viewer.update();
            this.notifyPositionChange();
        }
    }

    nextKyoku() {
        this.stopAutoPlay();
        if (this.isThrottled()) return;
        if (this.viewer.gameState.jumpToNextKyoku()) {
            this.viewer.update();
            this.notifyPositionChange();
        }
    }

    prevKyoku() {
        this.stopAutoPlay();
        if (this.isThrottled()) return;
        if (this.viewer.gameState.jumpToPrevKyoku()) {
            this.viewer.update();
            this.notifyPositionChange();
        }
    }

    toggleAutoPlay(btn: HTMLElement) {
        this.autoBtn = btn;
        if (this.autoPlayTimer) {
            this.stopAutoPlay();
        } else {
            btn.classList.add('active-btn');
            btn.setAttribute('aria-pressed', 'true');

            const loop = () => {
                if (!this.autoPlayTimer) return; // Stopped

                if (!this.viewer.gameState.stepForward()) {
                    this.stopAutoPlay();
                    return;
                }
                this.viewer.update();
                this.notifyPositionChange();

                // Check event type for delay
                const state = this.viewer.gameState.getState();
                const evt = state.lastEvent;
                let delay = 200;

                if (evt) {
                    if (evt.type === 'end_kyoku') {
                        delay = 3000;
                    } else if (
                        [
                            'pon',
                            'chi',
                            'kan',
                            'ankan',
                            'daiminkan',
                            'kakan',
                            'reach',
                            'reach_accepted',
                            'hora',
                        ].includes(evt.type)
                    ) {
                        delay = 1200; // 1s + 200ms standard
                    }
                }

                this.autoPlayTimer = window.setTimeout(loop, delay / this.playbackSpeed);
            };

            // Use timer ID to indicate active state, though setTimeout returns a different ID each time.
            // We can treat any non-null number as "active", but we need to store the specific timeout ID to cancel it.
            this.autoPlayTimer = window.setTimeout(loop, 200 / this.playbackSpeed);
        }
    }

    stopAutoPlay() {
        if (this.autoPlayTimer) {
            clearTimeout(this.autoPlayTimer);
            this.autoPlayTimer = null;
        }
        if (this.autoBtn) {
            this.autoBtn.classList.remove('active-btn');
            this.autoBtn.setAttribute('aria-pressed', 'false');
        }
    }

    destroy() {
        this.stopAutoPlay();
        this.listeners.abort();
    }

    toggleLog(btn: HTMLElement, panel: HTMLElement) {
        const display = panel.style.display;
        if (display === 'none' || !display) {
            panel.style.display = 'block';
            btn.classList.add('active-btn');
            btn.setAttribute('aria-pressed', 'true');
        } else {
            panel.style.display = 'none';
            btn.classList.remove('active-btn');
            btn.setAttribute('aria-pressed', 'false');
        }
    }
}
