import { Viewer } from './viewer';

export class ReplayController {
    viewer: Viewer;
    autoPlayTimer: number | null = null;
    private logBtn: HTMLElement | null = null;
    private autoBtn: HTMLElement | null = null;

    constructor(viewer: Viewer) {
        this.viewer = viewer;
    }

    setupKeyboardControls(target: HTMLElement | Window) {
        target.addEventListener('keydown', (e: any) => {
            if (e.key === 'ArrowRight') this.stepForward();
            if (e.key === 'ArrowLeft') this.stepBackward();
            if (e.key === 'ArrowUp') this.prevTurn();
            if (e.key === 'ArrowDown') this.nextTurn();
        });
    }

    setupWheelControls(target: HTMLElement) {
        let lastWheelTime = 0;
        const WHEEL_THROTTLE_MS = 100;

        target.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            const now = Date.now();
            if (now - lastWheelTime < WHEEL_THROTTLE_MS) return;
            lastWheelTime = now;

            if (e.deltaY > 0) {
                this.nextTurn();
            } else {
                this.prevTurn();
            }
        }, { passive: false });
    }

    stepForward() {
        if (this.viewer.gameState.stepForward()) this.viewer.update();
    }

    stepBackward() {
        if (this.viewer.gameState.stepBackward()) this.viewer.update();
    }

    nextTurn() {
        const vp = this.viewer.renderer.viewpoint;
        if (this.viewer.gameState.jumpToNextTurn(vp)) this.viewer.update();
    }

    prevTurn() {
        const vp = this.viewer.renderer.viewpoint;
        if (this.viewer.gameState.jumpToPrevTurn(vp)) this.viewer.update();
    }

    toggleAutoPlay(btn: HTMLElement) {
        this.autoBtn = btn;
        if (this.autoPlayTimer) {
            this.stopAutoPlay();
        } else {
            btn.classList.add('active-btn');

            const loop = () => {
                if (!this.autoPlayTimer) return; // Stopped

                if (!this.viewer.gameState.stepForward()) {
                    // End of logs
                    this.stopAutoPlay();
                    return;
                }
                this.viewer.update();

                // Check event type for delay
                const state = this.viewer.gameState.getState();
                const evt = state.lastEvent;
                let delay = 200;

                if (evt) {
                    if (evt.type === 'end_kyoku') {
                        delay = 3000;
                    } else if (['pon', 'chi', 'kan', 'ankan', 'daiminkan', 'kakan', 'reach', 'reach_accepted', 'hora'].includes(evt.type)) {
                        delay = 1200; // 1s + 200ms standard
                    }
                }

                this.autoPlayTimer = window.setTimeout(loop, delay);
            };

            // Use timer ID to indicate active state, though setTimeout returns a different ID each time.
            // We can treat any non-null number as "active", but we need to store the specific timeout ID to cancel it.
            this.autoPlayTimer = window.setTimeout(loop, 200);
        }
    }

    stopAutoPlay() {
        if (this.autoPlayTimer) {
            clearTimeout(this.autoPlayTimer);
            this.autoPlayTimer = null;
        }
        if (this.autoBtn) {
            this.autoBtn.classList.remove('active-btn');
        }
    }

    toggleLog(btn: HTMLElement, panel: HTMLElement) {
        this.logBtn = btn;
        const display = panel.style.display;
        if (display === 'none' || !display) {
            panel.style.display = 'block';
            btn.classList.add('active-btn');
        } else {
            panel.style.display = 'none';
            btn.classList.remove('active-btn');
        }
    }
}
