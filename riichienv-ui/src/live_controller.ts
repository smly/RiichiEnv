import { LiveViewer } from './live_viewer';

/**
 * Controller for live game mode.
 *
 * Unlike ReplayController, this does not support step-back or auto-play,
 * since events arrive in real-time from an external source.
 */
export class LiveController {
    viewer: LiveViewer;
    private logBtn: HTMLElement | null = null;

    constructor(viewer: LiveViewer) {
        this.viewer = viewer;
    }

    setupKeyboardControls(target: HTMLElement | Window) {
        target.addEventListener('keydown', (e: any) => {
            // In live mode, arrow keys change viewpoint instead of stepping
            if (e.key === 'ArrowRight') {
                this.cycleViewpoint(1);
            }
            if (e.key === 'ArrowLeft') {
                this.cycleViewpoint(-1);
            }
        });
    }

    private cycleViewpoint(delta: number) {
        const current = this.viewer.renderer.viewpoint;
        const pc = this.viewer.gameState.config.playerCount;
        const next = (current + delta + pc) % pc;
        this.viewer.renderer.viewpoint = next;
        this.viewer.update();
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
