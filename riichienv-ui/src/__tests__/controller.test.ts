import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ReplayController, type ViewerLike } from '../controller';

function makeButton() {
    const attributes = new Map<string, string>();
    return {
        classList: { add: vi.fn(), remove: vi.fn() },
        setAttribute: (name: string, value: string) => attributes.set(name, value),
        getAttribute: (name: string) => attributes.get(name),
    } as unknown as HTMLElement;
}

function makeViewer() {
    return {
        gameState: {
            stepForward: vi.fn(() => true),
            stepBackward: vi.fn(() => true),
            jumpTo: vi.fn(),
            getState: () => ({ lastEvent: { type: 'tsumo' } }),
        },
        update: vi.fn(),
        onPositionChange: vi.fn(),
    } as unknown as ViewerLike;
}

describe('Replay controls', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.stubGlobal('window', globalThis);
    });
    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
    });

    it('seeking pauses playback and reports the new position once', () => {
        const viewer = makeViewer();
        const controller = new ReplayController(viewer);
        const button = makeButton();
        controller.toggleAutoPlay(button);
        controller.seekTo(32);
        vi.advanceTimersByTime(2000);
        expect(viewer.gameState.jumpTo).toHaveBeenCalledWith(32);
        expect(viewer.gameState.stepForward).not.toHaveBeenCalled();
        expect(viewer.onPositionChange).toHaveBeenCalledTimes(1);
        expect(button.getAttribute('aria-pressed')).toBe('false');
    });

    it('manual stepping cancels scheduled playback', () => {
        const viewer = makeViewer();
        const controller = new ReplayController(viewer);
        controller.toggleAutoPlay(makeButton());
        controller.stepForward();
        vi.advanceTimersByTime(1000);
        expect(viewer.gameState.stepForward).toHaveBeenCalledTimes(1);
        expect(controller.autoPlayTimer).toBeNull();
    });

    it('applies playback speed to subsequent steps', () => {
        const viewer = makeViewer();
        const controller = new ReplayController(viewer);
        controller.playbackSpeed = 2;
        controller.toggleAutoPlay(makeButton());
        vi.advanceTimersByTime(99);
        expect(viewer.gameState.stepForward).not.toHaveBeenCalled();
        vi.advanceTimersByTime(101);
        expect(viewer.gameState.stepForward).toHaveBeenCalledTimes(2);
        controller.stopAutoPlay();
    });

    it('resets the play state when reaching the end', () => {
        const viewer = makeViewer();
        vi.mocked(viewer.gameState.stepForward).mockReturnValue(false);
        const controller = new ReplayController(viewer);
        const button = makeButton();
        controller.toggleAutoPlay(button);
        vi.advanceTimersByTime(1000);
        expect(controller.autoPlayTimer).toBeNull();
        expect(button.getAttribute('aria-pressed')).toBe('false');
        expect(viewer.onPositionChange).not.toHaveBeenCalled();
    });

    it('leaves arrow keys to the seek slider and speed selector', () => {
        const viewer = makeViewer();
        const controller = new ReplayController(viewer);
        let keydown: (event: any) => void = () => {};
        controller.setupKeyboardControls({
            addEventListener: (_name: string, handler: typeof keydown) => {
                keydown = handler;
            },
        } as unknown as HTMLElement);
        const preventDefault = vi.fn();
        keydown({ key: 'ArrowRight', target: { closest: () => ({}) }, preventDefault });
        expect(viewer.gameState.stepForward).not.toHaveBeenCalled();
        expect(preventDefault).not.toHaveBeenCalled();
        keydown({ key: 'ArrowRight', target: { closest: () => null }, preventDefault });
        expect(viewer.gameState.stepForward).toHaveBeenCalledTimes(1);
        expect(preventDefault).toHaveBeenCalledTimes(1);
    });

    it('removes keyboard listeners when a viewer is destroyed and remounted', () => {
        const target = new EventTarget();
        const oldViewer = makeViewer();
        const oldController = new ReplayController(oldViewer);
        oldController.setupKeyboardControls(target as unknown as HTMLElement);
        oldController.destroy();
        const newViewer = makeViewer();
        const newController = new ReplayController(newViewer);
        newController.setupKeyboardControls(target as unknown as HTMLElement);
        const event = new Event('keydown', { cancelable: true });
        Object.defineProperty(event, 'key', { value: 'ArrowRight' });
        target.dispatchEvent(event);
        expect(oldViewer.gameState.stepForward).not.toHaveBeenCalled();
        expect(newViewer.gameState.stepForward).toHaveBeenCalledTimes(1);
        newController.destroy();
    });
});
