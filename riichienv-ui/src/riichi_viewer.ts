import { computeKyokuKeyEvents, computeKyokuSummaries } from './analyzer';
import type { BaseViewer } from './base_viewer';
import type { KyokuInfo, KyokuKeyEvent, KyokuSummary, ViewerEventMap, ViewerOptions, ViewerPosition } from './types';
import { Viewer } from './viewer';
import { Viewer3D } from './viewer_3d';

type EventHandler<T> = (data: T) => void;

export class RiichiViewer {
    private _viewer: BaseViewer;
    private _listeners: { [K in keyof ViewerEventMap]?: Set<EventHandler<ViewerEventMap[K]>> } = {};
    private _lastKyokuIndex: number = -1;
    private _destroyed = false;

    private constructor(viewer: BaseViewer) {
        this._viewer = viewer;
        this._lastKyokuIndex = this._getCurrentKyokuIndex();

        // Wire up internal callbacks to emit events
        this._viewer.onPositionChange = () => {
            if (this._destroyed) return;
            const pos = this.getPosition();
            this._emit('positionChange', { kyokuIndex: pos.kyokuIndex, step: pos.step });

            if (pos.kyokuIndex !== this._lastKyokuIndex) {
                this._lastKyokuIndex = pos.kyokuIndex;
                const kyoku = this._viewer.gameState.kyokus[pos.kyokuIndex];
                if (kyoku) {
                    this._emit('kyokuChange', {
                        kyokuIndex: pos.kyokuIndex,
                        round: kyoku.round,
                        honba: kyoku.honba,
                    });
                }
            }
        };

        this._viewer.onViewpointChangeCallback = (viewpoint: number) => {
            if (this._destroyed) return;
            this._emit('viewpointChange', { viewpoint });
        };
    }

    static mount(container: HTMLElement | string, options: ViewerOptions): RiichiViewer {
        const el = typeof container === 'string' ? document.getElementById(container.replace(/^#/, '')) : container;
        if (!el) throw new Error(`Container ${container} not found`);

        const rendererType = options.renderer ?? '3d';
        const initialStep = options.initialPosition?.step;

        let viewer: BaseViewer;
        if (rendererType === '2d') {
            viewer = Viewer.fromElement(
                el,
                options.log,
                initialStep,
                options.perspective,
                options.freeze ?? false,
                undefined,
                undefined,
                options.players,
            );
        } else {
            viewer = Viewer3D.fromElement(
                el,
                options.log,
                initialStep,
                options.perspective,
                options.freeze ?? false,
                undefined,
                undefined,
                options.players,
            );
        }

        const rv = new RiichiViewer(viewer);

        // Handle initial kyoku position
        if (options.initialPosition?.kyoku !== undefined && initialStep === undefined) {
            rv.goToKyoku(options.initialPosition.kyoku);
        }

        return rv;
    }

    // Navigation
    goToKyoku(kyokuIndex: number) {
        this._viewer.gameState.jumpToKyoku(kyokuIndex);
        this._viewer.update();
        this._viewer.onPositionChange?.();
    }

    goToStep(stepIndex: number) {
        this._viewer.gameState.jumpTo(stepIndex);
        this._viewer.update();
        this._viewer.onPositionChange?.();
    }

    stepForward() {
        if (this._viewer.gameState.stepForward()) {
            this._viewer.update();
            this._viewer.onPositionChange?.();
        }
    }

    stepBackward() {
        if (this._viewer.gameState.stepBackward()) {
            this._viewer.update();
            this._viewer.onPositionChange?.();
        }
    }

    nextTurn() {
        const vp = this._viewer.renderer.viewpoint;
        if (this._viewer.gameState.jumpToNextTurn(vp)) {
            this._viewer.update();
            this._viewer.onPositionChange?.();
        }
    }

    prevTurn() {
        const vp = this._viewer.renderer.viewpoint;
        if (this._viewer.gameState.jumpToPrevTurn(vp)) {
            this._viewer.update();
            this._viewer.onPositionChange?.();
        }
    }

    nextKyoku() {
        if (this._viewer.gameState.jumpToNextKyoku()) {
            this._viewer.update();
            this._viewer.onPositionChange?.();
        }
    }

    prevKyoku() {
        if (this._viewer.gameState.jumpToPrevKyoku()) {
            this._viewer.update();
            this._viewer.onPositionChange?.();
        }
    }

    setViewpoint(playerIndex: number) {
        if (this._viewer.renderer.viewpoint !== playerIndex) {
            this._viewer.renderer.viewpoint = playerIndex;
            this._viewer.update();
            this._emit('viewpointChange', { viewpoint: playerIndex });
        }
    }

    toggleAutoPlay() {
        if (!this._viewer.controller) return;
        // Find or create a dummy button for the controller
        const btn = this._viewer.container.querySelector(
            '.icon-btn[title="Auto"], .icon-btn[title="Play/Pause"]',
        ) as HTMLElement;
        if (btn) {
            this._viewer.controller.toggleAutoPlay(btn);
        }
    }

    // State queries
    getPosition(): ViewerPosition {
        const gs = this._viewer.gameState;
        return {
            kyokuIndex: this._getCurrentKyokuIndex(),
            step: gs.cursor,
            totalSteps: gs.events.length,
        };
    }

    getKyokuList(): KyokuInfo[] {
        return this._viewer.gameState.kyokus.map((k, idx) => ({
            index: idx,
            round: k.round,
            honba: k.honba,
            scores: [...k.scores],
        }));
    }

    getViewpoint(): number {
        return this._viewer.renderer.viewpoint;
    }

    getPlayerNames(): string[] {
        return [...this._viewer.gameState.getState().playerNames];
    }

    // Analysis
    getKyokuSummaries(): KyokuSummary[] {
        return computeKyokuSummaries(this._viewer.gameState);
    }

    getKyokuKeyEvents(kyokuIndex: number): KyokuKeyEvent[] {
        return computeKyokuKeyEvents(this._viewer.gameState, kyokuIndex);
    }

    // Events
    on<K extends keyof ViewerEventMap>(event: K, handler: EventHandler<ViewerEventMap[K]>): void {
        const listeners = this._listeners as Record<K, Set<EventHandler<ViewerEventMap[K]>> | undefined>;
        (listeners[event] ??= new Set()).add(handler);
    }

    off<K extends keyof ViewerEventMap>(event: K, handler: EventHandler<ViewerEventMap[K]>): void {
        (this._listeners[event] as Set<EventHandler<ViewerEventMap[K]>> | undefined)?.delete(handler);
    }

    // Cleanup
    destroy() {
        this._destroyed = true;
        this._listeners = {};
        this._viewer.destroy();
    }

    private _emit<K extends keyof ViewerEventMap>(event: K, data: ViewerEventMap[K]) {
        const handlers = this._listeners[event] as Set<EventHandler<ViewerEventMap[K]>> | undefined;
        if (handlers) {
            for (const handler of handlers) {
                handler(data);
            }
        }
    }

    private _getCurrentKyokuIndex(): number {
        const gs = this._viewer.gameState;
        const cursor = gs.cursor;
        const kyokus = gs.kyokus;
        let idx = 0;
        for (let i = kyokus.length - 1; i >= 0; i--) {
            if (cursor > kyokus[i].index) {
                idx = i;
                break;
            }
        }
        return idx;
    }
}
