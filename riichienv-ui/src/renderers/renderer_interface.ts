import { BoardState } from '../types';

/**
 * Common interface for all renderer implementations (2D, 3D, etc.).
 *
 * Viewers interact with renderers exclusively through this interface,
 * allowing different rendering backends to be swapped in without
 * changing the viewer or game-state logic.
 */
export interface IRenderer {
    /** Index of the player shown at the bottom of the board. */
    viewpoint: number;

    /** Callback fired when the user clicks a player info area to change viewpoint. */
    onViewpointChange: ((pIdx: number) => void) | null;

    /** Callback fired when the user clicks the center info area. */
    onCenterClick: (() => void) | null;

    /** Render the current board state. */
    render(state: BoardState, debugPanel?: HTMLElement): void;

    /** Optional: resize the rendered board to fit a given width. */
    resize?(width: number): void;
}
