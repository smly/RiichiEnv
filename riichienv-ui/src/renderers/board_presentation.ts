import type { BoardState } from '../types';

export interface DisplayOptions {
    showOpponentHands: boolean;
    showWaits: boolean;
}

export const DEFAULT_DISPLAY_OPTIONS: Readonly<DisplayOptions> = {
    showOpponentHands: true,
    showWaits: true,
};

/** Filter board presentation without changing replay data or result information. */
export function presentBoard(
    state: BoardState,
    viewpoint: number,
    options: Readonly<DisplayOptions> = DEFAULT_DISPLAY_OPTIONS,
): BoardState {
    if (options.showOpponentHands && options.showWaits) return state;
    return {
        ...state,
        players: state.players.map((player, index) => ({
            ...player,
            // Keep tile count/order so the drawn tile retains its separate position.
            hand: options.showOpponentHands || index === viewpoint ? player.hand : player.hand.map(() => '?'),
            waits: options.showWaits ? player.waits : [],
        })),
    };
}
