import type { BoardState, PlayerState } from '../types';

/** The drawn tile stays separate through self-draw victory and round results,
 * regardless of which player is viewing the table. Ron adds no hand tile. */
export function drawnTileIndex(
    player: Pick<PlayerState, 'hand'>,
    playerIdx: number,
    state: Pick<BoardState, 'currentActor' | 'lastEvent'>,
): number {
    const event = state.lastEvent;
    if (state.currentActor !== playerIdx || player.hand.length % 3 !== 2 || !event) return -1;
    const hasDraw =
        ((event.type === 'tsumo' || event.type === 'reach') && event.actor === playerIdx) ||
        (event.type === 'hora' && event.actor === playerIdx && event.target === playerIdx) ||
        (event.type === 'end_kyoku' &&
            event.meta?.results?.some(
                (result: { actor: number; target: number }) =>
                    result.actor === playerIdx && result.target === playerIdx,
            ));
    return hasDraw ? player.hand.length - 1 : -1;
}
