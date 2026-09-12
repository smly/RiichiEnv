import { describe, expect, it, vi } from 'vitest';

vi.mock('../wasm/loader', () => ({ isWasmReady: () => false }));

import { createGameConfig3P, createGameConfig4P } from '../config';
import { GameState } from '../game_state';
import { relativeSeat } from '../renderers/seat_position';

function game(playerCount: number, dealer: number) {
    return new GameState(
        [
            {
                type: 'start_kyoku',
                bakaze: 'E',
                kyoku: dealer + 1,
                oya: dealer,
                scores: Array(playerCount).fill(25000),
                tehais: Array.from({ length: playerCount }, () => Array(13).fill('?')),
                dora_marker: '1p',
                honba: 0,
                kyotaku: 0,
            },
        ],
        playerCount === 3 ? createGameConfig3P() : createGameConfig4P(),
    );
}

describe('fixed physical table edges', () => {
    it.each([0, 1, 2])('preserves players and the empty seat for every viewpoint with dealer %i', (dealer) => {
        const { current } = game(3, dealer);
        // Rows: viewpoint player; columns: player index. Round winds do not affect placement.
        const expected = [
            [0, 1, 2],
            [3, 0, 1],
            [2, 3, 0],
        ];
        current.players.forEach((_, viewpoint) => {
            const seats = current.players.map((_, index) => {
                const seat = relativeSeat(current, index, viewpoint);
                expect(seat).toBe(expected[viewpoint][index]);
                return seat;
            });
            expect(new Set(seats).size).toBe(3);
            expect(seats).not.toContain([3, 2, 1][viewpoint]);
        });
    });
    it.each([0, 1, 2, 3])('preserves four-player positions with dealer %i', (dealer) => {
        const { current } = game(4, dealer);
        for (let viewpoint = 0; viewpoint < 4; viewpoint++) {
            for (let player = 0; player < 4; player++) {
                expect(relativeSeat(current, player, viewpoint)).toBe((player - viewpoint + 4) % 4);
            }
        }
    });
    it.each([0, 1, 2])('keeps called tiles facing their source across dealer changes (%i)', (dealer) => {
        const { current } = game(3, dealer);
        expect(relativeSeat(current, 0, 1)).toBe(3); // Player 0 is to player 1's left.
        expect(relativeSeat(current, 0, 2)).toBe(2); // Player 0 is opposite player 2.
        expect(relativeSeat(current, 1, 2)).toBe(3); // Player 1 is to player 2's left.
    });

    it('updates winds without changing positions when seeking forward and backward between rounds', () => {
        const events = [0, 1, 2].map((dealer) => ({
            type: 'start_kyoku',
            bakaze: 'E',
            kyoku: dealer + 1,
            oya: dealer,
            scores: [35000, 35000, 35000],
            tehais: Array.from({ length: 3 }, () => Array(13).fill('?')),
            dora_marker: '1p',
            honba: 0,
            kyotaku: 0,
        }));
        const gs = new GameState(events, createGameConfig3P());
        for (const step of [1, 2, 3, 2, 1]) {
            gs.jumpTo(step);
            const state = gs.getState();
            expect(state.players.map((player) => player.wind)).toEqual(
                [0, 1, 2].map((player) => (player - (step - 1) + 3) % 3),
            );
            for (let viewpoint = 0; viewpoint < 3; viewpoint++) {
                expect(state.players.map((_, player) => relativeSeat(state, player, viewpoint))).toEqual(
                    [0, 1, 2].map((player) => (player - viewpoint + 4) % 4),
                );
            }
        }
    });
});
