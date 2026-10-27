import { describe, expect, it, vi } from 'vitest';

vi.mock('../wasm/loader', () => ({ isWasmReady: () => false }));

import { createGameConfig3P, createGameConfig4P } from '../config';
import { GameState } from '../game_state';
import { presentBoard } from '../renderers/board_presentation';
import { drawnTileIndex } from '../renderers/hand_presentation';

function board(playerCount: number) {
    const game = new GameState(
        [
            {
                type: 'start_kyoku',
                bakaze: 'E',
                kyoku: 1,
                oya: 0,
                scores: Array(playerCount).fill(25000),
                tehais: Array.from({ length: playerCount }, () => Array(13).fill('5mr')),
                dora_marker: '1p',
                honba: 0,
                kyotaku: 0,
            },
        ],
        playerCount === 3 ? createGameConfig3P() : createGameConfig4P(),
    );
    const state = game.getState();
    state.players.forEach((player) => {
        player.waits = ['4p', '5m'];
        player.discards = [{ tile: '4p' }];
        player.melds = [{ type: 'pon', tiles: ['1p', '1p', '1p'], from: 0 }];
    });
    state.players[1].hand.push('4p');
    state.currentActor = 1;
    state.lastEvent = { type: 'tsumo', actor: 1, pai: '4p' };
    return state;
}

describe('display settings', () => {
    it.each([3, 4])('conceals every opponent including their draw from all %i-player viewpoints', (count) => {
        const state = board(count);
        const original = structuredClone(state);
        for (let viewpoint = 0; viewpoint < count; viewpoint++) {
            const displayed = presentBoard(state, viewpoint, { showOpponentHands: false, showWaits: true });
            displayed.players.forEach((player, index) => {
                expect(player.hand).toEqual(
                    index === viewpoint ? state.players[index].hand : Array(player.hand.length).fill('?'),
                );
                expect(player.hand).toHaveLength(state.players[index].hand.length);
                expect(player.waits).toEqual(['4p', '5m']);
                expect(player.melds).toEqual(original.players[index].melds);
                expect(player.discards).toEqual(original.players[index].discards);
            });
            expect(drawnTileIndex(displayed.players[1], 1, displayed)).toBe(13);
            expect(displayed.doraMarkers).toEqual(original.doraMarkers);
        }
        expect(state).toEqual(original);
        expect(presentBoard(state, 0).players[1].hand[13]).toBe('4p');
    });

    it.each([
        true,
        false,
    ])('clears all panel/highlight wait inputs independently of opponent visibility (%s)', (showOpponentHands) => {
        const state = board(4);
        const displayed = presentBoard(state, 0, { showOpponentHands, showWaits: false });
        expect(displayed.players.every((player) => player.waits?.length === 0)).toBe(true);
        expect(displayed.players[0].hand).toEqual(state.players[0].hand);
        expect(state.players.every((player) => player.waits?.length === 2)).toBe(true);
        expect(presentBoard(state, 0).players[1].waits).toEqual(['4p', '5m']);
    });
});
