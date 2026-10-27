import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../wasm/loader', () => ({ isWasmReady: () => true }));
vi.mock('../wasm/bridge', () => ({
    mjaiToTileId: (tile: string) => ('mps'.indexOf(tile[1]) * 9 + Number(tile[0]) - 1) * 4,
    tileIdToMjai: (id: number) => {
        const type = Math.floor(id / 4);
        const tile = `${(type % 9) + 1}${'mps'[Math.floor(type / 9)]}`;
        return [16, 52, 88].includes(id) ? `${tile}r` : tile;
    },
    calculateWaits: vi.fn(() => [12]), // 4p, in 34-tile notation
    calculateScore: vi.fn(() => null),
}));

import { GameState } from '../game_state';
import type { MjaiEvent } from '../types';
import { calculateWaits, mjaiToTileId } from '../wasm/bridge';

const hand = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '2p', '3p', '4p', '4p'];
const start: MjaiEvent = {
    type: 'start_kyoku',
    bakaze: 'E',
    kyoku: 1,
    oya: 0,
    honba: 0,
    kyotaku: 0,
    dora_marker: '1s',
    scores: [25000, 25000, 25000, 25000],
    tehais: [hand, [], [], []],
};

describe('waits while holding a drawn tile', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(calculateWaits).mockReturnValue([12]);
    });

    const events: MjaiEvent[] = [
        start,
        { type: 'tsumo', actor: 0, pai: '4p' },
        { type: 'hora', actor: 0, target: 0 },
        { type: 'end_kyoku' },
    ];

    it.each([2, 3, 4])('reconstructs pre-draw waits when seeking to event %i', (step) => {
        const gs = new GameState(events);
        gs.jumpTo(step);
        expect(gs.current.players[0].waits).toEqual(['4p']);
        expect(gs.current.players[0].hand).toEqual([...hand, '4p']);
        expect(calculateWaits).toHaveBeenLastCalledWith(hand.map(mjaiToTileId), []);
    });

    it('keeps waits through forward play and backward seeking', () => {
        const gs = new GameState(events);
        gs.jumpTo(2);
        gs.stepForward();
        expect(gs.current.players[0].waits).toEqual(['4p']);
        gs.stepBackward();
        expect(gs.current.players[0].waits).toEqual(['4p']);
        expect(gs.current.players[0].hand).toEqual([...hand, '4p']);
    });

    it('uses the full waiting hand after a discard and on ron', () => {
        const gs = new GameState([
            start,
            { type: 'tsumo', actor: 0, pai: '4p' },
            { type: 'dahai', actor: 0, pai: '4p', tsumogiri: true },
            { type: 'hora', actor: 1, target: 0 },
        ]);
        gs.jumpTo(4);
        expect(gs.current.players[0].hand).toHaveLength(13);
        expect(calculateWaits).toHaveBeenLastCalledWith(hand.map(mjaiToTileId), []);
    });

    it.each(['m', 'p', 's'])('uses a normal five for %s waits when seeking and advancing', (suit) => {
        vi.mocked(calculateWaits).mockReturnValue(['mps'.indexOf(suit) * 9 + 4]);
        const normal = `5${suit}`;
        const red = `${normal}r`;
        const gs = new GameState([
            start,
            { type: 'tsumo', actor: 0, pai: red },
            { type: 'dahai', actor: 0, pai: red, tsumogiri: true },
        ]);
        gs.jumpTo(2);
        expect(gs.current.players[0].waits).toEqual([normal]);
        expect(gs.current.players[0].hand).toContain(red);
        gs.stepForward();
        expect(gs.current.players[0].waits).toEqual([normal]);
        gs.jumpTo(3);
        expect(gs.current.players[0].waits).toEqual([normal]);
    });
});
