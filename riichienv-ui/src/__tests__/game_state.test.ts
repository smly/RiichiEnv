import { beforeEach, describe, expect, it, vi } from 'vitest';

// Mock WASM modules before importing GameState
vi.mock('../wasm/bridge', () => ({
    calculateWaits: vi.fn(() => null),
    calculateScore: vi.fn(() => null),
    mjaiToTileId: vi.fn(() => null),
    tileIdToMjai: vi.fn(() => null),
}));

vi.mock('../wasm/loader', () => ({
    isWasmReady: vi.fn(() => false),
}));

import { createGameConfig3P } from '../config';
import { GameState } from '../game_state';
import type { MjaiEvent } from '../types';

// Helper: create a minimal start_kyoku event
function makeStartKyoku(
    opts?: Partial<{
        bakaze: string;
        kyoku: number;
        honba: number;
        kyotaku: number;
        oya: number;
        scores: number[];
        dora_marker: string;
        tehais: string[][];
    }>,
): MjaiEvent {
    const playerCount = opts?.tehais?.length ?? 4;
    const defaultTehais = Array(playerCount)
        .fill(null)
        .map(() => ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '1p', '2p', '3p', '4p']);
    return {
        type: 'start_kyoku',
        bakaze: opts?.bakaze ?? 'E',
        kyoku: opts?.kyoku ?? 1,
        honba: opts?.honba ?? 0,
        kyotaku: opts?.kyotaku ?? 0,
        oya: opts?.oya ?? 0,
        scores: opts?.scores ?? [25000, 25000, 25000, 25000],
        dora_marker: opts?.dora_marker ?? '1s',
        tehais: opts?.tehais ?? defaultTehais,
    };
}

describe('GameState', () => {
    describe('initialization', () => {
        it('should create with empty events', () => {
            const gs = new GameState([]);
            expect(gs.events).toHaveLength(0);
            expect(gs.cursor).toBe(0);
        });

        it('should use default 4P config', () => {
            const gs = new GameState([]);
            expect(gs.config.playerCount).toBe(4);
        });

        it('should accept custom config', () => {
            const config = createGameConfig3P();
            const gs = new GameState([], config);
            expect(gs.config.playerCount).toBe(3);
        });

        it('should return initial state with playerCount', () => {
            const gs = new GameState([]);
            const state = gs.getState();
            expect(state.playerCount).toBe(4);
        });

        it('should have correct number of players in initial state', () => {
            const gs = new GameState([]);
            const state = gs.getState();
            expect(state.players).toHaveLength(4);
        });

        it('should have default scores in initial state', () => {
            const gs = new GameState([]);
            const state = gs.getState();
            state.players.forEach((p) => {
                expect(p.score).toBe(25000);
            });
        });

        it('should have correct initial wall remaining', () => {
            const gs = new GameState([]);
            const state = gs.getState();
            expect(state.wallRemaining).toBe(70);
        });

        it('should filter out start_game and end_game events', () => {
            const events: MjaiEvent[] = [{ type: 'start_game' }, makeStartKyoku(), { type: 'end_game' }];
            const gs = new GameState(events);
            expect(gs.events).toHaveLength(1);
            expect(gs.events[0].type).toBe('start_kyoku');
        });
    });

    describe('3P config initialization', () => {
        it('should create with 3 players', () => {
            const config = createGameConfig3P();
            const gs = new GameState([], config);
            const state = gs.getState();
            expect(state.playerCount).toBe(3);
            expect(state.players).toHaveLength(3);
        });

        it('should have 3P default scores', () => {
            const config = createGameConfig3P();
            const gs = new GameState([], config);
            const state = gs.getState();
            state.players.forEach((p) => {
                expect(p.score).toBe(35000);
            });
        });

        it('should have 3P initial wall remaining', () => {
            const config = createGameConfig3P();
            const gs = new GameState([], config);
            const state = gs.getState();
            expect(state.wallRemaining).toBe(55);
        });
    });

    describe('start_kyoku processing', () => {
        it('should set player hands from tehais', () => {
            const events: MjaiEvent[] = [makeStartKyoku()];
            const gs = new GameState(events);
            const state = gs.getState();
            state.players.forEach((p) => {
                expect(p.hand).toHaveLength(13);
            });
        });

        it('should assign winds based on oya', () => {
            const events: MjaiEvent[] = [makeStartKyoku({ oya: 1 })];
            const gs = new GameState(events);
            const state = gs.getState();
            // oya (player 1) should have wind 0 (East)
            expect(state.players[1].wind).toBe(0);
            // player 0 should be North (wind 3)
            expect(state.players[0].wind).toBe(3);
        });

        it('should set dora markers', () => {
            const events: MjaiEvent[] = [makeStartKyoku({ dora_marker: '5p' })];
            const gs = new GameState(events);
            const state = gs.getState();
            expect(state.doraMarkers).toEqual(['5p']);
        });

        it('should set round from bakaze and kyoku', () => {
            const events: MjaiEvent[] = [makeStartKyoku({ bakaze: 'S', kyoku: 2 })];
            const gs = new GameState(events);
            const state = gs.getState();
            // S2 = offset 4 + (2-1) = 5
            expect(state.round).toBe(5);
        });

        it('should reset conditions', () => {
            const events: MjaiEvent[] = [makeStartKyoku()];
            const gs = new GameState(events);
            const state = gs.getState();
            expect(state.conditions.callsMade).toBe(false);
            expect(state.conditions.afterKan).toBe(false);
            expect(state.conditions.ippatsu.every((v) => v === false)).toBe(true);
        });

        it('should reset wall remaining', () => {
            const events: MjaiEvent[] = [makeStartKyoku()];
            const gs = new GameState(events);
            const state = gs.getState();
            expect(state.wallRemaining).toBe(70);
        });
    });

    describe('tsumo processing', () => {
        it('should add tile to hand', () => {
            const events: MjaiEvent[] = [makeStartKyoku(), { type: 'tsumo', actor: 0, pai: '1s' }];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].hand).toHaveLength(14);
            expect(state.players[0].hand).toContain('1s');
        });

        it('should decrement wall remaining', () => {
            const events: MjaiEvent[] = [makeStartKyoku(), { type: 'tsumo', actor: 0, pai: '1s' }];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.wallRemaining).toBe(69);
        });

        it('should set current actor', () => {
            const events: MjaiEvent[] = [makeStartKyoku(), { type: 'tsumo', actor: 2, pai: '1s' }];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.currentActor).toBe(2);
        });
    });

    describe('dahai processing', () => {
        it('should remove tile from hand', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].hand).toHaveLength(13);
        });

        it('should add to discards', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].discards).toHaveLength(1);
            expect(state.players[0].discards[0].tile).toBe('1m');
        });

        it('should update condition tracking', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.firstTurnCompleted[0]).toBe(true);
            expect(state.conditions.turnCount).toBe(1);
        });
    });

    describe('call processing (pon/chi/kan)', () => {
        it('should set callsMade after pon', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'pon', actor: 2, target: 0, pai: '1m', consumed: ['1m', '1m'] },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.callsMade).toBe(true);
        });

        it('should clear all ippatsu after call', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'pon', actor: 2, target: 0, pai: '1m', consumed: ['1m', '1m'] },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.ippatsu.every((v) => v === false)).toBe(true);
        });

        it('should add meld to player', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'pon', actor: 2, target: 0, pai: '1m', consumed: ['1m', '1m'] },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[2].melds).toHaveLength(1);
            expect(state.players[2].melds[0].type).toBe('pon');
        });

        it('should set afterKan for daiminkan', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'daiminkan', actor: 2, target: 0, pai: '1m', consumed: ['1m', '1m', '1m'] },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.afterKan).toBe(true);
        });
    });

    describe('reach processing', () => {
        it('should set riichi on reach_accepted', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'reach_accepted', actor: 0 },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].riichi).toBe(true);
        });

        it('should set ippatsu on reach_accepted', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'reach_accepted', actor: 0 },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.ippatsu[0]).toBe(true);
        });

        it('should deduct 1000 points for riichi', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'reach_accepted', actor: 0 },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].score).toBe(24000);
            expect(state.kyotaku).toBe(1);
        });

        it('should detect double riichi (first turn, no calls)', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'reach_accepted', actor: 0 },
            ];
            const gs = new GameState(events);
            gs.jumpTo(events.length);
            const state = gs.getState();
            // First turn not completed before reach_accepted since dahai sets firstTurnCompleted
            // But the check is !callsMade && !firstTurnCompleted — by the time reach_accepted
            // is processed, firstTurnCompleted[0] is already true from the dahai, so
            // this should NOT be double riichi
            expect(state.conditions.doubleRiichi[0]).toBe(false);
        });
    });

    describe('step navigation', () => {
        let gs: GameState;

        beforeEach(() => {
            const events: MjaiEvent[] = [
                makeStartKyoku(),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'tsumo', actor: 1, pai: '2s' },
                { type: 'dahai', actor: 1, pai: '2m' },
            ];
            gs = new GameState(events);
        });

        it('should step forward', () => {
            const initialIdx = gs.current.eventIndex;
            const changed = gs.stepForward();
            expect(changed).toBe(true);
            expect(gs.current.eventIndex).toBe(initialIdx + 1);
        });

        it('should step backward', () => {
            gs.stepForward();
            const idx = gs.current.eventIndex;
            const changed = gs.stepBackward();
            expect(changed).toBe(true);
            expect(gs.current.eventIndex).toBe(idx - 1);
        });

        it('should not step forward past end', () => {
            // Step to end
            while (gs.stepForward()) {}
            const changed = gs.stepForward();
            expect(changed).toBe(false);
        });

        it('should not step backward below 1', () => {
            // Reset to beginning
            gs.jumpTo(1);
            const changed = gs.stepBackward();
            expect(changed).toBe(false);
        });

        it('should jump to specific index', () => {
            gs.jumpTo(3);
            expect(gs.current.eventIndex).toBe(3);
        });

        it('should clamp jump target', () => {
            gs.jumpTo(9999);
            expect(gs.current.eventIndex).toBe(gs.events.length);
        });
    });

    describe('kyoku management', () => {
        it('should detect kyoku checkpoints', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku({ bakaze: 'E', kyoku: 1 }),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'end_kyoku' },
                makeStartKyoku({ bakaze: 'E', kyoku: 2 }),
                { type: 'tsumo', actor: 0, pai: '2s' },
            ];
            const gs = new GameState(events);
            expect(gs.kyokus).toHaveLength(2);
            expect(gs.kyokus[0].round).toBe(0); // E1
            expect(gs.kyokus[1].round).toBe(1); // E2
        });

        it('should jump to kyoku by index', () => {
            const events: MjaiEvent[] = [
                makeStartKyoku({ bakaze: 'E', kyoku: 1 }),
                { type: 'tsumo', actor: 0, pai: '1s' },
                { type: 'end_kyoku' },
                makeStartKyoku({ bakaze: 'E', kyoku: 2, scores: [30000, 20000, 25000, 25000] }),
                { type: 'tsumo', actor: 0, pai: '2s' },
            ];
            const gs = new GameState(events);
            gs.jumpToKyoku(1);
            const state = gs.getState();
            expect(state.round).toBe(1); // E2
        });
    });

    describe('appendEvent (live mode)', () => {
        it('should append and process event', () => {
            const gs = new GameState([]);
            gs.appendEvent(makeStartKyoku());
            const state = gs.getState();
            expect(state.players[0].hand).toHaveLength(13);
        });

        it('should track kyokus on append', () => {
            const gs = new GameState([]);
            gs.appendEvent(makeStartKyoku());
            expect(gs.kyokus).toHaveLength(1);
        });

        it('should update totalEvents', () => {
            const gs = new GameState([]);
            gs.appendEvent(makeStartKyoku());
            expect(gs.current.totalEvents).toBe(1);
        });

        it('should ignore start_game and end_game', () => {
            const gs = new GameState([]);
            gs.appendEvent({ type: 'start_game' });
            expect(gs.events).toHaveLength(0);
        });
    });

    describe('kita processing', () => {
        it('should remove N tile from hand', () => {
            const tehais = Array(3)
                .fill(null)
                .map(() => ['1m', '9m', '1p', '9p', '1s', '9s', 'E', 'S', 'W', 'P', 'F', 'C', 'N']);
            const config = createGameConfig3P();
            const events: MjaiEvent[] = [
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000] }),
                { type: 'tsumo', actor: 0, pai: 'N' },
                { type: 'kita', actor: 0 },
            ];
            const gs = new GameState(events, config);
            gs.jumpTo(events.length);
            const state = gs.getState();
            // Hand should have 13 tiles (13 original - 1 N removed + 1 tsumo N - 1 kita N = 12? No:
            // After start_kyoku: 13 tiles. After tsumo: 14 tiles. After kita: 13 tiles (one N removed).
            expect(state.players[0].hand).toHaveLength(13);
            // One N should remain (from initial hand)
            expect(state.players[0].hand.filter((t) => t === 'N')).toHaveLength(1);
        });

        it('should increment kitaCount', () => {
            const tehais = Array(3)
                .fill(null)
                .map(() => ['1m', '9m', '1p', '9p', '1s', '9s', 'E', 'S', 'W', 'P', 'F', 'C', 'N']);
            const config = createGameConfig3P();
            const events: MjaiEvent[] = [
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000] }),
                { type: 'tsumo', actor: 0, pai: 'N' },
                { type: 'kita', actor: 0 },
            ];
            const gs = new GameState(events, config);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].kitaCount).toBe(1);
        });

        it('should increment kitaCount even when N is not in reconstructed hand', () => {
            const tehais = Array(3)
                .fill(null)
                .map(() => ['1m', '9m', '1p', '9p', '1s', '9s', 'E', 'S', 'W', 'P', 'F', 'C', '1m']);
            const config = createGameConfig3P();
            const events: MjaiEvent[] = [
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000] }),
                { type: 'kita', actor: 0 },
            ];
            const gs = new GameState(events, config);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].kitaCount).toBe(1);
        });

        it('should set afterKan for rinshan draw', () => {
            const tehais = Array(3)
                .fill(null)
                .map(() => ['1m', '9m', '1p', '9p', '1s', '9s', 'E', 'S', 'W', 'P', 'F', 'C', 'N']);
            const config = createGameConfig3P();
            const events: MjaiEvent[] = [
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000] }),
                { type: 'tsumo', actor: 0, pai: 'N' },
                { type: 'kita', actor: 0 },
            ];
            const gs = new GameState(events, config);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.afterKan).toBe(true);
        });

        it('should clear ippatsu on kita', () => {
            const tehais = Array(3)
                .fill(null)
                .map(() => ['1m', '9m', '1p', '9p', '1s', '9s', 'E', 'S', 'W', 'P', 'F', 'C', 'N']);
            const config = createGameConfig3P();
            const events: MjaiEvent[] = [
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000] }),
                { type: 'tsumo', actor: 0, pai: '1m' },
                { type: 'dahai', actor: 0, pai: '1m' },
                { type: 'reach_accepted', actor: 0 },
                { type: 'tsumo', actor: 1, pai: 'N' },
                { type: 'kita', actor: 1 },
            ];
            const gs = new GameState(events, config);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.conditions.ippatsu.every((v) => v === false)).toBe(true);
        });

        it('should reset kitaCount on new kyoku', () => {
            const tehais = Array(3)
                .fill(null)
                .map(() => ['1m', '9m', '1p', '9p', '1s', '9s', 'E', 'S', 'W', 'P', 'F', 'C', 'N']);
            const config = createGameConfig3P();
            const events: MjaiEvent[] = [
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000] }),
                { type: 'tsumo', actor: 0, pai: 'N' },
                { type: 'kita', actor: 0 },
                { type: 'end_kyoku' },
                makeStartKyoku({ tehais, scores: [35000, 35000, 35000], kyoku: 2 }),
            ];
            const gs = new GameState(events, config);
            gs.jumpTo(events.length);
            const state = gs.getState();
            expect(state.players[0].kitaCount).toBe(0);
        });
    });

    describe('condition tracking with config', () => {
        it('should have condition arrays matching player count for 4P', () => {
            const events: MjaiEvent[] = [makeStartKyoku()];
            const gs = new GameState(events);
            const state = gs.getState();
            expect(state.conditions.ippatsu).toHaveLength(4);
            expect(state.conditions.firstTurnCompleted).toHaveLength(4);
            expect(state.conditions.doubleRiichi).toHaveLength(4);
        });

        it('should have condition arrays matching player count for 3P', () => {
            const config = createGameConfig3P();
            const tehais3P = Array(3)
                .fill(null)
                .map(() => ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '1p', '2p', '3p', '4p']);
            const events: MjaiEvent[] = [makeStartKyoku({ tehais: tehais3P, scores: [35000, 35000, 35000] })];
            const gs = new GameState(events, config);
            const state = gs.getState();
            expect(state.conditions.ippatsu).toHaveLength(3);
            expect(state.conditions.firstTurnCompleted).toHaveLength(3);
            expect(state.conditions.doubleRiichi).toHaveLength(3);
        });
    });
});
