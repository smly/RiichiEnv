import { describe, expect, it } from 'vitest';
import { classifyHandLimit, computeKyokuSummaries } from '../analyzer';
import type { GameState } from '../game_state';
import type { MjaiEvent } from '../types';

const start = (options: Partial<MjaiEvent> = {}): MjaiEvent => ({
    type: 'start_kyoku',
    oya: 0,
    honba: 0,
    kyoku: 1,
    scores: [25000, 25000, 25000, 25000],
    ...options,
});
function game(events: MjaiEvent[], pc = 4): GameState {
    return {
        events,
        cursor: 1,
        config: { playerCount: pc },
        kyokus: events.flatMap((e, index) =>
            e.type === 'start_kyoku'
                ? [{ index, round: (e.kyoku ?? 1) - 1, honba: e.honba ?? 0, scores: e.scores }]
                : [],
        ),
    } as unknown as GameState;
}
const end: MjaiEvent = { type: 'end_kyoku' };

describe('kyoku summaries from unvisited raw logs', () => {
    it('includes riichi deposits and uses the next deal without replaying or mutating events', () => {
        const gs = game([
            start(),
            { type: 'reach', actor: 0 },
            { type: 'reach_accepted', actor: 0 },
            { type: 'reach_accepted', actor: 1 },
            { type: 'hora', actor: 0, target: 3, deltas: [9700, 0, 0, -7700] },
            end,
            start({ honba: 1, scores: [33700, 24000, 25000, 17300] }),
        ]);
        const original = JSON.stringify(gs);
        const [s] = computeKyokuSummaries(gs);
        expect(s.deltas).toEqual([8700, -1000, 0, -7700]);
        expect(s.result?.winners?.[0]).toMatchObject({ actor: 0, target: 3, isTsumo: false, points: 7700 });
        expect(s.result?.winners?.[0].limit).toBeUndefined();
        expect(s.playerActions.map((p) => p.riichi)).toEqual([true, true, false, false]);
        expect(s.playerActions[3].houjuu).toBe(true);
        expect(JSON.stringify(gs)).toBe(original);
    });

    it('does not charge a duplicated accepted riichi twice and calculates the last round', () => {
        const [s] = computeKyokuSummaries(
            game([
                start(),
                { type: 'reach', step: 2, actor: 2 },
                { type: 'reach_accepted', actor: 2 },
                { type: 'hora', actor: 1, target: 0, deltas: [-8000, 9000, 0, 0] },
                end,
            ]),
        );
        expect(s.endScores).toEqual([17000, 34000, 24000, 25000]);
        expect(s.result?.winners?.[0].limit).toBe('mangan');
        expect(s.deltasKnown).toBe(true);
    });

    it('accumulates multi-ron and does not count enriched results again', () => {
        const [s] = computeKyokuSummaries(
            game([
                start(),
                { type: 'hora', actor: 0, target: 2, deltas: [12000, 0, -12000, 0] },
                { type: 'hora', actor: 1, target: 2, deltas: [0, 12000, -12000, 0] },
                {
                    type: 'end_kyoku',
                    meta: {
                        results: [
                            { actor: 0, target: 2, score: { han: 5, fu: 30, points: 12000 } },
                            { actor: 1, target: 2 },
                        ],
                    },
                },
            ]),
        );
        expect(s.deltas).toEqual([12000, 12000, -24000, 0]);
        expect(s.result?.winners).toHaveLength(2);
        expect(s.result?.winners?.map((w) => w.limit)).toEqual(['mangan', 'haneman']);
    });

    it('respects absolute scores instead of adding the same deltas again', () => {
        const [s] = computeKyokuSummaries(
            game([
                start(),
                {
                    type: 'hora',
                    actor: 1,
                    target: 0,
                    scores: [17000, 33000, 25000, 25000],
                    deltas: [-8000, 8000, 0, 0],
                },
                end,
            ]),
        );
        expect(s.endScores).toEqual([17000, 33000, 25000, 25000]);
    });

    it('handles drawn rounds without double-counting end_kyoku metadata', () => {
        const draw = { deltas: [1500, -1500, 1500, -1500], reason: 'fanpai' };
        const [s] = computeKyokuSummaries(
            game([start(), { type: 'ryukyoku', ...draw }, { type: 'end_kyoku', meta: { ryukyoku: draw } }]),
        );
        expect(s.deltas).toEqual(draw.deltas);
        expect(s.playerActions.map((p) => p.tenpai)).toEqual([true, false, true, false]);
        expect(s.result?.type).toBe('ryukyoku');
    });

    it('supports metadata-only wins and distinguishes unknown settlement from zero', () => {
        const [s] = computeKyokuSummaries(
            game([
                start(),
                {
                    type: 'end_kyoku',
                    meta: { results: [{ actor: 1, target: 0, score: { han: 0, fu: 0, points: 64000, yaku: [48] } }] },
                },
            ]),
        );
        expect(s.result?.winners?.[0].limit).toBe('double-yakuman');
        expect(s.playerActions[1].hora).toBe(true);
        expect(s.playerActions[0].houjuu).toBe(true);
        expect(s.deltasKnown).toBe(false);
    });

    it('keeps an unfinished round distinct from a draw', () => {
        const [s] = computeKyokuSummaries(game([start(), { type: 'reach_accepted', actor: 2 }]));
        expect(s.result).toBeNull();
        expect(s.completed).toBe(false);
        expect(s.deltas).toEqual([0, 0, -1000, 0]);
    });

    it('uses the paid value when rules score a double-yakuman variant as single', () => {
        const [s] = computeKyokuSummaries(
            game([
                start(),
                {
                    type: 'end_kyoku',
                    meta: { results: [{ actor: 1, target: 0, score: { han: 13, fu: 0, points: 32000, yaku: [48] } }] },
                },
            ]),
        );
        expect(s.result?.winners?.[0].limit).toBe('yakuman');
    });

    it('reads tenpai and final scores from metadata-only draws', () => {
        const [s] = computeKyokuSummaries(
            game([
                start(),
                {
                    type: 'end_kyoku',
                    meta: {
                        ryukyoku: {
                            reason: 'fanpai',
                            scores: [28000, 24000, 24000, 24000],
                            deltas: [3000, -1000, -1000, -1000],
                        },
                    },
                },
            ]),
        );
        expect(s.playerActions.map((p) => p.tenpai)).toEqual([true, false, false, false]);
        expect(s.deltas).toEqual([3000, -1000, -1000, -1000]);
    });

    it.each([
        [4, 0, [12000, -4000, -4000, -4000], 'mangan'],
        [4, 0, [11700, -3900, -3900, -3900], undefined],
        [3, 0, [8000, -4000, -4000], 'mangan'],
        [3, 1, [-8000, 12000, -4000], 'baiman'],
        [4, 1, [-16000, 32000, -8000, -8000], 'yakuman'],
        [4, 1, [-32000, 64000, -16000, -16000], 'double-yakuman'],
        [4, 1, [-32000, 32000, 0, 0], 'yakuman'],
    ] as const)('infers tsumo limits for %i players (winner %i)', (pc, actor, deltas, limit) => {
        const [s] = computeKyokuSummaries(
            game(
                [
                    start({ scores: Array(pc).fill(35000) }),
                    { type: 'hora', actor, target: actor, deltas: [...deltas] },
                    end,
                ],
                pc,
            ),
        );
        expect(s.result?.winners?.[0].limit).toBe(limit);
    });

    it('removes honba from the hand value while preserving the actual movement', () => {
        const [s] = computeKyokuSummaries(
            game([start({ honba: 2 }), { type: 'hora', actor: 1, target: 0, deltas: [-8600, 8600, 0, 0] }, end]),
        );
        expect(s.result?.winners?.[0]).toMatchObject({ points: 8000, limit: 'mangan' });
        expect(s.deltas).toEqual([-8600, 8600, 0, 0]);
    });
});

describe('hand limit thresholds', () => {
    it.each([
        [4, 30, 0, undefined],
        [4, 40, 0, 'mangan'],
        [3, 70, 0, 'mangan'],
        [5, 20, 0, 'mangan'],
        [6, 30, 0, 'haneman'],
        [8, 30, 0, 'baiman'],
        [11, 30, 0, 'sanbaiman'],
        [13, 30, 0, 'yakuman'],
        [0, 0, 16000, 'double-yakuman'],
        [0, 0, 24000, '3x-yakuman'],
    ] as const)('%i han %i fu, base %i', (han, fu, base, limit) => {
        expect(classifyHandLimit(han, fu, base)).toBe(limit);
    });
});
