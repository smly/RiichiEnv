import { describe, expect, it } from 'vitest';
import { drawnTileIndex } from '../renderers/hand_presentation';
import type { MjaiEvent } from '../types';

describe('drawn tile presentation for every seat', () => {
    const player = { hand: [...Array(13).fill('1m'), '4p'] };

    it.each([0, 1, 2, 3])('preserves the separated tile through draw and win for player %i', (actor) => {
        const events: MjaiEvent[] = [
            { type: 'tsumo', actor, pai: '4p' },
            { type: 'reach', actor },
            { type: 'hora', actor, target: actor },
            { type: 'end_kyoku', meta: { results: [{ actor, target: actor }] } },
        ];
        for (const lastEvent of events) {
            expect(drawnTileIndex(player, actor, { currentActor: actor, lastEvent })).toBe(13);
            expect(drawnTileIndex(player, (actor + 1) % 4, { currentActor: actor, lastEvent })).toBe(-1);
        }
    });

    it.each([11, 8, 5, 2])('supports a drawn tile with %i concealed tiles after calls', (length) => {
        expect(
            drawnTileIndex({ hand: Array(length).fill('4p') }, 1, {
                currentActor: 1,
                lastEvent: { type: 'hora', actor: 1, target: 1 },
            }),
        ).toBe(length - 1);
    });

    it.each<MjaiEvent>([
        { type: 'dahai', actor: 0, pai: '4p' },
        { type: 'pon', actor: 0, target: 1 },
        { type: 'hora', actor: 0, target: 1 },
        { type: 'end_kyoku', meta: { results: [{ actor: 0, target: 1 }] } },
    ])('does not separate a tile for $type without a self draw', (lastEvent) => {
        expect(drawnTileIndex(player, 0, { currentActor: 0, lastEvent })).toBe(-1);
    });

    it('does not separate a tile from a thirteen-tile hand', () => {
        expect(
            drawnTileIndex({ hand: player.hand.slice(0, 13) }, 0, {
                currentActor: 0,
                lastEvent: { type: 'hora', actor: 0, target: 0 },
            }),
        ).toBe(-1);
    });
});
