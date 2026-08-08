import { describe, expect, it, vi } from 'vitest';
import { EventJournal } from '../event_journal';
import type { MjaiEvent } from '../types';

const startGame = '{"type":"start_game","names":["A","B","C","D"]}';

function startKyoku(kyoku: number): string {
    return JSON.stringify({
        type: 'start_kyoku',
        bakaze: 'E',
        kyoku,
        honba: 0,
        tehais: [['1m'], ['2m'], ['3m'], ['4m']],
    });
}

describe('EventJournal', () => {
    it('preserves raw events and indexes completed kyoku spans', () => {
        const customSpacing = '{ "type": "custom", "future": {"x": 1} }';
        const journal = EventJournal.fromEvents([startGame, startKyoku(1), customSpacing, '{"type":"end_kyoku"}']);

        expect(journal.rawEvents[2]).toBe(customSpacing);
        expect(journal.completedKyokuSpans).toEqual([
            {
                start: 1,
                end: 4,
                key: { bakaze: 'E', kyoku: 1, honba: 0 },
            },
        ]);
        expect(journal.eventsForKyoku(0).events).toEqual([startKyoku(1), customSpacing, '{"type":"end_kyoku"}']);
    });

    it('never exposes an in-progress kyoku start or private hand', () => {
        const journal = EventJournal.fromEvents([startGame, startKyoku(1), { type: 'tsumo', actor: 0, pai: '9m' }]);

        const safe = journal.spectatorSafePrefix();
        expect(safe).toMatchObject({ schemaVersion: 1, complete: false });
        expect(safe.to).toBe(1);
        expect(safe.parsedEvents.map((event) => event.type)).toEqual(['start_game']);
        expect(safe.events.join('\n')).not.toContain('tehais');
        expect(safe.events.join('\n')).not.toContain('tsumo');
    });

    it('fails closed when a private event appears outside kyoku boundaries', () => {
        const journal = EventJournal.fromEvents([
            startGame,
            { type: 'tsumo', actor: 0, pai: '9m' },
            { type: 'end_game' },
        ]);

        const safe = journal.spectatorSafePrefix();
        expect(safe).toMatchObject({ schemaVersion: 1, complete: true, from: 0, to: 1 });
        expect(safe.events).toEqual([startGame]);
        expect(safe.parsedEvents.map((event) => event.type)).toEqual(['start_game']);
    });

    it('never trusts unknown or private start_game fields as spectator metadata', () => {
        for (const unsafe of [
            '{"type":"start_game","tehais":[["1m"]]}',
            '{"type":"start_game","future":{"opaque":true}}',
            '{"type":"start_game","names":[{"tehai":["1m"]}]}',
        ]) {
            const journal = EventJournal.fromEvents([unsafe, { type: 'end_game' }]);
            expect(journal.spectatorCursor).toBe(0);
            expect(journal.spectatorSafePrefix().events).toEqual([]);
            expect(journal.rawEvents[0]).toBe(unsafe);
        }
    });

    it('rejects duplicate JSON keys without mutating the journal', () => {
        for (const duplicate of [
            '{"type":"private_wall=1m2m3m","type":"start_game"}',
            '{"type":"start_game","names":["PRIVATE"],"names":["A","B","C","D"]}',
            '{"type":"start_game","future":{"secret":1,"secret":2}}',
        ]) {
            const journal = new EventJournal();
            expect(() => journal.append(duplicate)).toThrow('duplicate JSON object key');
            expect(journal.revision).toBe(0);
            expect(journal.rawEvents).toEqual([]);
        }

        const privateKey = new EventJournal();
        try {
            privateKey.append('{"type":"start_game","TOKEN_SECRET":"PRIVATE","TOKEN_SECRET":"SECRET"}');
            throw new Error('expected duplicate key rejection');
        } catch (error) {
            expect(String(error)).toContain('duplicate JSON object key');
            expect(String(error)).not.toContain('TOKEN_SECRET');
            expect(String(error)).not.toContain('PRIVATE');
            expect(String(error)).not.toContain('SECRET');
        }
        expect(privateKey.revision).toBe(0);
    });

    it('withholds a standard shuffle seed while live and releases it after end_game', () => {
        const seededStart = '{"type":"start_game","names":["A","B","C","D"],"seed":[1,2]}';
        const journal = EventJournal.fromEvents([seededStart]);
        expect(journal.spectatorCursor).toBe(0);

        journal.appendMany([startKyoku(1), { type: 'end_kyoku' }, { type: 'end_game' }]);

        expect(journal.complete).toBe(true);
        expect(journal.spectatorCursor).toBe(journal.revision);
        expect(journal.spectatorSafePrefix().events[0]).toBe(seededStart);
    });

    it('does not let an unframed end_game release live-only metadata', () => {
        const seededStart = '{"type":"start_game","names":["A","B","C","D"],"seed":[1,2]}';
        const journal = EventJournal.fromEvents([seededStart, { type: 'end_game' }]);

        expect(journal.complete).toBe(true);
        expect(journal.spectatorCursor).toBe(0);
        expect(journal.spectatorSafePrefix().events).toEqual([]);

        const publicStart = EventJournal.fromEvents([startGame, '{"type":"end_game","wall":["PRIVATE"]}']);
        expect(publicStart.spectatorCursor).toBe(1);
        expect(publicStart.spectatorSafePrefix().events).toEqual([startGame]);

        const framedPrivate = EventJournal.fromEvents([
            startGame,
            startKyoku(1),
            { type: 'end_kyoku' },
            '{"type":"end_game","wall":["PRIVATE"]}',
        ]);
        expect(framedPrivate.spectatorCursor).toBe(3);
        expect(framedPrivate.spectatorSafePrefix().events.join('\n')).not.toContain('PRIVATE');
    });

    it('never declassifies a stream with a missing or duplicate start_game', () => {
        const missing = EventJournal.fromEvents([startKyoku(1), { type: 'end_kyoku' }, { type: 'end_game' }]);
        expect(missing.complete).toBe(true);
        expect(missing.spectatorCursor).toBe(0);

        const duplicate = EventJournal.fromEvents([
            startGame,
            startGame,
            startKyoku(1),
            { type: 'end_kyoku' },
            { type: 'end_game' },
        ]);
        expect(duplicate.complete).toBe(true);
        expect(duplicate.spectatorCursor).toBe(1);
        expect(duplicate.spectatorSafePrefix().events).toEqual([startGame]);
    });

    it('uses JavaScript safe-integer semantics for metadata and kyoku keys', () => {
        const safeStart = '{"type":"start_game","names":["A","B","C","D"],"id":1.0,"seed":[2e0]}';
        const numericKyoku = '{"type":"start_kyoku","bakaze":"E","kyoku":1.0,"honba":2e0}';
        const safe = EventJournal.fromEvents([safeStart, numericKyoku, { type: 'end_kyoku' }, { type: 'end_game' }]);
        expect(safe.completedKyokuSpans[0]?.key).toEqual({ bakaze: 'E', kyoku: 1, honba: 2 });
        expect(safe.spectatorCursor).toBe(safe.revision);

        const unsafe = EventJournal.fromEvents([
            '{"type":"start_game","id":9007199254740992}',
            startKyoku(1),
            { type: 'end_kyoku' },
            { type: 'end_game' },
        ]);
        expect(unsafe.spectatorCursor).toBe(0);

        const missingKey = EventJournal.fromEvents([
            startGame,
            '{"type":"start_kyoku","kyoku":1.5,"honba":256}',
            { type: 'end_kyoku' },
        ]);
        expect(missingKey.completedKyokuSpans[0]?.key).toEqual({ bakaze: null, kyoku: null, honba: null });

        const negativeZero = EventJournal.fromEvents([
            startGame,
            '{"type":"start_kyoku","kyoku":-0,"honba":-0.0}',
            { type: 'end_kyoku' },
        ]);
        expect(negativeZero.completedKyokuSpans[0]?.key).toEqual({ bakaze: null, kyoku: 0, honba: 0 });
        expect(Object.is(negativeZero.completedKyokuSpans[0]?.key.kyoku, -0)).toBe(false);
    });

    it('keeps a completed kyoku hidden until the next kyoku starts', () => {
        const journal = EventJournal.fromEvents([startGame, startKyoku(1), '{"type":"end_kyoku"}']);
        expect(journal.spectatorCursor).toBe(1);

        journal.append(startKyoku(2));

        expect(journal.spectatorCursor).toBe(3);
        expect(journal.spectatorSafePrefix().parsedEvents.map((event) => event.type)).toEqual([
            'start_game',
            'start_kyoku',
            'end_kyoku',
        ]);
        expect(journal.spectatorSafePrefix().events).not.toContain(startKyoku(2));
    });

    it('provides cursor deltas that concatenate to the safe prefix', () => {
        const journal = EventJournal.fromEvents([startGame]);
        const first = journal.spectatorEventsSince(0);
        journal.appendMany([startKyoku(1), { type: 'dahai', actor: 0, pai: '1m' }, { type: 'end_kyoku' }]);
        expect(journal.spectatorEventsSince(first.to).events).toEqual([]);

        journal.append(startKyoku(2));
        const second = journal.spectatorEventsSince(first.to);

        expect([...first.events, ...second.events]).toEqual(journal.spectatorSafePrefix().events);
        expect(second).toMatchObject({ from: 1, to: 4 });
    });

    it('releases the complete log only after end_game', () => {
        const journal = EventJournal.fromEvents([
            startGame,
            startKyoku(1),
            { type: 'end_kyoku' },
            startKyoku(2),
            { type: 'hora', actor: 0, target: 1 },
            { type: 'end_kyoku' },
        ]);
        expect(journal.spectatorCursor).toBe(3);
        expect(journal.spectatorCursor).toBeLessThan(journal.revision);

        journal.append({ type: 'end_game', scores: [30000, 20000, 25000, 25000] });

        expect(journal.complete).toBe(true);
        expect(journal.spectatorCursor).toBe(journal.revision);
        const safeDelta = journal.spectatorSafePrefix();
        expect(safeDelta).toMatchObject({ schemaVersion: 1, complete: true });
        const safeEvents = safeDelta.parsedEvents;
        expect(safeEvents[safeEvents.length - 1]?.type).toBe('end_game');
    });

    it('syncs only newly safe events into a LiveViewer-compatible sink', () => {
        const journal = EventJournal.fromEvents([startGame]);
        const received: MjaiEvent[] = [];
        const sink = {
            pushEvents: vi.fn((events: readonly MjaiEvent[]) => received.push(...events)),
        };

        let cursor = journal.syncSpectator(sink);
        journal.appendMany([startKyoku(1), { type: 'end_kyoku' }, startKyoku(2)]);
        cursor = journal.syncSpectator(sink, cursor);
        cursor = journal.syncSpectator(sink, cursor);

        expect(cursor).toBe(3);
        expect(received.map((event) => event.type)).toEqual(['start_game', 'start_kyoku', 'end_kyoku']);
        expect(sink.pushEvents).toHaveBeenCalledTimes(2);
    });

    it('rejects invalid boundaries without mutating the journal', () => {
        const journal = EventJournal.fromEvents([startGame]);
        expect(() => journal.append({ type: 'end_kyoku' })).toThrow('without a matching start_kyoku');
        expect(journal.revision).toBe(1);

        journal.append(startKyoku(1));
        expect(() => journal.append(startKyoku(2))).toThrow('before the current kyoku ended');
        expect(journal.revision).toBe(2);
    });

    it('round-trips JSONL with an optional trailing newline', () => {
        const jsonl = `${startGame}\n${startKyoku(1)}\n{"type":"end_kyoku"}`;
        const journal = EventJournal.fromJsonl(jsonl);

        expect(journal.toJsonl()).toBe(`${jsonl}\n`);
    });
});
