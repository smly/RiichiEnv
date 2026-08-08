import type { MjaiEvent } from './types';

export type EventCursor = number;

export interface KyokuKey {
    readonly bakaze: string | null;
    readonly kyoku: number | null;
    readonly honba: number | null;
}

/** Half-open event range for one completed kyoku. */
export interface CompletedKyokuSpan {
    readonly start: EventCursor;
    readonly end: EventCursor;
    readonly key: KyokuKey;
}

/** Cross-language schema-v1 transport envelope. */
export interface JournalDeltaWire {
    readonly schemaVersion: 1;
    readonly complete: boolean;
    readonly from: EventCursor;
    readonly to: EventCursor;
    /** Canonical raw MJAI JSON strings, matching core and WASM schema v1. */
    readonly events: readonly string[];
}

/** Browser convenience view; `parsedEvents` is not part of the wire schema. */
export interface JournalDelta extends JournalDeltaWire {
    /** Parsed convenience view for browser reducers and viewers. */
    readonly parsedEvents: readonly MjaiEvent[];
}

/** Structurally implemented by LiveViewer and other append-only consumers. */
export interface AppendOnlyEventSink {
    pushEvents(events: readonly MjaiEvent[]): void;
}

interface PendingKyoku {
    readonly start: EventCursor;
    readonly key: KyokuKey;
}

/**
 * JSON.parse uses last-key-wins semantics. The journal returns accepted input
 * verbatim, so accepting duplicate keys would let validation inspect a
 * different logical event from a downstream spectator parser. The input has
 * already passed JSON.parse when this scanner runs; it only enforces unique
 * object keys recursively.
 */
function assertNoDuplicateObjectKeys(rawEvent: string): void {
    let cursor = 0;

    const skipWhitespace = () => {
        while (
            cursor < rawEvent.length &&
            (rawEvent[cursor] === ' ' ||
                rawEvent[cursor] === '\n' ||
                rawEvent[cursor] === '\r' ||
                rawEvent[cursor] === '\t')
        ) {
            cursor += 1;
        }
    };

    const scanString = (): string => {
        const start = cursor;
        cursor += 1;
        while (cursor < rawEvent.length) {
            if (rawEvent[cursor] === '\\') {
                cursor += 2;
            } else if (rawEvent[cursor] === '"') {
                cursor += 1;
                return JSON.parse(rawEvent.slice(start, cursor)) as string;
            } else {
                cursor += 1;
            }
        }
        throw new Error('unterminated JSON string');
    };

    const scanValue = (): void => {
        skipWhitespace();
        const first = rawEvent[cursor];
        if (first === '{') {
            cursor += 1;
            skipWhitespace();
            const keys = new Set<string>();
            if (rawEvent[cursor] === '}') {
                cursor += 1;
                return;
            }
            while (cursor < rawEvent.length) {
                const key = scanString();
                if (keys.has(key)) throw new Error('duplicate JSON object key');
                keys.add(key);
                skipWhitespace();
                cursor += 1; // ':'; syntax was already checked by JSON.parse.
                scanValue();
                skipWhitespace();
                if (rawEvent[cursor] === '}') {
                    cursor += 1;
                    return;
                }
                cursor += 1; // ','
                skipWhitespace();
            }
        } else if (first === '[') {
            cursor += 1;
            skipWhitespace();
            if (rawEvent[cursor] === ']') {
                cursor += 1;
                return;
            }
            while (cursor < rawEvent.length) {
                scanValue();
                skipWhitespace();
                if (rawEvent[cursor] === ']') {
                    cursor += 1;
                    return;
                }
                cursor += 1; // ','
            }
        } else if (first === '"') {
            scanString();
        } else {
            while (
                cursor < rawEvent.length &&
                rawEvent[cursor] !== ',' &&
                rawEvent[cursor] !== ']' &&
                rawEvent[cursor] !== '}' &&
                rawEvent[cursor] !== ' ' &&
                rawEvent[cursor] !== '\n' &&
                rawEvent[cursor] !== '\r' &&
                rawEvent[cursor] !== '\t'
            ) {
                cursor += 1;
            }
        }
    };

    scanValue();
}

export function parseJournalEvent(rawEvent: string): MjaiEvent {
    let value: unknown;
    try {
        value = JSON.parse(rawEvent);
        assertNoDuplicateObjectKeys(rawEvent);
    } catch (error) {
        throw new Error(`Invalid MJAI JSON event: ${error instanceof Error ? error.message : String(error)}`);
    }

    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('MJAI event must be a JSON object');
    }

    const type = (value as Record<string, unknown>).type;
    if (typeof type !== 'string') {
        throw new Error("MJAI event must contain a string 'type' field");
    }
    return value as MjaiEvent;
}

function optionalU8(value: unknown): number | null {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > 255) return null;
    return value === 0 ? 0 : value;
}

function kyokuKey(event: MjaiEvent): KyokuKey {
    return {
        bakaze: typeof event.bakaze === 'string' ? event.bakaze : null,
        kyoku: optionalU8(event.kyoku),
        honba: optionalU8(event.honba),
    };
}

type StartGameSafety = 'public' | 'withhold-until-complete' | 'unsafe';

function classifyStartGame(event: MjaiEvent): StartGameSafety {
    let hasSeed = false;
    for (const [key, value] of Object.entries(event)) {
        let valid = false;
        if (key === 'type') valid = value === 'start_game';
        else if (key === 'names') valid = Array.isArray(value) && value.every((name) => typeof name === 'string');
        else if (key === 'id') {
            valid = typeof value === 'string' || (typeof value === 'number' && Number.isSafeInteger(value));
        } else if (key === 'seed') {
            hasSeed = true;
            valid =
                Array.isArray(value) && value.every((part) => typeof part === 'number' && Number.isSafeInteger(part));
        }
        if (!valid) return 'unsafe';
    }
    return hasSeed ? 'withhold-until-complete' : 'public';
}

function classifyEndGame(event: MjaiEvent): boolean {
    return Object.entries(event).every(([key, value]) => {
        if (key === 'type') return value === 'end_game';
        if (key === 'scores' || key === 'ranks') {
            return (
                Array.isArray(value) &&
                (value.length === 3 || value.length === 4) &&
                value.every((part) => typeof part === 'number' && Number.isSafeInteger(part))
            );
        }
        return false;
    });
}

/**
 * Append-only MJAI journal and replay timeline.
 *
 * Raw JSON is retained so unknown fields survive round-trips. The spectator
 * view has a fixed one-kyoku delay matching `riichienv-core::EventJournal`:
 * while a kyoku is in progress its `start_kyoku` and every later event are
 * withheld, while a structurally valid, spectator-safe `end_game` releases
 * the full journal.
 */
export class EventJournal {
    private readonly _rawEvents: string[] = [];
    private readonly _completedKyokus: CompletedKyokuSpan[] = [];
    private _currentKyoku: PendingKyoku | undefined;
    private _complete = false;
    private _sawInitialStartGame = false;
    private _completionTrusted = false;
    private _spectatorCeiling: EventCursor | undefined;
    private _liveSpectatorCeiling: EventCursor | undefined;

    static fromEvents(events: readonly (MjaiEvent | string)[]): EventJournal {
        const journal = new EventJournal();
        journal.appendMany(events);
        return journal;
    }

    static fromJsonl(jsonl: string): EventJournal {
        const journal = new EventJournal();
        for (const line of jsonl.split(/\r?\n/)) {
            if (line.trim().length > 0) journal.append(line);
        }
        return journal;
    }

    get revision(): EventCursor {
        return this._rawEvents.length;
    }

    get complete(): boolean {
        return this._complete;
    }

    get hasInProgressKyoku(): boolean {
        return this._currentKyoku !== undefined;
    }

    get currentKyokuStart(): EventCursor | null {
        return this._currentKyoku?.start ?? null;
    }

    get rawEvents(): readonly string[] {
        return [...this._rawEvents];
    }

    get events(): readonly MjaiEvent[] {
        return this._rawEvents.map(parseJournalEvent);
    }

    get completedKyokuSpans(): readonly CompletedKyokuSpan[] {
        return this._completedKyokus.map((span) => ({
            start: span.start,
            end: span.end,
            key: { ...span.key },
        }));
    }

    /** Append a raw JSON string or a JSON-serializable MJAI event. */
    append(event: MjaiEvent | string): EventCursor {
        const rawEvent = typeof event === 'string' ? event : this.serializeEvent(event);
        const parsed = parseJournalEvent(rawEvent);

        if (this._complete) {
            throw new Error('Cannot append an event after end_game');
        }
        if (parsed.type === 'start_kyoku' && this._currentKyoku) {
            throw new Error('start_kyoku encountered before the current kyoku ended');
        }
        if (parsed.type === 'end_kyoku' && !this._currentKyoku) {
            throw new Error('end_kyoku encountered without a matching start_kyoku');
        }
        if (parsed.type === 'end_game' && this._currentKyoku) {
            throw new Error('end_game encountered before the current kyoku ended');
        }

        const eventCursor = this.revision;
        this._rawEvents.push(rawEvent);

        if (parsed.type === 'start_game') {
            if (eventCursor === 0 && !this._sawInitialStartGame) {
                this._sawInitialStartGame = true;
            } else if (this._spectatorCeiling === undefined) {
                // Preserve duplicate/mid-stream markers for replay debugging,
                // but do not let a later end_game declassify their suffix.
                this._spectatorCeiling = eventCursor;
            }

            const safety = classifyStartGame(parsed);
            if (safety === 'unsafe' && this._spectatorCeiling === undefined) {
                // Raw unknown fields are retained, but extensions could carry
                // concealed data and therefore create a permanent ceiling.
                this._spectatorCeiling = eventCursor;
            } else if (safety === 'withhold-until-complete' && this._liveSpectatorCeiling === undefined) {
                this._liveSpectatorCeiling = eventCursor;
            }
        } else if (parsed.type === 'start_kyoku') {
            if (!this._sawInitialStartGame && this._spectatorCeiling === undefined) {
                this._spectatorCeiling = eventCursor;
            }
            this._currentKyoku = { start: eventCursor, key: kyokuKey(parsed) };
        } else if (parsed.type === 'end_kyoku') {
            const pending = this._currentKyoku;
            if (!pending) throw new Error('Internal EventJournal boundary error');
            this._currentKyoku = undefined;
            this._completedKyokus.push({
                start: pending.start,
                end: this.revision,
                key: pending.key,
            });
        } else if (parsed.type === 'end_game') {
            if (!classifyEndGame(parsed) && this._spectatorCeiling === undefined) {
                // Preserve extensions in the full journal, but unknown fields
                // may contain concealed data and are not spectator-safe.
                this._spectatorCeiling = eventCursor;
            }
            this._complete = true;
            this._completionTrusted =
                this._sawInitialStartGame && this._completedKyokus.length > 0 && this._spectatorCeiling === undefined;
            if (!this._completionTrusted && this._spectatorCeiling === undefined) {
                // Do not expose this marker or unknown fields carried by it
                // merely because an unframed stream claims to be complete.
                this._spectatorCeiling = eventCursor;
            }
        } else if (!this._currentKyoku && this._spectatorCeiling === undefined) {
            // Fail closed for truncated or malformed feeds. An event outside
            // explicit kyoku boundaries may contain concealed information;
            // no spectator view may cross it, even after a later end_game.
            this._spectatorCeiling = eventCursor;
        }

        return this.revision;
    }

    appendMany(events: readonly (MjaiEvent | string)[]): EventCursor {
        for (const event of events) this.append(event);
        return this.revision;
    }

    eventsSince(cursor: EventCursor): JournalDelta {
        return this.delta(cursor, this.revision);
    }

    eventsForKyoku(index: number): JournalDelta {
        const span = this.spanAt(index);
        return this.delta(span.start, span.end);
    }

    prefixThroughCompletedKyoku(index: number): JournalDelta {
        return this.delta(0, this.spanAt(index).end);
    }

    /** Safe end cursor for the fixed one-kyoku spectator delay. */
    get spectatorCursor(): EventCursor {
        let unconstrained: EventCursor;
        if (this._complete) {
            unconstrained = this.revision;
        } else if (this._currentKyoku) {
            unconstrained = this._completedKyokus[this._completedKyokus.length - 1]?.end ?? this._currentKyoku.start;
        } else if (this._completedKyokus.length >= 2) {
            unconstrained = this._completedKyokus[this._completedKyokus.length - 2].end;
        } else if (this._completedKyokus.length === 1) {
            unconstrained = this._completedKyokus[0].start;
        } else {
            unconstrained = this.revision;
        }
        const permanent = this._spectatorCeiling ?? Number.POSITIVE_INFINITY;
        const whileLive = this._completionTrusted
            ? Number.POSITIVE_INFINITY
            : (this._liveSpectatorCeiling ?? Number.POSITIVE_INFINITY);
        return Math.min(unconstrained, permanent, whileLive);
    }

    spectatorSafePrefix(): JournalDelta {
        return this.delta(0, this.spectatorCursor);
    }

    spectatorEventsSince(cursor: EventCursor): JournalDelta {
        return this.delta(cursor, this.spectatorCursor);
    }

    /** Push newly safe events to a LiveViewer-compatible sink exactly once. */
    syncSpectator(sink: AppendOnlyEventSink, cursor: EventCursor = 0): EventCursor {
        const delta = this.spectatorEventsSince(cursor);
        if (delta.parsedEvents.length > 0) sink.pushEvents(delta.parsedEvents);
        return delta.to;
    }

    toJsonl(): string {
        return this._rawEvents.length === 0 ? '' : `${this._rawEvents.join('\n')}\n`;
    }

    private serializeEvent(event: MjaiEvent): string {
        let rawEvent: string | undefined;
        try {
            rawEvent = JSON.stringify(event);
        } catch (error) {
            throw new Error(
                `Failed to serialize MJAI event: ${error instanceof Error ? error.message : String(error)}`,
            );
        }
        if (rawEvent === undefined) throw new Error('Failed to serialize MJAI event');
        return rawEvent;
    }

    private spanAt(index: number): CompletedKyokuSpan {
        if (!Number.isInteger(index) || index < 0 || index >= this._completedKyokus.length) {
            throw new RangeError(
                `Completed kyoku index ${index} is out of range (count=${this._completedKyokus.length})`,
            );
        }
        return this._completedKyokus[index];
    }

    private delta(from: EventCursor, to: EventCursor): JournalDelta {
        if (!Number.isInteger(from) || from < 0 || from > to || to > this.revision) {
            throw new RangeError(`Invalid event cursor range ${from}..${to} for revision ${this.revision}`);
        }
        const events = this._rawEvents.slice(from, to);
        return {
            schemaVersion: 1,
            complete: this.complete,
            from,
            to,
            events,
            parsedEvents: events.map(parseJournalEvent),
        };
    }
}
