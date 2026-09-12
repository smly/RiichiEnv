import { YAKU_MAP } from './constants';
import type { GameState } from './game_state';
import type { KyokuKeyEvent, KyokuPlayerAction, KyokuResult, KyokuSummary, KyokuWinner, MjaiEvent } from './types';

function meldLabel(type: string): string {
    switch (type) {
        case 'chi':
            return 'Chi';
        case 'pon':
            return 'Pon';
        case 'daiminkan':
        case 'ankan':
        case 'kakan':
            return 'Kan';
        default:
            return type;
    }
}

/** Classify from scoring metadata or the base payment (excluding honba/deposits). */
export function classifyHandLimit(
    han: number,
    fu: number,
    base: number = 0,
    yakumanCount: number = 0,
): string | undefined {
    const multiple = Math.max(yakumanCount, base >= 8000 ? Math.floor(base / 8000) : 0);
    if (multiple >= 2) return multiple === 2 ? 'double-yakuman' : `${multiple}x-yakuman`;
    if (multiple === 1 || han >= 13) return 'yakuman';
    if (han >= 11 || base >= 6000) return 'sanbaiman';
    if (han >= 8 || base >= 4000) return 'baiman';
    if (han >= 6 || base >= 3000) return 'haneman';
    if (han >= 5 || (han === 4 && fu >= 40) || (han === 3 && fu >= 70) || base >= 2000) return 'mangan';
    return undefined;
}

function summarizeWinner(event: MjaiEvent, dealer: number, honba: number, pc: number): KyokuWinner {
    const actor = event.actor!;
    const target = event.target ?? actor;
    const isTsumo = actor === target;
    const score = event.meta?.score ?? event.score ?? {};
    const han = score.han ?? event.han ?? 0;
    const fu = score.fu ?? event.fu ?? 0;
    const yakuIds: number[] = score.yaku ?? [];
    let points = score.points ?? 0;
    let base = 0;
    if (Array.isArray(event.deltas)) {
        if (isTsumo) {
            const losses = event.deltas.filter((d: number) => d < 0).map((d: number) => -d - honba * 100);
            // Dealer payment distinguishes 3900-all (4 han 30 fu) from mangan.
            const payment =
                actor === dealer ? Math.max(0, ...losses) : Math.max(0, -event.deltas[dealer] - honba * 100);
            // Liability payments can have only one payer; use their total below.
            if (losses.length === pc - 1) base = payment / 2;
            if (!points) points = losses.reduce((sum: number, value: number) => sum + value, 0);
        } else {
            const payment = Math.max(0, -event.deltas[target] - honba * 300);
            base = payment / (actor === dealer ? 6 : 4);
            if (!points) points = payment;
        }
    }
    if (!base && points) {
        const divisor = isTsumo ? (actor === dealer ? 2 * (pc - 1) : pc) : actor === dealer ? 6 : 4;
        base = points / divisor;
    }
    const yakumanCount = Math.max(
        score.yakuman_count ?? (score.yakuman === true ? 1 : 0),
        yakuIds
            .filter((id) => id >= 35 && id <= 50)
            .reduce((sum, id) => sum + ([47, 48, 49, 50].includes(id) ? 2 : 1), 0),
    );
    return {
        actor,
        target,
        isTsumo,
        points,
        han,
        fu,
        yaku: yakuIds.map((id) => YAKU_MAP[id] || `Yaku#${id}`),
        // Paid value respects rules that count a double-yakuman variant as single.
        limit: classifyHandLimit(han, fu, base, base > 0 ? 0 : yakumanCount),
    };
}

/** Scan raw MJAI without replaying or changing the viewer cursor. */
export function computeKyokuSummaries(gameState: GameState): KyokuSummary[] {
    const { events, kyokus } = gameState;
    const pc = gameState.config.playerCount;
    const validScores = (scores: unknown): scores is number[] =>
        Array.isArray(scores) && scores.length === pc && scores.every(Number.isFinite);
    return kyokus.map((k, idx) => {
        const start = events[k.index];
        const dealer = start.oya ?? k.round % pc;
        const endIdx = kyokus[idx + 1]?.index ?? events.length;
        const playerActions: KyokuPlayerAction[] = Array.from({ length: pc }, () => ({
            riichi: false,
            tenpai: false,
            houjuu: false,
            hora: false,
            tsumo: false,
            meldTypes: [],
        }));
        const accepted = new Set<number>();
        const wins = new Map<number, MjaiEvent>();
        let result: KyokuResult | null = null;
        let endScores = [...k.scores];
        let settled = false;
        let completed = false;
        for (let i = k.index + 1; i < endIdx; i++) {
            const evt = events[i];
            const action = evt.actor === undefined ? undefined : playerActions[evt.actor];
            if (evt.type === 'reach' || evt.type === 'reach_accepted') {
                if (action) action.riichi = true;
                if (
                    action &&
                    (evt.type === 'reach_accepted' || String(evt.step) === '2') &&
                    !accepted.has(evt.actor!)
                ) {
                    accepted.add(evt.actor!);
                    endScores[evt.actor!] -= 1000;
                }
                if (validScores(evt.scores)) endScores = [...evt.scores];
            }
            if (['chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(evt.type) && action)
                action.meldTypes.push(evt.type);
            if (evt.type === 'hora') wins.set(evt.actor!, evt);
            if (evt.type === 'hora' || evt.type === 'ryukyoku') {
                completed = true;
                if (validScores(evt.scores)) {
                    endScores = [...evt.scores];
                    settled = true;
                } else if (validScores(evt.deltas)) {
                    endScores = endScores.map((value, p) => value + evt.deltas[p]);
                    settled = true;
                }
                if (evt.type === 'ryukyoku') {
                    result = { type: 'ryukyoku', reason: evt.reason };
                    playerActions.forEach((p, actor) => {
                        p.tenpai = evt.tenpais?.[actor] === true || (evt.deltas?.[actor] ?? 0) > 0;
                    });
                }
            }
            if (evt.type === 'end_kyoku') {
                completed = true;
                for (const winner of evt.meta?.results ?? []) {
                    const raw = wins.get(winner.actor);
                    wins.set(winner.actor, {
                        ...raw,
                        ...winner,
                        type: 'hora',
                        meta: { ...raw?.meta, score: winner.score ?? raw?.meta?.score },
                    });
                }
                const draw = evt.meta?.ryukyoku;
                if (draw) {
                    result = { type: 'ryukyoku', reason: draw.reason };
                    playerActions.forEach((p, actor) => {
                        p.tenpai ||= draw.tenpais?.[actor] === true || (draw.deltas?.[actor] ?? 0) > 0;
                    });
                }
                const final = draw ?? evt;
                if (validScores(final.scores)) {
                    endScores = [...final.scores];
                    settled = true;
                } else if (!settled && validScores(final.deltas)) {
                    endScores = endScores.map((value, p) => value + final.deltas[p]);
                    settled = true;
                }
            }
        }
        if (wins.size) {
            const winners = [...wins.values()].map((e) => summarizeWinner(e, dealer, k.honba, pc));
            result = { type: 'hora', winners };
            for (const w of winners) {
                if (playerActions[w.actor])
                    Object.assign(playerActions[w.actor], { hora: true, tenpai: true, tsumo: w.isTsumo });
                if (!w.isTsumo && playerActions[w.target]) playerActions[w.target].houjuu = true;
            }
        }
        // The next deal is authoritative, including deposits and multi-ron settlements.
        const nextScores = kyokus[idx + 1]?.scores;
        if (validScores(nextScores)) {
            endScores = [...nextScores];
            settled = true;
        }
        return {
            index: idx,
            round: k.round,
            honba: k.honba,
            dealer,
            completed,
            startScores: [...k.scores],
            endScores,
            deltas: endScores.map((value, p) => value - k.scores[p]),
            deltasKnown: settled || !completed,
            result,
            playerActions,
        };
    });
}

/**
 * Compute key events within a kyoku by scanning events and walking game state.
 * This is heavier as it replays game state for tenpai tracking.
 */
export function computeKyokuKeyEvents(gameState: GameState, kyokuIndex: number): KyokuKeyEvent[] {
    const events = gameState.events;
    const kyokus = gameState.kyokus;
    const pc = gameState.config.playerCount;

    if (kyokuIndex < 0 || kyokuIndex >= kyokus.length) return [];

    const startIdx = kyokus[kyokuIndex].index;
    const endIdx = kyokuIndex + 1 < kyokus.length ? kyokus[kyokuIndex + 1].index : events.length;

    const keyEvents: KyokuKeyEvent[] = [];

    // Phase 1: Scan events for simple key events
    for (let i = startIdx; i < endIdx; i++) {
        const evt = events[i];
        switch (evt.type) {
            case 'reach':
                if (evt.actor !== undefined && (!evt.step || evt.step === '1' || evt.step === 1)) {
                    keyEvents.push({ step: i, type: 'reach', actor: evt.actor, label: 'Riichi' });
                }
                break;
            case 'chi':
            case 'pon':
            case 'daiminkan':
            case 'ankan':
            case 'kakan':
                if (evt.actor !== undefined) {
                    keyEvents.push({
                        step: i,
                        type: evt.type,
                        actor: evt.actor,
                        label: meldLabel(evt.type),
                        detail: tileSummary(evt),
                    });
                }
                break;
            case 'hora':
                if (evt.actor !== undefined) {
                    const isTsumo = evt.actor === evt.target;
                    keyEvents.push({
                        step: i,
                        type: 'hora',
                        actor: evt.actor,
                        label: isTsumo ? 'Tsumo' : 'Ron',
                    });
                }
                break;
            case 'ryukyoku':
                keyEvents.push({
                    step: i,
                    type: 'ryukyoku',
                    actor: -1,
                    label: 'Draw',
                    detail: evt.reason,
                });
                break;
        }
    }

    // Phase 2: Walk game state incrementally for tenpai tracking
    const savedCursor = gameState.cursor;

    gameState.jumpTo(startIdx + 1); // After start_kyoku
    const prevWaits: (string | undefined)[] = Array(pc).fill(undefined);

    // Walk forward one step at a time to avoid repeated jumpTo() + recomputeWaits() overhead
    while (gameState.cursor < endIdx) {
        const step = gameState.cursor;
        const evt = events[step];
        gameState.stepForward();

        // Only check waits after dahai/reach_accepted - that's when waits are recomputed
        if (evt.type === 'dahai' || evt.type === 'reach_accepted') {
            const state = gameState.getState();

            for (let p = 0; p < pc; p++) {
                const waits = state.players[p].waits;
                const waitsKey = waits ? [...waits].sort().join(',') : '';
                const prevKey = prevWaits[p] ?? '';

                if (prevKey === '' && waitsKey !== '') {
                    keyEvents.push({ step, type: 'tenpai', actor: p, label: 'Tenpai', detail: waits!.join(' ') });
                } else if (prevKey !== '' && waitsKey === '') {
                    keyEvents.push({ step, type: 'tenpai_lost', actor: p, label: 'Lost Tenpai' });
                } else if (prevKey !== '' && waitsKey !== '' && prevKey !== waitsKey) {
                    keyEvents.push({
                        step,
                        type: 'wait_change',
                        actor: p,
                        label: 'Wait Change',
                        detail: waits!.join(' '),
                    });
                }

                prevWaits[p] = waitsKey;
            }
        }
    }

    // Restore cursor
    gameState.jumpTo(savedCursor);

    // Sort by step
    keyEvents.sort((a, b) => a.step - b.step);

    return keyEvents;
}

function tileSummary(evt: MjaiEvent): string {
    const tiles = evt.consumed ? [...evt.consumed] : [];
    if (evt.pai) tiles.push(evt.pai);
    return tiles.join(' ');
}
