import { createGameConfig4P, type GameConfig } from './config';
import type { BoardState, ConditionTracker, MjaiEvent, PlayerConfig } from './types';
import {
    type ConditionsInput,
    calculateScore,
    calculateWaits,
    type MeldInput,
    mjaiToTileId,
    tileIdToMjai,
} from './wasm/bridge';
import { isWasmReady } from './wasm/loader';

// Helper to sort hand (simple alphanumeric sort for now, ideally strictly by tile order)
const sortHand = (hand: string[]) => {
    const order = (t: string) => {
        if (t === 'back') return 9999;

        let suit = '';
        let num = 0;
        let isRed = false;

        // Handle Honors (z)
        const honorMap: { [key: string]: number } = {
            E: 1,
            S: 2,
            W: 3,
            N: 4, // Winds
            P: 5,
            F: 6,
            C: 7, // Dragons (Haku/White, Hatsu/Green, Chun/Red)
        };

        if (honorMap[t]) {
            suit = 'z';
            num = honorMap[t];
        } else {
            // Handle Suited Tiles
            // Formats: "1m", "5mr", "0m" (if used)
            if (t.endsWith('r')) {
                isRed = true;
                suit = t.charAt(t.length - 2); // 5mr -> m
                num = parseInt(t.charAt(0), 10);
            } else {
                suit = t.charAt(t.length - 1); // 1m -> m
                num = parseInt(t.charAt(0), 10);
            }

            // Handle "0m" case if present in data (treat as Red 5)
            if (num === 0) {
                num = 5;
                isRed = true;
            }
        }

        const suitOrder: Record<string, number> = { m: 0, p: 100, s: 200, z: 300 };
        const redOffset = isRed ? 0.1 : 0;

        return (suitOrder[suit] ?? 900) + num + redOffset;
    };
    return [...hand].sort((a, b) => order(a) - order(b));
};

/** Convert PlayerState melds to WASM MeldInput format (136-encoding). */
function meldsToWasmInput(melds: { type: string; tiles: string[]; from: number }[]): MeldInput[] {
    return melds.map((m) => ({
        meld_type: m.type,
        tiles: m.tiles.map((t) => mjaiToTileId(t)).filter((id): id is number => id !== null),
    }));
}

/** Create a fresh ConditionTracker. */
function initialConditions(playerCount: number = 4): ConditionTracker {
    return {
        ippatsu: Array(playerCount).fill(false),
        afterKan: false,
        pendingChankan: false,
        callsMade: false,
        firstTurnCompleted: Array(playerCount).fill(false),
        turnCount: 0,
        doubleRiichi: Array(playerCount).fill(false),
    };
}

export class GameState {
    events: MjaiEvent[];
    cursor: number;
    kyokus: { index: number; round: number; honba: number; scores: number[] }[];
    readonly config: GameConfig;

    // Cache state at each step to allow fast jumping
    current: BoardState;

    // Player customization
    private _playerNames: string[];
    private _playerAvatars: (string | null)[];

    // Checkpoint cache: cursor position -> deep-cloned BoardState
    private stateCache: Map<number, BoardState> = new Map();
    private static readonly CHECKPOINT_INTERVAL = 20;

    // Flag to skip expensive WASM calculations during bulk replay
    private _isReplaying: boolean = false;

    constructor(events: MjaiEvent[], config?: GameConfig, playerConfigs?: PlayerConfig[]) {
        this.config = config ?? createGameConfig4P();
        const pc = this.config.playerCount;

        // Extract player names from start_game event in log
        const logNames = GameState.extractPlayerNames(events);

        // Build player names: option override > log names > default
        this._playerNames = Array(pc)
            .fill(0)
            .map((_, i) => playerConfigs?.[i]?.name || logNames[i] || `player${i}`);
        this._playerAvatars = Array(pc)
            .fill(0)
            .map((_, i) => playerConfigs?.[i]?.avatarUrl || null);

        // Filter out null events and start/end game events
        this.events = events.filter((e) => e && e.type !== 'start_game' && e.type !== 'end_game');
        this.cursor = 0;
        this.current = this.initialState();

        this.kyokus = this.getKyokuCheckpoints();

        // Jump to first meaningful state (start_kyoku + 1)
        const firstKyoku = this.events.findIndex((e) => e.type === 'start_kyoku');
        if (firstKyoku !== -1) {
            this.jumpTo(firstKyoku + 1);
        }
    }

    private static extractPlayerNames(events: MjaiEvent[]): string[] {
        for (const e of events) {
            if (e && e.type === 'start_game' && Array.isArray(e.names)) {
                return e.names;
            }
        }
        return [];
    }

    getState(): BoardState {
        return this.current;
    }

    jumpToKyoku(kyokuIndex: number) {
        if (kyokuIndex >= 0 && kyokuIndex < this.kyokus.length) {
            // Jump to the event index of the start_kyoku + 1
            this.jumpTo(this.kyokus[kyokuIndex].index + 1);
        }
    }

    // Returns list of indices where new rounds start
    getKyokuCheckpoints(): { index: number; round: number; honba: number; scores: number[] }[] {
        const checkpoints: { index: number; round: number; honba: number; scores: number[] }[] = [];
        this.events.forEach((e, i) => {
            if (e.type === 'start_kyoku') {
                checkpoints.push({
                    index: i,
                    round: this.getRoundIndex(e),
                    honba: e.honba || 0,
                    scores: e.scores || this.config.defaultScores,
                });
            }
        });
        return checkpoints;
    }

    initialState(): BoardState {
        const pc = this.config.playerCount;
        return {
            playerCount: pc,
            players: Array(pc)
                .fill(0)
                .map((_, i) => ({
                    hand: [],
                    discards: [],
                    melds: [],
                    score: this.config.defaultScores[i],
                    riichi: false,
                    pendingRiichi: false,
                    wind: 0,
                    kitaCount: 0,
                })),
            playerNames: [...this._playerNames],
            playerAvatars: [...this._playerAvatars],
            doraMarkers: [],
            round: 0,
            honba: 0,
            kyotaku: 0,
            wallRemaining: this.config.initialWallRemaining,
            currentActor: 0,
            eventIndex: 0,
            totalEvents: this.events.length,
            conditions: initialConditions(pc),
        };
    }

    // Returns true if state changed
    stepForward(): boolean {
        if (this.cursor >= this.events.length) return false;
        const event = this.events[this.cursor];
        this.processEvent(event);
        this.cursor++;
        this.current.eventIndex = this.cursor; // Sync

        // Cache state at kyoku boundaries and every N steps for fast backward jumps
        if (!this.stateCache.has(this.cursor)) {
            if (event.type === 'start_kyoku' || this.cursor % GameState.CHECKPOINT_INTERVAL === 0) {
                this.stateCache.set(this.cursor, this.cloneState(this.current));
            }
        }

        return true;
    }

    stepBackward(): boolean {
        // Prevent going back to 0 (before first start_kyoku)
        if (this.cursor <= 1) return false;
        this.jumpTo(this.cursor - 1);
        return true;
    }

    jumpTo(index: number) {
        if (index < 1) index = 1; // Enforce minimum 1
        if (index > this.events.length) index = this.events.length;

        // Skip expensive WASM calls during bulk replay
        const needsReplay = index !== this.cursor;
        if (needsReplay) this._isReplaying = true;

        if (index < this.cursor) {
            // Find nearest cached checkpoint at or before target index
            let bestCursor = 0;
            for (const cpCursor of this.stateCache.keys()) {
                if (cpCursor <= index && cpCursor > bestCursor) {
                    bestCursor = cpCursor;
                }
            }

            if (bestCursor > 0) {
                this.current = this.cloneState(this.stateCache.get(bestCursor)!);
                this.cursor = bestCursor;
            } else {
                this.reset();
            }
        }
        while (this.cursor < index) {
            this.stepForward();
        }

        if (needsReplay) {
            this._isReplaying = false;
            // Recompute waits for all players at the final position
            this.recomputeWaits();
        }
    }

    /** Recompute waits for all players at the current position (after replay). */
    private recomputeWaits(): void {
        if (!isWasmReady()) return;
        const pc = this.config.playerCount;
        for (let i = 0; i < pc; i++) {
            const p = this.current.players[i];
            p.waits = undefined;
            const tileIds = p.hand.map((t) => mjaiToTileId(t)).filter((id): id is number => id !== null);
            const meldInputs = meldsToWasmInput(p.melds);
            const expectedLen = 13 - meldInputs.length * 3;
            if (tileIds.length === expectedLen) {
                const waits34 = calculateWaits(tileIds, meldInputs);
                if (Array.isArray(waits34) && waits34.length > 0) {
                    p.waits = waits34.map((t34) => tileIdToMjai(t34 * 4)).filter((s): s is string => s !== null);
                }
            }
        }
    }

    // Jump to next turn for specific actor (tsumo event)
    jumpToNextTurn(actor: number): boolean {
        let target = -1;
        for (let i = this.cursor; i < this.events.length; i++) {
            const e = this.events[i];
            if ((e.type === 'tsumo' && e.actor === actor) || e.type === 'end_kyoku') {
                target = i;
                break;
            }
        }

        if (target !== -1) {
            // Jump to target + 1 (State AFTER tsumo)
            this.jumpTo(target + 1);
            return true;
        }
        return false;
    }

    // Jump to prev turn for specific actor
    jumpToPrevTurn(actor: number): boolean {
        let target = -1;
        // Search backwards from cursor - 2 (current event is cursor-1)
        for (let i = this.cursor - 2; i >= 0; i--) {
            const e = this.events[i];
            if ((e.type === 'tsumo' && e.actor === actor) || e.type === 'end_kyoku') {
                target = i;
                break;
            }
        }

        if (target !== -1) {
            this.jumpTo(target + 1);
            return true;
        }
        return false;
    }

    jumpToNextKyoku(): boolean {
        let target = -1;
        for (let i = this.cursor; i < this.events.length; i++) {
            if (this.events[i].type === 'start_kyoku') {
                target = i;
                break;
            }
        }
        if (target !== -1) {
            this.jumpTo(target + 1);
            return true;
        }
        return false;
    }

    jumpToPrevKyoku(): boolean {
        let target = -1;
        // Logic to jump to previous kyoku:
        // 1. Find start_kyoku of current round
        // 2. If we are at the start, find the previous one
        // Find the index of the start_kyoku for the CURRENT cursor position.
        let currentKyokuStart = -1;
        for (let i = this.cursor - 1; i >= 0; i--) {
            if (this.events[i].type === 'start_kyoku') {
                currentKyokuStart = i;
                break;
            }
        }

        // Now search backwards from there
        if (currentKyokuStart !== -1) {
            for (let i = currentKyokuStart - 1; i >= 0; i--) {
                if (this.events[i].type === 'start_kyoku') {
                    target = i;
                    break;
                }
            }
        }

        if (target !== -1) {
            this.jumpTo(target + 1);
            return true;
        }
        return false;
    }

    /**
     * Append a single event incrementally (for live mode).
     * Automatically advances the cursor to the latest event.
     */
    appendEvent(event: MjaiEvent): void {
        if (!event || event.type === 'start_game' || event.type === 'end_game') return;

        this.events.push(event);
        this.current.totalEvents = this.events.length;

        if (event.type === 'start_kyoku') {
            this.kyokus.push({
                index: this.events.length - 1,
                round: this.getRoundIndex(event),
                honba: event.honba || 0,
                scores: event.scores || this.config.defaultScores,
            });
        }

        this.processEvent(event);
        this.cursor++;
        this.current.eventIndex = this.cursor;
    }

    reset() {
        this.cursor = 0;
        this.current = this.initialState();
        this.stateCache.clear();
    }

    /** Deep clone a BoardState for checkpoint caching. */
    private cloneState(state: BoardState): BoardState {
        return {
            playerCount: state.playerCount,
            players: state.players.map((p) => ({
                hand: [...p.hand],
                discards: p.discards.map((d) => ({ ...d })),
                melds: p.melds.map((m) => ({ type: m.type, tiles: [...m.tiles], from: m.from })),
                score: p.score,
                riichi: p.riichi,
                pendingRiichi: p.pendingRiichi,
                wind: p.wind,
                waits: p.waits ? [...p.waits] : undefined,
                lastDrawnTile: p.lastDrawnTile,
                kitaCount: p.kitaCount,
            })),
            playerNames: [...state.playerNames],
            playerAvatars: [...state.playerAvatars],
            doraMarkers: [...state.doraMarkers],
            round: state.round,
            honba: state.honba,
            kyotaku: state.kyotaku,
            wallRemaining: state.wallRemaining,
            currentActor: state.currentActor,
            lastEvent: state.lastEvent,
            eventIndex: state.eventIndex,
            totalEvents: state.totalEvents,
            dahaiAnim: state.dahaiAnim,
            conditions: {
                ippatsu: [...state.conditions.ippatsu],
                afterKan: state.conditions.afterKan,
                pendingChankan: state.conditions.pendingChankan,
                chankanTarget: state.conditions.chankanTarget,
                callsMade: state.conditions.callsMade,
                firstTurnCompleted: [...state.conditions.firstTurnCompleted],
                turnCount: state.conditions.turnCount,
                doubleRiichi: [...state.conditions.doubleRiichi],
            },
        };
    }

    private getRoundIndex(e: MjaiEvent): number {
        const kyoku = (e.kyoku || 1) - 1;
        const bakaze = e.bakaze || 'E';
        const pc = this.config.playerCount;
        let offset = 0;
        if (bakaze === 'S') offset = pc;
        else if (bakaze === 'W') offset = pc * 2;
        else if (bakaze === 'N') offset = pc * 3;
        return offset + kyoku;
    }

    processEvent(e: MjaiEvent) {
        // Clear animation state
        this.current.dahaiAnim = undefined;

        switch (e.type) {
            case 'start_game':
                break;
            case 'start_kyoku':
                this.current.round = this.getRoundIndex(e);
                this.current.honba = e.honba || 0;
                this.current.kyotaku = e.kyotaku || 0;
                this.current.doraMarkers = [e.dora_marker];
                this.current.currentActor = e.oya;
                this.current.wallRemaining = this.config.initialWallRemaining;
                this.current.conditions = initialConditions(this.config.playerCount);
                this.current.players.forEach((p, i) => {
                    const tehai = e.tehais?.[i];
                    p.hand = tehai ? sortHand(tehai.map((t: string) => t)) : []; // Clone and sort
                    p.discards = [];
                    p.melds = [];
                    p.riichi = false;
                    p.pendingRiichi = false;
                    p.score = e.scores[i];
                    p.waits = undefined;
                    p.kitaCount = 0;
                    // Assign wind based on oya
                    p.wind = (i - e.oya + this.config.playerCount) % this.config.playerCount;
                    p.lastDrawnTile = undefined;
                });
                break;

            case 'tsumo':
                if (e.actor !== undefined && e.pai) {
                    this.current.players[e.actor].hand.push(e.pai);
                    this.current.players[e.actor].lastDrawnTile = e.pai;
                    // Do NOT sort hand here in renderer
                    this.current.currentActor = e.actor;
                    this.current.wallRemaining--;
                    // Clear pendingChankan on tsumo (rinshan draw after kan)
                    this.current.conditions.pendingChankan = false;
                    this.current.conditions.chankanTarget = undefined;
                }
                break;

            case 'dahai':
                if (e.actor !== undefined && e.pai) {
                    const p = this.current.players[e.actor];
                    const cond = this.current.conditions;
                    const discardIdx = p.hand.indexOf(e.pai);
                    // Note: discardIdx might be -1 if not found (shouldn't happen in valid)

                    if (discardIdx >= 0) {
                        p.hand.splice(discardIdx, 1);
                    }
                    p.hand = sortHand(p.hand);

                    // Find insert index of last drawn tile if te-dashi
                    let insertIdx = -1;
                    if (!e.tsumogiri && p.lastDrawnTile) {
                        // Find where the last drawn tile ended up after sort
                        // Note: If multiple same tiles, just pick first or last?
                        // Visually picking one is fine.
                        insertIdx = p.hand.indexOf(p.lastDrawnTile);
                    }

                    this.current.dahaiAnim = {
                        discardIdx: discardIdx,
                        insertIdx: insertIdx,
                        tsumogiri: !!e.tsumogiri,
                        drawnTile: p.lastDrawnTile,
                    };

                    // Riichi Logic
                    let isRiichi = false;

                    // Robust check: If pendingRiichi is true OR current event has reach flag OR immediately following a Reach event
                    const lastEv = this.current.lastEvent;
                    const prevWasReach =
                        lastEv &&
                        lastEv.type === 'reach' &&
                        lastEv.actor === e.actor &&
                        (!lastEv.step || lastEv.step === '1' || lastEv.step === 1);

                    // Lookahead: If next event is Reach (Step 1), this is the declaration tile
                    // (Handle cases where Dahai comes BEFORE Reach)
                    const nextEv = this.events[this.cursor + 1];
                    const nextIsReach =
                        nextEv &&
                        nextEv.type === 'reach' &&
                        nextEv.actor === e.actor &&
                        (!nextEv.step || nextEv.step === '1' || nextEv.step === 1);

                    if (p.pendingRiichi || e.reach || prevWasReach || nextIsReach) {
                        isRiichi = true;
                        p.pendingRiichi = false;
                    }

                    p.discards.push({ tile: e.pai, isRiichi, isTsumogiri: !!e.tsumogiri });

                    // Condition tracking
                    if (!cond.firstTurnCompleted[e.actor]) {
                        cond.firstTurnCompleted[e.actor] = true;
                    }
                    cond.turnCount++;
                    cond.ippatsu[e.actor] = false; // Turn passed without winning
                    cond.afterKan = false;

                    // WASM-first waits computation with melds
                    // Skip expensive WASM calls during bulk replay (jumpTo)
                    p.waits = undefined;
                    if (!this._isReplaying) {
                        if (isWasmReady()) {
                            const tileIds = p.hand
                                .map((t) => mjaiToTileId(t))
                                .filter((id): id is number => id !== null);
                            const meldInputs = meldsToWasmInput(p.melds);
                            const expectedLen = 13 - meldInputs.length * 3;
                            if (tileIds.length === expectedLen) {
                                const waits34 = calculateWaits(tileIds, meldInputs);
                                if (Array.isArray(waits34) && waits34.length > 0) {
                                    p.waits = waits34
                                        .map((t34) => tileIdToMjai(t34 * 4))
                                        .filter((s): s is string => s !== null);
                                }
                            }
                        }
                        // Fallback to meta only if WASM did not produce waits
                        if (!p.waits) {
                            p.waits = e.meta?.waits || undefined;
                        }
                    }

                    this.current.currentActor = e.actor;
                }
                break;

            case 'pon':
            case 'chi':
            case 'daiminkan':
                if (e.actor !== undefined && e.target !== undefined && e.pai && e.consumed) {
                    const p = this.current.players[e.actor];
                    e.consumed.forEach((t) => {
                        const idx = p.hand.indexOf(t);
                        if (idx >= 0) p.hand.splice(idx, 1);
                    });

                    // Add meld
                    p.melds.push({
                        type: e.type,
                        tiles: [...e.consumed, e.pai],
                        from: e.target,
                    });

                    p.waits = undefined;

                    this.current.currentActor = e.actor;

                    // Remove from target discard
                    const targetP = this.current.players[e.target];
                    if (targetP.discards.length > 0) {
                        const stolen = targetP.discards.pop();
                        // If stolen tile was Riichi declared, player must re-declare on next discard
                        if (stolen?.isRiichi) {
                            targetP.pendingRiichi = true;
                        }
                    }

                    // Condition tracking for calls
                    const cond = this.current.conditions;
                    cond.callsMade = true;
                    cond.ippatsu = Array(this.config.playerCount).fill(false);
                    if (e.type === 'daiminkan') {
                        cond.afterKan = true;
                    }
                }
                break;

            case 'ankan': // Closed Kan
                if (e.actor !== undefined && e.consumed) {
                    const p = this.current.players[e.actor];
                    e.consumed.forEach((t) => {
                        const idx = p.hand.indexOf(t);
                        if (idx >= 0) p.hand.splice(idx, 1);
                    });
                    p.melds.push({
                        type: e.type,
                        tiles: e.consumed, // all 4 tiles
                        from: e.actor,
                    });
                    p.waits = undefined;

                    // Condition tracking
                    const cond = this.current.conditions;
                    cond.callsMade = true;
                    cond.ippatsu = Array(this.config.playerCount).fill(false);
                    cond.afterKan = true;
                }
                break;

            case 'kakan': // Added Kan
                if (e.actor !== undefined && e.pai && e.consumed) {
                    const p = this.current.players[e.actor];
                    // MJAI spec: kakan event has pai (added tile) and consumed (array with just that tile)
                    const addedTile = e.pai;

                    // Remove from hand
                    const idx = p.hand.indexOf(addedTile);
                    if (idx >= 0) p.hand.splice(idx, 1);

                    // Find generic version for matching (ignore red/0)
                    const normalize = (t: string) => t.replace('0', '5').replace('r', '');
                    const targetNorm = normalize(addedTile);

                    // Find existing Pon
                    const pon = p.melds.find((m) => m.type === 'pon' && normalize(m.tiles[0]) === targetNorm);

                    if (pon) {
                        pon.type = 'kakan';
                        pon.tiles.push(addedTile);
                    } else {
                        console.warn('[GameState] Kakan: Could not find original Pon for', addedTile);
                        p.melds.push({
                            type: 'kakan',
                            tiles: [addedTile, addedTile, addedTile, addedTile], // Placeholder
                            from: e.actor,
                        });
                    }

                    // Condition tracking
                    const cond = this.current.conditions;
                    cond.callsMade = true;
                    cond.ippatsu = Array(this.config.playerCount).fill(false);
                    cond.afterKan = true;
                    cond.pendingChankan = true;
                    cond.chankanTarget = e.actor;

                    p.waits = undefined;
                }
                break;

            case 'kita':
                if (e.actor !== undefined) {
                    const p = this.current.players[e.actor];
                    // Always count kita for scoring context, even if hand reconstruction misses N.
                    p.kitaCount++;

                    // Remove 'N' tile from hand when available (fully-observed logs).
                    const idx = p.hand.indexOf('N');
                    if (idx >= 0) {
                        p.hand.splice(idx, 1);
                    } else {
                        console.warn('[GameState] Kita: Could not find N tile in hand for player', e.actor);
                    }

                    // Condition tracking: kita triggers rinshan draw, clears ippatsu
                    const cond = this.current.conditions;
                    cond.afterKan = true;
                    cond.ippatsu = Array(this.config.playerCount).fill(false);
                }
                break;

            case 'reach':
            case 'reach_accepted': // Handle distinct event type if present
                if (e.actor !== undefined) {
                    // Treat 'reach' without step as step 1 (declaration)
                    if (e.type === 'reach' && (!e.step || e.step === '1' || e.step === 1)) {
                        // Only set pending if we didn't just discard the declaration tile
                        const lastEv = this.current.lastEvent;
                        const prevWasDahai = lastEv && lastEv.type === 'dahai' && lastEv.actor === e.actor;

                        if (!prevWasDahai) {
                            this.current.players[e.actor].pendingRiichi = true;
                        }
                    }
                    if (e.type === 'reach_accepted' || (e.type === 'reach' && e.step === '2')) {
                        this.current.players[e.actor].riichi = true;
                        this.current.kyotaku += 1;
                        this.current.players[e.actor].score -= 1000;
                        this.current.players[e.actor].pendingRiichi = false;

                        // Set ippatsu for this player
                        this.current.conditions.ippatsu[e.actor] = true;

                        // Check double riichi: no calls made and first turn not completed
                        const cond = this.current.conditions;
                        if (!cond.callsMade && !cond.firstTurnCompleted[e.actor]) {
                            cond.doubleRiichi[e.actor] = true;
                        }
                    }
                }
                break;

            case 'dora':
                if (e.dora_marker) {
                    this.current.doraMarkers.push(e.dora_marker);
                }
                break;

            case 'hora':
            case 'ryukyoku':
                // Capture conditions at hora time for WASM scoring at end_kyoku
                if (e.type === 'hora' && e.actor !== undefined) {
                    const cond = this.current.conditions;
                    const isTsumo = e.actor === e.target;
                    e._horaConditions = {
                        ippatsu: cond.ippatsu[e.actor],
                        rinshan: isTsumo && cond.afterKan,
                        chankan: !isTsumo && cond.pendingChankan && cond.chankanTarget !== e.actor,
                        tsumoFirstTurn: !cond.callsMade && !cond.firstTurnCompleted[e.actor],
                        doubleRiichi: cond.doubleRiichi[e.actor],
                        wallRemaining: this.current.wallRemaining,
                    };
                }
                if (e.scores) {
                    this.current.players.forEach((p, i) => (p.score = e.scores[i]));
                }
                break;

            case 'end_kyoku': {
                // Check for preceding ryukyoku event
                let ryukyokuEvent: MjaiEvent | null = null;
                for (let i = this.cursor - 1; i >= 0; i--) {
                    const prev = this.events[i];
                    if (prev.type === 'start_kyoku') break;
                    if (prev.type === 'ryukyoku') {
                        ryukyokuEvent = prev;
                        break;
                    }
                }

                if (ryukyokuEvent) {
                    if (!e.meta) e.meta = {};
                    e.meta.ryukyoku = {
                        reason: ryukyokuEvent.reason,
                        deltas: ryukyokuEvent.deltas,
                        scores: ryukyokuEvent.scores,
                    };
                }

                // Build results from hora events if meta.results is missing
                if (!e.meta?.results) {
                    const horaResults: any[] = [];
                    for (let i = this.cursor - 1; i >= 0; i--) {
                        const prev = this.events[i];
                        if (prev.type === 'start_kyoku') break;
                        if (prev.type === 'hora') {
                            horaResults.push({
                                actor: prev.actor,
                                target: prev.target,
                            });
                        }
                    }
                    if (horaResults.length > 0) {
                        if (!e.meta) e.meta = {};
                        e.meta.results = horaResults;
                    }
                }

                // Enrich results with data from preceding hora events
                if (e.meta?.results) {
                    e.meta.results.forEach((res: any) => {
                        // 0. Check if pai is already in the result object (non-standard but possible)
                        if (res.pai) {
                            res.winningTile = res.pai;
                        }

                        let found = false;
                        for (let i = this.cursor - 1; i >= 0; i--) {
                            const prev = this.events[i];
                            if (prev.type === 'start_kyoku') {
                                break;
                            }

                            if (prev.type === 'hora') {
                                if (prev.actor === res.actor) {
                                    // Capture hora-time conditions for WASM scoring
                                    res._horaConditions = prev._horaConditions;

                                    if (prev.pai) {
                                        res.winningTile = prev.pai;
                                        res.uraMarkers = prev.ura_markers;
                                        found = true;
                                        break;
                                    } else {
                                        res.uraMarkers = prev.ura_markers; // Still capture ura markers if present

                                        // Infer winning tile:
                                        // If Ron (target != actor), winning tile is last discard of target.
                                        // If Tsumo (target == actor), winning tile is last tsumo of actor.

                                        const target = prev.target;
                                        const actor = prev.actor;

                                        // Search backwards from the hora event (index i)
                                        for (let j = i - 1; j >= 0; j--) {
                                            const e2 = this.events[j];
                                            if (e2.type === 'start_kyoku') break;

                                            if (target !== undefined && target !== actor) {
                                                // Ron: Look for dahai from target
                                                if (e2.type === 'dahai' && e2.actor === target) {
                                                    res.winningTile = e2.pai;
                                                    found = true;
                                                    break;
                                                }
                                            } else {
                                                // Tsumo: Look for tsumo from actor
                                                if (e2.type === 'tsumo' && e2.actor === actor) {
                                                    res.winningTile = e2.pai;
                                                    found = true;
                                                    break;
                                                }
                                            }
                                        }

                                        if (found) break; // Break outer loop if found
                                    }
                                }
                            }
                        }

                        if (!found && !res.winningTile) {
                            console.warn(`[ResultEnricher] Failed to find winning tile for actor ${res.actor}`);
                        }
                    });

                    // WASM score computation for each result
                    if (isWasmReady()) {
                        this.computeScoresViaWasm(e.meta.results);
                    }
                }
                break;
            }
        }
        this.current.lastEvent = e;
    }

    /** Compute scores via WASM for each hora result at end_kyoku time. */
    private computeScoresViaWasm(results: any[]) {
        results.forEach((res: any) => {
            const actor = res.actor;
            const target = res.target;
            const isTsumo = actor === target;
            const winningTile = res.winningTile;

            if (winningTile === undefined) return; // Can't compute without winning tile

            const player = this.current.players[actor];
            const winTileId = mjaiToTileId(winningTile);
            if (winTileId === null) return;

            // Build hand tiles (136-encoding), excluding the winning tile for tsumo
            const handForScoring = [...player.hand];
            if (isTsumo) {
                // For tsumo, hand has 14 tiles; remove the win tile to get 13
                const winIdx = handForScoring.indexOf(winningTile);
                if (winIdx >= 0) handForScoring.splice(winIdx, 1);
            }

            const tileIds = handForScoring.map((t) => mjaiToTileId(t)).filter((id): id is number => id !== null);

            const meldInputs = meldsToWasmInput(player.melds);

            // Dora indicators
            const doraIds = this.current.doraMarkers
                .map((t) => mjaiToTileId(t))
                .filter((id): id is number => id !== null);

            // Ura dora indicators
            const uraIds = (res.uraMarkers || [])
                .map((t: string) => mjaiToTileId(t))
                .filter((id: number | null): id is number => id !== null);

            // Build conditions from hora-time snapshot
            const horaC = res._horaConditions || {};
            const roundWind = Math.floor(this.current.round / this.config.playerCount); // 0=E, 1=S, 2=W, 3=N

            const pc = this.config.playerCount;
            const conditions: ConditionsInput = {
                tsumo: isTsumo,
                riichi: player.riichi,
                double_riichi: horaC.doubleRiichi || false,
                ippatsu: horaC.ippatsu || false,
                haitei: horaC.wallRemaining === 0 && isTsumo,
                houtei: horaC.wallRemaining === 0 && !isTsumo,
                rinshan: horaC.rinshan || false,
                chankan: horaC.chankan || false,
                tsumo_first_turn: horaC.tsumoFirstTurn || false,
                player_wind: player.wind,
                round_wind: roundWind,
                honba: this.current.honba,
                kita_count: player.kitaCount,
                is_sanma: pc === 3,
            };

            const wasmResult = calculateScore(tileIds, meldInputs, winTileId, doraIds, uraIds, conditions);

            if (wasmResult?.is_win) {
                // Convert WASM result to renderer format
                let points: number;
                const koCount = this.config.playerCount - 2;
                if (!isTsumo) {
                    points = wasmResult.ron_agari;
                } else if (player.wind === 0) {
                    // Dealer tsumo
                    points = wasmResult.tsumo_agari_ko * (this.config.playerCount - 1);
                } else {
                    // Non-dealer tsumo
                    points = wasmResult.tsumo_agari_oya + wasmResult.tsumo_agari_ko * koCount;
                }

                res.score = {
                    han: wasmResult.han,
                    fu: wasmResult.fu,
                    points: points,
                    yaku: wasmResult.yaku,
                };
            }
            // If WASM failed, res.score retains its original meta value (fallback)
        });
    }
}
