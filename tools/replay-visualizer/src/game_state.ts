import { MjaiEvent, PlayerState, BoardState } from './types';

// Helper to sort hand (simple alphanumeric sort for now, ideally strictly by tile order)
const sortHand = (hand: string[]) => {
    const order = (t: string) => {
        if (t === 'back') return 9999;

        let suit = '';
        let num = 0;
        let isRed = false;

        // Handle Honors (z)
        const honorMap: { [key: string]: number } = {
            'E': 1, 'S': 2, 'W': 3, 'N': 4, // Winds
            'P': 5, 'F': 6, 'C': 7          // Dragons (Haku/White, Hatsu/Green, Chun/Red)
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
                num = parseInt(t.charAt(0));
            } else {
                suit = t.charAt(t.length - 1); // 1m -> m
                num = parseInt(t.charAt(0));
            }

            // Handle "0m" case if present in data (treat as Red 5)
            if (num === 0) {
                num = 5;
                isRed = true;
            }
        }

        const suitOrder: Record<string, number> = { 'm': 0, 'p': 100, 's': 200, 'z': 300 };
        const redOffset = isRed ? 0.1 : 0;

        return (suitOrder[suit] ?? 900) + num + redOffset;
    };
    return [...hand].sort((a, b) => order(a) - order(b));
};

export class GameState {
    events: MjaiEvent[];
    cursor: number;

    // Cache state at each step to allow fast jumping
    // For MVP, we might re-compute from nearest checkpoint or just replay from 0.
    // Let's implement full replay from 0 for simplicity first, or incremental update.
    current: BoardState;

    constructor(events: MjaiEvent[]) {
        // Filter out null events and start/end game events
        this.events = events.filter(e => e && e.type !== 'start_game' && e.type !== 'end_game');
        this.cursor = 0;
        this.current = this.initialState();

        // Jump to first meaningful state (start_kyoku + 1)
        const firstKyoku = this.events.findIndex(e => e.type === 'start_kyoku');
        if (firstKyoku !== -1) {
            this.jumpTo(firstKyoku + 1);
        }
    }

    // Returns list of indices where new rounds start
    getKyokuCheckpoints(): { index: number, round: number, honba: number, scores: number[] }[] {
        const checkpoints: { index: number, round: number, honba: number, scores: number[] }[] = [];
        this.events.forEach((e, i) => {
            if (e.type === 'start_kyoku') {
                checkpoints.push({
                    index: i,
                    round: this.getRoundIndex(e),
                    honba: e.honba || 0,
                    scores: e.scores || [25000, 25000, 25000, 25000]
                });
            }
        });
        return checkpoints;
    }

    initialState(): BoardState {
        return {
            players: Array(4).fill(0).map(() => ({
                hand: [],
                discards: [],
                melds: [],
                score: 25000,
                riichi: false,
                pendingRiichi: false,
                wind: 0
            })),
            doraMarkers: [],
            round: 0,
            honba: 0,
            kyotaku: 0,
            wallRemaining: 70,
            currentActor: 0,
            eventIndex: 0,
            totalEvents: this.events.length
        };
    }

    // Returns true if state changed
    stepForward(): boolean {
        if (this.cursor >= this.events.length) return false;
        const event = this.events[this.cursor];
        this.processEvent(event);
        this.cursor++;
        this.current.eventIndex = this.cursor; // Sync
        return true;
    }

    stepBackward(): boolean {
        // Prevent going back to 0 (before first start_kyoku)
        if (this.cursor <= 1) return false;
        const target = this.cursor - 1;
        this.reset();
        while (this.cursor < target) {
            this.stepForward();
        }
        return true;
    }

    jumpTo(index: number) {
        if (index < 1) index = 1; // Enforce minimum 1
        if (index > this.events.length) index = this.events.length;

        if (index < this.cursor) {
            this.reset();
        }
        while (this.cursor < index) {
            this.stepForward();
        }
    }

    // Jump to next turn for specific actor (tsumo event)
    jumpToNextTurn(actor: number): boolean {
        let target = -1;
        for (let i = this.cursor; i < this.events.length; i++) {
            const e = this.events[i];
            if (e.type === 'tsumo' && e.actor === actor) {
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
            if (e.type === 'tsumo' && e.actor === actor) {
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
        // Search backwards. If we are currently AT a start_kyoku (cursor is just after it), 
        // we likely want the PREVIOUS one, so search from cursor - 2.
        // But if we are mid-game, finding the *current* start_kyoku is basically "Restart Kyoku".
        // The user asked for "Previous Kyoku", which implies index - 1.

        // Let's interpret "Prev Kyoku" as:
        // 1. Find the start_kyoku of the CURRENT round first.
        // 2. If we are far into the round, maybe just jumping to START of current is okay?
        // User probably expects: Round 1 -> Round 2 -> [Prev] -> Round 1.

        // Algorithm:
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

    reset() {
        this.cursor = 0;
        this.current = this.initialState();
    }

    private getRoundIndex(e: MjaiEvent): number {
        const kyoku = (e.kyoku || 1) - 1;
        const bakaze = e.bakaze || 'E';
        let offset = 0;
        if (bakaze === 'S') offset = 4;
        else if (bakaze === 'W') offset = 8;
        else if (bakaze === 'N') offset = 12;
        return offset + kyoku;
    }

    processEvent(e: MjaiEvent) {
        switch (e.type) {
            case 'start_game':
                break;
            case 'start_kyoku':
                this.current.round = this.getRoundIndex(e);
                this.current.honba = e.honba || 0;
                this.current.kyotaku = e.kyotaku || 0;
                this.current.doraMarkers = [e.dora_marker];
                this.current.currentActor = e.oya;
                this.current.players.forEach((p, i) => {
                    p.hand = sortHand(e.tehais[i].map((t: string) => t)); // Clone and sort
                    p.discards = [];
                    p.melds = [];
                    p.riichi = false;
                    p.pendingRiichi = false;
                    p.score = e.scores[i];
                    p.waits = undefined;
                    // Assign wind based on oya
                    p.wind = (i - e.oya + 4) % 4;
                });
                break;

            case 'tsumo':
                if (e.actor !== undefined && e.pai) {
                    this.current.players[e.actor].hand.push(e.pai);
                    // Do NOT sort hand here. 
                    // User wants the drawn tile to be visually separated on the right.
                    // Renderer separates the LAST tile. So we just push it.
                    // this.current.players[e.actor].hand = sortHand(this.current.players[e.actor].hand);
                    this.current.currentActor = e.actor;
                }
                break;

            case 'dahai':
                if (e.actor !== undefined && e.pai) {
                    const p = this.current.players[e.actor];
                    const idx = p.hand.indexOf(e.pai);
                    if (idx >= 0) {
                        p.hand.splice(idx, 1);
                    }
                    p.hand = sortHand(p.hand);

                    // Riichi Logic
                    let isRiichi = false;
                    if (p.pendingRiichi || e.reach) {
                        isRiichi = true;
                        p.pendingRiichi = false;
                    }

                    p.discards.push({ tile: e.pai, isRiichi, isTsumogiri: !!e.tsumogiri });
                    p.waits = e.meta?.waits;
                    this.current.currentActor = e.actor;
                }
                break;

            case 'pon':
            case 'chi':
            case 'daiminkan':
                if (e.actor !== undefined && e.target !== undefined && e.pai && e.consumed) {
                    const p = this.current.players[e.actor];
                    e.consumed.forEach(t => {
                        const idx = p.hand.indexOf(t);
                        if (idx >= 0) p.hand.splice(idx, 1);
                    });

                    // Add meld
                    p.melds.push({
                        type: e.type,
                        tiles: [...e.consumed, e.pai],
                        from: e.target
                    });

                    p.waits = undefined;

                    this.current.currentActor = e.actor;

                    // Remove from target discard
                    const targetP = this.current.players[e.target];
                    if (targetP.discards.length > 0) {
                        const stolen = targetP.discards.pop();
                        // If stolen tile was Riichi declared, player must re-declare on next discard
                        if (stolen && stolen.isRiichi) {
                            targetP.pendingRiichi = true;
                        }
                    }
                }
                break;

            case 'ankan': // Closed Kan
                if (e.actor !== undefined && e.consumed) {
                    const p = this.current.players[e.actor];
                    e.consumed.forEach(t => {
                        const idx = p.hand.indexOf(t);
                        if (idx >= 0) p.hand.splice(idx, 1);
                    });
                    p.melds.push({
                        type: e.type,
                        tiles: e.consumed, // all 4 tiles
                        from: e.actor
                    });
                    p.waits = undefined;
                }
                break;

            case 'kakan': // Added Kan
                if (e.actor !== undefined && e.pai && e.consumed) {
                    const p = this.current.players[e.actor];
                    // Remove the added tile from hand (usually e.consumed has it, or just e.pai)
                    // MJAI spec: kakan event has pai (added tile) and consumed (array with just that tile)
                    const addedTile = e.pai;

                    // Remove from hand
                    const idx = p.hand.indexOf(addedTile);
                    if (idx >= 0) p.hand.splice(idx, 1);

                    // Find generic version for matching (ignore red/0)
                    const normalize = (t: string) => t.replace('0', '5').replace('r', '');
                    const targetNorm = normalize(addedTile);

                    // Find existing Pon
                    const pon = p.melds.find(m => m.type === 'pon' && normalize(m.tiles[0]) === targetNorm);

                    if (pon) {
                        pon.type = 'kakan';
                        pon.tiles.push(addedTile);
                    } else {
                        // Fallback: This shouldn't happen in valid logs, but prevent crash
                        console.warn("[GameState] Kakan: Could not find original Pon for", addedTile);
                        p.melds.push({
                            type: 'kakan',
                            tiles: [addedTile, addedTile, addedTile, addedTile], // Placeholder
                            from: e.actor
                        });
                    }
                    p.waits = undefined;
                }
                break;

            case 'reach':
            case 'reach_accepted': // Handle distinct event type if present
                if (e.actor !== undefined) {
                    if (e.type === 'reach' && e.step === '1') {
                        this.current.players[e.actor].pendingRiichi = true;
                    }
                    if (e.type === 'reach_accepted' || (e.type === 'reach' && e.step === '2')) {
                        this.current.players[e.actor].riichi = true;
                        this.current.kyotaku += 1;
                        this.current.players[e.actor].score -= 1000;
                        this.current.players[e.actor].pendingRiichi = false;
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
                if (e.scores) {
                    this.current.players.forEach((p, i) => p.score = e.scores[i]);
                }
                break;
        }
        this.current.lastEvent = e;
    }
}
