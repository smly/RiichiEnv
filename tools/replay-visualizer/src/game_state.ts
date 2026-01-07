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
        this.events = events;
        this.cursor = 0;
        this.current = this.initialState();

        // Skip initial events (start_game, etc.) and jump to first meaningful state
        const firstKyoku = this.events.findIndex(e => e.type === 'start_kyoku');
        if (firstKyoku !== -1) {
            // Jump to firstKyoku + 1 to ensure the start_kyoku event is processed
            this.jumpTo(firstKyoku + 1);
        } else if (this.events.length > 2) {
            this.jumpTo(2);
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
        if (this.cursor <= 0) return false;
        const target = this.cursor - 1;
        this.reset();
        while (this.cursor < target) {
            this.stepForward();
        }
        return true;
    }

    jumpTo(index: number) {
        if (index < 0) index = 0;
        if (index > this.events.length) index = this.events.length;

        if (index < this.cursor) {
            this.reset();
        }
        while (this.cursor < index) {
            this.stepForward();
        }
    }

    // Jump to next/prev turn for a specific actor
    stepTurn(forward: boolean, actor: number) {
        let target = this.cursor;
        const len = this.events.length;

        if (forward) {
            for (let i = this.cursor + 1; i < len; i++) {
                const e = this.events[i];
                if (e.actor === actor && ['tsumo', 'dahai', 'pon', 'chi', 'kan', 'daiminkan', 'kakan', 'ankan'].includes(e.type)) {
                    target = i;
                    break;
                }
            }
        } else {
            // Step back once to avoid staying on current event if we are on an actor's event
            for (let i = this.cursor - 2; i >= 0; i--) {
                const e = this.events[i];
                if (e.actor === actor && ['tsumo', 'dahai', 'pon', 'chi', 'kan', 'daiminkan', 'kakan', 'ankan'].includes(e.type)) {
                    target = i;
                    break;
                }
            }
        }
        // Jump to index + 1 to show the STATE AFTER the action
        this.jumpTo(target + 1);
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
                    this.current.players[e.actor].hand = sortHand(this.current.players[e.actor].hand);
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

            case 'kakan': // Added Kan
            case 'ankan': // Closed Kan
                if (e.actor !== undefined && e.consumed) {
                    const p = this.current.players[e.actor];
                    e.consumed.forEach(t => {
                        const idx = p.hand.indexOf(t);
                        if (idx >= 0) p.hand.splice(idx, 1);
                    });
                    p.melds.push({
                        type: e.type,
                        tiles: e.consumed, // all tiles involved
                        from: e.actor
                    });
                    p.waits = undefined; // Clear waits
                }
                break;

            case 'reach':
                if (e.actor !== undefined) {
                    if (e.step === '1') {
                        this.current.players[e.actor].pendingRiichi = true;
                    }
                    if (e.step === '2') {
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
