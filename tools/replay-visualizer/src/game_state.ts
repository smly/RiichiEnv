import { MjaiEvent, PlayerState, BoardState } from './types';

// Helper to sort hand (simple alphanumeric sort for now, ideally strictly by tile order)
const sortHand = (hand: string[]) => {
    const order = (t: string) => {
        // 1m..9m, 1p..9p, 1s..9s, 1z..7z. red (0/5xr) treated nicely?
        // MJAI: 5mr, 5pr, 5sr.
        if (t === 'back') return 9999;

        let suit = t.slice(-1); // m, p, s, z, r
        let num = parseInt(t[0]);

        // Handle red 5s: 5mr -> suit=r, but we want it near 5m.
        if (t.endsWith('r')) {
            suit = t.slice(-2, -1); // 'm' from '5mr'
            num = 5;
        }

        const suitOrder = { 'm': 0, 'p': 10, 's': 20, 'z': 30 };
        // @ts-ignore
        return (suitOrder[suit] || 0) * 10 + num + (t.endsWith('r') ? 0.5 : 0);
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
    }

    // Returns list of indices where new rounds start
    getKyokuCheckpoints(): { index: number, round: number, honba: number }[] {
        const checkpoints: { index: number, round: number, honba: number }[] = [];
        this.events.forEach((e, i) => {
            if (e.type === 'start_kyoku') {
                checkpoints.push({
                    index: i,
                    round: (e.kyoku || 1) - 1,
                    honba: e.honba || 0
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
                wind: 0
            })),
            doraMarkers: [],
            round: 0,
            honba: 0,
            kyotaku: 0,
            wallRemaining: 70, // Start estimate or parse start_kyoku
            currentActor: 0
        };
    }

    // Returns true if state changed
    stepForward(): boolean {
        if (this.cursor >= this.events.length) return false;
        const event = this.events[this.cursor];
        this.processEvent(event);
        this.cursor++;
        return true;
    }

    // Very naive step back: reset and replay to cursor-1
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

    reset() {
        this.cursor = 0;
        this.current = this.initialState();
    }

    processEvent(e: MjaiEvent) {
        switch (e.type) {
            case 'start_game':
                break;
            case 'start_kyoku':
                this.current.round = (e.kyoku || 1) - 1;
                this.current.honba = e.honba || 0;
                this.current.kyotaku = e.kyotaku || 0;
                this.current.doraMarkers = [e.dora_marker];
                this.current.currentActor = e.oya;
                this.current.players.forEach((p, i) => {
                    p.hand = e.tehais[i].map((t: string) => t); // Clone
                    p.discards = [];
                    p.melds = [];
                    p.riichi = false;
                    p.score = e.scores[i];
                });
                break;

            case 'tsumo':
                // actor draws pai
                if (e.actor !== undefined && e.pai) {
                    this.current.players[e.actor].hand.push(e.pai);
                    this.current.currentActor = e.actor;
                }
                break;

            case 'dahai':
                if (e.actor !== undefined && e.pai) {
                    const p = this.current.players[e.actor];
                    // Remove from hand. 
                    // Note: MJAI 'pai' matches the tile string exactly usually.
                    const idx = p.hand.indexOf(e.pai);
                    if (idx >= 0) {
                        p.hand.splice(idx, 1);
                    } else {
                        // Tsumogiri (sometimes implied if not found?)
                        // Or maybe we missed the tsumo event?
                    }
                    p.hand = sortHand(p.hand);
                    p.discards.push(e.pai);
                    this.current.currentActor = e.actor;
                }
                break;

            case 'pon':
            case 'chi':
            case 'daiminkan':
                if (e.actor !== undefined && e.target !== undefined && e.pai && e.consumed) {
                    const p = this.current.players[e.actor];
                    // Remove consumed tiles from hand
                    e.consumed.forEach(t => {
                        const idx = p.hand.indexOf(t);
                        if (idx >= 0) p.hand.splice(idx, 1);
                    });

                    // Add meld
                    // e.pai is the stolen tile
                    // We need to visually represent who it came from (relative index)
                    p.melds.push({
                        type: e.type,
                        tiles: [...e.consumed, e.pai], // Naive
                        from: e.target
                    });

                    this.current.currentActor = e.actor;

                    // Remove the stolen tile from target's discard (it was just discarded)
                    const targetP = this.current.players[e.target];
                    if (targetP.discards.length > 0) {
                        targetP.discards.pop();
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
                }
                break;

            case 'reach':
                // Step 1: Declare (prior to discard)
                // Step 2: Success (step attribute usually, or inferred)
                if (e.actor !== undefined) {
                    // We'll mark them as riichi pending or just riichi
                    // Usually MJAI has 'reach' output BEFORE the discard. 
                    // Real Riichi is established after discard passes. 
                    // For viz, we can show logic immediately.
                    // However, simple boolean toggle for now:
                    if (e.step === '2') {
                        this.current.players[e.actor].riichi = true;
                        this.current.kyotaku += 1;
                        this.current.players[e.actor].score -= 1000;
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
                // End of round logic if needed
                if (e.scores) {
                    // Update scores
                    this.current.players.forEach((p, i) => p.score = e.scores[i]);
                }
                break;
        }
    }
}
