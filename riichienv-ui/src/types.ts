export interface MjaiEvent {
    type: string;
    actor?: number;
    target?: number;
    pai?: string;
    consumed?: string[];
    meta?: {
        waits?: string[];
        score?: {
            han: number;
            fu: number;
            points: number;
            yaku: number[];
        };
        [key: string]: any;
    };
    [key: string]: any;
}

export interface Tile {
    tile: string;
    isRiichi?: boolean;
    isTsumogiri?: boolean;
}

export interface PlayerState {
    hand: string[];
    discards: Tile[]; // Changed from string[] to Tile[]
    melds: { type: string; tiles: string[]; from: number }[];
    score: number;
    riichi: boolean;
    pendingRiichi?: boolean; // Waiting for discard to mark as riichi
    wind: number;
    waits?: string[];
    lastDrawnTile?: string;
}

export interface ConditionTracker {
    ippatsu: boolean[];              // [p0, p1, p2, p3]
    afterKan: boolean;               // Post-kan flag (for rinshan)
    pendingChankan: boolean;         // Kakan pending flag (for chankan)
    chankanTarget?: number;          // Actor who declared kakan
    callsMade: boolean;              // Any call made in this kyoku
    firstTurnCompleted: boolean[];   // Per-player first dahai completed
    turnCount: number;               // Total dahai count in kyoku
    doubleRiichi: boolean[];         // Per-player double riichi declared
}

export interface BoardState {
    playerCount: number;
    players: PlayerState[];
    doraMarkers: string[];
    round: number; // Kyoku (0-indexed, 0=E1)
    honba: number;
    kyotaku: number;
    wallRemaining: number;
    currentActor: number;
    lastEvent?: MjaiEvent;
    eventIndex: number;
    totalEvents: number;
    dahaiAnim?: {
        discardIdx: number;
        insertIdx: number;
        tsumogiri: boolean;
        drawnTile?: string;
    };
    conditions: ConditionTracker;
}
