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
}

export interface BoardState {
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
}
