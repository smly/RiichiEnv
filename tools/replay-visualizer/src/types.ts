export interface MjaiEvent {
    type: string;
    actor?: number;
    target?: number;
    pai?: string;
    consumed?: string[];
    [key: string]: any;
}

export interface PlayerState {
    hand: string[];
    discards: string[]; // List of tile strings
    melds: { type: string; tiles: string[]; from: number }[];
    score: number;
    riichi: boolean;
    wind: number; // 0=East, 1=South, etc. (relative to Oya? No, absolute for seat)
}

export interface BoardState {
    players: PlayerState[];
    doraMarkers: string[];
    round: number; // Kyoku (0-indexed, 0=E1)
    honba: number;
    kyotaku: number;
    wallRemaining: number;
    currentActor: number;
}
