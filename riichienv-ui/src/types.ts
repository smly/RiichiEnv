import type { Locale } from './i18n/index';
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
    kitaCount: number;
}

export interface ConditionTracker {
    ippatsu: boolean[]; // [p0, p1, p2, p3]
    afterKan: boolean; // Post-kan flag (for rinshan)
    pendingChankan: boolean; // Kakan pending flag (for chankan)
    chankanTarget?: number; // Actor who declared kakan
    callsMade: boolean; // Any call made in this kyoku
    firstTurnCompleted: boolean[]; // Per-player first dahai completed
    turnCount: number; // Total dahai count in kyoku
    doubleRiichi: boolean[]; // Per-player double riichi declared
}

// --- Public API types ---

export interface PlayerConfig {
    name?: string;
    avatarUrl?: string;
}

export interface ViewerOptions {
    language?: Locale;
    log: MjaiEvent[];
    renderer?: '2d' | '3d';
    perspective?: number;
    freeze?: boolean;
    initialPosition?: { kyoku?: number; step?: number };
    players?: PlayerConfig[];
}

export interface ViewerPosition {
    kyokuIndex: number;
    step: number;
    totalSteps: number;
}

export interface KyokuInfo {
    index: number;
    round: number;
    honba: number;
    scores: number[];
}

export type ViewerEventMap = {
    positionChange: { kyokuIndex: number; step: number };
    kyokuChange: { kyokuIndex: number; round: number; honba: number };
    viewpointChange: { viewpoint: number };
};

// --- Analysis types ---

export interface KyokuSummary {
    index: number;
    dealer?: number;
    completed?: boolean;
    deltasKnown?: boolean;
    round: number;
    honba: number;
    startScores: number[];
    endScores: number[];
    deltas: number[];
    result: KyokuResult | null;
    playerActions: KyokuPlayerAction[];
}

export interface KyokuResult {
    type: 'hora' | 'ryukyoku';
    winners?: KyokuWinner[];
    reason?: string;
}

export interface KyokuWinner {
    actor: number;
    target: number;
    isTsumo: boolean;
    points: number;
    han: number;
    fu: number;
    yaku: string[];
    limit?: string;
}

export interface KyokuPlayerAction {
    riichi: boolean;
    tenpai: boolean;
    houjuu: boolean;
    hora: boolean;
    tsumo: boolean;
    meldTypes: string[];
}

export interface KyokuKeyEvent {
    step: number;
    type: string;
    actor: number;
    label: string;
    detail?: string;
}

// --- Internal types ---

export interface BoardState {
    playerCount: number;
    players: PlayerState[];
    playerNames: string[];
    playerAvatars: (string | null)[];
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
