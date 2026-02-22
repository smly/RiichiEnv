/**
 * Game and layout configuration for multi-variant viewer support.
 *
 * Centralizes all player-count-dependent constants so that
 * 4-player and 3-player (sanma) variants can share the same
 * rendering and game-state logic.
 */

export interface GameConfig {
    /** Number of players (4 for standard, 3 for sanma). */
    playerCount: number;
    /** Starting score for each player. */
    defaultScores: number[];
    /** MJAI wind identifiers per seat. */
    winds: string[];
    /** Sprite-sheet keys for wind kanji display. */
    windCharKeys: string[];
    /** Initial wall tile count at start of each kyoku. */
    initialWallRemaining: number;
}

export interface LayoutConfig {
    /** Base board dimension in px (square). */
    boardSize: number;
    /** Rotation angles (degrees) for each player position relative to viewpoint. */
    playerAngles: number[];
    /** Overall content area width (board + sidebar). */
    contentWidth: number;
    /** Overall content area height. */
    contentHeight: number;
    /** View area dimension (board container). */
    viewAreaSize: number;
}

// ---------------------------------------------------------------------------
// 4-player (standard) presets
// ---------------------------------------------------------------------------

export function createGameConfig4P(): GameConfig {
    return {
        playerCount: 4,
        defaultScores: [25000, 25000, 25000, 25000],
        winds: ['E', 'S', 'W', 'N'],
        windCharKeys: ['東_red', '南', '西', '北'],
        initialWallRemaining: 70,
    };
}

export function createLayoutConfig4P(): LayoutConfig {
    return {
        boardSize: 800,
        playerAngles: [0, -90, 180, 90],
        contentWidth: 970,
        contentHeight: 900,
        viewAreaSize: 880,
    };
}

// ---------------------------------------------------------------------------
// 3-player (sanma) presets
// ---------------------------------------------------------------------------

export function createGameConfig3P(): GameConfig {
    return {
        playerCount: 3,
        defaultScores: [35000, 35000, 35000],
        winds: ['E', 'S', 'W'],
        windCharKeys: ['東_red', '南', '西'],
        initialWallRemaining: 55,
    };
}

export function createLayoutConfig3P(): LayoutConfig {
    return {
        boardSize: 800,
        playerAngles: [0, -120, 120],
        contentWidth: 970,
        contentHeight: 900,
        viewAreaSize: 880,
    };
}

// ---------------------------------------------------------------------------
// 3D layout configuration (16:9 perspective view)
// ---------------------------------------------------------------------------

export interface LayoutConfig3D {
    /** View area width in px. */
    viewAreaWidth: number;
    /** View area height in px. */
    viewAreaHeight: number;
    /** Overall content area width (same as viewAreaWidth; sidebar is overlay). */
    contentWidth: number;
    /** Overall content area height. */
    contentHeight: number;
    /** Table surface dimension in px (square). */
    tableSize: number;
    /** CSS perspective value in px (higher = weaker perspective / telephoto). */
    perspective: number;
    /** Table tilt angle in degrees (rotateX). */
    tiltAngle: number;
    /** Hand layer height at bottom of viewport in px. */
    handLayerHeight: number;
    /** Tile dimensions [width, height] per rendering context. */
    tileSizes: {
        riverTile: [number, number];
        opponentTile: [number, number];
        ownTile: [number, number];
        doraTile: [number, number];
        meldTileTable: [number, number];
        meldTileOwn: [number, number];
    };
}

export function createLayout3DConfig4P(): LayoutConfig3D {
    return {
        viewAreaWidth: 1280,
        viewAreaHeight: 720,
        contentWidth: 1280,
        contentHeight: 720,
        tableSize: 750,
        perspective: 1500,
        tiltAngle: 35,
        handLayerHeight: 120,
        tileSizes: {
            riverTile: [26, 36],
            opponentTile: [22, 30],
            ownTile: [50, 70],
            doraTile: [18, 25],
            meldTileTable: [20, 28],
            meldTileOwn: [40, 56],
        },
    };
}

export function createLayout3DConfig3P(): LayoutConfig3D {
    return {
        viewAreaWidth: 1280,
        viewAreaHeight: 720,
        contentWidth: 1280,
        contentHeight: 720,
        tableSize: 880,
        perspective: 1500,
        tiltAngle: 48,
        handLayerHeight: 120,
        tileSizes: {
            riverTile: [26, 36],
            opponentTile: [22, 30],
            ownTile: [50, 70],
            doraTile: [18, 25],
            meldTileTable: [20, 28],
            meldTileOwn: [40, 56],
        },
    };
}
