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
