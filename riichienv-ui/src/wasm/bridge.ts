/**
 * High-level TypeScript API wrapping raw WASM calls.
 *
 * All functions return null when WASM is not loaded (graceful degradation).
 * Tile IDs use 136-encoding (0-135) unless noted otherwise.
 */

import { getWasm, isWasmReady } from './loader';

export interface ScoreResult {
    is_win: boolean;
    yakuman: boolean;
    han: number;
    fu: number;
    ron_agari: number;
    tsumo_agari_oya: number;
    tsumo_agari_ko: number;
    yaku: number[];
}

export interface MeldInput {
    meld_type: string; // 'chi' | 'pon' | 'daiminkan' | 'ankan' | 'kakan'
    tiles: number[];   // tile IDs in 136-encoding
}

export interface ConditionsInput {
    tsumo?: boolean;
    riichi?: boolean;
    double_riichi?: boolean;
    ippatsu?: boolean;
    haitei?: boolean;
    houtei?: boolean;
    rinshan?: boolean;
    chankan?: boolean;
    tsumo_first_turn?: boolean;
    player_wind?: number;
    round_wind?: number;
    honba?: number;
}

/**
 * Calculate wait tiles for a hand.
 * Returns tile types (34-encoding: tile_id / 4) or null if WASM unavailable.
 */
export function calculateWaits(tiles136: number[], melds: MeldInput[] = []): number[] | null {
    if (!isWasmReady()) return null;
    const wasm = getWasm()!;
    try {
        return wasm.calc_waits(
            JSON.stringify(tiles136),
            JSON.stringify(melds)
        );
    } catch (e) {
        console.warn('[WASM] calculateWaits failed:', e);
        return null;
    }
}

/**
 * Calculate shanten number for a hand.
 * Returns -1 for tenpai, 0 for iishanten, etc.
 */
export function calculateShanten(tiles136: number[]): number | null {
    if (!isWasmReady()) return null;
    const wasm = getWasm()!;
    try {
        return wasm.calc_shanten(JSON.stringify(tiles136));
    } catch (e) {
        console.warn('[WASM] calculateShanten failed:', e);
        return null;
    }
}

/**
 * Calculate score for a winning hand.
 */
export function calculateScore(
    tiles136: number[],
    melds: MeldInput[],
    winTile: number,
    doraIndicators: number[],
    uraIndicators: number[],
    conditions: ConditionsInput
): ScoreResult | null {
    if (!isWasmReady()) return null;
    const wasm = getWasm()!;
    try {
        return wasm.calc_score(
            JSON.stringify(tiles136),
            JSON.stringify(melds),
            winTile,
            JSON.stringify(doraIndicators),
            JSON.stringify(uraIndicators),
            JSON.stringify(conditions)
        );
    } catch (e) {
        console.warn('[WASM] calculateScore failed:', e);
        return null;
    }
}

/**
 * Convert MJAI tile string (e.g. "5mr", "1m", "E") to 136-encoding tile ID.
 */
export function mjaiToTileId(mjai: string): number | null {
    if (!isWasmReady()) return null;
    const wasm = getWasm()!;
    try {
        const result = wasm.mjai_to_tile_id(mjai);
        return result !== undefined ? result : null;
    } catch (e) {
        return null;
    }
}

/**
 * Convert 136-encoding tile ID to MJAI tile string.
 */
export function tileIdToMjai(tid: number): string | null {
    if (!isWasmReady()) return null;
    const wasm = getWasm()!;
    try {
        return wasm.tile_id_to_mjai(tid);
    } catch (e) {
        return null;
    }
}

/**
 * Check if a hand is tenpai.
 */
export function checkTenpai(tiles136: number[], melds: MeldInput[] = []): boolean | null {
    if (!isWasmReady()) return null;
    const wasm = getWasm()!;
    try {
        return wasm.is_tenpai(
            JSON.stringify(tiles136),
            JSON.stringify(melds)
        );
    } catch (e) {
        console.warn('[WASM] checkTenpai failed:', e);
        return null;
    }
}
