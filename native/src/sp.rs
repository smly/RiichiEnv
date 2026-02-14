//! Single Player (SP) Calculator for Mahjong
//!
//! Computes tenpai probability, win probability, and expected value (EV)
//! for each possible discard over future turns.
//!
//! Ported from Mortal's libriichi/src/algo/sp/ with full optimizations:
//! - Lookup-table shanten calculation (O(1) per call)
//! - AHashMap for fast state caching
//! - Rc<Values> for zero-copy cache sharing
//! - Pre-computed probability tables
//! - Const generics for compile-time loop optimization
//! - State discard/undo_discard for in-place mutation
//! - Uses RiichiEnv's existing agari/yaku/score infrastructure

use std::rc::Rc;

use ahash::AHashMap;

use crate::score;
use crate::types::{Hand, Meld, TILE_MAX};
use crate::yaku::{self, YakuContext};

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of future draws to consider
pub const MAX_TSUMOS_LEFT: usize = 17;

/// Shanten threshold: only compute SP tables for shanten <= this value
const SHANTEN_THRES: i8 = 3;

/// Maximum tiles that can be left in wall (34*4 - 1 hand - 13 initial dead)
const MAX_TILES_LEFT: usize = 34 * 4 - 1 - 13;

// ============================================================================
// Shanten tables (Nyanten / Cryolite algorithm)
// ============================================================================

/// Shupai (number suit) hash table: [9 tiles][15 total_count][5 mentsu_count]
/// Used to compute a cumulative hash over a suit's tile counts.
#[rustfmt::skip]
const SHUPAI_TABLE: [[[u32; 5]; 15]; 9] = [
    [ // i = 0
        [0,0,0,0,0], [0,139150,0,0,0], [0,105150,244300,0,0],
        [0,75750,180900,320050,0], [0,51810,127560,232710,371860],
        [0,33490,85300,161050,266200], [0,0,33490,85300,161050],
        [0,0,0,33490,85300], [0,0,0,0,33490],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 1
        [0,0,0,0,0], [0,43130,0,0,0], [0,34995,78125,0,0],
        [0,27120,62115,105245,0], [0,19980,47100,82095,125225],
        [0,13925,33905,61025,96020], [0,9130,23055,43035,70155],
        [0,5595,14725,28650,48630], [0,3180,8775,17905,31830],
        [0,1660,4840,10435,19565], [0,0,1660,4840,10435],
        [0,0,0,1660,4840], [0,0,0,0,1660],
        [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 2
        [0,0,0,0,0], [0,11880,0,0,0], [0,10374,22254,0,0],
        [0,8688,19062,30942,0], [0,6937,15625,25999,37879],
        [0,5251,12188,20876,31250], [0,3745,8996,15933,24621],
        [0,2499,6244,11495,18432], [0,1548,4047,7792,13043],
        [0,882,2430,4929,8674], [0,456,1338,2886,5385],
        [0,210,666,1548,3096], [0,84,294,750,1632],
        [0,28,112,322,778], [0,0,28,112,322],
    ],
    [ // i = 3
        [0,0,0,0,0], [0,2878,0,0,0], [0,2693,5571,0,0],
        [0,2438,5131,8009,0], [0,2118,4556,7249,10127],
        [0,1753,3871,6309,9002], [0,1372,3125,5243,7681],
        [0,1007,2379,4132,6250], [0,687,1694,3066,4819],
        [0,432,1119,2126,3498], [0,247,679,1366,2373],
        [0,126,373,805,1492], [0,56,182,429,861],
        [0,21,77,203,450], [0,6,27,83,209],
    ],
    [ // i = 4
        [0,0,0,0,0], [0,620,0,0,0], [0,610,1230,0,0],
        [0,590,1200,1820,0], [0,555,1145,1755,2375],
        [0,503,1058,1648,2258], [0,435,938,1493,2083],
        [0,355,790,1293,1848], [0,270,625,1060,1563],
        [0,190,460,815,1250], [0,122,312,582,937],
        [0,70,192,382,652], [0,35,105,227,417],
        [0,15,50,120,242], [0,5,20,55,125],
    ],
    [ // i = 5
        [0,0,0,0,0], [0,125,0,0,0], [0,125,250,0,0],
        [0,125,250,375,0], [0,124,249,374,499],
        [0,121,245,370,495], [0,115,236,360,485],
        [0,105,220,341,465], [0,90,195,310,431],
        [0,72,162,267,382], [0,53,125,215,320],
        [0,35,88,160,250], [0,20,55,108,180],
        [0,10,30,65,118], [0,4,14,34,69],
    ],
    [ // i = 6
        [0,0,0,0,0], [0,25,0,0,0], [0,25,50,0,0],
        [0,25,50,75,0], [0,25,50,75,100],
        [0,25,50,75,100], [0,25,50,75,100],
        [0,25,50,75,100], [0,24,49,74,99],
        [0,22,46,71,96], [0,19,41,65,90],
        [0,15,34,56,80], [0,10,25,44,66],
        [0,6,16,31,50], [0,3,9,19,34],
    ],
    [ // i = 7
        [0,0,0,0,0], [0,5,0,0,0], [0,5,10,0,0],
        [0,5,10,15,0], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,4,9,14,19],
        [0,3,7,12,17], [0,2,5,9,14],
    ],
    [ // i = 8
        [0,0,0,0,0], [0,1,0,0,0], [0,1,2,0,0],
        [0,1,2,3,0], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
    ],
];

/// Zipai (honor suit) hash table: [7 tiles][15 total_count][5 mentsu_count]
#[rustfmt::skip]
const ZIPAI_TABLE: [[[u32; 5]; 15]; 7] = [
    [ // i = 0
        [0,0,0,0,0], [0,11880,0,0,0], [0,10374,22254,0,0],
        [0,8688,19062,30942,0], [0,6937,15625,25999,37879],
        [0,5251,12188,20876,31250], [0,0,5251,12188,20876],
        [0,0,0,5251,12188], [0,0,0,0,5251],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
        [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 1
        [0,0,0,0,0], [0,2878,0,0,0], [0,2693,5571,0,0],
        [0,2438,5131,8009,0], [0,2118,4556,7249,10127],
        [0,1753,3871,6309,9002], [0,1372,3125,5243,7681],
        [0,1007,2379,4132,6250], [0,687,1694,3066,4819],
        [0,432,1119,2126,3498], [0,0,432,1119,2126],
        [0,0,0,432,1119], [0,0,0,0,432],
        [0,0,0,0,0], [0,0,0,0,0],
    ],
    [ // i = 2
        [0,0,0,0,0], [0,620,0,0,0], [0,610,1230,0,0],
        [0,590,1200,1820,0], [0,555,1145,1755,2375],
        [0,503,1058,1648,2258], [0,435,938,1493,2083],
        [0,355,790,1293,1848], [0,270,625,1060,1563],
        [0,190,460,815,1250], [0,122,312,582,937],
        [0,70,192,382,652], [0,35,105,227,417],
        [0,15,50,120,242], [0,0,15,50,120],
    ],
    [ // i = 3
        [0,0,0,0,0], [0,125,0,0,0], [0,125,250,0,0],
        [0,125,250,375,0], [0,124,249,374,499],
        [0,121,245,370,495], [0,115,236,360,485],
        [0,105,220,341,465], [0,90,195,310,431],
        [0,72,162,267,382], [0,53,125,215,320],
        [0,35,88,160,250], [0,20,55,108,180],
        [0,10,30,65,118], [0,4,14,34,69],
    ],
    [ // i = 4
        [0,0,0,0,0], [0,25,0,0,0], [0,25,50,0,0],
        [0,25,50,75,0], [0,25,50,75,100],
        [0,25,50,75,100], [0,25,50,75,100],
        [0,25,50,75,100], [0,24,49,74,99],
        [0,22,46,71,96], [0,19,41,65,90],
        [0,15,34,56,80], [0,10,25,44,66],
        [0,6,16,31,50], [0,3,9,19,34],
    ],
    [ // i = 5
        [0,0,0,0,0], [0,5,0,0,0], [0,5,10,0,0],
        [0,5,10,15,0], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,5,10,15,20],
        [0,5,10,15,20], [0,4,9,14,19],
        [0,3,7,12,17], [0,2,5,9,14],
    ],
    [ // i = 6
        [0,0,0,0,0], [0,1,0,0,0], [0,1,2,0,0],
        [0,1,2,3,0], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
        [0,1,2,3,4], [0,1,2,3,4],
    ],
];

/// Key lookup tables for hierarchical shanten compression (Nyanten/Cryolite).
/// Loaded from binary files generated by scripts/convert_nyanten_tables.py.
static SHUPAI_KEYS: &[u8; 405_350] = include_bytes!("data/nyanten_shupai_keys.bin");
static ZIPAI_KEYS: &[u8; 43_130] = include_bytes!("data/nyanten_zipai_keys.bin");
static KEYS1: &[u8; 15_876] = include_bytes!("data/nyanten_keys1.bin"); // 126 * 126
static KEYS2: &[u8; 22_680] = include_bytes!("data/nyanten_keys2.bin"); // 180 * 126
static KEYS3: &[u8; 49_500] = include_bytes!("data/nyanten_keys3.bin"); // 180 * 55 * 5

/// Compute cumulative hash for a 9-tile number suit (man/pin/sou).
#[inline]
fn hash_shupai(tiles: &[u8]) -> usize {
    let mut n: usize = 0;
    let mut h: usize = 0;
    for (i, &c) in tiles.iter().enumerate() {
        let c = c as usize;
        n += c;
        h += SHUPAI_TABLE[i][n][c] as usize;
    }
    h
}

/// Compute cumulative hash for the 7 honor tiles (zipai).
#[inline]
fn hash_zipai(tiles: &[u8]) -> usize {
    let mut n: usize = 0;
    let mut h: usize = 0;
    for (i, &c) in tiles.iter().enumerate() {
        let c = c as usize;
        n += c;
        h += ZIPAI_TABLE[i][n][c] as usize;
    }
    h
}

/// Normal (standard) form shanten using Nyanten hierarchical key lookup. O(1).
fn calc_normal(tiles: &[u8; TILE_MAX], len_div3: u8) -> i8 {
    let m = len_div3 as usize;
    let k0_m = SHUPAI_KEYS[hash_shupai(&tiles[0..9])] as usize;
    let k0_p = SHUPAI_KEYS[hash_shupai(&tiles[9..18])] as usize;
    let k1 = KEYS1[k0_m * 126 + k0_p] as usize;
    let k0_s = SHUPAI_KEYS[hash_shupai(&tiles[18..27])] as usize;
    let k2 = KEYS2[k1 * 126 + k0_s] as usize;
    let k0_z = ZIPAI_KEYS[hash_zipai(&tiles[27..34])] as usize;
    let replacement = KEYS3[(k2 * 55 + k0_z) * 5 + m];
    (replacement as i8) - 1
}

/// Chiitoitsu (seven pairs) shanten.
fn calc_chitoi(tiles: &[u8; TILE_MAX]) -> i8 {
    let mut pairs = 0u8;
    let mut kinds = 0u8;
    for &c in tiles.iter() {
        if c > 0 {
            kinds += 1;
            if c >= 2 {
                pairs += 1;
            }
        }
    }
    let redunct = 7u8.saturating_sub(kinds) as i8;
    7 - pairs as i8 + redunct - 1
}

/// Kokushi (thirteen orphans) shanten.
fn calc_kokushi(tiles: &[u8; TILE_MAX]) -> i8 {
    const TERMINALS: [usize; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut kinds = 0i8;
    let mut has_pair = false;
    for &idx in &TERMINALS {
        if tiles[idx] > 0 {
            kinds += 1;
            if tiles[idx] >= 2 {
                has_pair = true;
            }
        }
    }
    14 - kinds - has_pair as i8 - 1
}

/// Calculate shanten considering all forms (normal, chitoi, kokushi).
/// This is the primary shanten function used throughout the SP calculator.
pub fn calc_shanten_from_counts(tehai: &[u8; TILE_MAX], tehai_len_div3: u8) -> i8 {
    let mut shanten = calc_normal(tehai, tehai_len_div3);
    if shanten <= 0 || tehai_len_div3 < 4 {
        return shanten;
    }
    shanten = shanten.min(calc_chitoi(tehai));
    if shanten > 0 {
        shanten.min(calc_kokushi(tehai))
    } else {
        shanten
    }
}

// ============================================================================
// Probability pre-computation tables
// ============================================================================

fn build_tsumo_prob_table<const MAX_TSUMO: usize>(n_left_tiles: usize) -> [[f32; MAX_TSUMO]; 4] {
    let mut table = [[0.; MAX_TSUMO]; 4];
    for (i, row) in table.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (i + 1) as f32 / (n_left_tiles - j) as f32;
        }
    }
    table
}

fn build_not_tsumo_prob_table<const MAX_TSUMO: usize>(
    n_left_tiles: usize,
) -> [[f32; MAX_TSUMO]; MAX_TILES_LEFT + 1] {
    let mut table = [[0.; MAX_TSUMO]; MAX_TILES_LEFT + 1];
    for (i, row) in table.iter_mut().enumerate().take(n_left_tiles + 1) {
        row[0] = 1.;
        for j in 0..(MAX_TSUMO - 1).min(n_left_tiles - i) {
            row[j + 1] = row[j] * (n_left_tiles - i - j) as f32 / (n_left_tiles - j) as f32;
        }
    }
    table
}

// ============================================================================
// Data types
// ============================================================================

/// Diagnostic statistics from one SP calculation
#[derive(Debug, Default, Clone)]
pub struct SPCalcStats {
    pub shanten: i8,
    pub tsumos_left: usize,
    pub can_discard: bool,
    pub num_candidates: usize,
    pub draw_cache_hits: u64,
    pub draw_cache_misses: u64,
    pub discard_cache_hits: u64,
    pub discard_cache_misses: u64,
    pub get_score_calls: u64,
    pub draw_slow_calls: u64,
    pub discard_slow_calls: u64,
    pub rc_values_created: u64,
    pub draw_cache_size: usize,
    pub discard_cache_size: usize,
}

/// Result for a single discard candidate
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Tile type (0-33) to discard, 255 = no discard (draw phase)
    pub tile: u8,
    /// Probability of reaching tenpai by turn i (cumulative)
    pub tenpai_probs: [f32; MAX_TSUMOS_LEFT],
    /// Probability of winning by turn i (cumulative)
    pub win_probs: [f32; MAX_TSUMOS_LEFT],
    /// Expected point value at turn i
    pub exp_values: [f32; MAX_TSUMOS_LEFT],
    /// Number of effective tiles (sum of counts)
    pub num_required_tiles: u8,
    /// Tiles that would improve shanten after this discard: (tile_type, count) pairs
    pub required_tiles: Vec<(u8, u8)>,
    /// Whether this discard worsens shanten
    pub shanten_down: bool,
}

/// State for the SP calculation
#[derive(Clone, PartialEq, Eq, Hash)]
struct State {
    tehai: [u8; TILE_MAX],
    tiles_in_wall: [u8; TILE_MAX],
    akas_in_hand: [bool; 3],
    akas_in_wall: [bool; 3],
    n_extra_tsumo: u8,
    /// Cached sum of tiles_in_wall (updated incrementally)
    sum_left: u8,
}

impl State {
    fn discard(&mut self, tile: u8) {
        self.tehai[tile as usize] -= 1;
        if is_aka_tile_type(tile) {
            let idx = aka_index(tile);
            if self.akas_in_hand[idx] && self.tehai[tile as usize] == 0 {
                self.akas_in_hand[idx] = false;
            }
        }
    }

    fn undo_discard(&mut self, tile: u8) {
        self.tehai[tile as usize] += 1;
        // Note: aka tracking is approximate for 5m/5p/5s in undo_discard.
        // The discard() method conservatively manages aka_in_hand flags.
    }

    fn deal(&mut self, tile: u8) {
        self.tiles_in_wall[tile as usize] -= 1;
        self.sum_left -= 1;
        if is_aka_tile_type(tile) {
            let idx = aka_index(tile);
            if self.akas_in_wall[idx] && self.tiles_in_wall[tile as usize] == 0 {
                self.akas_in_wall[idx] = false;
                self.akas_in_hand[idx] = true;
            }
        }
        self.tehai[tile as usize] += 1;
    }

    fn undo_deal(&mut self, tile: u8) {
        self.tehai[tile as usize] -= 1;
        if is_aka_tile_type(tile) {
            let idx = aka_index(tile);
            if self.akas_in_hand[idx] && self.tiles_in_wall[tile as usize] == 0 {
                self.akas_in_hand[idx] = false;
                self.akas_in_wall[idx] = true;
            }
        }
        self.tiles_in_wall[tile as usize] += 1;
        self.sum_left += 1;
    }

    fn get_discard_tiles(&self, shanten: i8, tehai_len_div3: u8) -> ([DiscardTile; 34], usize) {
        let mut result = [DiscardTile {
            tile: 0,
            shanten_diff: 0,
        }; 34];
        let mut len = 0;
        let mut tehai = self.tehai;
        for tid in 0..TILE_MAX {
            if tehai[tid] == 0 {
                continue;
            }
            tehai[tid] -= 1;
            let shanten_after = calc_shanten_from_counts(&tehai, tehai_len_div3);
            tehai[tid] += 1;
            result[len] = DiscardTile {
                tile: tid as u8,
                shanten_diff: shanten_after - shanten,
            };
            len += 1;
        }
        (result, len)
    }

    fn get_draw_tiles(&self, shanten: i8, tehai_len_div3: u8) -> ([DrawTile; 34], usize) {
        let mut result = [DrawTile {
            tile: 0,
            count: 0,
            shanten_diff: 0,
        }; 34];
        let mut len = 0;
        let mut tehai = self.tehai;
        for tid in 0..TILE_MAX {
            if self.tiles_in_wall[tid] == 0 {
                continue;
            }
            tehai[tid] += 1;
            let shanten_after = calc_shanten_from_counts(&tehai, tehai_len_div3);
            tehai[tid] -= 1;
            let shanten_diff = shanten_after - shanten;
            result[len] = DrawTile {
                tile: tid as u8,
                count: self.tiles_in_wall[tid],
                shanten_diff,
            };
            len += 1;
        }
        (result, len)
    }

    fn get_required_tiles(&self, tehai_len_div3: u8) -> ([(u8, u8); 34], usize) {
        let mut tehai = self.tehai;
        let shanten = calc_shanten_from_counts(&tehai, tehai_len_div3);
        let mut result = [(0u8, 0u8); 34];
        let mut len = 0;
        for tid in 0..TILE_MAX {
            if self.tiles_in_wall[tid] == 0 {
                continue;
            }
            tehai[tid] += 1;
            let shanten_after = calc_shanten_from_counts(&tehai, tehai_len_div3);
            tehai[tid] -= 1;
            if shanten_after < shanten {
                result[len] = (tid as u8, self.tiles_in_wall[tid]);
                len += 1;
            }
        }
        (result, len)
    }
}

/// Draw tile info
#[derive(Clone, Copy)]
struct DrawTile {
    tile: u8,
    count: u8,
    shanten_diff: i8,
}

/// Discard tile info
#[derive(Clone, Copy)]
struct DiscardTile {
    tile: u8,
    shanten_diff: i8,
}

/// Cached values for a state (const generic for compile-time optimization)
struct Values<const MAX_TSUMO: usize> {
    tenpai_probs: [f32; MAX_TSUMO],
    win_probs: [f32; MAX_TSUMO],
    exp_values: [f32; MAX_TSUMO],
}

impl<const MAX_TSUMO: usize> Default for Values<MAX_TSUMO> {
    fn default() -> Self {
        Self {
            tenpai_probs: [0.; MAX_TSUMO],
            win_probs: [0.; MAX_TSUMO],
            exp_values: [0.; MAX_TSUMO],
        }
    }
}

/// Scores or Values returned from draw step
enum ScoresOrValues<const MAX_TSUMO: usize> {
    /// Score for han+0, han+1, han+2, han+3 (pre-computed for turn-dependent bonuses)
    Scores([f32; 4]),
    Values(Rc<Values<MAX_TSUMO>>),
}

/// Type aliases for caches indexed by shanten level
type StateCache<const MAX_TSUMO: usize> =
    [AHashMap<State, Rc<Values<MAX_TSUMO>>>; SHANTEN_THRES as usize + 1];

// ============================================================================
// SP Calculator
// ============================================================================

/// The main SP calculator (immutable configuration)
pub struct SPCalculator {
    /// Number of melds (for hand length calculation)
    tehai_len_div3: u8,
    /// Wind info for yaku/score calculation
    bakaze: u8,
    jikaze: u8,
    /// Whether the hand is closed (menzen)
    is_menzen: bool,
    /// Whether the player is oya (dealer)
    is_oya: bool,
    /// Dora indicator tiles (34-format)
    dora_indicators: Vec<u8>,
    /// Number of dora in open melds
    num_doras_in_fuuro: u8,
    /// Number of aka dora in open melds
    num_aka_in_fuuro: u8,
    /// Open meld info for yaku calculation
    melds: Vec<Meld>,
    /// Whether to consider shanten-down (worsening shanten)
    pub calc_shanten_down: bool,
    /// Whether to assume riichi (= is_menzen for SP)
    prefer_riichi: bool,
    /// Whether to calculate double riichi bonus (turn 0)
    calc_double_riichi: bool,
    /// Whether to calculate haitei bonus (last turn)
    calc_haitei: bool,
}

/// Key for score cache: (tehai, akas_in_hand, win_tile)
#[derive(Clone, PartialEq, Eq, Hash)]
struct ScoreCacheKey {
    tehai: [u8; TILE_MAX],
    akas_in_hand: [bool; 3],
    win_tile: u8,
}

/// Mutable calculator state (with const generic MAX_TSUMO)
struct SPCalculatorState<'a, const MAX_TSUMO: usize> {
    sup: &'a SPCalculator,
    state: State,
    tsumo_prob_table: &'a [[f32; MAX_TSUMO]; 4],
    not_tsumo_prob_table: &'a [[f32; MAX_TSUMO]; MAX_TILES_LEFT + 1],
    discard_cache: StateCache<MAX_TSUMO>,
    draw_cache: StateCache<MAX_TSUMO>,
    score_cache: AHashMap<ScoreCacheKey, Option<[f32; 4]>>,
    // Profiling counters
    draw_cache_hits: u64,
    draw_cache_misses: u64,
    discard_cache_hits: u64,
    discard_cache_misses: u64,
    get_score_calls: u64,
    draw_slow_calls: u64,
    discard_slow_calls: u64,
    rc_values_created: u64,
}

impl SPCalculator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tehai_len_div3: u8,
        bakaze: u8,
        jikaze: u8,
        is_menzen: bool,
        is_oya: bool,
        dora_indicators: Vec<u8>,
        num_doras_in_fuuro: u8,
        num_aka_in_fuuro: u8,
        melds: Vec<Meld>,
    ) -> Self {
        Self {
            tehai_len_div3,
            bakaze,
            jikaze,
            is_menzen,
            is_oya,
            dora_indicators,
            num_doras_in_fuuro,
            num_aka_in_fuuro,
            melds,
            calc_shanten_down: false,
            prefer_riichi: is_menzen,
            calc_double_riichi: false,
            calc_haitei: true,
        }
    }

    /// Calculate SP table for the given hand state.
    pub fn calc(
        &self,
        tehai: &[u8; TILE_MAX],
        tiles_seen: &[u8; TILE_MAX],
        akas_in_hand: &[bool; 3],
        akas_seen: &[bool; 3],
        tsumos_left: usize,
        can_discard: bool,
    ) -> Vec<Candidate> {
        self.calc_with_stats(
            tehai,
            tiles_seen,
            akas_in_hand,
            akas_seen,
            tsumos_left,
            can_discard,
        )
        .0
    }

    /// Calculate SP table and return diagnostic statistics.
    pub fn calc_with_stats(
        &self,
        tehai: &[u8; TILE_MAX],
        tiles_seen: &[u8; TILE_MAX],
        akas_in_hand: &[bool; 3],
        akas_seen: &[bool; 3],
        tsumos_left: usize,
        can_discard: bool,
    ) -> (Vec<Candidate>, SPCalcStats) {
        let cur_shanten = calc_shanten_from_counts(tehai, self.tehai_len_div3);
        let mut stats = SPCalcStats {
            shanten: cur_shanten,
            tsumos_left: tsumos_left.min(MAX_TSUMOS_LEFT),
            can_discard,
            ..Default::default()
        };

        if cur_shanten > SHANTEN_THRES || tsumos_left == 0 {
            return (Vec::new(), stats);
        }

        let mut tiles_in_wall = [0u8; TILE_MAX];
        for i in 0..TILE_MAX {
            tiles_in_wall[i] = 4u8.saturating_sub(tiles_seen[i]);
        }
        let akas_in_wall = [!akas_seen[0], !akas_seen[1], !akas_seen[2]];

        let sum_left: u8 = tiles_in_wall.iter().sum();
        let state = State {
            tehai: *tehai,
            tiles_in_wall,
            akas_in_hand: *akas_in_hand,
            akas_in_wall,
            n_extra_tsumo: 0,
            sum_left,
        };

        let tsumos_left = tsumos_left.min(MAX_TSUMOS_LEFT);
        let n_left_tiles = sum_left as usize;

        if n_left_tiles == 0 {
            return (Vec::new(), stats);
        }

        // Use const generics via macro to enable compile-time optimization
        macro_rules! static_expand {
            ($($n:literal),*) => {
                match tsumos_left {
                    $($n => {
                        let tsumo_prob_table = build_tsumo_prob_table::<$n>(n_left_tiles);
                        let not_tsumo_prob_table = build_not_tsumo_prob_table::<$n>(n_left_tiles);
                        let mut calc_state = SPCalculatorState::<$n> {
                            sup: self,
                            state,
                            tsumo_prob_table: &tsumo_prob_table,
                            not_tsumo_prob_table: &not_tsumo_prob_table,
                            discard_cache: Default::default(),
                            draw_cache: Default::default(),
                            score_cache: AHashMap::new(),
                            draw_cache_hits: 0,
                            draw_cache_misses: 0,
                            discard_cache_hits: 0,
                            discard_cache_misses: 0,
                            get_score_calls: 0,
                            draw_slow_calls: 0,
                            discard_slow_calls: 0,
                            rc_values_created: 0,
                        };
                        let candidates = calc_state.calc(can_discard, cur_shanten);
                        stats.num_candidates = candidates.len();
                        stats.draw_cache_hits = calc_state.draw_cache_hits;
                        stats.draw_cache_misses = calc_state.draw_cache_misses;
                        stats.discard_cache_hits = calc_state.discard_cache_hits;
                        stats.discard_cache_misses = calc_state.discard_cache_misses;
                        stats.get_score_calls = calc_state.get_score_calls;
                        stats.draw_slow_calls = calc_state.draw_slow_calls;
                        stats.discard_slow_calls = calc_state.discard_slow_calls;
                        stats.rc_values_created = calc_state.rc_values_created;
                        stats.draw_cache_size = calc_state.draw_cache.iter().map(|m| m.len()).sum();
                        stats.discard_cache_size = calc_state.discard_cache.iter().map(|m| m.len()).sum();
                        (candidates, stats)
                    },)*
                    _ => (Vec::new(), stats),
                }
            }
        }

        static_expand!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
    }
}

// ============================================================================
// SPCalculatorState implementation (const generic MAX_TSUMO)
// ============================================================================

impl<const MAX_TSUMO: usize> SPCalculatorState<'_, MAX_TSUMO> {
    fn calc(&mut self, can_discard: bool, cur_shanten: i8) -> Vec<Candidate> {
        if can_discard {
            self.analyze_discard(cur_shanten)
        } else {
            self.analyze_draw(cur_shanten)
        }
    }

    fn analyze_discard(&mut self, shanten: i8) -> Vec<Candidate> {
        let (discard_tiles, discard_len) = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);
        let mut candidates = Vec::with_capacity(discard_len);

        for &DiscardTile { tile, shanten_diff } in &discard_tiles[..discard_len] {
            if shanten_diff == 0 {
                // Normal discard (maintains shanten)
                self.state.discard(tile);
                let (required, req_len) = self.state.get_required_tiles(self.sup.tehai_len_div3);
                let num_required: u8 = required[..req_len].iter().map(|r| r.1).sum();
                let required_vec = required[..req_len].to_vec();
                let values = self.draw(shanten);
                self.state.undo_discard(tile);

                let mut tenpai_probs = values.tenpai_probs;
                if shanten == 0 {
                    tenpai_probs = [1.0; MAX_TSUMO];
                }

                candidates.push(self.make_candidate(
                    tile,
                    &tenpai_probs,
                    &values.win_probs,
                    &values.exp_values,
                    num_required,
                    required_vec,
                    false,
                ));
            } else if self.sup.calc_shanten_down && shanten_diff == 1 && shanten < SHANTEN_THRES {
                // Shanten-down discard
                self.state.discard(tile);
                let (required, req_len) = self.state.get_required_tiles(self.sup.tehai_len_div3);
                let num_required: u8 = required[..req_len].iter().map(|r| r.1).sum();
                let required_vec = required[..req_len].to_vec();
                self.state.n_extra_tsumo += 1;
                let values = self.draw(shanten + 1);
                self.state.n_extra_tsumo -= 1;
                self.state.undo_discard(tile);

                candidates.push(self.make_candidate(
                    tile,
                    &values.tenpai_probs,
                    &values.win_probs,
                    &values.exp_values,
                    num_required,
                    required_vec,
                    true,
                ));
            }
        }

        // Sort by EV descending, then win prob, then num_required
        candidates.sort_by(|a, b| {
            let ev_cmp = b.exp_values[0]
                .partial_cmp(&a.exp_values[0])
                .unwrap_or(std::cmp::Ordering::Equal);
            if ev_cmp != std::cmp::Ordering::Equal {
                return ev_cmp;
            }
            let wp_cmp = b.win_probs[0]
                .partial_cmp(&a.win_probs[0])
                .unwrap_or(std::cmp::Ordering::Equal);
            if wp_cmp != std::cmp::Ordering::Equal {
                return wp_cmp;
            }
            b.num_required_tiles.cmp(&a.num_required_tiles)
        });

        candidates
    }

    fn analyze_draw(&mut self, shanten: i8) -> Vec<Candidate> {
        let (required, req_len) = self.state.get_required_tiles(self.sup.tehai_len_div3);
        let num_required: u8 = required[..req_len].iter().map(|r| r.1).sum();
        let required_vec = required[..req_len].to_vec();
        let values = self.draw(shanten);

        let mut tenpai_probs = values.tenpai_probs;
        if shanten == 0 {
            tenpai_probs = [1.0; MAX_TSUMO];
        }

        vec![self.make_candidate(
            255, // sentinel: no discard
            &tenpai_probs,
            &values.win_probs,
            &values.exp_values,
            num_required,
            required_vec,
            false,
        )]
    }

    /// Convert const-generic arrays to fixed-size Candidate
    #[allow(clippy::too_many_arguments)]
    fn make_candidate(
        &self,
        tile: u8,
        tenpai_probs: &[f32; MAX_TSUMO],
        win_probs: &[f32; MAX_TSUMO],
        exp_values: &[f32; MAX_TSUMO],
        num_required_tiles: u8,
        required_tiles: Vec<(u8, u8)>,
        shanten_down: bool,
    ) -> Candidate {
        let mut c = Candidate {
            tile,
            tenpai_probs: [0.0; MAX_TSUMOS_LEFT],
            win_probs: [0.0; MAX_TSUMOS_LEFT],
            exp_values: [0.0; MAX_TSUMOS_LEFT],
            num_required_tiles,
            required_tiles,
            shanten_down,
        };
        let n = MAX_TSUMO.min(MAX_TSUMOS_LEFT);
        c.tenpai_probs[..n].copy_from_slice(&tenpai_probs[..n]);
        c.win_probs[..n].copy_from_slice(&win_probs[..n]);
        c.exp_values[..n].copy_from_slice(&exp_values[..n]);
        // Clamp probabilities to [0, 1] and values to >= 0
        for i in 0..MAX_TSUMOS_LEFT {
            c.tenpai_probs[i] = c.tenpai_probs[i].clamp(0.0, 1.0);
            c.win_probs[i] = c.win_probs[i].clamp(0.0, 1.0);
            c.exp_values[i] = c.exp_values[i].max(0.0);
        }
        c
    }

    // ---- Draw step ----

    fn draw(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        if let Some(cached) = self.draw_cache[shanten as usize].get(&self.state) {
            self.draw_cache_hits += 1;
            return Rc::clone(cached);
        }
        self.draw_cache_misses += 1;
        self.draw_slow(shanten)
    }

    fn draw_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.draw_slow_calls += 1;
        let mut tenpai_probs = [0.; MAX_TSUMO];
        let mut win_probs = [0.; MAX_TSUMO];
        let mut exp_values = [0.; MAX_TSUMO];

        let (draw_tiles, draw_len) = self.state.get_draw_tiles(shanten, self.sup.tehai_len_div3);

        // Sum of required (useful) tiles
        let sum_required_tiles: u8 = draw_tiles[..draw_len]
            .iter()
            .filter(|d| d.shanten_diff == -1)
            .map(|d| d.count)
            .sum();
        let not_tsumo_probs = &self.not_tsumo_prob_table[sum_required_tiles as usize];

        let assume_riichi = self.sup.prefer_riichi;

        for dt in &draw_tiles[..draw_len] {
            if dt.shanten_diff != -1 || dt.count == 0 {
                continue;
            }

            self.state.deal(dt.tile);
            let scores_or_values = if shanten > 0 {
                ScoresOrValues::Values(self.discard(shanten - 1))
            } else {
                match self.get_score(dt.tile) {
                    Some(scores) => ScoresOrValues::Scores(scores),
                    None => {
                        self.state.undo_deal(dt.tile);
                        continue;
                    }
                }
            };
            self.state.undo_deal(dt.tile);

            let tsumo_probs = &self.tsumo_prob_table[dt.count as usize - 1];

            for i in 0..MAX_TSUMO {
                let m = not_tsumo_probs[i];
                if m == 0. {
                    break;
                }
                let inv_m = 1.0 / m;

                for j in i..MAX_TSUMO {
                    let n = not_tsumo_probs[j];
                    if n == 0. {
                        break;
                    }
                    let prob = tsumo_probs[j] * n * inv_m;

                    match &scores_or_values {
                        ScoresOrValues::Scores(scores) => {
                            let win_ippatsu = assume_riichi && j == i;
                            let win_double_riichi =
                                assume_riichi && self.sup.calc_double_riichi && i == 0;
                            let win_haitei = self.sup.calc_haitei && j == MAX_TSUMO - 1;
                            let han_plus = win_double_riichi as usize
                                + win_ippatsu as usize
                                + win_haitei as usize;
                            win_probs[i] += prob;
                            exp_values[i] += prob * scores[han_plus.min(3)];
                        }
                        ScoresOrValues::Values(next_values) => {
                            if shanten == 1 {
                                tenpai_probs[i] += prob;
                            }
                            if j < MAX_TSUMO - 1 {
                                if shanten > 1 {
                                    tenpai_probs[i] += prob * next_values.tenpai_probs[j + 1];
                                }
                                win_probs[i] += prob * next_values.win_probs[j + 1];
                                exp_values[i] += prob * next_values.exp_values[j + 1];
                            }
                        }
                    }
                }
            }
        }

        self.rc_values_created += 1;
        let values = Rc::new(Values {
            tenpai_probs,
            win_probs,
            exp_values,
        });
        self.draw_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));
        values
    }

    // ---- Discard step (pick optimal discard) ----

    fn discard(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        if let Some(cached) = self.discard_cache[shanten as usize].get(&self.state) {
            self.discard_cache_hits += 1;
            return Rc::clone(cached);
        }
        self.discard_cache_misses += 1;
        self.discard_slow(shanten)
    }

    fn discard_slow(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        self.discard_slow_calls += 1;
        let (discard_tiles, discard_len) = self
            .state
            .get_discard_tiles(shanten, self.sup.tehai_len_div3);

        let mut max_tenpai_probs = [f32::MIN; MAX_TSUMO];
        let mut max_win_probs = [f32::MIN; MAX_TSUMO];
        let mut max_exp_values = [f32::MIN; MAX_TSUMO];
        let mut max_values_i32 = [i32::MIN; MAX_TSUMO];
        let mut max_tiles = [0u8; MAX_TSUMO]; // Phase C: tile tracking for tie-breaking
        let mut has_any = false;

        for &DiscardTile { tile, shanten_diff } in &discard_tiles[..discard_len] {
            if shanten_diff == 0 {
                self.state.discard(tile);
                let values = self.draw(shanten);
                self.state.undo_discard(tile);

                for i in 0..MAX_TSUMO {
                    let value = values.exp_values[i] as i32;
                    if value > max_values_i32[i]
                        || (value == max_values_i32[i] && tile > max_tiles[i])
                    {
                        max_tenpai_probs[i] = values.tenpai_probs[i];
                        max_win_probs[i] = values.win_probs[i];
                        max_exp_values[i] = values.exp_values[i];
                        max_values_i32[i] = value;
                        max_tiles[i] = tile;
                    }
                }
                has_any = true;
            } else if self.sup.calc_shanten_down
                && self.state.n_extra_tsumo == 0
                && shanten_diff == 1
                && shanten < SHANTEN_THRES
            {
                self.state.discard(tile);
                self.state.n_extra_tsumo += 1;
                let values = self.draw(shanten + 1);
                self.state.n_extra_tsumo -= 1;
                self.state.undo_discard(tile);

                for i in 0..MAX_TSUMO {
                    let value = values.exp_values[i] as i32;
                    if value > max_values_i32[i]
                        || (value == max_values_i32[i] && tile > max_tiles[i])
                    {
                        max_tenpai_probs[i] = values.tenpai_probs[i];
                        max_win_probs[i] = values.win_probs[i];
                        max_exp_values[i] = values.exp_values[i];
                        max_values_i32[i] = value;
                        max_tiles[i] = tile;
                    }
                }
                has_any = true;
            }
        }

        self.rc_values_created += 1;
        let values = if has_any {
            Rc::new(Values {
                tenpai_probs: max_tenpai_probs,
                win_probs: max_win_probs,
                exp_values: max_exp_values,
            })
        } else {
            Rc::new(Values::default())
        };
        self.discard_cache[shanten as usize].insert(self.state.clone(), Rc::clone(&values));
        values
    }

    // ---- Score calculation ----

    /// Returns scores for han+0, han+1, han+2, han+3 (for turn-dependent bonuses).
    /// Returns None if the hand has no yaku.
    fn get_score(&mut self, win_tile: u8) -> Option<[f32; 4]> {
        self.get_score_calls += 1;

        let cache_key = ScoreCacheKey {
            tehai: self.state.tehai,
            akas_in_hand: self.state.akas_in_hand,
            win_tile,
        };
        if let Some(cached) = self.score_cache.get(&cache_key) {
            return *cached;
        }

        let result = self.get_score_slow(win_tile);
        self.score_cache.insert(cache_key, result);
        result
    }

    fn get_score_slow(&self, win_tile: u8) -> Option<[f32; 4]> {
        let mut hand = Hand::default();
        for i in 0..TILE_MAX {
            for _ in 0..self.state.tehai[i] {
                hand.add(i as u8);
            }
        }

        let mut dora_count: u8 = self.sup.num_doras_in_fuuro;
        for &indicator in &self.sup.dora_indicators {
            let dora_tile = next_dora_tile(indicator);
            dora_count += self.state.tehai[dora_tile as usize];
        }

        let mut aka_count: u8 = self.sup.num_aka_in_fuuro;
        for (i, &has_aka) in self.state.akas_in_hand.iter().enumerate() {
            if has_aka {
                let aka_tile_type = match i {
                    0 => 4,
                    1 => 13,
                    2 => 22,
                    _ => continue,
                };
                if self.state.tehai[aka_tile_type] > 0 {
                    aka_count += 1;
                }
            }
        }

        let ctx = YakuContext {
            is_menzen: self.sup.is_menzen,
            is_reach: self.sup.is_menzen,
            is_ippatsu: false,
            is_tsumo: true,
            is_haitei: false,
            is_houtei: false,
            is_rinshan: false,
            is_chankan: false,
            is_tsumo_first_turn: false,
            is_daburu_reach: false,
            dora_count,
            aka_dora: aka_count,
            ura_dora_count: 0,
            bakaze: self.sup.bakaze,
            jikaze: self.sup.jikaze,
        };

        let yaku_result = yaku::calculate_yaku(&hand, &self.sup.melds, &ctx, win_tile);
        if yaku_result.han == 0 {
            return None;
        }

        // Yakuman: all 4 entries are the same score
        if yaku_result.yakuman_count > 0 {
            let s =
                score::calculate_score(yaku_result.han, yaku_result.fu, self.sup.is_oya, true, 0);
            return Some([s.total as f32; 4]);
        }

        // Normal hand: compute scores for han+0, han+1, han+2, han+3
        let mut scores = [0.0f32; 4];
        for (i, score_val) in scores.iter_mut().enumerate() {
            let s = score::calculate_score(
                yaku_result.han + i as u8,
                yaku_result.fu,
                self.sup.is_oya,
                true,
                0,
            );
            *score_val = s.total as f32;
        }
        Some(scores)
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Check if a tile type can have an aka (red five)
fn is_aka_tile_type(tile: u8) -> bool {
    tile == 4 || tile == 13 || tile == 22
}

/// Get the aka index (0=5m, 1=5p, 2=5s) for a tile type
fn aka_index(tile: u8) -> usize {
    match tile {
        4 => 0,
        13 => 1,
        22 => 2,
        _ => 0,
    }
}

/// Get the next tile in dora sequence (dora indicator -> actual dora)
pub fn next_dora_tile(indicator: u8) -> u8 {
    if indicator < 27 {
        let suit = indicator / 9;
        let num = indicator % 9;
        suit * 9 + (num + 1) % 9
    } else if indicator < 31 {
        27 + (indicator - 27 + 1) % 4
    } else {
        31 + (indicator - 31 + 1) % 3
    }
}

// ============================================================================
// Feature encoding
// ============================================================================

/// Encode SP results into feature channels.
///
/// Returns a flat f32 array of shape (123, 34) = 4182 values:
///   - 2 channels: max EV (two scales: /100000, /30000)
///   - 68 channels: required tiles (34 normal + 34 shanten-down)
///   - 2 channels: best discard indicators
///   - 51 channels: SP probability table (3 metrics × 17 turns)
///
/// Total: 2 + 68 + 2 + 51 = 123 channels × 34 tile positions
pub fn encode_sp_features(candidates: &[Candidate], can_discard: bool) -> Vec<f32> {
    let num_channels = 123;
    let mut arr = vec![0.0f32; num_channels * 34];

    if candidates.is_empty() {
        return arr;
    }

    // Early exit: if first candidate has zero probability, skip encoding
    // (matches Mortal's check for shanten >= 4 or all-zero probs)
    if candidates[0].tenpai_probs[0] <= 0.0 {
        return arr;
    }

    let set = |arr: &mut Vec<f32>, ch: usize, tile: usize, val: f32| {
        arr[ch * 34 + tile] = val;
    };

    // 1. Max EV (2 channels, broadcast)
    let max_ev = candidates
        .iter()
        .map(|c| c.exp_values[0])
        .fold(0.0f32, f32::max);

    let ev_scale = if max_ev >= 1.0 { 1.0 / max_ev } else { 0.0 };

    for t in 0..34 {
        set(&mut arr, 0, t, (max_ev / 100000.0).min(1.0));
        set(&mut arr, 1, t, (max_ev / 30000.0).min(1.0));
    }

    // 2. Required tiles (68 channels: 34 normal + 34 shanten-down)
    //    For each discard candidate, encode which tiles would improve shanten
    //    at the required tile's position. Matches Mortal v4 encoding.
    if can_discard {
        for cand in candidates {
            if cand.tile >= 34 {
                continue;
            }
            let discard_tid = cand.tile as usize;
            let offset = if cand.shanten_down { 34 } else { 0 };
            for &(req_tile, _count) in &cand.required_tiles {
                let required_tid = req_tile as usize;
                set(&mut arr, 2 + offset + discard_tid, required_tid, 1.0);
            }
        }
    }

    // 3. Best discard / required tiles indicators (2 channels: 70, 71)
    if can_discard {
        // Mortal: max_by(CandidateColumn::NotShantenDown) = prefer !shanten_down,
        // then most required_tiles. Since calc_shanten_down=false, all candidates
        // have shanten_down=false, so this selects the one with most required tiles.
        if let Some(best) = candidates
            .iter()
            .filter(|c| !c.shanten_down && c.tile < 34)
            .max_by_key(|c| c.num_required_tiles)
        {
            set(&mut arr, 70, best.tile as usize, 1.0);
        } else if let Some(best) = candidates
            .iter()
            .filter(|c| c.tile < 34)
            .max_by_key(|c| c.num_required_tiles)
        {
            set(&mut arr, 70, best.tile as usize, 1.0);
        }
        // Channel 71: unused (Mortal skips with idx += 2, only writes ch 70)
    } else {
        // When can_discard=false, encode required tiles of best candidate
        // Mortal writes to channel 71 (skips 2*34 + 1 = 69, then writes 1)
        if let Some(cand) = candidates.first() {
            for &(req_tile, _count) in &cand.required_tiles {
                set(&mut arr, 71, req_tile as usize, 1.0);
            }
        }
    }

    // 4. SP probability table (51 channels: 3 × 17 turns)
    //    Mortal uses take_while(|&&p| p > 0.) to stop at first zero probability.
    //    Since arrays are zero-initialized, writing 0.0 is a no-op, so the result
    //    is identical whether we use take_while or loop all 17 turns.
    if can_discard {
        for cand in candidates {
            if cand.tile >= 34 {
                continue;
            }
            let tid = cand.tile as usize;

            for turn in 0..MAX_TSUMOS_LEFT {
                if cand.tenpai_probs[turn] <= 0.0 {
                    break;
                }
                set(&mut arr, 72 + turn, tid, cand.tenpai_probs[turn]);
                set(&mut arr, 72 + 17 + turn, tid, cand.win_probs[turn]);
                set(
                    &mut arr,
                    72 + 34 + turn,
                    tid,
                    (cand.exp_values[turn] * ev_scale).min(1.0),
                );
            }
        }
    } else if let Some(cand) = candidates.first() {
        for turn in 0..MAX_TSUMOS_LEFT {
            if cand.tenpai_probs[turn] <= 0.0 {
                break;
            }
            for t in 0..34 {
                set(&mut arr, 72 + turn, t, cand.tenpai_probs[turn]);
                set(&mut arr, 72 + 17 + turn, t, cand.win_probs[turn]);
                set(
                    &mut arr,
                    72 + 34 + turn,
                    t,
                    (cand.exp_values[turn] * ev_scale).min(1.0),
                );
            }
        }
    }

    arr
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_shanten_tenpai() {
        // 1m2m3m 4m5m6m 7m8m9m 1p2p3p 4p -> tenpai
        let mut tehai = [0u8; 34];
        for t in tehai.iter_mut().take(9) {
            *t = 1;
        }
        tehai[9] = 1;
        tehai[10] = 1;
        tehai[11] = 1;
        tehai[12] = 1;
        assert_eq!(calc_shanten_from_counts(&tehai, 4), 0);
    }

    #[test]
    fn test_calc_shanten_complete() {
        // 1m2m3m 4m5m6m 7m8m9m 1p1p2p3p (13 tiles, tenpai = 0)
        let mut tehai = [0u8; 34];
        for t in tehai.iter_mut().take(9) {
            *t = 1;
        }
        tehai[9] = 2; // 1p pair
        tehai[10] = 1; // 2p
        tehai[11] = 1; // 3p
                       // 13 tiles (3*4+1), tenpai for 1p/4p
        assert_eq!(calc_shanten_from_counts(&tehai, 4), 0);

        // 55m = pair, len_div3=0 => complete hand = -1
        let mut tehai2 = [0u8; 34];
        tehai2[4] = 2; // 55m
        assert_eq!(calc_shanten_from_counts(&tehai2, 0), -1);
    }

    #[test]
    fn test_calc_shanten_mortal_cases() {
        // From Mortal's test suite
        fn hand(desc: &str) -> [u8; 34] {
            // Simple parser: "1111m 333p 222s 444z"
            let mut tiles = [0u8; 34];
            let mut nums = Vec::new();
            for c in desc.chars() {
                match c {
                    '0'..='9' => nums.push(c.to_digit(10).unwrap() as usize),
                    'm' => {
                        for &n in &nums {
                            tiles[n - 1] += 1;
                        }
                        nums.clear();
                    }
                    'p' => {
                        for &n in &nums {
                            tiles[9 + n - 1] += 1;
                        }
                        nums.clear();
                    }
                    's' => {
                        for &n in &nums {
                            tiles[18 + n - 1] += 1;
                        }
                        nums.clear();
                    }
                    'z' => {
                        for &n in &nums {
                            tiles[27 + n - 1] += 1;
                        }
                        nums.clear();
                    }
                    ' ' => {}
                    _ => {}
                }
            }
            tiles
        }

        // calc_3n_plus_1 tests
        assert_eq!(
            calc_shanten_from_counts(&hand("1111m 333p 222s 444z"), 4),
            1
        );
        assert_eq!(
            calc_shanten_from_counts(&hand("147m 258p 369s 1234z"), 4),
            6
        );
        assert_eq!(calc_shanten_from_counts(&hand("468m 33346p 7s"), 3), 2);
        assert_eq!(calc_shanten_from_counts(&hand("147m 258p 3s"), 2), 4);
        assert_eq!(calc_shanten_from_counts(&hand("4455s"), 1), 0);
        assert_eq!(calc_shanten_from_counts(&hand("7z"), 0), 0);
        assert_eq!(
            calc_shanten_from_counts(&hand("15559m 19p 19s 1234z"), 4),
            3
        );
        assert_eq!(
            calc_shanten_from_counts(&hand("9999m 6677p 88s 335z"), 4),
            2
        );
        assert_eq!(
            calc_shanten_from_counts(&hand("19m 19p 159s 123456z"), 4),
            1
        );

        // calc_3n_plus_2 tests
        assert_eq!(
            calc_shanten_from_counts(&hand("2344456m 14p 127s 2z 7p"), 4),
            3
        );
        assert_eq!(
            calc_shanten_from_counts(&hand("2344456m 14p 127s 2z 5p"), 4),
            2
        );
        assert_eq!(calc_shanten_from_counts(&hand("344455667p 1139s 9m"), 4), 2);
        assert_eq!(calc_shanten_from_counts(&hand("344455667p 1139s 9p"), 4), 1);
        assert_eq!(
            calc_shanten_from_counts(&hand("122334m 678p 37s 22z 5s"), 4),
            0
        );
        assert_eq!(
            calc_shanten_from_counts(&hand("122334m 678p 12s 22z 4s"), 4),
            0
        );
        assert_eq!(
            calc_shanten_from_counts(&hand("12223456m 78889p 2m"), 4),
            -1
        );
        assert_eq!(calc_shanten_from_counts(&hand("34778p"), 1), 0);
        assert_eq!(calc_shanten_from_counts(&hand("34s"), 0), 0);
        assert_eq!(calc_shanten_from_counts(&hand("55m"), 0), -1);
    }

    #[test]
    fn test_next_dora_tile() {
        assert_eq!(next_dora_tile(0), 1);
        assert_eq!(next_dora_tile(8), 0);
        assert_eq!(next_dora_tile(27), 28);
        assert_eq!(next_dora_tile(30), 27);
        assert_eq!(next_dora_tile(33), 31);
    }

    #[test]
    fn test_encode_sp_empty() {
        let arr = encode_sp_features(&[], true);
        assert_eq!(arr.len(), 123 * 34);
        assert!(arr.iter().all(|&v| v == 0.0));
    }
}
