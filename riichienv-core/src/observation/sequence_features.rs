//! Sequence feature encoding for transformer models.
//!
//! Produces sparse tokens, numeric features, progression (action history),
//! and candidate (legal action) features suitable for embedding-based
//! transformer architectures.
//!
//! Based on the Kanachan v3 encoding (subset — Room and Grade removed):
//! <https://github.com/Cryolite/kanachan/wiki/%5Bv3%5DNotes-on-Training-Data>

use crate::action::{Action, ActionType};
use crate::parser::mjai_to_tid;

use super::Observation;

// ── Constants ────────────────────────────────────────────────────────────────
// These constants are `pub` for Python-side consumption (see riichienv-ml).
#[allow(dead_code)]
pub const SPARSE_VOCAB_SIZE: usize = 442;
#[allow(dead_code)]
pub const SPARSE_PAD: u16 = 441;
pub const MAX_SPARSE_LEN: usize = 25;

/// Progression tuple dimensions: (actor, type, moqie, liqi, from)
#[allow(dead_code)]
pub const PROG_DIMS: [u16; 5] = [5, 277, 3, 3, 5];
pub const MAX_PROG_LEN: usize = 512;
#[allow(dead_code)]
pub const PROG_PAD: [u16; 5] = [4, 276, 2, 2, 4];

/// Candidate tuple dimensions: (type, moqie, liqi, from)
#[allow(dead_code)]
pub const CAND_DIMS: [u16; 4] = [280, 3, 3, 4];
#[allow(dead_code)]
pub const MAX_CAND_LEN: usize = 64;
#[allow(dead_code)]
pub const CAND_PAD: [u16; 4] = [279, 2, 2, 3];

pub const NUM_NUMERIC: usize = 12;

// ── Tile conversions ─────────────────────────────────────────────────────────

/// Convert a 136-tile ID to kan37 (37 tiles, red fives distinct).
///
/// Layout: 0=red5m, 1-9=1m-9m, 10=red5p, 11-19=1p-9p,
///         20=red5s, 21-29=1s-9s, 30-36=E/S/W/N/P/F/C
pub fn tile_id_to_kan37(tile_id: u32) -> u8 {
    // Red fives
    if tile_id == 16 {
        return 0; // red 5m
    }
    if tile_id == 52 {
        return 10; // red 5p
    }
    if tile_id == 88 {
        return 20; // red 5s
    }

    let tile_type = (tile_id / 4) as u8; // 0-33
    tile_type_to_kan37(tile_type)
}

/// Convert a tile type (0-33, no red distinction) to kan37.
/// For 5m/5p/5s this returns the non-red version.
fn tile_type_to_kan37(tile_type: u8) -> u8 {
    match tile_type {
        0..=8 => tile_type + 1,   // 1m-9m → 1-9
        9..=17 => tile_type + 2,  // 1p-9p → 11-19
        18..=26 => tile_type + 3, // 1s-9s → 21-29
        27..=33 => tile_type + 3, // honors → 30-36
        _ => 0,
    }
}

/// Convert a tile type (0-33) to kan34 (identity).
#[inline]
#[allow(dead_code)]
pub fn tile_type_to_kan34(tile_type: u8) -> u8 {
    tile_type
}

/// Convert MJAI tile string to kan37.
fn mjai_tile_to_kan37(mjai: &str) -> Option<u8> {
    let tid = mjai_to_tid(mjai)?;
    Some(tile_id_to_kan37(tid as u32))
}

// ── Meld pattern encoding ────────────────────────────────────────────────────

/// Encode a chi (sequence) call into 0-89.
///
/// 90 patterns = 3 suits × 30 per suit.
/// Per suit: 7 base sequences × positions + red-five variants.
///
/// For each suit the 30 slots are indexed by the lowest tile type in the
/// sequence (relative to suit start, 0-6) and a sub-index that encodes
/// which tile was called and whether a red five is involved.
///
/// `consumed` = sorted tile IDs of the 2 tiles from hand.
/// `called_tile` = tile ID claimed from discard.
pub fn encode_chi(consumed: &[u8], called_tile: u8) -> u16 {
    let mut all_tiles = vec![called_tile];
    all_tiles.extend_from_slice(consumed);
    all_tiles.sort();

    // Determine suit from the first tile
    let first_type = all_tiles[0] / 4;
    let suit = first_type / 9; // 0=m, 1=p, 2=s
    let suit_base = suit * 9;
    let seq_start = first_type - suit_base; // 0-6

    // Which position was the called tile?
    let called_type = called_tile / 4;
    let call_pos = called_type - suit_base - seq_start; // 0, 1, or 2

    // Check if any tile is a red five
    let has_red = all_tiles.iter().any(|&t| t == 16 || t == 52 || t == 88);
    let five_in_seq = (suit_base + 4) >= (suit_base + seq_start)
        && (suit_base + 4) <= (suit_base + seq_start + 2);
    let involves_five = five_in_seq && (seq_start..=seq_start + 2).contains(&4);

    // Base index per sequence start (0-6), 3 call positions + red variants
    // Sequences not containing 5: 3 patterns each
    // Sequences containing 5: 3 normal + red variants for each call position = up to 6
    let suit_offset = (suit as u16) * 30;

    // Compute per-suit offset
    let mut offset: u16 = 0;
    for s in 0..seq_start {
        let seq_has_five = (s..=s + 2).contains(&4);
        offset += if seq_has_five { 6 } else { 3 };
    }

    let sub_idx = if involves_five && has_red {
        3 + call_pos // red variant
    } else {
        call_pos
    };

    suit_offset + offset + sub_idx as u16
}

/// Encode a pon (triplet) call into 0-39.
///
/// 40 patterns = 3 suits × 11 per suit + 7 honors.
/// Per suit: 8 non-five tiles (3 each = too many; actually 1 pattern each)
///   + 3 five-variants (normal, red-in-hand, red-called) = 11.
///     Honors: 7 × 1 = 7.
///
/// `consumed` = sorted tile IDs of the 2 tiles from hand.
/// `called_tile` = tile ID claimed from discard.
pub fn encode_pon(consumed: &[u8], called_tile: u8) -> u16 {
    let called_type = called_tile / 4;
    let suit = called_type / 9;

    if suit == 3 {
        // Honor: simple index 0-6
        let honor_idx = called_type - 27;
        return 33 + honor_idx as u16; // 33..39
    }

    let suit_base = suit * 9;
    let rank = called_type - suit_base; // 0-8

    let suit_offset = (suit as u16) * 11;

    if rank == 4 {
        // Five tile: 3 variants
        let called_is_red = called_tile == 16 || called_tile == 52 || called_tile == 88;
        let consumed_has_red = consumed.iter().any(|&t| t == 16 || t == 52 || t == 88);

        let sub_idx = if called_is_red {
            2 // red five was called
        } else if consumed_has_red {
            1 // red five in hand
        } else {
            0 // no red five
        };
        // ranks 0-3 take 4 slots, then five variants at offset 4
        suit_offset + 4 + sub_idx
    } else {
        // Non-five tile: rank maps directly, but skip the 3 five-slots
        // ranks 0-3 → 0-3, rank 5 → 7, rank 6 → 8, rank 7 → 9, rank 8 → 10
        let idx = if rank < 4 {
            rank as u16
        } else {
            // ranks 5-8 map to indices 7-10 (after 0-3 + 3 five-variants)
            (rank as u16) + 2
        };
        suit_offset + idx
    }
}

/// Relative seat: (target - actor + n_players - 1) % n_players
/// For 4P: 0=shimocha(right), 1=toimen(across), 2=kamicha(left)
fn relative_from(actor: u8, target: u8) -> u8 {
    ((target as i8 - actor as i8 + 3) % 4) as u8
}

/// Parse "consumed" array from MJAI event JSON → Vec<u8> of tile IDs.
fn parse_consumed_tids_from_value(v: &serde_json::Value) -> Vec<u8> {
    let mut tids = Vec::new();
    if let Some(arr) = v["consumed"].as_array() {
        for item in arr {
            if let Some(s) = item.as_str()
                && let Some(tid) = mjai_to_tid(s)
            {
                tids.push(tid);
            }
        }
    }
    tids
}

/// Process a single MJAI event for the progression cache.
///
/// Called incrementally from GameState::_push_mjai_event to avoid
/// O(N²) JSON re-parsing in encode_seq_progression.
///
/// `pending_reach_actor` is mutable state tracking who declared reach.
pub fn process_single_event_progression(
    event: &serde_json::Value,
    pending_reach_actor: &mut Option<u8>,
) -> Option<[u16; 5]> {
    let event_type = event["type"].as_str()?;

    match event_type {
        "start_kyoku" => Some([4, 0, 2, 2, 4]),
        "reach" => {
            if let Some(actor) = event["actor"].as_u64() {
                *pending_reach_actor = Some(actor as u8);
            }
            None
        }
        "dahai" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let k37 = mjai_tile_to_kan37(pai)?;
            let type_idx = 1 + k37 as u16;
            let moqie = if event["tsumogiri"].as_bool().unwrap_or(false) {
                1
            } else {
                0
            };
            let liqi = if *pending_reach_actor == Some(actor) {
                *pending_reach_actor = None;
                1
            } else {
                0
            };
            Some([actor as u16, type_idx, moqie, liqi, 4])
        }
        "chi" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let target = event["target"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let called_tid = mjai_to_tid(pai)?;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.len() < 2 {
                return None;
            }
            let type_idx = 38 + encode_chi(&consumed, called_tid);
            let rel = relative_from(actor, target);
            Some([actor as u16, type_idx, 2, 2, rel as u16])
        }
        "pon" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let target = event["target"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let called_tid = mjai_to_tid(pai)?;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.len() < 2 {
                return None;
            }
            let type_idx = 128 + encode_pon(&consumed, called_tid);
            let rel = relative_from(actor, target);
            Some([actor as u16, type_idx, 2, 2, rel as u16])
        }
        "daiminkan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let target = event["target"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let k37 = mjai_tile_to_kan37(pai)?;
            let type_idx = 168 + k37 as u16;
            let rel = relative_from(actor, target);
            Some([actor as u16, type_idx, 2, 2, rel as u16])
        }
        "ankan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let consumed = parse_consumed_tids_from_value(event);
            if consumed.is_empty() {
                return None;
            }
            let tile34 = consumed[0] / 4;
            let type_idx = 205 + tile34 as u16;
            Some([actor as u16, type_idx, 2, 2, 4])
        }
        "kakan" => {
            let actor = event["actor"].as_u64().unwrap_or(0) as u8;
            let pai = event["pai"].as_str().unwrap_or("?");
            if pai == "?" {
                return None;
            }
            let k37 = mjai_tile_to_kan37(pai)?;
            let type_idx = 239 + k37 as u16;
            Some([actor as u16, type_idx, 2, 2, 4])
        }
        _ => None,
    }
}

// ── Sparse features ──────────────────────────────────────────────────────────

impl Observation {
    /// Encode sparse features: variable-length u16 indices (max 25).
    ///
    /// Offsets:
    /// - 0-1: game style (0=tonpuusen, 1=hanchan)
    /// - 2-5: seat (player_id)
    /// - 6-8: chang / round wind (E/S/W)
    /// - 9-12: ju / dealer round (0-3)
    /// - 13-82: tiles remaining (0-69)
    /// - 83-267: dora indicators (5 slots × 37 tiles)
    /// - 268-403: hand tile instances (tile_id 0-135 → offset 268 + instance index)
    /// - 404-440: drawn tile (kan37)
    /// - 441: padding
    pub fn encode_seq_sparse(&self, game_style: u8) -> Vec<u16> {
        let mut tokens: Vec<u16> = Vec::with_capacity(MAX_SPARSE_LEN);

        // 1. Game style (offset 0-1)
        tokens.push(game_style.min(1) as u16);

        // 2. Seat (offset 2-5)
        tokens.push(2 + self.player_id.min(3) as u16);

        // 3. Chang / round wind (offset 6-8)
        tokens.push(6 + self.round_wind.min(2) as u16);

        // 4. Ju / dealer (offset 9-12)
        tokens.push(9 + self.oya.min(3) as u16);

        // 5. Tiles remaining (offset 13-82)
        let tiles_remaining = self.count_tiles_remaining();
        tokens.push(13 + tiles_remaining.min(69));

        // 6. Dora indicators (offset 83-267, 5 slots × 37)
        for (i, &dora_tid) in self.dora_indicators.iter().enumerate() {
            if i >= 5 {
                break;
            }
            let k37 = tile_id_to_kan37(dora_tid);
            tokens.push(83 + (i as u16) * 37 + k37 as u16);
        }

        // 7. Hand tiles (offset 268-403)
        // Use tile_id as 136-space instance index, but mapped to
        // a compact hand representation. We encode each tile in hand
        // as offset 268 + tile_id (0-135). Max ~14 tiles.
        let my_hand = &self.hands[self.player_id as usize];
        for &tid in my_hand {
            let tid = tid as u16;
            if tid < 136 {
                tokens.push(268 + tid);
            }
        }

        // 8. Drawn tile (offset 404-440)
        if let Some(drawn) = self.get_drawn_tile() {
            let k37 = tile_id_to_kan37(drawn as u32);
            tokens.push(404 + k37 as u16);
        }

        tokens
    }

    /// Count approximate tiles remaining in the wall.
    fn count_tiles_remaining(&self) -> u16 {
        let n = 4; // 4 players
        let total_tiles: u32 = 136; // 4P

        let mut used: u32 = 0;
        // Hands
        for i in 0..n {
            used += self.hands[i].len() as u32;
        }
        // Discards
        for i in 0..n {
            used += self.discards[i].len() as u32;
        }
        // Melds (only count tiles not already in hand)
        for i in 0..n {
            for meld in &self.melds[i] {
                used += meld.tiles.len() as u32;
            }
        }
        // Dora indicators
        used += self.dora_indicators.len() as u32;

        // Dead wall has 14 tiles (minus dora indicators already counted)
        // Initial deal: 13*4 = 52 tiles. Remaining = total - 14(dead) - used
        let wall_size = total_tiles.saturating_sub(14 + used);
        wall_size as u16
    }

    /// Get the last drawn tile for the current player (from tsumo event).
    fn get_drawn_tile(&self) -> Option<u8> {
        // Walk events backwards to find last tsumo for this player
        for event_str in self.events.iter().rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) {
                let event_type = v["type"].as_str().unwrap_or("");
                if event_type == "tsumo" {
                    let actor = v["actor"].as_u64();
                    if actor == Some(self.player_id as u64)
                        && let Some(pai) = v["pai"].as_str()
                        && pai != "?"
                    {
                        return mjai_to_tid(pai);
                    }
                }
                // Stop at decision-relevant events
                if event_type == "dahai"
                    || event_type == "chi"
                    || event_type == "pon"
                    || event_type == "daiminkan"
                {
                    break;
                }
            }
        }
        None
    }

    // ── Numeric features ─────────────────────────────────────────────────

    /// Encode numeric features: 12 floats.
    ///
    /// [0] honba (current)
    /// [1] riichi deposits (current)
    /// [2-5] scores (self, right, opposite, left) relative to player_id
    /// [6] honba (round start)
    /// [7] riichi deposits (round start)
    /// [8-11] scores at round start (self-relative)
    pub fn encode_seq_numeric(&self) -> [f32; NUM_NUMERIC] {
        let mut out = [0.0f32; NUM_NUMERIC];
        let pid = self.player_id as usize;

        // Current state
        out[0] = self.honba as f32;
        out[1] = self.riichi_sticks as f32;

        // Scores (self-relative rotation)
        for i in 0..4 {
            let seat = (pid + i) % 4;
            out[2 + i] = self.scores[seat] as f32;
        }

        // Round-start values from start_kyoku event
        let (start_honba, start_riichi, start_scores) = self.parse_start_kyoku_info();
        out[6] = start_honba as f32;
        out[7] = start_riichi as f32;
        for i in 0..4 {
            let seat = (pid + i) % 4;
            out[8 + i] = start_scores[seat] as f32;
        }

        out
    }

    /// Parse start_kyoku event for initial round state.
    fn parse_start_kyoku_info(&self) -> (u32, u32, [i32; 4]) {
        for event_str in &self.events {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str)
                && v["type"].as_str() == Some("start_kyoku")
            {
                let honba = v["honba"].as_u64().unwrap_or(0) as u32;
                let kyotaku = v["kyotaku"].as_u64().unwrap_or(0) as u32;
                let mut scores = [0i32; 4];
                if let Some(arr) = v["scores"].as_array() {
                    for (i, val) in arr.iter().enumerate().take(4) {
                        scores[i] = val.as_i64().unwrap_or(0) as i32;
                    }
                }
                return (honba, kyotaku, scores);
            }
        }
        (self.honba as u32, self.riichi_sticks, self.scores)
    }

    // ── Progression features ─────────────────────────────────────────────

    /// Encode progression (action history) as variable-length 5-tuples.
    ///
    /// Each tuple: (actor, type, moqie, liqi, from)
    /// - actor: 0-3 (seats), 4 (marker/padding)
    /// - type: 0-276 (see plan for encoding)
    /// - moqie: 0=tedashi, 1=tsumogiri, 2=N/A
    /// - liqi: 0=no riichi, 1=with riichi, 2=N/A
    /// - from: 0-2 (relative seat), 4=N/A
    pub fn encode_seq_progression(&self) -> Vec<[u16; 5]> {
        // Fast path: use pre-computed progression from GameState
        if let Some(ref cached) = self.cached_progression {
            return cached.clone();
        }

        // Fallback: parse events from JSON (for deserialized Observations, replays, etc.)
        let mut prog: Vec<[u16; 5]> = Vec::with_capacity(128);
        let mut pending_reach_actor: Option<u8> = None;

        for event_str in &self.events {
            let v = match serde_json::from_str::<serde_json::Value>(event_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = match v["type"].as_str() {
                Some(t) => t,
                None => continue,
            };

            match event_type {
                "start_kyoku" => {
                    // Beginning-of-round marker
                    prog.push([4, 0, 2, 2, 4]);
                }
                "reach" => {
                    // Track reach for the next dahai
                    if let Some(actor) = v["actor"].as_u64() {
                        pending_reach_actor = Some(actor as u8);
                    }
                }
                "dahai" => {
                    let actor = v["actor"].as_u64().unwrap_or(0) as u8;
                    let pai = v["pai"].as_str().unwrap_or("?");
                    let tsumogiri = v["tsumogiri"].as_bool().unwrap_or(false);

                    if pai == "?" {
                        continue; // masked tile, skip
                    }

                    let k37 = match mjai_tile_to_kan37(pai) {
                        Some(k) => k,
                        None => continue,
                    };

                    let type_idx = 1 + k37 as u16; // 1-37
                    let moqie = if tsumogiri { 1 } else { 0 };
                    let liqi = if pending_reach_actor == Some(actor) {
                        pending_reach_actor = None;
                        1
                    } else {
                        0
                    };

                    prog.push([actor as u16, type_idx, moqie, liqi, 4]);
                }
                "chi" => {
                    let actor = v["actor"].as_u64().unwrap_or(0) as u8;
                    let target = v["target"].as_u64().unwrap_or(0) as u8;
                    let pai = v["pai"].as_str().unwrap_or("?");

                    if pai == "?" {
                        continue;
                    }

                    let called_tid = match mjai_to_tid(pai) {
                        Some(t) => t,
                        None => continue,
                    };

                    let consumed = self.parse_consumed_tids(&v);
                    if consumed.len() < 2 {
                        continue;
                    }

                    let chi_enc = encode_chi(&consumed, called_tid);
                    let type_idx = 38 + chi_enc; // 38-127
                    let rel = relative_from(actor, target);

                    prog.push([actor as u16, type_idx, 2, 2, rel as u16]);
                }
                "pon" => {
                    let actor = v["actor"].as_u64().unwrap_or(0) as u8;
                    let target = v["target"].as_u64().unwrap_or(0) as u8;
                    let pai = v["pai"].as_str().unwrap_or("?");

                    if pai == "?" {
                        continue;
                    }

                    let called_tid = match mjai_to_tid(pai) {
                        Some(t) => t,
                        None => continue,
                    };

                    let consumed = self.parse_consumed_tids(&v);
                    if consumed.len() < 2 {
                        continue;
                    }

                    let pon_enc = encode_pon(&consumed, called_tid);
                    let type_idx = 128 + pon_enc; // 128-167
                    let rel = relative_from(actor, target);

                    prog.push([actor as u16, type_idx, 2, 2, rel as u16]);
                }
                "daiminkan" => {
                    let actor = v["actor"].as_u64().unwrap_or(0) as u8;
                    let target = v["target"].as_u64().unwrap_or(0) as u8;
                    let pai = v["pai"].as_str().unwrap_or("?");

                    if pai == "?" {
                        continue;
                    }

                    let k37 = match mjai_tile_to_kan37(pai) {
                        Some(k) => k,
                        None => continue,
                    };

                    let type_idx = 168 + k37 as u16; // 168-204
                    let rel = relative_from(actor, target);

                    prog.push([actor as u16, type_idx, 2, 2, rel as u16]);
                }
                "ankan" => {
                    let actor = v["actor"].as_u64().unwrap_or(0) as u8;

                    // For ankan, get tile type from consumed tiles
                    let consumed = self.parse_consumed_tids(&v);
                    if consumed.is_empty() {
                        continue;
                    }

                    let tile34 = consumed[0] / 4;
                    let type_idx = 205 + tile34 as u16; // 205-238

                    prog.push([actor as u16, type_idx, 2, 2, 4]);
                }
                "kakan" => {
                    let actor = v["actor"].as_u64().unwrap_or(0) as u8;
                    let pai = v["pai"].as_str().unwrap_or("?");

                    if pai == "?" {
                        continue;
                    }

                    let k37 = match mjai_tile_to_kan37(pai) {
                        Some(k) => k,
                        None => continue,
                    };

                    let type_idx = 239 + k37 as u16; // 239-275

                    prog.push([actor as u16, type_idx, 2, 2, 4]);
                }
                _ => {
                    // tsumo, dora, reach_accepted, etc. — not progression events
                }
            }

            if prog.len() >= MAX_PROG_LEN {
                break;
            }
        }

        prog
    }

    /// Parse "consumed" array from MJAI event JSON → Vec<u8> of tile IDs.
    fn parse_consumed_tids(&self, v: &serde_json::Value) -> Vec<u8> {
        let mut tids = Vec::new();
        if let Some(arr) = v["consumed"].as_array() {
            for item in arr {
                if let Some(s) = item.as_str()
                    && let Some(tid) = mjai_to_tid(s)
                {
                    tids.push(tid);
                }
            }
        }
        tids
    }

    // ── Candidate features ───────────────────────────────────────────────

    /// Encode candidate (legal action) features as variable-length 4-tuples.
    ///
    /// Each tuple: (type, moqie, liqi, from)
    /// - type: 0-279
    /// - moqie: 0=tedashi, 1=tsumogiri, 2=N/A
    /// - liqi: 0=no, 1=yes, 2=N/A
    /// - from: 0-2 (relative seat), 3=self
    pub fn encode_seq_candidates(&self) -> Vec<[u16; 4]> {
        let mut cands: Vec<[u16; 4]> = Vec::with_capacity(64);
        let pid = self.player_id;

        // Check if there's a pending reach (riichi action in legal_actions)
        let has_riichi = self
            ._legal_actions
            .iter()
            .any(|a| a.action_type == ActionType::Riichi);

        for action in &self._legal_actions {
            let tuple = self.encode_candidate_action(action, pid, has_riichi);
            if let Some(t) = tuple {
                cands.push(t);
            }
        }

        cands
    }

    /// Encode a single legal action as a candidate 4-tuple.
    fn encode_candidate_action(
        &self,
        action: &Action,
        pid: u8,
        _has_riichi: bool,
    ) -> Option<[u16; 4]> {
        match action.action_type {
            ActionType::Discard => {
                let tile = action.tile?;
                let k37 = tile_id_to_kan37(tile as u32);
                let type_idx = k37 as u16; // 0-36

                // Determine moqie (tedashi vs tsumogiri)
                let moqie = if self.is_tsumogiri_candidate(tile) {
                    1
                } else {
                    0
                };

                Some([type_idx, moqie, 2, 3]) // from=3 (self)
            }
            ActionType::Riichi => {
                // Riichi is encoded as discard + liqi=1
                // It will appear alongside the discard candidates
                // Skip the riichi action itself; the discard actions
                // that follow riichi are the actual candidates
                None
            }
            ActionType::Ankan => {
                let first = *action.consume_tiles.first()?;
                let tile34 = first / 4;
                let type_idx = 37 + tile34 as u16; // 37-70

                Some([type_idx, 2, 2, 3])
            }
            ActionType::Kakan => {
                let tile = action
                    .tile
                    .or_else(|| action.consume_tiles.first().copied())?;
                let k37 = tile_id_to_kan37(tile as u32);
                let type_idx = 71 + k37 as u16; // 71-107

                Some([type_idx, 2, 2, 3])
            }
            ActionType::Tsumo => Some([108, 2, 2, 3]),
            ActionType::KyushuKyuhai => Some([109, 2, 2, 3]),
            ActionType::Pass => Some([110, 2, 2, 3]),
            ActionType::Chi => {
                let called_tile = action.tile?;
                let consumed = &action.consume_tiles;
                if consumed.len() < 2 {
                    return None;
                }

                let chi_enc = encode_chi(consumed, called_tile);
                let type_idx = 111 + chi_enc; // 111-200

                // from = relative seat of the discard source
                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);

                Some([type_idx, 2, 2, rel as u16])
            }
            ActionType::Pon => {
                let called_tile = action.tile?;
                let consumed = &action.consume_tiles;
                if consumed.len() < 2 {
                    return None;
                }

                let pon_enc = encode_pon(consumed, called_tile);
                let type_idx = 201 + pon_enc; // 201-240

                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);

                Some([type_idx, 2, 2, rel as u16])
            }
            ActionType::Daiminkan => {
                let tile = action.tile?;
                let k37 = tile_id_to_kan37(tile as u32);
                let type_idx = 241 + k37 as u16; // 241-277

                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);

                Some([type_idx, 2, 2, rel as u16])
            }
            ActionType::Ron => {
                let target = self.find_last_discard_actor()?;
                let rel = relative_from(pid, target);
                Some([278, 2, 2, rel as u16])
            }
            ActionType::Kita => None, // 3P only, not supported
        }
    }

    /// Check if discarding this tile would be tsumogiri.
    fn is_tsumogiri_candidate(&self, tile: u8) -> bool {
        if let Some(drawn) = self.get_drawn_tile() {
            drawn == tile
        } else {
            false
        }
    }

    /// Find the actor of the last discard (for chi/pon/kan/ron response).
    fn find_last_discard_actor(&self) -> Option<u8> {
        for event_str in self.events.iter().rev() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(event_str) {
                let event_type = v["type"].as_str().unwrap_or("");
                if event_type == "dahai" || event_type == "kakan" {
                    return v["actor"].as_u64().map(|a| a as u8);
                }
            }
        }
        None
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_id_to_kan37() {
        // Red fives
        assert_eq!(tile_id_to_kan37(16), 0); // red 5m → 0
        assert_eq!(tile_id_to_kan37(52), 10); // red 5p → 10
        assert_eq!(tile_id_to_kan37(88), 20); // red 5s → 20

        // 1m (tile_id 0-3, type 0) → kan37 = 1
        assert_eq!(tile_id_to_kan37(0), 1);
        assert_eq!(tile_id_to_kan37(3), 1);

        // 5m non-red (tile_id 17-19, type 4) → kan37 = 5
        assert_eq!(tile_id_to_kan37(17), 5);

        // 9m (tile_id 32-35, type 8) → kan37 = 9
        assert_eq!(tile_id_to_kan37(32), 9);

        // 1p (tile_id 36-39, type 9) → kan37 = 11
        assert_eq!(tile_id_to_kan37(36), 11);

        // 1s (tile_id 72-75, type 18) → kan37 = 21
        assert_eq!(tile_id_to_kan37(72), 21);

        // East wind (tile_id 108-111, type 27) → kan37 = 30
        assert_eq!(tile_id_to_kan37(108), 30);

        // Chun / Red dragon (tile_id 132-135, type 33) → kan37 = 36
        assert_eq!(tile_id_to_kan37(132), 36);
    }

    #[test]
    fn test_relative_from() {
        // Player 0 calling from player 3 → kamicha (left) = 2
        assert_eq!(relative_from(0, 3), 2);
        // Player 0 calling from player 1 → shimocha (right) = 0
        assert_eq!(relative_from(0, 1), 0);
        // Player 0 calling from player 2 → toimen (across) = 1
        assert_eq!(relative_from(0, 2), 1);
        // Player 2 calling from player 3 → shimocha = 0
        assert_eq!(relative_from(2, 3), 0);
    }

    #[test]
    fn test_encode_chi_basic() {
        // Chi: 1m-2m-3m, called 1m from discard
        // tile IDs: 1m=0, 2m=4, 3m=8 (first copies)
        // suit=0, seq_start=0, call_pos=0
        let consumed = [4u8, 8]; // 2m, 3m from hand
        let called = 0u8; // 1m called
        let enc = encode_chi(&consumed, called);
        assert_eq!(enc, 0); // suit_offset=0, base=0, sub=0

        // Chi: 1m-2m-3m, called 2m
        let consumed = [0u8, 8]; // 1m, 3m from hand
        let called = 4u8; // 2m called
        let enc = encode_chi(&consumed, called);
        assert_eq!(enc, 1); // sub=1 (middle)
    }

    #[test]
    fn test_encode_pon_honor() {
        // Pon: East wind
        // tile IDs: E=108, 109, 110
        let consumed = [109u8, 110];
        let called = 108u8;
        let enc = encode_pon(&consumed, called);
        assert_eq!(enc, 33); // first honor
    }

    #[test]
    fn test_encode_pon_five_red() {
        // Pon: 5m with red five in consumed
        // tile IDs: 5m = 16(red), 17, 18, 19
        let consumed = [16u8, 17]; // red 5m + normal 5m
        let called = 18u8; // normal 5m called
        let enc = encode_pon(&consumed, called);
        // suit_offset=0, rank=4 → five variants at offset 4
        // consumed has red → sub_idx=1
        assert_eq!(enc, 5); // 0 + 4 + 1

        // Pon: 5m with red five called
        let consumed = [17u8, 18];
        let called = 16u8; // red 5m called
        let enc = encode_pon(&consumed, called);
        assert_eq!(enc, 6); // 0 + 4 + 2
    }

    #[test]
    fn test_sparse_vocab_bounds() {
        // Verify all sparse offsets are within vocab
        assert!(441 < SPARSE_VOCAB_SIZE as u16);

        // Dora max: 83 + 4*37 + 36 = 83 + 148 + 36 = 267
        assert!(83 + 4 * 37 + 36 < SPARSE_VOCAB_SIZE as u16);

        // Hand max: 268 + 135 = 403
        assert!(268 + 135 < SPARSE_VOCAB_SIZE as u16);

        // Drawn tile max: 404 + 36 = 440
        assert!(404 + 36 < SPARSE_VOCAB_SIZE as u16);
    }

    #[test]
    fn test_progression_type_bounds() {
        // Verify type indices stay within PROG_DIMS[1] = 277
        let prog_type_max: u16 = 277;
        assert!(37 <= prog_type_max); // dahai max: 1 + 36 = 37
        assert!(38 + 89 <= prog_type_max); // chi max: 38 + 89 = 127
        assert!(128 + 39 <= prog_type_max); // pon max: 128 + 39 = 167
        assert!(168 + 36 <= prog_type_max); // daiminkan max: 168 + 36 = 204
        assert!(205 + 33 <= prog_type_max); // ankan max: 205 + 33 = 238
        assert!(239 + 36 <= prog_type_max); // kakan max: 239 + 36 = 275
    }

    #[test]
    fn test_candidate_type_bounds() {
        // Verify type indices stay within CAND_DIMS[0] = 280
        let cand_type_max: u16 = 280;
        assert!(36 < cand_type_max); // discard max: 36
        assert!(37 + 33 < cand_type_max); // ankan max: 70
        assert!(71 + 36 < cand_type_max); // kakan max: 107
        assert!(108 < cand_type_max); // tsumo
        assert!(109 < cand_type_max); // kyushu
        assert!(110 < cand_type_max); // pass
        assert!(111 + 89 < cand_type_max); // chi max: 200
        assert!(201 + 39 < cand_type_max); // pon max: 240
        assert!(241 + 36 < cand_type_max); // daiminkan max: 277
        assert!(278 < cand_type_max); // ron
    }
}
