//! Deal-in Risk EV (DREV) features.
//!
//! Distinct from `sp` — SP answers "what's the EV of MY hand if I pick this
//! discard"; DREV answers "what's the danger of dealing into an OPPONENT if I
//! pick this discard". The two are conceptually orthogonal so they live in
//! separate modules and produce separate channel blocks.
//!
//! Currently emits the static, no-opponent-SP layer:
//!   - genbutsu (anpai) — count of opponents to whom each tile is in their
//!     discard pile, normalised by `n_active_opponents`.
//!   - suji-safe — for each tile, fraction of opponents that have discarded
//!     a suji-blocker (T±3 within suit). Tells the model how protected each
//!     tile is against ryanmen waits.
//!   - kabe / ryanmen no-chance — for each tile, whether *every* possible
//!     ryanmen partner pair is blocked because at least one partner is
//!     four-copies visible. Pure structural signal from `tiles_seen`.
//!
//! Future steps will add reach-state weighting, per-opponent breakdowns,
//! and SP-derived deal-in EV.

use crate::observation::Observation;
use crate::types::TILE_MAX;

/// Total number of channels emitted by `encode_drev_into`.
///
/// Channel layout:
///   - ch 0: anpai (genbutsu fraction across active opponents)
///   - ch 1: suji-safe fraction
///   - ch 2: ryanmen no-chance flag (kabe)
///   - ch 3: reach_genbutsu_norm — per-tile genbutsu fraction restricted to
///     opponents that have declared reach. The most actionable danger
///     signal because reach freezes the wait set, so any tile in a reached
///     opponent's discard pile is permanent genbutsu against ron from them.
///     0 when no opponents have reached.
///   - ch 4: n_reach_norm (broadcast) — count of reached opponents divided
///     by `n_active_opponents`. Range [0, 1]. Lets the model interpret ch 3
///     in context: a 0.5 signal at ch 3 means very different things when
///     n_reach_norm is 0.33 vs 1.0.
///   - ch 5..8 ( 3): opp_tenpai_prob[slot] (broadcast). Heuristic estimate
///     of `P(opp_i is tenpai)`: 1.0 for reached opps, otherwise a turn /
///     open-meld / discard-count based prior. Slot ordering matches
///     `opp_safe_mask`. Inactive slots stay 0.
///   - ch 8 ( 1): per-tile threat aggregate (the DREV proxy). For each
///     tile, sum over active opponents of `(1 - safety_i_T) * tenpai_i`,
///     divided by `n_active_opponents`. `safety_i_T` per opp =
///     `max(genbutsu, suji, kabe)`. Range [0, 1], higher = more dangerous.
///     Combines the static safety signals with the threat heuristic into a
///     single deal-in-likelihood proxy.
pub const DREV_CHANNELS: usize = 9;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrevInput {
    /// Per-opponent genbutsu mask. `opp_safe_mask[i]` is a bitmask over
    /// `0..TILE_MAX`; bit `t` is set iff opponent `i` has discarded tile `t`
    /// at least once. Slots are relative-seat ordered:
    ///   slot 0 = shimocha (next), slot 1 = toimen, slot 2 = kamicha.
    /// For 3p, slot 2 stays 0 and `n_active_opponents` is 2.
    pub opp_safe_mask: [u64; 3],
    /// Number of non-self opponents currently at the table (3 for 4p, 2 for
    /// 3p). Used as the denominator for safety-fraction features.
    pub n_active_opponents: u8,
    /// Number of copies of each tile visible to the player (own hand, all
    /// melds, all discards, dora indicators), capped at 4. Used for the
    /// kabe / ryanmen no-chance check.
    #[serde(with = "serde_tiles_seen")]
    pub tiles_seen: [u8; TILE_MAX],
    /// Per-opponent reach state. Same slot ordering as `opp_safe_mask`.
    /// `true` iff the opponent has declared reach. Inactive opponents
    /// (3p kamicha slot, turn-0 4p) stay `false`.
    #[serde(default)]
    pub opp_reach: [bool; 3],
    /// Per-opponent count of *open* melds (chi/pon/open-kan). Closed kans
    /// are excluded — they don't reveal hand structure. Used as a prior
    /// component in the tenpai-probability heuristic: more open melds → the
    /// hand is closer to completion.
    #[serde(default)]
    pub opp_open_melds: [u8; 3],
    /// Per-opponent discard count (number of tiles already discarded). A
    /// proxy for "how deep into the round" each opponent is — late-game
    /// players are more likely to be tenpai.
    #[serde(default)]
    pub opp_discard_count: [u8; 3],
}

mod serde_tiles_seen {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::TILE_MAX;

    pub fn serialize<S: Serializer>(v: &[u8; TILE_MAX], s: S) -> Result<S::Ok, S::Error> {
        v.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; TILE_MAX], D::Error> {
        let v: Vec<u8> = Vec::deserialize(d)?;
        if v.len() != TILE_MAX {
            return Err(serde::de::Error::custom(format!(
                "expected {} elements, got {}",
                TILE_MAX,
                v.len()
            )));
        }
        let mut out = [0u8; TILE_MAX];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

impl DrevInput {
    pub fn from_observation(obs: &Observation) -> Self {
        let player_idx = obs.player_id as usize;
        let mut opp_safe_mask = [0u64; 3];
        let mut opp_reach = [false; 3];
        let mut opp_open_melds = [0u8; 3];
        let mut opp_discard_count = [0u8; 3];
        let mut n_active_opponents = 0u8;
        for slot in 0..3usize {
            let opp_idx = ((obs.player_id as usize) + slot + 1) % 4;
            if opp_idx == player_idx {
                continue;
            }
            let has_activity =
                !obs.hands[opp_idx].is_empty() || !obs.discards[opp_idx].is_empty();
            if !has_activity {
                continue;
            }
            n_active_opponents += 1;
            let mut mask = 0u64;
            for &tile in &obs.discards[opp_idx] {
                let tile_type = (tile / 4) as usize;
                if tile_type < TILE_MAX {
                    mask |= 1u64 << tile_type;
                }
            }
            opp_safe_mask[slot] = mask;
            opp_reach[slot] = obs.riichi_declared[opp_idx];
            opp_open_melds[slot] = obs.melds[opp_idx]
                .iter()
                .filter(|m| m.opened)
                .count()
                .min(255) as u8;
            opp_discard_count[slot] = obs.discards[opp_idx].len().min(255) as u8;
        }

        let mut tiles_seen = [0u8; TILE_MAX];
        let mut bump = |tile: u32| {
            let tile_type = (tile / 4) as usize;
            if tile_type < TILE_MAX {
                tiles_seen[tile_type] = tiles_seen[tile_type].saturating_add(1).min(4);
            }
        };
        for hand in obs.hands.iter() {
            for &tile in hand {
                bump(tile);
            }
        }
        for melds in obs.melds.iter() {
            for meld in melds {
                for &tile in &meld.tiles {
                    bump(tile as u32);
                }
            }
        }
        for discards in obs.discards.iter() {
            for &tile in discards {
                bump(tile);
            }
        }
        for &tile in &obs.dora_indicators {
            bump(tile);
        }

        Self {
            opp_safe_mask,
            n_active_opponents,
            tiles_seen,
            opp_reach,
            opp_open_melds,
            opp_discard_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DrevResult {
    /// Per-tile genbutsu score in [0, 1].
    pub anpai_norm: [f32; TILE_MAX],
    /// Per-tile suji-safe score in [0, 1]. For honor tiles always 0.
    pub suji_norm: [f32; TILE_MAX],
    /// Per-tile ryanmen no-chance flag in {0.0, 1.0}. Pure structural signal
    /// from `tiles_seen` — independent of opponent identity. For honor tiles
    /// always 0 (no ryanmen-style danger applicable).
    pub kabe: [f32; TILE_MAX],
    /// Per-tile reach-restricted genbutsu score in [0, 1]: fraction of
    /// reached opponents to whom the tile is genbutsu. 0 when no opponents
    /// have declared reach.
    pub reach_genbutsu_norm: [f32; TILE_MAX],
    /// Scalar in [0, 1]: count of reached opponents divided by
    /// `n_active_opponents`. Broadcast across all tile cells in ch 4.
    pub n_reach_norm: f32,
    /// Per-opponent tenpai-probability heuristic in [0, 1]. Slot ordering
    /// matches `opp_safe_mask`. 1.0 for reached opponents.
    pub opp_tenpai_prob: [f32; 3],
    /// Per-tile threat aggregate in [0, 1]: deal-in proxy combining
    /// per-opponent safety (max of genbutsu/suji/kabe) with the per-opponent
    /// tenpai prior. Higher = more dangerous to discard.
    pub threat: [f32; TILE_MAX],
}

pub fn calculate_drev(input: &DrevInput) -> DrevResult {
    let anpai_norm = anpai_norm(input);
    let suji_norm = suji_norm(input);
    let kabe = kabe_nochance(input);
    let (reach_genbutsu_norm, n_reach_norm) = reach_signals(input);
    let opp_tenpai_prob = opp_tenpai_prob(input);
    let threat = threat_aggregate(input, &kabe, &opp_tenpai_prob);
    DrevResult {
        anpai_norm,
        suji_norm,
        kabe,
        reach_genbutsu_norm,
        n_reach_norm,
        opp_tenpai_prob,
        threat,
    }
}

/// Heuristic per-opponent tenpai probability. Reached opponents are 1.0;
/// otherwise the prior is a coarse function of (open melds, discards). The
/// constants are intentionally loose — they're meant to give the model a
/// monotone signal, not a calibrated probability. Real calibration belongs
/// in a learned head, not in this hand-coded prior.
fn opp_tenpai_prob(input: &DrevInput) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for slot in 0..3 {
        let active = input.opp_safe_mask[slot] != 0
            || input.opp_open_melds[slot] != 0
            || input.opp_discard_count[slot] != 0
            || input.opp_reach[slot];
        if !active {
            continue;
        }
        if input.opp_reach[slot] {
            out[slot] = 1.0;
            continue;
        }
        let melds = input.opp_open_melds[slot] as f32;
        let discards = input.opp_discard_count[slot] as f32;
        // Loose linear blend: deeper into the round + more open melds → more
        // likely tenpai. Capped at 0.85 (we never claim >reach-level certainty
        // for non-reach opps).
        let prior = 0.05 + 0.05 * discards.min(18.0) + 0.15 * melds.min(3.0);
        out[slot] = prior.clamp(0.0, 0.85);
    }
    out
}

/// For each tile T, compute a deal-in-likelihood proxy:
///   sum over active opps of `(1 - safety_i_T) * tenpai_i` / n_active.
/// where `safety_i_T` = max over (genbutsu, suji-blocker-credit, kabe).
///
/// We re-derive per-opp safety here rather than reusing the averaged
/// `anpai_norm` / `suji_norm`, because the threat aggregate must consider
/// per-opponent threat * per-opponent safety — not the averaged versions.
fn threat_aggregate(
    input: &DrevInput,
    kabe: &[f32; TILE_MAX],
    tenpai: &[f32; 3],
) -> [f32; TILE_MAX] {
    let mut out = [0.0f32; TILE_MAX];
    if input.n_active_opponents == 0 {
        return out;
    }
    let inv_n = 1.0 / input.n_active_opponents as f32;
    for tile in 0..TILE_MAX {
        let bit = 1u64 << tile;
        let kabe_t = kabe[tile];
        // Suji blockers for this tile, computed once.
        let suit_base = (tile / 9) * 9;
        let num = tile % 9;
        let (blocker_low, blocker_high, n_blockers) = if tile < 27 {
            let lo = num.checked_sub(3).map(|n| suit_base + n);
            let hi = if num + 3 <= 8 { Some(suit_base + num + 3) } else { None };
            let n = (lo.is_some() as u8) + (hi.is_some() as u8);
            (lo, hi, n)
        } else {
            (None, None, 0)
        };
        let inv_blockers = if n_blockers > 0 { 1.0 / n_blockers as f32 } else { 0.0 };

        let mut total = 0.0f32;
        for slot in 0..3 {
            if tenpai[slot] == 0.0 {
                continue;
            }
            let mask = input.opp_safe_mask[slot];
            // Genbutsu against this opp.
            let genbutsu = (mask & bit != 0) as u32 as f32;
            // Suji credit per opp.
            let mut suji_hit = 0u8;
            if let Some(b) = blocker_low
                && mask & (1u64 << b) != 0
            {
                suji_hit += 1;
            }
            if let Some(b) = blocker_high
                && mask & (1u64 << b) != 0
            {
                suji_hit += 1;
            }
            let suji = suji_hit as f32 * inv_blockers;
            // Per-opp safety = max of (genbutsu, suji, kabe).
            let safety = genbutsu.max(suji).max(kabe_t).min(1.0);
            total += (1.0 - safety) * tenpai[slot];
        }
        out[tile] = (total * inv_n).clamp(0.0, 1.0);
    }
    out
}

fn reach_signals(input: &DrevInput) -> ([f32; TILE_MAX], f32) {
    let mut per_tile = [0.0f32; TILE_MAX];

    let n_reach = input.opp_reach.iter().filter(|&&r| r).count() as u8;
    let n_reach_norm = if input.n_active_opponents == 0 {
        0.0
    } else {
        n_reach as f32 / input.n_active_opponents as f32
    };

    if n_reach == 0 {
        return (per_tile, n_reach_norm);
    }

    let inv_reach = 1.0 / n_reach as f32;
    for tile in 0..TILE_MAX {
        let bit = 1u64 << tile;
        let mut hit = 0u32;
        for slot in 0..3 {
            if input.opp_reach[slot] && input.opp_safe_mask[slot] & bit != 0 {
                hit += 1;
            }
        }
        if hit > 0 {
            per_tile[tile] = hit as f32 * inv_reach;
        }
    }
    (per_tile, n_reach_norm)
}

fn anpai_norm(input: &DrevInput) -> [f32; TILE_MAX] {
    let mut out = [0.0f32; TILE_MAX];
    if input.n_active_opponents == 0 {
        return out;
    }
    let inv_n = 1.0 / input.n_active_opponents as f32;
    for tile in 0..TILE_MAX {
        let bit = 1u64 << tile;
        let mut count = 0u32;
        for slot in 0..3 {
            if input.opp_safe_mask[slot] & bit != 0 {
                count += 1;
            }
        }
        out[tile] = count as f32 * inv_n;
    }
    out
}

fn suji_norm(input: &DrevInput) -> [f32; TILE_MAX] {
    let mut out = [0.0f32; TILE_MAX];
    if input.n_active_opponents == 0 {
        return out;
    }
    let inv_n = 1.0 / input.n_active_opponents as f32;
    for tile in 0..TILE_MAX {
        if tile >= 27 {
            continue; // honors have no suji
        }
        let suit_base = (tile / 9) * 9;
        let num = tile % 9;
        // Suji blockers for tile T: T-3 and T+3 within same suit.
        let blocker_low = num.checked_sub(3).map(|n| suit_base + n);
        let blocker_high = if num + 3 <= 8 { Some(suit_base + num + 3) } else { None };
        let n_blockers = (blocker_low.is_some() as u8) + (blocker_high.is_some() as u8);
        if n_blockers == 0 {
            continue;
        }
        let inv_blockers = 1.0 / n_blockers as f32;

        let mut total = 0.0f32;
        for slot in 0..3 {
            let mask = input.opp_safe_mask[slot];
            let mut hit = 0u8;
            if let Some(b) = blocker_low
                && mask & (1u64 << b) != 0
            {
                hit += 1;
            }
            if let Some(b) = blocker_high
                && mask & (1u64 << b) != 0
            {
                hit += 1;
            }
            total += hit as f32 * inv_blockers;
        }
        out[tile] = total * inv_n;
    }
    out
}

/// Ryanmen no-chance per tile: tile T is "no-chance" iff *every* potential
/// ryanmen partner pair is impossible because at least one partner tile is
/// four-copies visible.
///
/// T can be a ryanmen wait via two partner pairs (within suit, num 0..=8):
///   - High pair (num+1, num+2) — needs `num + 2 <= 8`. If absent, the high
///     pair is trivially "blocked" (can't form ryanmen at all).
///   - Low pair (num-2, num-1) — needs `num >= 2`. If absent, the low pair
///     is trivially "blocked".
/// A pair is "blocked" if either partner has `tiles_seen >= 4`.
/// Tile T is no-chance iff *both* pairs are blocked.
fn kabe_nochance(input: &DrevInput) -> [f32; TILE_MAX] {
    let mut out = [0.0f32; TILE_MAX];
    let ts = &input.tiles_seen;
    for tile in 0..27usize {
        let suit_base = (tile / 9) * 9;
        let num = tile % 9;

        let high_blocked = if num + 2 > 8 {
            true
        } else {
            ts[suit_base + num + 1] >= 4 || ts[suit_base + num + 2] >= 4
        };
        let low_blocked = if num < 2 {
            true
        } else {
            ts[suit_base + num - 2] >= 4 || ts[suit_base + num - 1] >= 4
        };
        if high_blocked && low_blocked {
            out[tile] = 1.0;
        }
    }
    out
}

pub fn encode_drev(result: &DrevResult) -> Vec<f32> {
    let mut buf = vec![0.0f32; DREV_CHANNELS * TILE_MAX];
    encode_drev_into(result, &mut buf, 0);
    buf
}

pub fn encode_drev_into(result: &DrevResult, buf: &mut [f32], ch_offset: usize) {
    if buf.len() < (ch_offset + DREV_CHANNELS) * TILE_MAX {
        return;
    }
    let row = |ch: usize| (ch_offset + ch) * TILE_MAX;
    for tile in 0..TILE_MAX {
        let v = result.anpai_norm[tile];
        if v > 0.0 {
            buf[row(0) + tile] = v;
        }
        let v = result.suji_norm[tile];
        if v > 0.0 {
            buf[row(1) + tile] = v;
        }
        let v = result.kabe[tile];
        if v > 0.0 {
            buf[row(2) + tile] = v;
        }
        let v = result.reach_genbutsu_norm[tile];
        if v > 0.0 {
            buf[row(3) + tile] = v;
        }
        let v = result.threat[tile];
        if v > 0.0 {
            buf[row(8) + tile] = v;
        }
    }
    if result.n_reach_norm > 0.0 {
        for tile in 0..TILE_MAX {
            buf[row(4) + tile] = result.n_reach_norm;
        }
    }
    // Per-opp tenpai prob: ch 5 (slot 0), 6 (slot 1), 7 (slot 2). Each is a
    // scalar broadcast across all 34 cells.
    for slot in 0..3 {
        let v = result.opp_tenpai_prob[slot];
        if v > 0.0 {
            for tile in 0..TILE_MAX {
                buf[row(5 + slot) + tile] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_with(
        opp_safe_mask: [u64; 3],
        n_active_opponents: u8,
        tiles_seen: [u8; TILE_MAX],
    ) -> DrevInput {
        DrevInput {
            opp_safe_mask,
            n_active_opponents,
            tiles_seen,
            opp_reach: [false; 3],
            opp_open_melds: [0; 3],
            opp_discard_count: [0; 3],
        }
    }

    fn input_with_reach(
        opp_safe_mask: [u64; 3],
        n_active_opponents: u8,
        tiles_seen: [u8; TILE_MAX],
        opp_reach: [bool; 3],
    ) -> DrevInput {
        DrevInput {
            opp_safe_mask,
            n_active_opponents,
            tiles_seen,
            opp_reach,
            opp_open_melds: [0; 3],
            opp_discard_count: [0; 3],
        }
    }

    #[test]
    fn anpai_zero_when_no_active_opponents() {
        let result = calculate_drev(&input_with([0; 3], 0, [0; TILE_MAX]));
        for v in result.anpai_norm {
            assert_eq!(v, 0.0);
        }
        for v in result.suji_norm {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn anpai_fraction_matches_opponent_count() {
        // shimocha discards 1m; toimen discards 1m and 9m; kamicha silent.
        let input = input_with(
            [(1u64 << 0), (1u64 << 0) | (1u64 << 8), 0],
            3,
            [0; TILE_MAX],
        );
        let result = calculate_drev(&input);
        let third = 1.0f32 / 3.0;
        assert!((result.anpai_norm[0] - 2.0 * third).abs() < 1e-6, "1m");
        assert!((result.anpai_norm[8] - third).abs() < 1e-6, "9m");
    }

    #[test]
    fn three_p_only_two_opponents_normalises_correctly() {
        let input = input_with(
            [(1u64 << 0), (1u64 << 0), 0],
            2,
            [0; TILE_MAX],
        );
        let result = calculate_drev(&input);
        assert!((result.anpai_norm[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn suji_score_uses_t_minus_3_and_t_plus_3() {
        // Single opponent (slot 0) discards 4m (tile 3). Then:
        //   - 1m (tile 0): suji blocker is 4m (T+3). 1 of 1 blockers hit → 1.0
        //   - 7m (tile 6): suji blocker is 4m (T-3). 1 of 1 → 1.0
        //   - 4m (tile 3): blockers are 1m and 7m. 0 of 2 → 0.0
        //   - 5m (tile 4): blockers are 2m and 8m. 0 of 2 → 0.0
        let input = input_with([(1u64 << 3), 0, 0], 1, [0; TILE_MAX]);
        let result = calculate_drev(&input);
        assert!((result.suji_norm[0] - 1.0).abs() < 1e-6);
        assert!((result.suji_norm[6] - 1.0).abs() < 1e-6);
        assert!(result.suji_norm[3].abs() < 1e-6);
        assert!(result.suji_norm[4].abs() < 1e-6);
    }

    #[test]
    fn suji_score_double_sided_middle_tile_partial_credit() {
        // Single opponent discards 1m AND 7m. Then 4m has BOTH blockers hit.
        // suji[4m] = 2 / 2 = 1.0 (full suji).
        let input = input_with(
            [(1u64 << 0) | (1u64 << 6), 0, 0],
            1,
            [0; TILE_MAX],
        );
        let result = calculate_drev(&input);
        assert!((result.suji_norm[3] - 1.0).abs() < 1e-6, "4m fully suji");
        // 4m is a "double-sided" tile with 2 blockers; if only one was discarded
        // (e.g., only 1m), credit would be 0.5.
        let input2 = input_with([(1u64 << 0), 0, 0], 1, [0; TILE_MAX]);
        let r2 = calculate_drev(&input2);
        assert!((r2.suji_norm[3] - 0.5).abs() < 1e-6, "4m half suji");
    }

    #[test]
    fn suji_zero_for_honors() {
        let input = input_with(
            [u64::MAX, u64::MAX, u64::MAX], // every tile dropped
            3,
            [0; TILE_MAX],
        );
        let result = calculate_drev(&input);
        for t in 27..TILE_MAX {
            assert_eq!(result.suji_norm[t], 0.0, "tile {t} (honor)");
        }
    }

    #[test]
    fn kabe_nochance_blocks_when_partners_exhausted() {
        // 1m (tile 0): only ryanmen pair is (2m, 3m). If 2m is 4-visible,
        // the high pair is blocked; the low pair is auto-blocked (no T-2,
        // T-1 within suit). So 1m is no-chance.
        let mut tiles_seen = [0u8; TILE_MAX];
        tiles_seen[1] = 4; // 2m fully visible
        let input = input_with([0; 3], 0, tiles_seen);
        let result = calculate_drev(&input);
        assert_eq!(result.kabe[0], 1.0, "1m blocked by 2m exhaustion");
    }

    #[test]
    fn kabe_nochance_for_middle_tile_requires_both_sides() {
        // 5m (tile 4): pairs are (6m, 7m) and (3m, 4m). Need both blocked.
        // Block only one side → not no-chance.
        let mut ts = [0u8; TILE_MAX];
        ts[5] = 4; // 6m exhausted → blocks high pair (6,7)
        let r = calculate_drev(&input_with([0; 3], 0, ts));
        assert_eq!(r.kabe[4], 0.0, "5m needs both sides blocked");

        // Now block low side too: 3m exhausted → blocks low pair (3,4)
        let mut ts2 = ts;
        ts2[2] = 4; // 3m exhausted
        let r2 = calculate_drev(&input_with([0; 3], 0, ts2));
        assert_eq!(r2.kabe[4], 1.0, "5m fully blocked");
    }

    #[test]
    fn kabe_zero_for_honors() {
        let ts = [4u8; TILE_MAX]; // every tile fully visible
        let r = calculate_drev(&input_with([0; 3], 0, ts));
        for t in 27..TILE_MAX {
            assert_eq!(r.kabe[t], 0.0, "honor {t}");
        }
    }

    #[test]
    fn reach_genbutsu_zero_when_no_reach() {
        let input = input_with([(1u64 << 0); 3], 3, [0; TILE_MAX]);
        let result = calculate_drev(&input);
        assert_eq!(result.n_reach_norm, 0.0);
        for v in result.reach_genbutsu_norm {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn reach_genbutsu_only_counts_reached_opponents() {
        // 3 active opps, only slot 1 (toimen) is in reach. Toimen has discarded
        // tile 7, shimocha (non-reach) has discarded tile 7 too (irrelevant
        // to reach_genbutsu). reach_genbutsu_norm[7] should be 1.0 (1/1
        // reached opps with the tile).
        let input = input_with_reach(
            [(1u64 << 7), (1u64 << 7), 0],
            3,
            [0; TILE_MAX],
            [false, true, false],
        );
        let result = calculate_drev(&input);
        assert!((result.n_reach_norm - 1.0 / 3.0).abs() < 1e-6);
        assert!((result.reach_genbutsu_norm[7] - 1.0).abs() < 1e-6);
        // Tile 0 (1m): only shimocha (non-reach) discarded? actually no
        // discards of 1m. So reach_genbutsu[0] should be 0.
        assert_eq!(result.reach_genbutsu_norm[0], 0.0);
    }

    #[test]
    fn reach_genbutsu_fraction_with_two_reachers() {
        // Slots 0 and 1 are both in reach. Slot 0 has tile 5, slot 1 has tile 9.
        // reach_genbutsu_norm[5] = 1/2, reach_genbutsu_norm[9] = 1/2.
        let input = input_with_reach(
            [(1u64 << 5), (1u64 << 9), 0],
            3,
            [0; TILE_MAX],
            [true, true, false],
        );
        let result = calculate_drev(&input);
        assert!((result.n_reach_norm - 2.0 / 3.0).abs() < 1e-6);
        assert!((result.reach_genbutsu_norm[5] - 0.5).abs() < 1e-6);
        assert!((result.reach_genbutsu_norm[9] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn opp_tenpai_prob_is_one_for_reached_opp() {
        let input = input_with_reach(
            [(1u64 << 0), 0, 0],
            3,
            [0; TILE_MAX],
            [true, false, false],
        );
        let result = calculate_drev(&input);
        assert_eq!(result.opp_tenpai_prob[0], 1.0);
    }

    #[test]
    fn opp_tenpai_prob_zero_for_inactive_slot() {
        // Only slot 0 is active (has discards), slots 1/2 fully inert.
        let input = input_with([(1u64 << 0), 0, 0], 1, [0; TILE_MAX]);
        let result = calculate_drev(&input);
        assert!(result.opp_tenpai_prob[0] > 0.0, "active slot should have a prior");
        assert_eq!(result.opp_tenpai_prob[1], 0.0, "inactive slot");
        assert_eq!(result.opp_tenpai_prob[2], 0.0, "inactive slot");
    }

    #[test]
    fn opp_tenpai_prob_grows_with_discards_and_melds() {
        let mut early = input_with([(1u64 << 0), 0, 0], 1, [0; TILE_MAX]);
        early.opp_discard_count[0] = 3;
        let mut late = early.clone();
        late.opp_discard_count[0] = 12;
        late.opp_open_melds[0] = 2;
        let r_early = calculate_drev(&early);
        let r_late = calculate_drev(&late);
        assert!(
            r_late.opp_tenpai_prob[0] > r_early.opp_tenpai_prob[0],
            "deeper / open-meld hand should have higher prior"
        );
    }

    #[test]
    fn threat_is_zero_when_no_opponents_tenpai() {
        // No reach, no melds, no discards → tenpai prior = 0 → threat = 0.
        let input = input_with([0; 3], 0, [0; TILE_MAX]);
        let result = calculate_drev(&input);
        for v in result.threat {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn threat_drops_for_genbutsu_against_threatening_opp() {
        // Slot 0 reached. Their discards include tile 5. Tile 5 is genbutsu
        // against slot 0 → safety_0_5 = 1.0 → threat contribution from slot 0 = 0.
        // Other slots inactive → no contribution. So threat[5] = 0.
        // Tile 0 (1m): NOT in slot 0's discards. safety_0 ≈ 0 (no suji, no
        // kabe in this fixture) → threat[0] = 1.0 × 1.0 / 3 = 0.333...
        let input = input_with_reach(
            [(1u64 << 5), 0, 0],
            3,
            [0; TILE_MAX],
            [true, false, false],
        );
        let result = calculate_drev(&input);
        assert_eq!(result.threat[5], 0.0, "tile 5 is genbutsu vs slot 0");
        assert!(
            (result.threat[0] - 1.0 / 3.0).abs() < 1e-6,
            "tile 0 threat = (1 - 0) * 1.0 / 3 active"
        );
    }

    #[test]
    fn threat_combines_kabe_with_tenpai_prob() {
        // Slot 0 reached, tile 0 (1m) — but 2m is exhausted → kabe[1m] = 1.
        // Per-opp safety_0_T=0 = max(genbutsu=0, suji=0, kabe=1) = 1.
        // Threat[0] = (1 - 1) * 1.0 / 3 = 0.
        let mut ts = [0u8; TILE_MAX];
        ts[1] = 4; // 2m exhausted → kabe-blocks 1m
        let input = input_with_reach([0; 3], 3, ts, [true, false, false]);
        let result = calculate_drev(&input);
        assert_eq!(result.threat[0], 0.0, "kabe blocks even reached opp");
    }

    #[test]
    fn encode_drev_writes_all_channels() {
        let mut ts = [0u8; TILE_MAX];
        ts[1] = 4; // makes 1m no-chance
        let input = input_with_reach(
            [(1u64 << 5), 0, 0],
            2,
            ts,
            [true, false, false], // shimocha in reach
        );
        let result = calculate_drev(&input);
        let buf = encode_drev(&result);
        assert_eq!(buf.len(), DREV_CHANNELS * TILE_MAX);
        // ch 0 (anpai): tile 5 has 1/2 (one of two active opponents).
        assert!((buf[5] - 0.5).abs() < 1e-6, "anpai ch 0");
        // ch 1 (suji): tile 2 (= 3m) — only blocker is T+3 = tile 5 (6m).
        //   blocker hit fraction for slot 0 = 1/1, averaged over 2 active = 0.5
        assert!((buf[TILE_MAX + 2] - 0.5).abs() < 1e-6, "suji ch 1 for 3m");
        // ch 2 (kabe): tile 0 (1m) is no-chance because 2m exhausted.
        assert_eq!(buf[2 * TILE_MAX + 0], 1.0, "kabe ch 2 for 1m");
        // ch 3 (reach_genbutsu): only shimocha is in reach; she discarded
        // tile 5. reach_genbutsu_norm[5] = 1/1 = 1.0.
        assert!((buf[3 * TILE_MAX + 5] - 1.0).abs() < 1e-6, "reach_genbutsu ch 3");
        // ch 4 (n_reach_norm broadcast): 1 reached / 2 active = 0.5, every cell.
        for t in 0..TILE_MAX {
            assert!(
                (buf[4 * TILE_MAX + t] - 0.5).abs() < 1e-6,
                "n_reach_norm ch 4 cell {t}"
            );
        }
        // ch 5 (slot 0 tenpai prob, broadcast): shimocha is reached → 1.0.
        for t in 0..TILE_MAX {
            assert_eq!(buf[5 * TILE_MAX + t], 1.0, "tenpai slot 0 ch 5 cell {t}");
        }
        // ch 6, 7: slot 1 / 2 inactive → 0.
        for t in 0..TILE_MAX {
            assert_eq!(buf[6 * TILE_MAX + t], 0.0);
            assert_eq!(buf[7 * TILE_MAX + t], 0.0);
        }
        // ch 8 (threat): tile 5 is genbutsu vs slot 0 → safety = 1 → 0.
        assert_eq!(buf[8 * TILE_MAX + 5], 0.0, "threat ch 8 tile 5 (genbutsu)");
        // tile 0 (1m): kabe = 1 → safety = 1 → 0.
        assert_eq!(buf[8 * TILE_MAX + 0], 0.0, "threat ch 8 tile 0 (kabe)");
        // tile 8 (9m): kabe? num=8, high pair (9,10) doesn't exist → blocked.
        // low pair (7,8) = tiles 6,7. Neither exhausted → not blocked.
        // So kabe[8] = 0. tile 8 in slot 0's mask? Only bit 5 is set → no.
        // Suji blockers for tile 8 (num=8): T-3=tile 5 (in mask!), T+3 absent.
        // Suji credit = 1/1 = 1.0 → safety = 1 → threat = 0.
        assert_eq!(buf[8 * TILE_MAX + 8], 0.0, "threat ch 8 tile 8 (suji)");
        // tile 13 (5p): no genbutsu, no suji, no kabe → safety = 0 → threat
        // = (1 - 0) × 1.0 / 2 active = 0.5.
        assert!(
            (buf[8 * TILE_MAX + 13] - 0.5).abs() < 1e-6,
            "threat ch 8 tile 13 (no safety)"
        );
    }
}
