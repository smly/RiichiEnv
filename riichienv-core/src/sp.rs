use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::rc::Rc;

use crate::feature_context::{FeatureContext, FeatureContext3P};
use crate::hand_evaluator::HandEvaluator;
use crate::hand_evaluator_3p::HandEvaluator3P;
use crate::observation::Observation;
use crate::observation_3p::Observation3P;
use crate::shanten;
use crate::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};

pub const SP_MAX_TURNS: usize = 17;
/// SP channel layout:
///  - ch  0..  2  ( 2): broadcast max_ev (100k / 30k normalisation)
///  - ch  2.. 36  (34): per-discard required_tiles[discard][tile] = 1
///  - ch 36.. 70  (34): per-discard yaku_progress_tiles[discard][tile] = 1
///  - ch     70   ( 1): one-hot best_required discard tile
///  - ch     71   ( 1): one-hot best_yaku_progress discard tile
///  - ch 72.. 89  (17): tenpai_probs per (discard, turn)
///  - ch 89..106  (17): win_probs per (discard, turn)
///  - ch 106..123 (17): exp_values per (discard, turn)
///  - ch 123..135 (12): per-discard yaku-mask flags (one bit each, see
///    `yaku_mask_bits`); cell `(123+bit, discard_tile)` = 1 if achievable
///  - ch 135..138 ( 3): per-discard scoring stats — at cell `(.., discard_tile)`,
///    min_point/100k, mean_point/100k, max_point/100k respectively
///  - ch 138..172 (34): per-discard future wait map. Channel `138+discard`
///    has cell `tile` = 1 if drawing that tile after the discard is
///    structurally useful (= `potentially_effective_for_draw`). Superset of
///    `required_tiles` (ch 2..36) — at tenpai they coincide; pre-tenpai it
///    captures the broader set of tiles that *could* progress the hand.
///  - ch 172..178 ( 6): per-discard target-point achievement probability.
///    For target thresholds `target_points::TARGETS = [1k, 2k, 4k, 8k, 12k,
///    16k]`, channel `172+k` has cell `discard_tile` =
///    `P(score ≥ T_k | win at this discard)`. Populated at shanten=0 only.
///
/// Total: 178 channels. Deal-in Risk EV (DREV) features live in a sibling
/// `drev` module and are concatenated downstream in the observation encoder.
pub const SP_CHANNELS: usize = 178;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpInput {
    #[serde(with = "serde_arrays")]
    pub tehai: [u8; TILE_MAX],
    pub akas_in_hand: [bool; 3],
    #[serde(with = "serde_arrays")]
    pub tiles_seen: [u8; TILE_MAX],
    /// `[5mr, 5pr, 5sr]` — true if the corresponding red 5 is already visible
    /// (own hand, any player's melds, dora indicators, any player's discards).
    /// `false` means the red could still be in the wall and the DP will branch
    /// draw events into "drew normal 5x" vs "drew the red 5x" sub-paths
    /// (matching Mortal's `akas_in_wall` semantics).
    #[serde(default)]
    pub akas_seen: [bool; 3],
    pub dora_indicators: Vec<u8>,
    pub melds: Vec<Meld>,
    pub bakaze: u8,
    pub jikaze: u8,
    pub is_menzen: bool,
    pub can_riichi: bool,
    pub can_double_riichi: bool,
    pub tsumos_left: u8,
    pub discard_candidates: Vec<u8>,
}

/// Sanma SP input. The calculation retains the canonical 34-tile indexing
/// internally and excludes 2m through 8m from the wall. Encoders compact the
/// spatial axis to the 27 tile types used by every other 3P feature block.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpInput3P {
    #[serde(with = "serde_arrays")]
    pub tehai: [u8; TILE_MAX],
    pub akas_in_hand: [bool; 3],
    #[serde(with = "serde_arrays")]
    pub tiles_seen: [u8; TILE_MAX],
    #[serde(default)]
    pub akas_seen: [bool; 3],
    pub dora_indicators: Vec<u8>,
    pub melds: Vec<Meld>,
    pub bakaze: u8,
    pub jikaze: u8,
    pub is_menzen: bool,
    pub can_riichi: bool,
    pub can_double_riichi: bool,
    pub tsumos_left: u8,
    pub discard_candidates: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpVariant {
    FourPlayer,
    ThreePlayer,
}

mod serde_arrays {
    //! [u8; 34] doesn't have built-in serde support; emit/parse as Vec<u8> with
    //! a length check.
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

/// Compute the per-discard "future wait map": for each tile in 0..34, mark
/// 1.0 if drawing that tile after the given discard is structurally relevant
/// to hand progression. Uses `potentially_effective_for_draw` semantics — a
/// superset of `required_tiles` (which only marks tiles that *immediately*
/// reduce shanten). Future-wait extends to "tiles within ±2 of any in-hand
/// tile in the same suit, OR matching an in-hand tile" — these are the
/// candidates the network may want to weigh for hand-shape planning beyond
/// the immediate next draw.
#[inline]
fn future_wait_map(after_discard: &[u8; TILE_MAX], remaining: &[u8; TILE_MAX]) -> [f32; TILE_MAX] {
    let mut out = [0.0f32; TILE_MAX];
    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || after_discard[tile] >= 4 {
            continue;
        }
        if potentially_effective_for_draw(after_discard, tile) {
            out[tile] = remaining[tile] as f32;
        }
    }
    out
}

/// Bit positions for `SpCandidate.yaku_mask`. Each bit indicates that the
/// corresponding yaku is achievable by SOME wait tile of this discard's
/// resulting hand (= structural shape allows it). For non-optimal discards
/// (= shanten-down), the mask is left at 0 because per-wait yaku detection
/// is not run for those candidates.
pub mod yaku_mask_bits {
    pub const TANYAO: u32 = 1 << 0;
    pub const YAKUHAI_DRAGON: u32 = 1 << 1;
    pub const YAKUHAI_ROUND: u32 = 1 << 2;
    pub const YAKUHAI_SEAT: u32 = 1 << 3;
    pub const HONITSU: u32 = 1 << 4;
    pub const CHINITSU: u32 = 1 << 5;
    pub const HONROUTOU: u32 = 1 << 6;
    pub const TOITOI: u32 = 1 << 7;
    pub const SANSHOKU_DOUKOU: u32 = 1 << 8;
    pub const SHOUSANGEN: u32 = 1 << 9;
    pub const RIICHI: u32 = 1 << 10;
    pub const MENZEN_TSUMO: u32 = 1 << 11;
    /// Number of yaku-mask bits actually consumed by the encoder.
    pub const N_BITS: usize = 12;
}

/// Tsumo-total point thresholds used to encode "target-point achievement
/// probability" features. For each candidate (discard), we report
/// `P(score ≥ T_k | win at this discard)` = wait-count-weighted fraction of
/// waits scoring ≥ T_k. Targets cover the natural breakpoints between hand
/// classes (kid tsumo totals): 1han, 2han, 3han, mangan, haneman, baiman.
pub mod target_points {
    pub const TARGETS: [f32; 6] = [1000.0, 2000.0, 4000.0, 8000.0, 12_000.0, 16_000.0];
    pub const N: usize = TARGETS.len();
}

#[derive(Debug, Clone)]
pub struct SpCandidate {
    pub tile: u8,
    pub tenpai_probs: [f32; SP_MAX_TURNS],
    pub win_probs: [f32; SP_MAX_TURNS],
    pub exp_values: [f32; SP_MAX_TURNS],
    pub required_tiles: [f32; TILE_MAX],
    pub yaku_progress_tiles: [f32; TILE_MAX],
    pub num_required_tiles: f32,
    pub num_yaku_progress_tiles: f32,
    /// Bitmask of yaku achievable by SOME wait of this discard. See
    /// `yaku_mask_bits` for bit positions. Populated for tenpai-maintaining
    /// (optimal) discards; left at 0 for shanten-down (non-optimal) ones.
    pub yaku_mask: u32,
    /// Lowest tsumo total point (yen) over the wait set. 0 if no scored waits.
    pub min_point: f32,
    /// Wait-count-weighted mean tsumo total point. Same value as
    /// `ScoringSummary.mean_point` but exposed per candidate.
    pub mean_point: f32,
    /// Highest tsumo total point over the wait set.
    pub max_point: f32,
    /// Future wait candidate map: for each tile, the count of remaining
    /// instances if drawing it would be structurally useful for the hand's
    /// future progression (= `potentially_effective_for_draw`). Superset of
    /// `required_tiles`.
    pub future_wait_tiles: [f32; TILE_MAX],
    /// Target-point achievement probabilities. `point_achievement_probs[k]`
    /// = `P(score ≥ target_points::TARGETS[k] | win at this discard)`,
    /// computed as wait-count-weighted fraction over the wait set. Populated
    /// at shanten=0 only; left at 0 for shanten-down candidates (the network
    /// can rely on `mean_point`/`max_point` for those).
    pub point_achievement_probs: [f32; target_points::N],
}

#[derive(Debug, Clone)]
pub struct SpResult {
    pub candidates: Vec<SpCandidate>,
}

impl SpInput {
    pub fn from_observation(obs: &Observation) -> Self {
        let context = FeatureContext::new_unchecked(obs);
        Self::from_feature_context(&context)
    }

    pub fn from_feature_context(context: &FeatureContext<'_>) -> Self {
        let obs = context.observation();
        let player_idx = context.player_index();

        let rel_seat = (obs.player_id + 4 - obs.oya) % 4;
        let can_riichi = obs.riichi_declared[player_idx] || obs.scores[player_idx] >= 1000;
        let can_double_riichi = can_riichi
            && obs.discards.iter().all(Vec::is_empty)
            && obs.melds.iter().all(Vec::is_empty);

        Self {
            tehai: *context.hand_counts(),
            akas_in_hand: context.akas_in_hand(),
            tiles_seen: *context.visible_counts_capped(),
            akas_seen: context.akas_seen(),
            dora_indicators: obs.dora_indicators.iter().map(|&x| x as u8).collect(),
            melds: obs.melds[player_idx].clone(),
            bakaze: obs.round_wind,
            jikaze: 27 + rel_seat,
            is_menzen: obs.melds[player_idx].iter().all(|m| !m.opened),
            can_riichi,
            can_double_riichi,
            tsumos_left: remaining_self_draws(obs),
            discard_candidates: context.discard_candidates().to_vec(),
        }
    }
}

impl SpInput3P {
    pub fn from_observation(obs: &Observation3P) -> Self {
        let context = FeatureContext3P::new_unchecked(obs);
        Self::from_feature_context(&context)
    }

    pub fn from_feature_context(context: &FeatureContext3P<'_>) -> Self {
        let obs = context.observation();
        let player_idx = context.player_index();
        let rel_seat = (obs.player_id + 3 - obs.oya) % 3;
        let can_riichi = obs.riichi_declared[player_idx] || obs.scores[player_idx] >= 1000;
        let can_double_riichi = can_riichi
            && obs.discards.iter().all(Vec::is_empty)
            && obs.melds.iter().all(Vec::is_empty);

        Self {
            tehai: *context.hand_counts(),
            akas_in_hand: context.akas_in_hand(),
            tiles_seen: *context.visible_counts_capped(),
            akas_seen: context.akas_seen(),
            dora_indicators: obs.dora_indicators.iter().map(|&tile| tile as u8).collect(),
            melds: obs.melds[player_idx].clone(),
            bakaze: obs.round_wind,
            jikaze: 27 + rel_seat,
            is_menzen: obs.melds[player_idx].iter().all(|meld| !meld.opened),
            can_riichi,
            can_double_riichi,
            tsumos_left: remaining_self_draws_3p(obs),
            discard_candidates: context.discard_candidates().to_vec(),
        }
    }

    fn as_common_input(&self) -> SpInput {
        SpInput {
            tehai: self.tehai,
            akas_in_hand: self.akas_in_hand,
            tiles_seen: self.tiles_seen,
            akas_seen: self.akas_seen,
            dora_indicators: self.dora_indicators.clone(),
            melds: self.melds.clone(),
            bakaze: self.bakaze,
            jikaze: self.jikaze,
            is_menzen: self.is_menzen,
            can_riichi: self.can_riichi,
            can_double_riichi: self.can_double_riichi,
            tsumos_left: self.tsumos_left,
            discard_candidates: self.discard_candidates.clone(),
        }
    }
}

/// DEBUG-ONLY: expose the leaf scoring path so the Mortal-comparison harness
/// can localize EV interpretation differences. Returns (han, fu, total) for a
/// menzen-tsumo or open-tsumo win on `win_tile` from the 13-tile `tehai_13`.
#[doc(hidden)]
pub fn __debug_score_for_win(
    input: &SpInput,
    tehai_13: &[u8; TILE_MAX],
    win_tile: u8,
) -> Option<(u32, u32, u32)> {
    let base = base_score_tsumo(
        input,
        tehai_13,
        win_tile,
        input.akas_in_hand,
        SpVariant::FourPlayer,
    )?;
    Some((base.han, base.fu, base.total))
}

/// Initial `akas_in_wall` from the input. Mortal's convention:
///   `akas_in_wall = !akas_seen` (after `akas_seen` is OR-ed with `akas_in_hand`,
///   since own akas are trivially "seen" by the player).
fn initial_akas_in_wall(input: &SpInput) -> [bool; 3] {
    let mut wall = [true; 3];
    for i in 0..3 {
        if input.akas_seen[i] || input.akas_in_hand[i] {
            wall[i] = false;
        }
    }
    wall
}

/// Update `akas_in_hand` to reflect discarding `tile` from `counts` (pre-discard).
/// Mortal's convention: when only one copy of a 5x is in hand and the red is in
/// hand, the discarded tile *is* the red. Otherwise the discard removes a
/// regular and aka stays.
fn aka_after_discard(akas: [bool; 3], counts: &[u8; TILE_MAX], tile: u8) -> [bool; 3] {
    let red_idx = match tile {
        4 => 0,
        13 => 1,
        22 => 2,
        _ => return akas,
    };
    if akas[red_idx] && counts[tile as usize] == 1 {
        let mut next = akas;
        next[red_idx] = false;
        next
    } else {
        akas
    }
}

pub fn calculate_sp(input: &SpInput) -> SpResult {
    calculate_sp_for_variant(input, SpVariant::FourPlayer)
}

pub fn calculate_sp_3p(input: &SpInput3P) -> SpResult {
    let common = input.as_common_input();
    calculate_sp_for_variant(&common, SpVariant::ThreePlayer)
}

fn calculate_sp_for_variant(input: &SpInput, variant: SpVariant) -> SpResult {
    let raw_discard_tiles: Vec<u8> = if input.discard_candidates.is_empty() {
        input
            .tehai
            .iter()
            .enumerate()
            .filter_map(|(tile, &count)| (count > 0).then_some(tile as u8))
            .collect()
    } else {
        input.discard_candidates.clone()
    };

    let remaining = remaining_counts_for_variant(input, variant);
    let total_remaining: f32 = remaining.iter().map(|&x| x as f32).sum::<f32>().max(1.0);

    let mut dp = DpContext::new(input, variant);

    // First pass: compute the post-discard shanten for every candidate so we
    // know which discards are "shanten-maintaining" (= optimal). Mortal-style
    // pre-filtering: only the optimal-shanten discards run the full DP; the
    // shanten-down ones use the cheap probability_series approximation.
    let mut prepared: Vec<(u8, [u8; TILE_MAX], i8)> = Vec::with_capacity(raw_discard_tiles.len());
    let mut best_shanten = i8::MAX;
    for tile in raw_discard_tiles {
        let tile_idx = tile as usize;
        if tile_idx >= TILE_MAX || input.tehai[tile_idx] == 0 {
            continue;
        }
        let mut after_discard = input.tehai;
        after_discard[tile_idx] -= 1;
        let s = dp.shanten(&after_discard);
        best_shanten = best_shanten.min(s);
        prepared.push((tile, after_discard, s));
    }

    let mut candidates = Vec::with_capacity(prepared.len());
    for (tile, after_discard, shanten_after) in prepared {
        let is_optimal = shanten_after == best_shanten;

        // At tenpai (the hot per-candidate case), `required_tiles`,
        // `score_waits`, and `yaku_progress_tiles` all enumerate the same wait
        // set; fuse them into one 34-tile pass with cached shanten lookups.
        let (required_tiles, mut scoring, yaku_progress_tiles) = if shanten_after == 0 {
            fused_tenpai_pass(&mut dp, &after_discard, &remaining)
        } else {
            let req = required_tiles(&mut dp, &after_discard, &remaining, shanten_after);
            let sc = score_waits(&mut dp, &after_discard, &remaining);
            let yp = yaku_progress_tiles(&mut dp, &after_discard, &remaining, shanten_after);
            (req, sc, yp)
        };
        let num_required_tiles = required_tiles.iter().sum::<f32>();
        let num_yaku_progress_tiles = yaku_progress_tiles.iter().sum::<f32>();

        // `mean_point` is consumed only by `series_for_candidate_approx`
        // (shanten-down or shanten ≥ 4 paths). Tenpai+optimal and shanten
        // 1..=3+optimal both ignore it, so skip the fallback in those cases.
        let needs_mean_point = !is_optimal || shanten_after > SHANTEN_THRES;
        if needs_mean_point && scoring.mean_point <= 0.0 {
            scoring.mean_point = rough_point_estimate(input, &after_discard, variant);
        }

        // Aka tracking through the outer-loop discard: if `tile` is a 5x and
        // the only copy was the red one, the player no longer has it.
        let post_discard_akas = aka_after_discard(input.akas_in_hand, &input.tehai, tile);

        let (tenpai_probs, win_probs, exp_values) = if is_optimal {
            let waits = (shanten_after == 0).then_some(&required_tiles);
            series_for_candidate(
                &mut dp,
                &after_discard,
                &remaining,
                shanten_after,
                num_required_tiles,
                scoring.wait_count,
                scoring.mean_point,
                total_remaining,
                post_discard_akas,
                waits,
            )
        } else {
            // Shanten-down discard: skip the expensive DP and use the
            // closed-form probability_series approximation. Same code path the
            // shanten ≥ 4 fallback already takes.
            series_for_candidate_approx(
                shanten_after,
                num_required_tiles,
                scoring.wait_count,
                scoring.mean_point,
                total_remaining,
                (input.tsumos_left as usize).min(SP_MAX_TURNS),
            )
        };

        let future_wait_tiles = future_wait_map(&after_discard, &remaining);

        let mut point_achievement_probs = [0.0f32; target_points::N];
        if scoring.wait_count > 0.0 {
            let inv = 1.0 / scoring.wait_count;
            for k in 0..target_points::N {
                point_achievement_probs[k] = (scoring.point_buckets[k] * inv).clamp(0.0, 1.0);
            }
        }

        candidates.push(SpCandidate {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles,
            yaku_progress_tiles,
            num_required_tiles,
            num_yaku_progress_tiles,
            yaku_mask: scoring.yaku_mask,
            min_point: scoring.min_point,
            mean_point: scoring.mean_point,
            max_point: scoring.max_point,
            future_wait_tiles,
            point_achievement_probs,
        });
    }

    candidates.sort_by(|a, b| {
        b.exp_values[0]
            .partial_cmp(&a.exp_values[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.tile.cmp(&b.tile))
    });

    SpResult { candidates }
}

pub fn encode_sp(result: &SpResult) -> Vec<f32> {
    let mut buf = vec![0.0f32; SP_CHANNELS * TILE_MAX];
    encode_sp_into(result, &mut buf, 0);
    buf
}

pub fn encode_sp_into(result: &SpResult, buf: &mut [f32], ch_offset: usize) {
    encode_sp_into_layout(result, buf, ch_offset, TILE_MAX, Some);
}

pub fn encode_sp_3p(result: &SpResult) -> Vec<f32> {
    const TILE_TYPES_3P: usize = 27;
    let mut buf = vec![0.0f32; SP_CHANNELS * TILE_TYPES_3P];
    encode_sp_3p_into(result, &mut buf, 0);
    buf
}

pub fn encode_sp_3p_into(result: &SpResult, buf: &mut [f32], ch_offset: usize) {
    const TILE_TYPES_3P: usize = 27;
    encode_sp_into_layout(result, buf, ch_offset, TILE_TYPES_3P, |tile| match tile {
        0 => Some(0),
        1..=7 => None,
        8..=33 => Some(tile - 7),
        _ => None,
    });
}

fn encode_sp_into_layout(
    result: &SpResult,
    buf: &mut [f32],
    ch_offset: usize,
    tile_types: usize,
    compact: impl Fn(usize) -> Option<usize> + Copy,
) {
    if buf.len() < (ch_offset + SP_CHANNELS) * tile_types {
        return;
    }

    let Some(best) = result.candidates.first() else {
        return;
    };

    let max_ev = best.exp_values[0].max(0.0);
    broadcast_layout(
        buf,
        ch_offset,
        0,
        (max_ev.min(100_000.0)) / 100_000.0,
        tile_types,
    );
    broadcast_layout(
        buf,
        ch_offset,
        1,
        (max_ev.min(30_000.0)) / 30_000.0,
        tile_types,
    );

    for candidate in &result.candidates {
        let discard = candidate.tile as usize;
        if discard >= TILE_MAX {
            continue;
        }
        for tile in 0..TILE_MAX {
            let Some(tile_column) = compact(tile) else {
                continue;
            };
            if candidate.required_tiles[tile] > 0.0 {
                set_layout(buf, ch_offset, 2 + discard, tile_column, 1.0, tile_types);
            }
            if candidate.yaku_progress_tiles[tile] > 0.0 {
                set_layout(
                    buf,
                    ch_offset,
                    2 + TILE_MAX + discard,
                    tile_column,
                    1.0,
                    tile_types,
                );
            }
        }
    }

    if let Some(best_required) = result.candidates.iter().max_by(|a, b| {
        a.num_required_tiles
            .partial_cmp(&b.num_required_tiles)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.tile.cmp(&a.tile))
    }) && let Some(tile) = compact(best_required.tile as usize)
    {
        set_layout(buf, ch_offset, 70, tile, 1.0, tile_types);
    }
    if let Some(best_yaku) = result.candidates.iter().max_by(|a, b| {
        a.num_yaku_progress_tiles
            .partial_cmp(&b.num_yaku_progress_tiles)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.tile.cmp(&a.tile))
    }) && best_yaku.num_yaku_progress_tiles > 0.0
        && let Some(tile) = compact(best_yaku.tile as usize)
    {
        set_layout(buf, ch_offset, 71, tile, 1.0, tile_types);
    }

    let ev_scale = if max_ev >= 1.0 { 1.0 / max_ev } else { 0.0 };
    // Channel base offsets after the 123-channel legacy block.
    const YAKU_MASK_BASE: usize = 72 + SP_MAX_TURNS * 3; // = 123
    const SCORING_BASE: usize = YAKU_MASK_BASE + yaku_mask_bits::N_BITS; // = 135
    const FUTURE_WAIT_BASE: usize = SCORING_BASE + 3; // = 138
    const TARGET_PROB_BASE: usize = FUTURE_WAIT_BASE + TILE_MAX; // = 172
    /// Normalisation for tsumo total points. Mangan ≈ 8000, kazoe yakuman ≈
    /// 32000; pick 100k as the same upper bound used by `max_ev` ch 0 so the
    /// scale matches across channels.
    const POINT_NORM: f32 = 100_000.0;
    for candidate in &result.candidates {
        let discard = candidate.tile as usize;
        if discard >= TILE_MAX {
            continue;
        }
        let Some(discard_column) = compact(discard) else {
            continue;
        };
        for turn in 0..SP_MAX_TURNS {
            set_layout(
                buf,
                ch_offset,
                72 + turn,
                discard_column,
                candidate.tenpai_probs[turn],
                tile_types,
            );
            set_layout(
                buf,
                ch_offset,
                72 + SP_MAX_TURNS + turn,
                discard_column,
                candidate.win_probs[turn],
                tile_types,
            );
            set_layout(
                buf,
                ch_offset,
                72 + SP_MAX_TURNS * 2 + turn,
                discard_column,
                (candidate.exp_values[turn] * ev_scale).clamp(0.0, 1.0),
                tile_types,
            );
        }

        // Yaku-mask per-discard one-hot. Bit b of `candidate.yaku_mask` →
        // channel YAKU_MASK_BASE + b at the discard-tile cell.
        for b in 0..yaku_mask_bits::N_BITS {
            if candidate.yaku_mask & (1u32 << b) != 0 {
                set_layout(
                    buf,
                    ch_offset,
                    YAKU_MASK_BASE + b,
                    discard_column,
                    1.0,
                    tile_types,
                );
            }
        }

        // Scoring stats per-discard: min/mean/max normalised by POINT_NORM.
        if candidate.mean_point > 0.0 || candidate.max_point > 0.0 {
            set_layout(
                buf,
                ch_offset,
                SCORING_BASE,
                discard_column,
                (candidate.min_point / POINT_NORM).clamp(0.0, 1.0),
                tile_types,
            );
            set_layout(
                buf,
                ch_offset,
                SCORING_BASE + 1,
                discard_column,
                (candidate.mean_point / POINT_NORM).clamp(0.0, 1.0),
                tile_types,
            );
            set_layout(
                buf,
                ch_offset,
                SCORING_BASE + 2,
                discard_column,
                (candidate.max_point / POINT_NORM).clamp(0.0, 1.0),
                tile_types,
            );
        }

        // Per-discard future wait map: channel `FUTURE_WAIT_BASE + discard`
        // has cell `tile` = 1 if drawing that tile after this discard is
        // potentially-effective. One-hot, not count-weighted, to match the
        // existing `required_tiles` channel encoding (ch 2..36).
        for tile in 0..TILE_MAX {
            if candidate.future_wait_tiles[tile] > 0.0
                && let Some(tile_column) = compact(tile)
            {
                set_layout(
                    buf,
                    ch_offset,
                    FUTURE_WAIT_BASE + discard,
                    tile_column,
                    1.0,
                    tile_types,
                );
            }
        }

        // Per-discard target-point achievement probabilities at the discard
        // cell. Each of `target_points::N` thresholds gets its own channel.
        for k in 0..target_points::N {
            let p = candidate.point_achievement_probs[k];
            if p > 0.0 {
                set_layout(
                    buf,
                    ch_offset,
                    TARGET_PROB_BASE + k,
                    discard_column,
                    p,
                    tile_types,
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ScoringSummary {
    wait_count: f32,
    /// Wait-count-weighted mean point (used by `series_for_candidate_approx`
    /// for shanten-down/high-shanten paths). Tenpai+optimal candidates do NOT
    /// consume this — `tenpai_series_from_waits` ignores it.
    mean_point: f32,
    /// Lowest score across waits (0 if none scored).
    min_point: f32,
    /// Highest score across waits.
    max_point: f32,
    /// OR of yaku flag bits achievable for SOME wait. See `yaku_mask_bits`.
    yaku_mask: u32,
    /// Wait-count-weighted accumulator: `point_buckets[k]` = sum of weights
    /// of scored waits whose tsumo total ≥ `target_points::TARGETS[k]`.
    /// Divide by `wait_count` to get achievement probability.
    point_buckets: [f32; target_points::N],
}

fn remaining_self_draws(obs: &Observation) -> u8 {
    let self_discards = obs.discards[obs.player_id as usize].len();
    17usize.saturating_sub(self_discards).min(SP_MAX_TURNS) as u8
}

fn remaining_self_draws_3p(obs: &Observation3P) -> u8 {
    let self_discards = obs.discards[obs.player_id as usize].len();
    17usize.saturating_sub(self_discards).min(SP_MAX_TURNS) as u8
}

#[cfg(test)]
fn remaining_counts(input: &SpInput) -> [u8; TILE_MAX] {
    remaining_counts_for_variant(input, SpVariant::FourPlayer)
}

fn remaining_counts_for_variant(input: &SpInput, variant: SpVariant) -> [u8; TILE_MAX] {
    let mut remaining = [0u8; TILE_MAX];
    for (tile, out) in remaining.iter_mut().enumerate() {
        *out = if variant == SpVariant::ThreePlayer && (1..=7).contains(&tile) {
            0
        } else {
            4u8.saturating_sub(input.tiles_seen[tile].min(4))
        };
    }
    remaining
}

#[cfg(test)]
fn shanten_of_counts(counts: &[u8; TILE_MAX]) -> i8 {
    shanten_of_counts_for_variant(counts, SpVariant::FourPlayer)
}

fn shanten_of_counts_for_variant(counts: &[u8; TILE_MAX], variant: SpVariant) -> i8 {
    let len_div3 = counts.iter().sum::<u8>() / 3;
    match variant {
        SpVariant::FourPlayer => shanten::calc_shanten_from_counts(counts, len_div3),
        SpVariant::ThreePlayer => shanten::calc_shanten_from_counts_3p(counts, len_div3),
    }
}

/// Incremental shanten: given pre-computed k0 bytes for the hand BEFORE the
/// `+1[tile]` perturbation, compute shanten of the perturbed hand by
/// re-hashing only the affected suit/honor slice. `next` already reflects the
/// perturbation. `len_div3` corresponds to `next` (= sum/3 of post-add counts).
#[inline]
fn shanten_after_add_incremental(
    next: &[u8; TILE_MAX],
    tile: usize,
    base_k0_m: u8,
    base_k0_p: u8,
    base_k0_s: u8,
    base_k0_z: u8,
    len_div3: u8,
    variant: SpVariant,
) -> i8 {
    if variant == SpVariant::ThreePlayer {
        return shanten::calc_shanten_from_counts_3p(next, len_div3);
    }
    let (km, kp, ks, kz) = if tile < 9 {
        (
            shanten::k0_shupai_for(&next[0..9]),
            base_k0_p,
            base_k0_s,
            base_k0_z,
        )
    } else if tile < 18 {
        (
            base_k0_m,
            shanten::k0_shupai_for(&next[9..18]),
            base_k0_s,
            base_k0_z,
        )
    } else if tile < 27 {
        (
            base_k0_m,
            base_k0_p,
            shanten::k0_shupai_for(&next[18..27]),
            base_k0_z,
        )
    } else {
        (
            base_k0_m,
            base_k0_p,
            base_k0_s,
            shanten::k0_zipai_for(&next[27..34]),
        )
    };
    let mut s = shanten::shanten_normal_from_k0s(km, kp, ks, kz, len_div3);
    if s > 0 && len_div3 >= 4 {
        s = s.min(shanten::shanten_chitoi_kokushi_floor(next));
    }
    s
}

fn required_tiles(
    dp: &mut DpContext<'_>,
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    current_shanten: i8,
) -> [f32; TILE_MAX] {
    let mut out = [0.0; TILE_MAX];
    if current_shanten < 0 {
        return out;
    }
    // Incremental shanten: cache base k0 bytes once, recompute only the
    // affected suit per tile (saves 3 hash_shupai/zipai per iteration).
    let base_k0_m = shanten::k0_shupai_for(&counts[0..9]);
    let base_k0_p = shanten::k0_shupai_for(&counts[9..18]);
    let base_k0_s = shanten::k0_shupai_for(&counts[18..27]);
    let base_k0_z = shanten::k0_zipai_for(&counts[27..34]);
    let len_div3_next = (counts.iter().sum::<u8>() + 1) / 3;
    let mut next = *counts;
    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }
        if !potentially_effective_for_draw(counts, tile) {
            continue;
        }
        next[tile] += 1;
        let s_after = shanten_after_add_incremental(
            &next,
            tile,
            base_k0_m,
            base_k0_p,
            base_k0_s,
            base_k0_z,
            len_div3_next,
            dp.variant,
        );
        next[tile] -= 1;
        if s_after < current_shanten {
            out[tile] = remaining[tile] as f32;
        }
    }
    out
}

/// Fused tenpai (`shanten == 0`) pass: in one 34-tile loop, compute
/// `required_tiles`, the `ScoringSummary`, and `yaku_progress_tiles`. All three
/// of these enumerate the same wait set and share the same `score_tsumo`
/// evaluations, so doing them separately costs ~3× the wall-time at tenpai
/// (which is the hottest per-discard case in real play). We use the cached
/// `dp.shanten` / `dp.score_tsumo` paths so the inner shanten check and the
/// per-wait point lookup are both hash-amortized.
fn fused_tenpai_pass(
    dp: &mut DpContext<'_>,
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
) -> ([f32; TILE_MAX], ScoringSummary, [f32; TILE_MAX]) {
    let mut required = [0.0f32; TILE_MAX];
    let mut yaku_progress = [0.0f32; TILE_MAX];
    let mut scoring = ScoringSummary::default();
    // Riichi guarantees a yaku, so under the assumed-riichi path we can skip
    // the per-wait `score_tsumo` lookup entirely (yaku_progress = required).
    // Mean point is also dead computation at tenpai+optimal — the consumer
    // (`tenpai_series_from_waits`) computes its own per-wait score vector.
    let assume_riichi = dp.input.is_menzen && dp.input.can_riichi;

    // Hand-level structural yaku flags (computed once over `counts + melds`)
    // so each wait tile becomes a constant-time post-win shape check instead
    // of a `score_tsumo` invocation. Catches tanyao / honitsu / chinitsu /
    // yakuhai-in-meld / yakuhai-by-completion — which together cover the
    // majority of open-hand tenpai candidates.
    let bakaze_yh = norm_wind_for_yakuhai(dp.input.bakaze);
    let jikaze_yh = norm_wind_for_yakuhai(dp.input.jikaze);
    let mut yh = [0u8; 5];
    let mut n_yh = 3usize;
    yh[0] = 31;
    yh[1] = 32;
    yh[2] = 33;
    if (27..=30).contains(&bakaze_yh) {
        yh[n_yh] = bakaze_yh;
        n_yh += 1;
    }
    if (27..=30).contains(&jikaze_yh) && jikaze_yh != bakaze_yh {
        yh[n_yh] = jikaze_yh;
        n_yh += 1;
    }
    let yakuhai_in_meld = !assume_riichi
        && dp.input.melds.iter().any(|m| {
            m.tiles.iter().any(|&t| {
                let tt = t / 4;
                yh[..n_yh].contains(&tt)
            })
        });
    // In-hand yakuhai kotsu in pre-discard 13-tile hand: any wait gives yaku.
    let yakuhai_kotsu_pre = !assume_riichi && yh[..n_yh].iter().any(|&y| counts[y as usize] >= 3);

    // Decompose yakuhai by class (dragon/round/seat) for yaku-mask reporting.
    // A class is "wait-invariant present" if any meld is that class OR any
    // count_pre[that_tile] ≥ 3.
    let yakuhai_dragon_pre = matches!(counts[31], 3..)
        || matches!(counts[32], 3..)
        || matches!(counts[33], 3..)
        || dp
            .input
            .melds
            .iter()
            .any(|m| m.tiles.iter().any(|&t| matches!(t / 4, 31 | 32 | 33)));
    let yakuhai_round_pre = (27..=30).contains(&bakaze_yh)
        && (counts[bakaze_yh as usize] >= 3
            || dp
                .input
                .melds
                .iter()
                .any(|m| m.tiles.iter().any(|&t| t / 4 == bakaze_yh)));
    let yakuhai_seat_pre = (27..=30).contains(&jikaze_yh)
        && (counts[jikaze_yh as usize] >= 3
            || dp
                .input
                .melds
                .iter()
                .any(|m| m.tiles.iter().any(|&t| t / 4 == jikaze_yh)));

    // Shousangen pre-check: 2 dragon kotsu + 1 dragon pair in (counts + melds).
    // Wait-invariant.
    let mut dragon_kotsu_count = 0u8;
    let mut dragon_pair_present = false;
    for d in [31u8, 32, 33] {
        let mut c = counts[d as usize];
        for m in &dp.input.melds {
            for &t in &m.tiles {
                if t / 4 == d {
                    c = c.saturating_add(1);
                }
            }
        }
        if c >= 3 {
            dragon_kotsu_count += 1;
        } else if c == 2 {
            dragon_pair_present = true;
        }
    }
    let shousangen_pre = dragon_kotsu_count >= 2 && dragon_pair_present;

    // counts_plus_melds: pre-win 13-tile + melds. Post-win shape = + wait tile.
    let mut full_pre = *counts;
    for meld in &dp.input.melds {
        for &t136 in &meld.tiles {
            let tt = (t136 / 4) as usize;
            if tt < TILE_MAX {
                full_pre[tt] = full_pre[tt].saturating_add(1);
            }
        }
    }
    let mut suits_pre = 0u8;
    let mut has_yaocchi_pre = false;
    let mut has_z_pre = false;
    let mut has_simple_pre = false;
    let mut has_singleton_pre = false;
    for t in 0..27usize {
        let c = full_pre[t];
        if c > 0 {
            suits_pre |= 1 << (t / 9);
            if t % 9 == 0 || t % 9 == 8 {
                has_yaocchi_pre = true;
            } else {
                has_simple_pre = true;
            }
            if c == 1 {
                has_singleton_pre = true;
            }
        }
    }
    for t in 27..34usize {
        let c = full_pre[t];
        if c > 0 {
            has_z_pre = true;
            has_yaocchi_pre = true;
            if c == 1 {
                has_singleton_pre = true;
            }
        }
    }
    let n_suits_pre = suits_pre.count_ones();
    let single_suit_pre = n_suits_pre <= 1;
    let main_suit_pre = if n_suits_pre == 1 {
        Some(suits_pre.trailing_zeros() as u8)
    } else {
        None
    };
    // Sanshoku doukou pre-check: 3 kotsu of same number (n) across 3 suits in
    // the 13-tile (or with a wait that completes an existing pair).
    let mut sanshoku_doukou_pre_full = false;
    for n in 0..9usize {
        if full_pre[n] >= 3 && full_pre[n + 9] >= 3 && full_pre[n + 18] >= 3 {
            sanshoku_doukou_pre_full = true;
            break;
        }
    }

    // Pre-compute base k0 bytes for the 13-tile `counts` hand. The inner
    // shanten loop perturbs 1 suit at a time, so cached k0s for the other
    // 3 suits avoid 3 redundant hash_shupai/zipai calls per iteration.
    let base_k0_m = crate::shanten::k0_shupai_for(&counts[0..9]);
    let base_k0_p = crate::shanten::k0_shupai_for(&counts[9..18]);
    let base_k0_s = crate::shanten::k0_shupai_for(&counts[18..27]);
    let base_k0_z = crate::shanten::k0_zipai_for(&counts[27..34]);
    // len_div3 of the post-add hand (sum + 1) / 3.
    let len_div3_next = (counts.iter().sum::<u8>() + 1) / 3;

    let mut next = *counts;
    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }
        // A tile completes the 13-tile hand only if it sits adjacent to or
        // matches an existing tile. Skip the (cached) shanten lookup entirely
        // for isolated tiles.
        if !potentially_effective_for_draw(counts, tile) {
            continue;
        }
        next[tile] += 1;
        let s_after = shanten_after_add_incremental(
            &next,
            tile,
            base_k0_m,
            base_k0_p,
            base_k0_s,
            base_k0_z,
            len_div3_next,
            dp.variant,
        );
        next[tile] -= 1;
        if s_after >= 0 {
            // Not a wait — drawing this tile does not complete the hand.
            continue;
        }
        required[tile] = remaining[tile] as f32;

        // Compute per-wait structural yaku flags (these are also the gating
        // conditions for the "skip score_tsumo" short-circuit). Then, for
        // ANY wait that has yaku, OR the corresponding mask bits into
        // `scoring.yaku_mask`, and update min/mean/max via score_tsumo.
        let tile_u = tile as u8;
        let tile_is_yaocchi = if tile < 27 {
            matches!(tile % 9, 0 | 8)
        } else {
            true
        };
        let tile_is_honor = tile >= 27;
        let tile_suit = if tile < 27 {
            Some((tile / 9) as u8)
        } else {
            None
        };
        let post_tanyao = !has_yaocchi_pre && !tile_is_yaocchi;
        let post_chinitsu = single_suit_pre
            && !has_z_pre
            && match main_suit_pre {
                Some(s) => tile_suit == Some(s),
                None => false,
            };
        let post_honitsu = single_suit_pre
            && has_z_pre
            && match main_suit_pre {
                Some(s) => tile_suit == Some(s) || tile_is_honor,
                None => tile_is_honor,
            };
        let post_single_suit = post_chinitsu || post_honitsu;
        let post_honroutou = !has_simple_pre && tile_is_yaocchi;
        let post_toitoi = !has_singleton_pre && counts[tile] == 2;

        // Per-wait yakuhai-completion: wait IS a yakuhai tile, and counts
        // already had 2 of it (so it completes the kotsu).
        let waits_complete_dragon = matches!(tile_u, 31 | 32 | 33) && counts[tile] == 2;
        let waits_complete_round =
            (27..=30).contains(&bakaze_yh) && tile_u == bakaze_yh && counts[tile] == 2;
        let waits_complete_seat =
            (27..=30).contains(&jikaze_yh) && tile_u == jikaze_yh && counts[tile] == 2;

        let post_yakuhai_completion =
            waits_complete_dragon || waits_complete_round || waits_complete_seat;

        let has_struct_yaku = yakuhai_in_meld
            || yakuhai_kotsu_pre
            || sanshoku_doukou_pre_full
            || shousangen_pre
            || post_tanyao
            || post_single_suit
            || post_honroutou
            || post_toitoi
            || post_yakuhai_completion;

        // Update scoring (min/mean/max + yaku_mask) for waits that have yaku.
        // `assume_riichi` always has yaku via riichi itself; `has_struct_yaku`
        // identifies the per-wait structural source. For all other cases,
        // fall through to score_tsumo to discover lean-path yaku (sanshoku
        // doujun / ittsu / sanankou / pinfu / iipeikou / etc.).
        let wait_has_yaku = assume_riichi || has_struct_yaku;
        if wait_has_yaku {
            yaku_progress[tile] = remaining[tile] as f32;
            // Mask: union of applicable yaku for THIS wait.
            let mut mask: u32 = 0;
            if assume_riichi {
                mask |= yaku_mask_bits::RIICHI;
            }
            if dp.input.is_menzen {
                mask |= yaku_mask_bits::MENZEN_TSUMO;
            }
            if post_tanyao {
                mask |= yaku_mask_bits::TANYAO;
            }
            if yakuhai_dragon_pre || waits_complete_dragon {
                mask |= yaku_mask_bits::YAKUHAI_DRAGON;
            }
            if yakuhai_round_pre || waits_complete_round {
                mask |= yaku_mask_bits::YAKUHAI_ROUND;
            }
            if yakuhai_seat_pre || waits_complete_seat {
                mask |= yaku_mask_bits::YAKUHAI_SEAT;
            }
            if post_chinitsu {
                mask |= yaku_mask_bits::CHINITSU;
            }
            if post_honitsu {
                mask |= yaku_mask_bits::HONITSU;
            }
            if post_honroutou {
                mask |= yaku_mask_bits::HONROUTOU;
            }
            if post_toitoi {
                mask |= yaku_mask_bits::TOITOI;
            }
            if sanshoku_doukou_pre_full {
                mask |= yaku_mask_bits::SANSHOKU_DOUKOU;
            }
            if shousangen_pre {
                mask |= yaku_mask_bits::SHOUSANGEN;
            }
            scoring.yaku_mask |= mask;

            // Min/mean/max point: score this wait via the lean tsumo path.
            // Even under riichi (where we skipped this before), the
            // distribution per wait is needed for the new min/max channels.
            if let Some(point) = dp.score_tsumo(counts, remaining, tile_u, ScoreMods::default()) {
                merge_point(&mut scoring, point, remaining[tile] as f32);
            }
            continue;
        }

        // No structural yaku → fall back to per-wait score_tsumo. If it
        // succeeds, the lean path detected a decomp-dependent yaku
        // (sanshoku doujun / ittsu / sanankou / etc.). We can't easily
        // categorize which without re-running the lean path, so leave the
        // mask bits unset for this wait; the channel will still be 0 if
        // no other wait sets them, and the network can fall back on
        // `yaku_progress_tiles` (the binary "any yaku" channel) for that
        // info.
        if let Some(point) = dp.score_tsumo(counts, remaining, tile_u, ScoreMods::default()) {
            yaku_progress[tile] = remaining[tile] as f32;
            merge_point(&mut scoring, point, remaining[tile] as f32);
        }
    }
    (required, scoring, yaku_progress)
}

fn yaku_progress_tiles(
    dp: &mut DpContext<'_>,
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    current_shanten: i8,
) -> [f32; TILE_MAX] {
    let mut out = [0.0; TILE_MAX];
    if current_shanten > 1 {
        return out;
    }

    let assume_riichi = dp.input.is_menzen && dp.input.can_riichi;

    // Incremental shanten state for the post-discard 13-tile.
    let base_k0_m = shanten::k0_shupai_for(&counts[0..9]);
    let base_k0_p = shanten::k0_shupai_for(&counts[9..18]);
    let base_k0_s = shanten::k0_shupai_for(&counts[18..27]);
    let base_k0_z = shanten::k0_zipai_for(&counts[27..34]);
    let len_div3_next = (counts.iter().sum::<u8>() + 1) / 3;
    let mut drawn = *counts;

    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }
        if !potentially_effective_for_draw(counts, tile) {
            continue;
        }

        if current_shanten == 0 {
            if dp
                .score_tsumo(counts, remaining, tile as u8, ScoreMods::default())
                .is_some()
            {
                out[tile] = remaining[tile] as f32;
            }
            continue;
        }

        drawn[tile] += 1;
        let s_drawn = shanten_after_add_incremental(
            &drawn,
            tile,
            base_k0_m,
            base_k0_p,
            base_k0_s,
            base_k0_z,
            len_div3_next,
            dp.variant,
        );
        if s_drawn >= current_shanten {
            drawn[tile] -= 1;
            continue;
        }
        // current_shanten == 1, drawn shanten ≤ 0 (= 14-tile is tenpai or agari).
        // Under assume_riichi, the inner has_yaku_tenpai_after_best_discard call
        // is logically equivalent to `shanten(drawn) == 0` (already known true):
        // by min monotonicity, shanten(14) = min over discards of shanten(13'),
        // so shanten(drawn) ≤ 0 implies some discard yields a 13' at shanten 0.
        // Riichi guarantees yaku, so we count this wait without paying the
        // 14-tile shanten sweep + cache lookup.
        if assume_riichi {
            out[tile] = remaining[tile] as f32;
            drawn[tile] -= 1;
            continue;
        }
        let mut next_remaining = *remaining;
        next_remaining[tile] -= 1;
        let yaku_present = has_yaku_tenpai_after_best_discard(dp, &drawn, &next_remaining);
        drawn[tile] -= 1;
        if yaku_present {
            out[tile] = remaining[tile] as f32;
        }
    }
    out
}

fn score_waits(
    dp: &mut DpContext<'_>,
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
) -> ScoringSummary {
    let mut scoring = ScoringSummary::default();
    if dp.shanten(counts) != 0 {
        return scoring;
    }

    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }
        if !potentially_effective_for_draw(counts, tile) {
            continue;
        }
        if let Some(point) = dp.score_tsumo(counts, remaining, tile as u8, ScoreMods::default()) {
            let weight = remaining[tile] as f32;
            merge_point(&mut scoring, point, weight);
        }
    }
    scoring
}

fn merge_point(scoring: &mut ScoringSummary, point: f32, weight: f32) {
    let old_weighted_sum = scoring.mean_point * scoring.wait_count;
    let was_empty = scoring.wait_count == 0.0;
    scoring.wait_count += weight;
    if scoring.wait_count > 0.0 {
        scoring.mean_point = (old_weighted_sum + point * weight) / scoring.wait_count;
    }
    if was_empty || point < scoring.min_point {
        scoring.min_point = point;
    }
    if point > scoring.max_point {
        scoring.max_point = point;
    }
    // Per-target weight accumulator. The targets are sorted ascending, so we
    // can break early at the first miss.
    for (k, &threshold) in target_points::TARGETS.iter().enumerate() {
        if point >= threshold {
            scoring.point_buckets[k] += weight;
        } else {
            break;
        }
    }
}

fn has_yaku_tenpai_after_best_discard(
    dp: &mut DpContext<'_>,
    counts_14: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
) -> bool {
    // ── Hot fast path: pre-call base flags imply yaku ────────────────────
    // These checks are cheap (no full-array build, no cache hash) and catch
    // the vast majority of open-hand SP samples (tanyao, single-suit,
    // yakuhai-in-meld, and the all-yaocchi shape). Only when none fire do we
    // pay the YakuTenpaiKey hashing + per-tenpai-count enumeration.
    if dp.base_full_yakuhai_in_meld {
        return true;
    }
    // Tanyao on the FINAL 14-tile = (counts_14 + melds) all non-yaocchi.
    // counts_14 = base_tehai - outer_discard + wait. Both base and melds
    // contribute. If base_full has no yaocchi (BASE_NO_YAOCCHI), the only
    // way counts_14+melds gains a yaocchi is via the wait tile. So:
    //   tanyao_at_counts_14 == counts_14 has no yaocchi (since base+melds is
    //   yaocchi-free, counts_14+melds is yaocchi-free iff counts_14 is).
    if dp.base_full_no_yaocchi {
        let mut yaocchi = false;
        for t in [0usize, 8, 9, 17, 18, 26] {
            if counts_14[t] > 0 {
                yaocchi = true;
                break;
            }
        }
        if !yaocchi {
            for t in 27..34usize {
                if counts_14[t] > 0 {
                    yaocchi = true;
                    break;
                }
            }
        }
        if !yaocchi {
            return true;
        }
    }
    // Honitsu/Chinitsu: base_full is single-suit. counts_14+melds preserves
    // single-suit iff counts_14 only contains tiles in that suit (or honors
    // for honitsu: base_full has honors).
    if let Some(s) = dp.base_full_single_suit {
        let suit_base = (s as usize) * 9;
        let mut outside_suit = false;
        for t in 0..27usize {
            if counts_14[t] > 0 && (t < suit_base || t >= suit_base + 9) {
                outside_suit = true;
                break;
            }
        }
        if !outside_suit {
            // Honors are allowed only if base_full has honors (= honitsu).
            // If base_full has no honors (= chinitsu), counts_14 must also
            // have no honors. Either way, accept.
            let counts_has_z = (27..34).any(|t| counts_14[t] > 0);
            if dp.base_full_has_z || !counts_has_z {
                return true;
            }
        }
    }
    // In-hand yakuhai kotsu in counts_14 (≥3 of any yakuhai tile).
    for i in 0..dp.base_n_yakuhai {
        if counts_14[dp.base_yakuhai[i] as usize] >= 3 {
            return true;
        }
    }

    let key = YakuTenpaiKey {
        counts: *counts_14,
        remaining: *remaining,
        elapsed_turns: 0,
        turns_left: 0,
    };
    if let Some(&cached) = dp.yaku_tenpai_cache.get(&key) {
        return cached;
    }

    // Riichi guarantees yaku, so if we can riichi we only need to confirm
    // that *some* discard reaches shanten==0. Skip the per-wait score_tsumo
    // loop entirely and short-circuit on the first tenpai-reaching discard.
    // This dominates the tenpai SP cost in our profile (96% of 77μs/sample
    // when most discards drop to shanten=1) for menzen+can_riichi inputs.
    let assume_riichi = dp.input.is_menzen && dp.input.can_riichi;
    if assume_riichi {
        // Per-suit incremental shanten: drop 1 tile per iteration → only that
        // suit's k0 needs refresh. Saves 3 hash_shupai/zipai per shanten check.
        let r_base_k0_m = crate::shanten::k0_shupai_for(&counts_14[0..9]);
        let r_base_k0_p = crate::shanten::k0_shupai_for(&counts_14[9..18]);
        let r_base_k0_s = crate::shanten::k0_shupai_for(&counts_14[18..27]);
        let r_base_k0_z = crate::shanten::k0_zipai_for(&counts_14[27..34]);
        let r_len_div3 = counts_14.iter().sum::<u8>().saturating_sub(1) / 3;
        let mut reaches_tenpai = false;
        let mut next = *counts_14;
        for discard in 0..TILE_MAX {
            if counts_14[discard] == 0 {
                continue;
            }
            next[discard] -= 1;
            let s = shanten_after_add_incremental(
                &next,
                discard,
                r_base_k0_m,
                r_base_k0_p,
                r_base_k0_s,
                r_base_k0_z,
                r_len_div3,
                dp.variant,
            );
            next[discard] += 1;
            if s == 0 {
                reaches_tenpai = true;
                break;
            }
        }
        dp.yaku_tenpai_cache.insert(key, reaches_tenpai);
        return reaches_tenpai;
    }

    // ── Structural shape-only short-circuit (non-riichi) ────────────────
    // These yaku are determined entirely by the *full* 14-tile hand shape
    // (counts_14 + meld tiles) and are invariant to which tile is discarded
    // best-shanten-wise:
    //   - yakuhai-in-meld (pon/kan of dragon/seat/round wind)
    //   - tanyao (full has no yaocchi)
    //   - honitsu / chinitsu (full uses only 1 numbered suit)
    //   - honroutou (full all yaocchi → kotsu-only decomp guaranteed)
    //   - toitoi (no count == 1 in full → 4 kotsu + 1 pair only shape)
    //   - sanshoku doukou (some n with counts[n]≥3, counts[n+9]≥3, counts[n+18]≥3)
    //   - in-hand yakuhai kotsu (any yakuhai tile at counts ≥ 3 in counts_14)
    // Any of these → yaku guaranteed without per-wait score_tsumo.
    {
        let bakaze = norm_wind_for_yakuhai(dp.input.bakaze);
        let jikaze = norm_wind_for_yakuhai(dp.input.jikaze);
        let mut yh = [31u8, 32, 33, 0, 0];
        let mut n_yh = 3usize;
        if (27..=30).contains(&bakaze) {
            yh[n_yh] = bakaze;
            n_yh += 1;
        }
        if (27..=30).contains(&jikaze) && jikaze != bakaze {
            yh[n_yh] = jikaze;
            n_yh += 1;
        }
        let yakuhai_in_meld = dp
            .input
            .melds
            .iter()
            .any(|m| m.tiles.iter().any(|&t| yh[..n_yh].contains(&(t / 4))));
        if yakuhai_in_meld {
            dp.yaku_tenpai_cache.insert(key, true);
            return true;
        }
        // In-hand yakuhai kotsu (≥3 in counts_14 — a kotsu survives at least
        // one discard if there are still ≥2 left after).
        for i in 0..n_yh {
            if counts_14[yh[i] as usize] >= 3 {
                dp.yaku_tenpai_cache.insert(key, true);
                return true;
            }
        }
        let mut full = *counts_14;
        for meld in &dp.input.melds {
            for &t136 in &meld.tiles {
                let tt = (t136 / 4) as usize;
                if tt < TILE_MAX {
                    full[tt] = full[tt].saturating_add(1);
                }
            }
        }
        let mut has_terminal = false;
        let mut has_simple = false;
        let mut suits_present = 0u8;
        let mut any_singleton = false; // ≥1 tile at exactly count=1
        for tile in 0..27usize {
            let c = full[tile];
            if c > 0 {
                suits_present |= 1 << (tile / 9);
                if tile % 9 == 0 || tile % 9 == 8 {
                    has_terminal = true;
                } else {
                    has_simple = true;
                }
                if c == 1 {
                    any_singleton = true;
                }
            }
        }
        for tile in 27..34usize {
            let c = full[tile];
            if c > 0 {
                has_terminal = true;
                if c == 1 {
                    any_singleton = true;
                }
            }
        }
        let n_numbered = suits_present.count_ones();
        let tanyao = !has_terminal;
        let single_suit = n_numbered <= 1;
        let all_yaocchi = !has_simple;
        // Toitoi: shape allows only kotsu+pair (no shuntsu possible since
        // shuntsu requires 3 different adjacent tiles, each at count ≥ 1 — so
        // a singleton is required for any shuntsu). No singleton ⇒ toitoi.
        let toitoi = !any_singleton;
        // Sanshoku doukou: ≥3 of some same number across all 3 suits.
        let mut sanshoku_doukou = false;
        for n in 0..9usize {
            if full[n] >= 3 && full[n + 9] >= 3 && full[n + 18] >= 3 {
                sanshoku_doukou = true;
                break;
            }
        }
        if tanyao || single_suit || all_yaocchi || toitoi || sanshoku_doukou {
            dp.yaku_tenpai_cache.insert(key, true);
            return true;
        }

        // ── Yaku-impossibility check ────────────────────────────────────
        // None of the immediate-fire yaku above applied. Now check whether
        // ANY yaku could *possibly* fire under (best_discard, some wait).
        // If not, return false directly — no Pass 1 / Pass 2 enumeration.
        // This catches "no-yaku" hands (the dominant cost in open+other:
        // hands with floating yaocchi that can't be reduced via 1-discard).
        //
        // For "yaku-possible" we need ≥ 1 of:
        //   - tanyao_via_drop: yaocchi count in `full` ≤ 1 (= 1-discard
        //     can clear all yaocchi).
        //   - single-suit-via-drop: ≤ 1 numbered suit OR (2 suits + secondary
        //     count ≤ 1 → can drop the secondary's 1 tile).
        //   - toitoi-via-drop: singletons ≤ 1.
        //   - honroutou-via-drop: simples ≤ 1.
        //   - yakuhai_meld (already checked above).
        //   - in-hand yakuhai count ≥ 2 (= shanpon-completion possible).
        //   - sanshoku_doukou-via-add: some n with full[n], full[n+9],
        //     full[n+18] all ≥ 2.
        //   - sanshoku_doujun-via-add: some n with shuntsu candidates in
        //     all 3 suits (each tile in (n, n+1, n+2) ≥ 1 in each suit).
        //   - ittsu-via-add: some suit covers 1-9 with ≥ 1 each (or ≥ 0
        //     with 1 missing tile fillable from remaining).
        //   - sanankou-via-add: ≥ 2 kotsu in counts_14 (third can be added).
        //   - junchan/chanta-via-shape: no "middle simple" (= no full[t]>0
        //     for t with t<27 and t%9 ∈ {3,4,5}).
        //
        // We compute these from the same `full` we already built.
        let mut yaocchi_count_full = 0u8;
        let mut simple_count = 0u8;
        let mut singleton_total = 0u8;
        let mut middle_simple = false;
        let mut secondary_suit_count = 0u8;
        let mut suit_counts = [0u8; 3];
        for t in 0..27usize {
            let c = full[t];
            if c == 0 {
                continue;
            }
            suit_counts[t / 9] = suit_counts[t / 9].saturating_add(c);
            if t % 9 == 0 || t % 9 == 8 {
                yaocchi_count_full = yaocchi_count_full.saturating_add(c);
            } else {
                simple_count = simple_count.saturating_add(c);
                if (t % 9) >= 3 && (t % 9) <= 5 {
                    middle_simple = true;
                }
            }
            if c == 1 {
                singleton_total += 1;
            }
        }
        for t in 27..34usize {
            let c = full[t];
            if c == 0 {
                continue;
            }
            yaocchi_count_full = yaocchi_count_full.saturating_add(c);
            if c == 1 {
                singleton_total += 1;
            }
        }
        // Smallest-non-zero numbered-suit count (= "secondary" after primary).
        let mut sorted_suits = suit_counts;
        sorted_suits.sort_unstable_by(|a, b| b.cmp(a));
        if sorted_suits[1] > 0 {
            secondary_suit_count = sorted_suits[1];
        }

        let tanyao_via_drop = yaocchi_count_full <= 1;
        let single_suit_via_drop = secondary_suit_count <= 1; // primary + ≤1 secondary
        let toitoi_via_drop = singleton_total <= 1;
        let honroutou_via_drop = simple_count <= 1;

        let mut yakuhai_count_in_counts = 0u8;
        for i in 0..n_yh {
            yakuhai_count_in_counts =
                yakuhai_count_in_counts.saturating_add(counts_14[yh[i] as usize]);
        }
        let yakuhai_possible = yakuhai_count_in_counts >= 2;

        let mut sanshoku_doukou_via_add = false;
        for n in 0..9usize {
            if full[n] >= 2 && full[n + 9] >= 2 && full[n + 18] >= 2 {
                sanshoku_doukou_via_add = true;
                break;
            }
        }
        let mut sanshoku_doujun_via_add = false;
        for n in 0..7usize {
            let m_ok = full[n] >= 1 && full[n + 1] >= 1 && full[n + 2] >= 1;
            let p_ok = full[n + 9] >= 1 && full[n + 10] >= 1 && full[n + 11] >= 1;
            let s_ok = full[n + 18] >= 1 && full[n + 19] >= 1 && full[n + 20] >= 1;
            if m_ok && p_ok && s_ok {
                sanshoku_doujun_via_add = true;
                break;
            }
        }
        let ittsu_via_add = (0..9).all(|n| full[n] >= 1)
            || (0..9).all(|n| full[n + 9] >= 1)
            || (0..9).all(|n| full[n + 18] >= 1);

        let mut kotsu_in_hand = 0u8;
        for t in 0..TILE_MAX {
            if counts_14[t] >= 3 {
                kotsu_in_hand += 1;
            }
        }
        let sanankou_via_add = kotsu_in_hand >= 2;

        let junchan_chanta_via_shape = !middle_simple;

        let any_yaku_possible = tanyao_via_drop
            || single_suit_via_drop
            || toitoi_via_drop
            || honroutou_via_drop
            || yakuhai_possible
            || sanshoku_doukou_via_add
            || sanshoku_doujun_via_add
            || ittsu_via_add
            || sanankou_via_add
            || junchan_chanta_via_shape;

        if !any_yaku_possible {
            // Provably no yaku for any (best_discard, wait) — skip enumeration.
            dp.yaku_tenpai_cache.insert(key, false);
            return false;
        }
    }

    // Non-riichi path: pick the best (= lowest-shanten) discard, then for each
    // tenpai-shaped result check if some wait tile yields a yaku.
    // Stack-allocated `tenpai_counts` (max 14 distinct discards keep the hand
    // tenpai, so 14 slots suffice) avoids the per-call heap allocation.
    let mut best_shanten = i8::MAX;
    let mut tenpai_counts: [[u8; TILE_MAX]; 14] = [[0u8; TILE_MAX]; 14];
    let mut tenpai_discards: [u8; 14] = [0u8; 14];
    let mut n_tenpai = 0usize;
    let mut next = *counts_14;

    // Per-suit incremental shanten state for the tenpai_counts collection.
    // Each iteration drops 1 tile from counts_14 → only that suit's k0 changes.
    let coll_base_k0_m = crate::shanten::k0_shupai_for(&counts_14[0..9]);
    let coll_base_k0_p = crate::shanten::k0_shupai_for(&counts_14[9..18]);
    let coll_base_k0_s = crate::shanten::k0_shupai_for(&counts_14[18..27]);
    let coll_base_k0_z = crate::shanten::k0_zipai_for(&counts_14[27..34]);
    let coll_len_div3_drop = counts_14.iter().sum::<u8>().saturating_sub(1) / 3;
    // Iterate yaocchi tiles first so the (likely tanyao-yielding) yaocchi-
    // discard tenpai_count appears at low indices in `tenpai_counts`. This
    // lets Pass 1 fire structural-tanyao on the first iteration for tanyao-
    // bias hands where the only yaocchi was the just-drawn wait or a single
    // floating yaocchi in the original tehai.
    const TILE_ORDER_YAOCCHI_FIRST: [u8; 34] = [
        // Yaocchi: 6 terminals + 7 honors
        0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, // Then non-yaocchi: 21 simples
        1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25,
    ];
    for &discard in &TILE_ORDER_YAOCCHI_FIRST {
        let di = discard as usize;
        if counts_14[di] == 0 {
            continue;
        }
        next[di] -= 1;
        let s = shanten_after_add_incremental(
            &next,
            di,
            coll_base_k0_m,
            coll_base_k0_p,
            coll_base_k0_s,
            coll_base_k0_z,
            coll_len_div3_drop,
            dp.variant,
        );
        if s < best_shanten {
            best_shanten = s;
            n_tenpai = 0;
        }
        if s == best_shanten && s == 0 && n_tenpai < tenpai_counts.len() {
            tenpai_counts[n_tenpai] = next;
            tenpai_discards[n_tenpai] = discard;
            n_tenpai += 1;
        }
        next[di] += 1;
    }

    // Per-tenpai_count structural+yakuhai short-circuits. Each tenpai_count is
    // a 13-tile hand (post-best-discard from drawn). We refresh structural
    // flags per tenpai_count since discarding a tile can flip them (e.g.,
    // dropping the only yaocchi turns a non-tanyao 14-tile into tanyao).
    let bakaze = norm_wind_for_yakuhai(dp.input.bakaze);
    let jikaze = norm_wind_for_yakuhai(dp.input.jikaze);
    let mut yakuhai_tiles = [31u8, 32, 33, 0, 0];
    let mut n_yh = 3usize;
    if (27..=30).contains(&bakaze) {
        yakuhai_tiles[n_yh] = bakaze;
        n_yh += 1;
    }
    if (27..=30).contains(&jikaze) && jikaze != bakaze {
        yakuhai_tiles[n_yh] = jikaze;
        n_yh += 1;
    }

    // Build the meld portion of `full` once (it's invariant across tenpai_counts).
    let mut meld_full = [0u8; TILE_MAX];
    for meld in &dp.input.melds {
        for &t136 in &meld.tiles {
            let tt = (t136 / 4) as usize;
            if tt < TILE_MAX {
                meld_full[tt] = meld_full[tt].saturating_add(1);
            }
        }
    }

    // Pre-compute full14 (= counts_14 + melds) features ONCE. Per-tenpai_count
    // features are then derived in O(1) by adjusting for the single tile_X
    // that was discarded — instead of re-scanning 34 tiles per tenpai_count.
    // Identifies: per-suit count, yaocchi/simple totals, singleton count, etc.
    let mut full14 = *counts_14;
    for t in 0..TILE_MAX {
        full14[t] = full14[t].saturating_add(meld_full[t]);
    }
    let mut full14_suit_counts = [0u8; 3];
    let mut full14_z_count = 0u8;
    let mut full14_yaocchi = 0u8;
    let mut full14_simple = 0u8;
    let mut full14_singletons = 0u8;
    for t in 0..27usize {
        let c = full14[t];
        if c == 0 {
            continue;
        }
        full14_suit_counts[t / 9] = full14_suit_counts[t / 9].saturating_add(c);
        if t % 9 == 0 || t % 9 == 8 {
            full14_yaocchi = full14_yaocchi.saturating_add(c);
        } else {
            full14_simple = full14_simple.saturating_add(c);
        }
        if c == 1 {
            full14_singletons += 1;
        }
    }
    for t in 27..34usize {
        let c = full14[t];
        if c == 0 {
            continue;
        }
        full14_z_count = full14_z_count.saturating_add(c);
        full14_yaocchi = full14_yaocchi.saturating_add(c);
        if c == 1 {
            full14_singletons += 1;
        }
    }

    // Pass 1: cheap structural shape checks across ALL tenpai_counts.
    // Per-tenpai_count features are O(1) deltas off full14's pre-computed flags.
    let mut struct_yaku_found = false;
    'pass1: for k in 0..n_tenpai {
        let counts = &tenpai_counts[k];
        let tile_x = tenpai_discards[k] as usize;
        let cx = full14[tile_x]; // count of tile_X in full14
        debug_assert!(cx >= 1);

        // Hand-side yakuhai (kotsu in 13-tile = counts ≥ 3): wait-invariant.
        for i in 0..n_yh {
            if counts[yakuhai_tiles[i] as usize] >= 3 {
                struct_yaku_found = true;
                break 'pass1;
            }
        }

        // Derive full13's structural flags from full14 by removing 1 of tile_X.
        let tile_x_yaocchi = if tile_x < 27 {
            tile_x % 9 == 0 || tile_x % 9 == 8
        } else {
            true
        };
        let tile_x_simple = !tile_x_yaocchi;
        let tile_x_suit = if tile_x < 27 {
            Some((tile_x / 9) as u8)
        } else {
            None
        };

        let yaocchi13 = if tile_x_yaocchi {
            full14_yaocchi - 1
        } else {
            full14_yaocchi
        };
        let simple13 = if tile_x_simple {
            full14_simple - 1
        } else {
            full14_simple
        };
        // Singletons: cx==1 → singleton drops to 0 (-1). cx==2 → was pair, now singleton (+1).
        let singletons13 = if cx == 1 {
            full14_singletons - 1
        } else if cx == 2 {
            full14_singletons + 1
        } else {
            full14_singletons
        };
        // Suit counts: subtract 1 from tile_X's suit (or honor pile).
        let mut suit_counts13 = full14_suit_counts;
        let mut z_count13 = full14_z_count;
        if let Some(s) = tile_x_suit {
            suit_counts13[s as usize] -= 1;
        } else {
            z_count13 -= 1;
        }
        let suits13_mask: u8 = ((suit_counts13[0] > 0) as u8)
            | (((suit_counts13[1] > 0) as u8) << 1)
            | (((suit_counts13[2] > 0) as u8) << 2);
        let n_suits13 = suits13_mask.count_ones();

        // Sanshoku doukou pre-check (wait-invariant; depends on full13 only).
        // counts of the 3 suits at the same n must all be ≥ 3 in full13.
        // For incremental update: subtracting 1 from tile_X may break a
        // pre-existing sanshoku-doukou-shape. Just check full13 fresh — it's
        // 9 iters × 3 reads, much cheaper than computing full13.
        // Build full13 inline by reading full14 with adjustment for tile_X.
        let read_full13 = |t: usize| -> u8 {
            if t == tile_x {
                full14[t] - 1
            } else {
                full14[t]
            }
        };
        let mut sanshoku_doukou_full13 = false;
        for n in 0..9usize {
            if read_full13(n) >= 3 && read_full13(n + 9) >= 3 && read_full13(n + 18) >= 3 {
                sanshoku_doukou_full13 = true;
                break;
            }
        }
        if sanshoku_doukou_full13 {
            struct_yaku_found = true;
            break 'pass1;
        }

        let main_suit = if n_suits13 == 1 {
            Some(suits13_mask.trailing_zeros() as u8)
        } else {
            None
        };

        let can_tanyao = yaocchi13 == 0;
        let can_single_suit = n_suits13 <= 1;
        let can_honroutou = simple13 == 0;
        let can_toitoi = singletons13 == 0;

        if !can_tanyao && !can_single_suit && !can_honroutou && !can_toitoi {
            // Only yakuhai-by-completion remains. Check the ≤5 yakuhai tiles
            // directly (vs scanning all 34 in the wait loop).
            for i in 0..n_yh {
                let t = yakuhai_tiles[i] as usize;
                if counts[t] == 2 && remaining[t] > 0 {
                    struct_yaku_found = true;
                    break 'pass1;
                }
            }
            continue;
        }

        // Per-wait structural checks. Iterate tile properties precomputed by
        // class (yaocchi-vs-simple, suit, honor) so each iteration is O(1).
        for tile in 0..TILE_MAX {
            if remaining[tile] == 0 || counts[tile] >= 4 {
                continue;
            }
            let tile_u = tile as u8;
            let tile_yaocchi = if tile < 27 {
                matches!(tile % 9, 0 | 8)
            } else {
                true
            };
            let tile_honor = tile >= 27;
            let tile_suit = if tile < 27 {
                Some((tile / 9) as u8)
            } else {
                None
            };
            // Yakuhai by wait completing kotsu.
            for i in 0..n_yh {
                if tile_u == yakuhai_tiles[i] && counts[tile] == 2 {
                    struct_yaku_found = true;
                    break 'pass1;
                }
            }
            // Tanyao.
            if can_tanyao && !tile_yaocchi {
                struct_yaku_found = true;
                break 'pass1;
            }
            // Honitsu / Chinitsu (with honor allowed if z present in full13).
            if can_single_suit
                && match main_suit {
                    Some(s) => tile_suit == Some(s) || (tile_honor && z_count13 > 0),
                    None => tile_honor && z_count13 > 0,
                }
            {
                struct_yaku_found = true;
                break 'pass1;
            }
            // Honroutou-shape.
            if can_honroutou && tile_yaocchi {
                struct_yaku_found = true;
                break 'pass1;
            }
            // Toitoi-shape + shanpon-completion.
            if can_toitoi && counts[tile] == 2 {
                struct_yaku_found = true;
                break 'pass1;
            }
        }
    }
    let result = if struct_yaku_found {
        true
    } else {
        // Pass 2: fallback to per-wait yaku-presence check.
        // Use `base_score_tsumo` directly (cache key = counts + win_tile + akas)
        // instead of `score_tsumo` (cache key adds 34-byte `remaining`). The
        // smaller key trims hashing overhead by ~3× per call. Both return
        // `Some` iff yaku exists; for presence-only check we don't need
        // dora/aka/timing adjustments.
        let akas = dp.input.akas_in_hand;
        tenpai_counts[..n_tenpai].iter().any(|counts| {
            (0..TILE_MAX).any(|tile| {
                if remaining[tile] == 0 || counts[tile] >= 4 {
                    return false;
                }
                dp.base_score_tsumo(counts, tile as u8, akas).is_some()
            })
        })
    };
    dp.yaku_tenpai_cache.insert(key, result);
    result
}

/// Normalize a wind-tile id to its 27..=30 (E/S/W/N) form so the yakuhai
/// fast-path can compare to existing hand counts directly.
#[inline]
fn norm_wind_for_yakuhai(w: u8) -> u8 {
    if (27..=30).contains(&w) {
        w
    } else {
        27 + (w & 0b11)
    }
}

fn probability_series(
    shanten_after_discard: i8,
    required_count: f32,
    wait_count: f32,
    total_remaining: f32,
    tsumos_left: usize,
) -> ([f32; SP_MAX_TURNS], [f32; SP_MAX_TURNS]) {
    let mut tenpai = [0.0; SP_MAX_TURNS];
    let mut win = [0.0; SP_MAX_TURNS];
    let horizon = tsumos_left.min(SP_MAX_TURNS);

    for turn in 1..=horizon {
        let idx = turn - 1;
        if shanten_after_discard <= 0 {
            tenpai[idx] = 1.0;
            win[idx] = at_least_one_prob(wait_count, total_remaining, turn);
        } else if shanten_after_discard == 1 {
            tenpai[idx] = at_least_one_prob(required_count, total_remaining, turn);
            win[idx] = improve_then_win_prob(required_count, wait_count, total_remaining, turn);
        } else {
            let p = (required_count / total_remaining).clamp(0.0, 1.0);
            tenpai[idx] = binomial_at_least(turn, shanten_after_discard as usize, p);
            win[idx] = tenpai[idx] * (wait_count / total_remaining).clamp(0.0, 1.0);
        }
    }

    (tenpai, win)
}

fn series_for_candidate(
    dp: &mut DpContext<'_>,
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    shanten_after_discard: i8,
    required_count: f32,
    wait_count: f32,
    mean_point: f32,
    total_remaining: f32,
    akas_in_hand: [bool; 3],
    waits_for_tenpai: Option<&[f32; TILE_MAX]>,
) -> (
    [f32; SP_MAX_TURNS],
    [f32; SP_MAX_TURNS],
    [f32; SP_MAX_TURNS],
) {
    let horizon = (dp.input.tsumos_left as usize).min(SP_MAX_TURNS);
    // Fast path: at tenpai we already discovered the wait set during
    // `fused_tenpai_pass`; reuse it instead of going through `draw_dp` (which
    // would re-enumerate 34 tiles, hash a `DpKey`, and iterate the same waits).
    if shanten_after_discard == 0
        && let Some(waits) = waits_for_tenpai
    {
        return dp.tenpai_series_from_waits(counts, remaining, waits, akas_in_hand);
    }
    if shanten_after_discard <= SHANTEN_THRES {
        return dp.series_with_akas(counts, remaining, horizon, akas_in_hand);
    }
    series_for_candidate_approx(
        shanten_after_discard,
        required_count,
        wait_count,
        mean_point,
        total_remaining,
        horizon,
    )
}

/// Cheap closed-form approximation used for non-optimal (shanten-down) discards
/// and for hands at shanten ≥ 4 where running the full DP is wasteful.
fn series_for_candidate_approx(
    shanten_after_discard: i8,
    required_count: f32,
    wait_count: f32,
    mean_point: f32,
    total_remaining: f32,
    horizon: usize,
) -> (
    [f32; SP_MAX_TURNS],
    [f32; SP_MAX_TURNS],
    [f32; SP_MAX_TURNS],
) {
    let (tenpai, win) = probability_series(
        shanten_after_discard,
        required_count,
        wait_count,
        total_remaining,
        horizon,
    );
    let mut ev = [0.0; SP_MAX_TURNS];
    for i in 0..SP_MAX_TURNS {
        ev[i] = win[i] * mean_point;
    }
    (tenpai, win, ev)
}

const SHANTEN_THRES: i8 = 3;
/// 13枚の手牌と、自摸可能な残り牌の最大合計枚数 (= 34*4 - 13 - 1).
const MAX_TILES_LEFT: usize = TILE_MAX * 4 - 1 - 13;

/// SP内DPキャッシュ専用の FxHash 実装。`[u8; 34]` 等のキーを SipHash より大幅に高速にハッシュする。
/// 依存ゼロで wasm バイナリサイズへの影響もない。
#[derive(Default, Clone, Copy)]
struct FxHasher64 {
    hash: u64,
}

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher64 {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.hash;
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            // SAFETY: chunks_exact yields exactly 8-byte slices.
            let v = u64::from_ne_bytes(unsafe { *(chunk.as_ptr() as *const [u8; 8]) });
            h = h.rotate_left(5) ^ v;
            h = h.wrapping_mul(FX_SEED);
        }
        for &b in chunks.remainder() {
            h = h.rotate_left(5) ^ (b as u64);
            h = h.wrapping_mul(FX_SEED);
        }
        self.hash = h;
    }
    #[inline]
    fn write_u8(&mut self, b: u8) {
        self.hash = (self.hash.rotate_left(5) ^ (b as u64)).wrapping_mul(FX_SEED);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher64>>;

/// 各「絶対巡目 i」に対するDP値ベクトル。
/// `i` は「DPホライズンの先頭 (turn 0) から数えた現在巡目」で、
/// `Values.tenpai[i]` などは「i 巡目以降にこの状態から到達する各事象の確率/期待値」。
#[derive(Clone)]
struct Values {
    tenpai: [f32; SP_MAX_TURNS],
    win: [f32; SP_MAX_TURNS],
    exp: [f32; SP_MAX_TURNS],
}

impl Default for Values {
    fn default() -> Self {
        Self {
            tenpai: [0.0; SP_MAX_TURNS],
            win: [0.0; SP_MAX_TURNS],
            exp: [0.0; SP_MAX_TURNS],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DpKey {
    counts: [u8; TILE_MAX],
    remaining: [u8; TILE_MAX],
    /// `[5mr, 5pr, 5sr]` — true if the corresponding red 5 is still in the
    /// player's in-hand portion. Updated by discard transitions.
    akas_in_hand: [bool; 3],
    /// `[5mr, 5pr, 5sr]` — true if the corresponding red 5 might still be in
    /// the wall (= !seen so far). Drawing a 5x with the red still in the wall
    /// branches into "drew normal 5x" vs "drew the red"; the red branch sets
    /// `akas_in_hand[i] = true` and `akas_in_wall[i] = false`.
    akas_in_wall: [bool; 3],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
struct ScoreMods {
    ippatsu: bool,
    double_riichi: bool,
    haitei: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ScoreKey {
    counts: [u8; TILE_MAX],
    remaining: [u8; TILE_MAX],
    win_tile: u8,
    mods: ScoreMods,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BaseScoreKey {
    counts: [u8; TILE_MAX],
    win_tile: u8,
    akas_in_hand: [bool; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ScoreVecKey {
    counts_14: [u8; TILE_MAX],
    remaining_after_win: [u8; TILE_MAX],
    win_tile: u8,
    /// Aka state at the leaf — affects han via aka dora count.
    akas_in_hand: [bool; 3],
}

#[derive(Debug, Clone, Copy)]
struct BaseScore {
    total: u32,
    han: u32,
    fu: u32,
    is_oya: bool,
    riichi: bool,
    apply_ura: bool,
    honba: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct YakuTenpaiKey {
    counts: [u8; TILE_MAX],
    remaining: [u8; TILE_MAX],
    elapsed_turns: u8,
    turns_left: u8,
}

struct DpContext<'a> {
    input: &'a SpInput,
    variant: SpVariant,
    /// 計算したいホライズン (= 入力 tsumos_left).
    horizon: usize,
    /// 山+他家手牌などの未公開牌合計枚数 (= sum(remaining)).
    n_left_tiles: u32,

    /// `tsumo_prob[c][j]`: 残り c+1 枚の特定種別を巡目 j で引く確率。
    /// 内部で残り山枚数の減少 `(n_left_tiles - j)` を反映済み。
    tsumo_prob: [[f32; SP_MAX_TURNS]; 4],
    /// `not_tsumo_prob[total][j]`: 有効牌が合計 `total` 枚あるとき、巡目 j-1 までに
    /// 一度も引けなかった確率。`[*][0] = 1.0`.
    not_tsumo_prob: Vec<[f32; SP_MAX_TURNS]>,

    /// shanten ごとの DP キャッシュ (0..=SHANTEN_THRES).
    discard_cache: [FxHashMap<DpKey, Rc<Values>>; (SHANTEN_THRES + 1) as usize],
    draw_cache: [FxHashMap<DpKey, Rc<Values>>; (SHANTEN_THRES + 1) as usize],

    /// leaf (tenpai) 時の点数表 (timing-han 0..=3).
    score_vec_cache: FxHashMap<ScoreVecKey, Option<[f32; 4]>>,

    score_cache: FxHashMap<ScoreKey, Option<u32>>,
    base_score_cache: FxHashMap<BaseScoreKey, Option<BaseScore>>,
    yaku_tenpai_cache: FxHashMap<YakuTenpaiKey, bool>,

    /// Once-per-SP-call structural yaku flags computed on the FULL pre-discard
    /// 14-tile hand (`input.tehai + meld_tiles`). Lets the per-wait
    /// `has_yaku_tenpai_after_best_discard` and `fused_tenpai_pass` skip the
    /// per-call full-array build + 34-tile rescan when the structural property
    /// is invariant under any (outer_discard, wait) pair.
    base_full_no_yaocchi: bool,
    base_full_single_suit: Option<u8>, // None = >1 numbered suit; Some(s) = only suit s
    base_full_has_z: bool,
    base_full_yakuhai_in_meld: bool,
    /// Yakuhai tile types (3 dragons + bakaze + jikaze if wind tiles).
    base_yakuhai: [u8; 5],
    base_n_yakuhai: usize,
}

impl<'a> DpContext<'a> {
    fn new(input: &'a SpInput, variant: SpVariant) -> Self {
        let horizon = (input.tsumos_left as usize).min(SP_MAX_TURNS).max(1);
        let remaining = remaining_counts_for_variant(input, variant);
        let n_left_tiles = remaining.iter().map(|&v| v as u32).sum::<u32>();
        let tsumo_prob = build_tsumo_prob_table(n_left_tiles, horizon);
        let not_tsumo_prob = build_not_tsumo_prob_table(n_left_tiles, horizon);

        // Pre-compute once-per-call structural flags on `input.tehai + melds`.
        let mut base_full = input.tehai;
        for meld in &input.melds {
            for &t136 in &meld.tiles {
                let tt = (t136 / 4) as usize;
                if tt < TILE_MAX {
                    base_full[tt] = base_full[tt].saturating_add(1);
                }
            }
        }
        let mut suits = 0u8;
        let mut has_yaocchi = false;
        let mut has_simple = false;
        let mut has_z = false;
        for t in 0..27usize {
            if base_full[t] > 0 {
                suits |= 1 << (t / 9);
                if t % 9 == 0 || t % 9 == 8 {
                    has_yaocchi = true;
                } else {
                    has_simple = true;
                }
            }
        }
        for t in 27..34usize {
            if base_full[t] > 0 {
                has_z = true;
                has_yaocchi = true;
                break;
            }
        }
        let n_suits = suits.count_ones();
        let _ = has_simple;
        let base_full_no_yaocchi = !has_yaocchi;
        let base_full_has_z = has_z;
        let base_full_single_suit = if n_suits == 1 {
            Some(suits.trailing_zeros() as u8)
        } else {
            None
        };

        let bakaze = norm_wind_for_yakuhai(input.bakaze);
        let jikaze = norm_wind_for_yakuhai(input.jikaze);
        let mut base_yakuhai = [31u8, 32, 33, 0, 0];
        let mut base_n_yakuhai = 3usize;
        if (27..=30).contains(&bakaze) {
            base_yakuhai[base_n_yakuhai] = bakaze;
            base_n_yakuhai += 1;
        }
        if (27..=30).contains(&jikaze) && jikaze != bakaze {
            base_yakuhai[base_n_yakuhai] = jikaze;
            base_n_yakuhai += 1;
        }
        let base_full_yakuhai_in_meld = input.melds.iter().any(|m| {
            m.tiles.iter().any(|&t| {
                let tt = t / 4;
                base_yakuhai[..base_n_yakuhai].contains(&tt)
            })
        });

        Self {
            input,
            variant,
            horizon,
            n_left_tiles,
            tsumo_prob,
            not_tsumo_prob,
            discard_cache: Default::default(),
            draw_cache: Default::default(),
            score_vec_cache: FxHashMap::default(),
            score_cache: FxHashMap::default(),
            base_score_cache: FxHashMap::default(),
            yaku_tenpai_cache: FxHashMap::default(),
            base_full_no_yaocchi,
            base_full_single_suit,
            base_full_has_z,
            base_full_yakuhai_in_meld,
            base_yakuhai,
            base_n_yakuhai,
        }
    }

    fn shanten(&mut self, counts: &[u8; TILE_MAX]) -> i8 {
        // Direct compute: empirically the [u8; 34] cache lookup overhead
        // (~12 ns hash+probe) is comparable to `calc_normal`'s nyanten cascade
        // on a warm L1, so the cache amortizes only marginally. Removing it
        // also frees ~2 KB of hashbrown table per SP run.
        shanten_of_counts_for_variant(counts, self.variant)
    }

    fn score_tsumo(
        &mut self,
        counts: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        win_tile: u8,
        mods: ScoreMods,
    ) -> Option<f32> {
        // Helpers used here (score_waits, yaku_progress) call us with the
        // initial input.akas_in_hand state — they don't run inside the DP.
        let akas = self.input.akas_in_hand;
        if self.input.dora_indicators.is_empty() {
            return self
                .base_score_tsumo(counts, win_tile, akas)
                .map(|base| {
                    score_from_base(
                        self.input,
                        base,
                        counts,
                        remaining,
                        win_tile,
                        mods,
                        self.variant,
                    )
                })
                .or_else(|| {
                    mods.haitei
                        .then(|| {
                            exact_score_tsumo(self.input, counts, win_tile, mods, self.variant)
                        })
                        .flatten()
                });
        }

        let key = ScoreKey {
            counts: *counts,
            remaining: *remaining,
            win_tile,
            mods,
        };
        if let Some(&cached) = self.score_cache.get(&key) {
            return cached.map(|point| point as f32);
        }

        let score = self
            .base_score_tsumo(counts, win_tile, akas)
            .map(|base| {
                score_from_base(
                    self.input,
                    base,
                    counts,
                    remaining,
                    win_tile,
                    mods,
                    self.variant,
                ) as u32
            })
            .or_else(|| {
                mods.haitei
                    .then(|| exact_score_tsumo(self.input, counts, win_tile, mods, self.variant))
                    .flatten()
                    .map(|point| point as u32)
            });
        self.score_cache.insert(key, score);
        score.map(|point| point as f32)
    }

    fn base_score_tsumo(
        &mut self,
        counts: &[u8; TILE_MAX],
        win_tile: u8,
        akas_in_hand: [bool; 3],
    ) -> Option<BaseScore> {
        let key = BaseScoreKey {
            counts: *counts,
            win_tile,
            akas_in_hand,
        };
        if let Some(&cached) = self.base_score_cache.get(&key) {
            return cached;
        }

        let score = base_score_tsumo(self.input, counts, win_tile, akas_in_hand, self.variant);
        self.base_score_cache.insert(key, score);
        score
    }

    /// 入力 `remaining` の合計枚数に応じて確率テーブルを構築/再構築する。
    /// テーブル依存のキャッシュ (discard_cache, draw_cache, score_vec_cache) も同時に無効化する。
    fn ensure_prob_tables(&mut self, n_left: u32, horizon: usize) {
        if self.n_left_tiles == n_left && self.horizon == horizon {
            return;
        }
        self.horizon = horizon.max(1);
        self.n_left_tiles = n_left;
        self.tsumo_prob = build_tsumo_prob_table(n_left, self.horizon);
        self.not_tsumo_prob = build_not_tsumo_prob_table(n_left, self.horizon);
        for c in &mut self.discard_cache {
            c.clear();
        }
        for c in &mut self.draw_cache {
            c.clear();
        }
        // score_vec_cache は (counts, remaining, win_tile) で識別され、
        // 確率テーブルに依存しないので再利用可能。
    }

    /// `series[i]`: 「i+1 巡先までに到達する各事象の確率/期待値」を返す。
    #[cfg(test)]
    fn series(
        &mut self,
        counts: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        tsumos_left: usize,
    ) -> (
        [f32; SP_MAX_TURNS],
        [f32; SP_MAX_TURNS],
        [f32; SP_MAX_TURNS],
    ) {
        // Default: use the input's initial aka state. Callers that already
        // adjusted aka (e.g. discarded the red 5xr externally) should call
        // `series_with_akas` directly.
        self.series_with_akas(counts, remaining, tsumos_left, self.input.akas_in_hand)
    }

    fn series_with_akas(
        &mut self,
        counts: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        tsumos_left: usize,
        akas_in_hand: [bool; 3],
    ) -> (
        [f32; SP_MAX_TURNS],
        [f32; SP_MAX_TURNS],
        [f32; SP_MAX_TURNS],
    ) {
        let mut tenpai = [0.0; SP_MAX_TURNS];
        let mut win = [0.0; SP_MAX_TURNS];
        let mut ev = [0.0; SP_MAX_TURNS];

        let horizon = tsumos_left.min(SP_MAX_TURNS);
        if horizon == 0 {
            return (tenpai, win, ev);
        }

        let n_left = remaining.iter().map(|&v| v as u32).sum::<u32>();
        self.ensure_prob_tables(n_left, horizon);

        let s = self.shanten(counts);
        if s < 0 || s > SHANTEN_THRES {
            return (tenpai, win, ev);
        }

        // The generic DP stores values indexed by the absolute draw at which
        // a state is entered.  For an already-tenpai hand, the public series
        // is instead the cumulative probability from the current draw
        // through each horizon.  Use the closed-form path so the first
        // channel has denominator `n_left`, not `n_left - horizon + 1`.
        if s == 0 {
            let waits = required_tiles(self, counts, remaining, s);
            return self.tenpai_series_from_waits(counts, remaining, &waits, akas_in_hand);
        }

        let key = DpKey {
            counts: *counts,
            remaining: *remaining,
            akas_in_hand,
            akas_in_wall: initial_akas_in_wall(self.input),
        };
        let values = self.draw_dp(key, s);

        // 「i+1 巡先 = 残り i+1 巡」を Values[N-1-i] に対応付ける。
        for i in 0..horizon {
            let v_idx = horizon - 1 - i;
            tenpai[i] = values.tenpai[v_idx];
            win[i] = values.win[v_idx];
            ev[i] = values.exp[v_idx];
        }
        (tenpai, win, ev)
    }

    /// Closed-form `series_with_akas` for the tenpai (`shanten==0`) case.
    /// Skips the full `draw_dp` cache lookup + 34-tile re-enumeration by taking
    /// the wait set as input. Wait tiles come from `fused_tenpai_pass`, so the
    /// caller has already paid the shanten checks.
    ///
    /// Per-call this avoids: (a) `DpKey` hashing for the draw cache, (b) a
    /// second full 34-tile loop in `draw_dp_slow`, (c) repeated cached-shanten
    /// lookups that we just performed. The arithmetic (per-wait probability
    /// series, aka draw branching, ippatsu/double-riichi/haitei bonuses) is
    /// the same as `draw_dp_slow` at `shanten == 0`.
    fn tenpai_series_from_waits(
        &mut self,
        counts_13: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        wait_tiles: &[f32; TILE_MAX],
        akas_in_hand: [bool; 3],
    ) -> (
        [f32; SP_MAX_TURNS],
        [f32; SP_MAX_TURNS],
        [f32; SP_MAX_TURNS],
    ) {
        let mut tenpai_out = [0.0f32; SP_MAX_TURNS];
        let mut win_out = [0.0f32; SP_MAX_TURNS];
        let mut exp_out = [0.0f32; SP_MAX_TURNS];

        let horizon = (self.input.tsumos_left as usize).min(SP_MAX_TURNS);
        if horizon == 0 {
            return (tenpai_out, win_out, exp_out);
        }
        for i in 0..horizon {
            tenpai_out[i] = 1.0;
        }

        let n_left = remaining.iter().map(|&v| v as u32).sum::<u32>();
        self.ensure_prob_tables(n_left, horizon);

        let mut sum_required: u32 = 0;
        for &c in wait_tiles {
            sum_required += c as u32;
        }
        if sum_required == 0 {
            return (tenpai_out, win_out, exp_out);
        }

        let assume_riichi = self.input.is_menzen && self.input.can_riichi;
        let calc_double_riichi = assume_riichi && self.input.can_double_riichi;
        let last_turn_idx = horizon - 1;
        let akas_in_wall = initial_akas_in_wall(self.input);

        let total_idx = (sum_required as usize).min(self.not_tsumo_prob.len() - 1);
        let not_tsumo: [f32; SP_MAX_TURNS] = self.not_tsumo_prob[total_idx];

        // Exact first-win probabilities by draw.  Prefix-summing these below
        // yields P(win within h draws), matching the public feature contract.
        let mut win_at = [0.0f32; SP_MAX_TURNS];
        let mut exp_at = [0.0f32; SP_MAX_TURNS];

        let accumulate = |dp: &mut Self,
                          tile: u8,
                          sub_count: u32,
                          scoring_akas: [bool; 3],
                          win_at: &mut [f32; SP_MAX_TURNS],
                          exp_at: &mut [f32; SP_MAX_TURNS]| {
            if sub_count == 0 {
                return;
            }
            let scores = dp.score_vector_for_win(counts_13, remaining, tile, scoring_akas);
            let Some(scores) = scores else { return };
            let tsumo_row: [f32; SP_MAX_TURNS] = dp.tsumo_prob[(sub_count as usize - 1).min(3)];
            for draw in 0..horizon {
                let prob = tsumo_row[draw] * not_tsumo[draw];
                if prob == 0.0 {
                    continue;
                }
                let han_plus = (calc_double_riichi as usize
                    + (assume_riichi && draw == 0) as usize
                    + (draw == last_turn_idx) as usize)
                    .min(3);
                win_at[draw] += prob;
                exp_at[draw] += prob * scores[han_plus];
            }
        };

        for tile_idx in 0..TILE_MAX {
            let count = wait_tiles[tile_idx] as u32;
            if count == 0 {
                continue;
            }
            let tile = tile_idx as u8;
            // Mortal-style aka draw branching for 5m/5p/5s waits.
            let red_idx = match tile {
                4 => Some(0),
                13 => Some(1),
                22 => Some(2),
                _ => None,
            };
            let split = red_idx
                .map(|i| akas_in_wall[i] && !akas_in_hand[i])
                .unwrap_or(false);
            if split {
                let i = red_idx.unwrap();
                let mut akas_red = akas_in_hand;
                akas_red[i] = true;
                if count >= 2 {
                    accumulate(
                        self,
                        tile,
                        count - 1,
                        akas_in_hand,
                        &mut win_at,
                        &mut exp_at,
                    );
                }
                accumulate(self, tile, 1, akas_red, &mut win_at, &mut exp_at);
            } else {
                accumulate(self, tile, count, akas_in_hand, &mut win_at, &mut exp_at);
            }
        }

        let mut cumulative_win = 0.0f32;
        let mut cumulative_exp = 0.0f32;
        for draw in 0..horizon {
            cumulative_win += win_at[draw];
            cumulative_exp += exp_at[draw];
            win_out[draw] = cumulative_win.clamp(0.0, 1.0);
            exp_out[draw] = cumulative_exp.max(0.0);
        }
        (tenpai_out, win_out, exp_out)
    }

    fn draw_dp(&mut self, key: DpKey, shanten: i8) -> Rc<Values> {
        debug_assert!((0..=SHANTEN_THRES).contains(&shanten));
        if let Some(v) = self.draw_cache[shanten as usize].get(&key) {
            return Rc::clone(v);
        }
        let v = Rc::new(self.draw_dp_slow(&key, shanten));
        self.draw_cache[shanten as usize].insert(key, Rc::clone(&v));
        v
    }

    fn discard_dp(&mut self, key: DpKey, shanten: i8) -> Rc<Values> {
        debug_assert!((0..=SHANTEN_THRES).contains(&shanten));
        if let Some(v) = self.discard_cache[shanten as usize].get(&key) {
            return Rc::clone(v);
        }
        let v = Rc::new(self.discard_dp_slow(&key, shanten));
        self.discard_cache[shanten as usize].insert(key, Rc::clone(&v));
        v
    }

    /// 14 枚の手牌から、現 shanten を維持する打牌のうち最善のものを選び、その Values を返す。
    fn discard_dp_slow(&mut self, key: &DpKey, shanten: i8) -> Values {
        let horizon = self.horizon;
        let mut best_tenpai = [f32::MIN; SP_MAX_TURNS];
        let mut best_win = [f32::MIN; SP_MAX_TURNS];
        let mut best_exp = [f32::MIN; SP_MAX_TURNS];
        // Track the tile picked at each turn so we can break ties using
        // discard priority (mirrors Mortal's `cmp_discard_priority`). Without
        // tie-breaking, our impl picks the lowest tile_id at ties; Mortal
        // prefers honors > terminals > middle > aka, which downstream changes
        // which intermediate state we recurse into and accumulates 0.05–0.13
        // tenpai_prob divergence on high-shanten DP paths.
        let mut best_tile = [u8::MAX; SP_MAX_TURNS];
        let mut any_valid = false;

        // Per-suit incremental shanten state for the discard enumeration.
        // Each iteration drops 1 tile from key.counts → only that suit's k0
        // refreshes. Saves 3 hashes per shanten check inside the DP recursion.
        let dd_base_k0_m = crate::shanten::k0_shupai_for(&key.counts[0..9]);
        let dd_base_k0_p = crate::shanten::k0_shupai_for(&key.counts[9..18]);
        let dd_base_k0_s = crate::shanten::k0_shupai_for(&key.counts[18..27]);
        let dd_base_k0_z = crate::shanten::k0_zipai_for(&key.counts[27..34]);
        let dd_len_div3_drop = key.counts.iter().sum::<u8>().saturating_sub(1) / 3;
        let mut next_counts = key.counts;

        for tile in 0..TILE_MAX {
            if key.counts[tile] == 0 {
                continue;
            }
            next_counts[tile] -= 1;
            let s = shanten_after_add_incremental(
                &next_counts,
                tile,
                dd_base_k0_m,
                dd_base_k0_p,
                dd_base_k0_s,
                dd_base_k0_z,
                dd_len_div3_drop,
                self.variant,
            );
            if s != shanten {
                // 向聴維持のみ。向聴落としは現状サポートしない。
                next_counts[tile] += 1;
                continue;
            }
            // Aka tracking: when the only copy of a 5x in hand is discarded
            // and the player is holding the red one, that red is now gone.
            let next_akas = aka_after_discard(key.akas_in_hand, &key.counts, tile as u8);
            let next_key = DpKey {
                counts: next_counts,
                remaining: key.remaining,
                akas_in_hand: next_akas,
                akas_in_wall: key.akas_in_wall,
            };
            let v = self.draw_dp(next_key, shanten);
            for i in 0..horizon {
                // Mortal-style tie-break: when EVs are equal at i32 (= yen)
                // precision, prefer the higher-discard-priority tile (honors
                // > terminals > middle > aka). This matches Mortal's
                // `cmp_discard_priority` and keeps intermediate-state choices
                // aligned, which reduces tenpai_prob accumulating drift.
                let cur_i = v.exp[i] as i32;
                let best_i = best_exp[i] as i32;
                let take = !any_valid
                    || cur_i > best_i
                    || (cur_i == best_i
                        && discard_priority(tile as u8) > discard_priority(best_tile[i]));
                if take {
                    best_tenpai[i] = v.tenpai[i];
                    best_win[i] = v.win[i];
                    best_exp[i] = v.exp[i];
                    best_tile[i] = tile as u8;
                }
            }
            any_valid = true;
            // Restore for next iteration.
            next_counts[tile] += 1;
        }

        let mut out = Values::default();
        if !any_valid {
            return out;
        }
        for i in 0..horizon {
            out.tenpai[i] = if best_tenpai[i] == f32::MIN {
                0.0
            } else {
                best_tenpai[i]
            };
            out.win[i] = if best_win[i] == f32::MIN {
                0.0
            } else {
                best_win[i]
            };
            out.exp[i] = if best_exp[i] == f32::MIN {
                0.0
            } else {
                best_exp[i]
            };
        }
        out
    }

    /// 13 枚の手牌から「自摸を1回引く」DP。shanten==0 ならその自摸が和了牌。
    fn draw_dp_slow(&mut self, key: &DpKey, shanten: i8) -> Values {
        let horizon = self.horizon;
        let assume_riichi = self.input.is_menzen && self.input.can_riichi;
        let calc_double_riichi = assume_riichi && self.input.can_double_riichi;
        let last_turn_idx = horizon - 1;

        // 有効牌を列挙し、合計枚数を計算する。
        // Per-suit incremental shanten: cache base k0 once, recompute only
        // the affected suit per candidate tile. For shanten ≥ 2 hands this
        // path is called millions of times by the DP recursion — saving 3
        // hash_shupai/zipai per iteration is significant.
        let dp_base_k0_m = crate::shanten::k0_shupai_for(&key.counts[0..9]);
        let dp_base_k0_p = crate::shanten::k0_shupai_for(&key.counts[9..18]);
        let dp_base_k0_s = crate::shanten::k0_shupai_for(&key.counts[18..27]);
        let dp_base_k0_z = crate::shanten::k0_zipai_for(&key.counts[27..34]);
        let dp_len_div3_next = (key.counts.iter().sum::<u8>() + 1) / 3;
        let mut effective: [(u8, u8); 34] = [(0, 0); 34];
        let mut n_eff: usize = 0;
        let mut sum_required: u32 = 0;
        let mut next_counts = key.counts;
        for tile in 0..TILE_MAX {
            let count = key.remaining[tile];
            if count == 0 || key.counts[tile] >= 4 {
                continue;
            }
            // Pre-filter: a tile cannot reduce shanten unless it's adjacent to
            // an existing hand tile. For honors, "adjacent" means count >= 1.
            // For numbered tiles, anywhere within ±2 in the same suit. This
            // filters ~50% of tiles before paying the shanten lookup.
            if !potentially_effective_for_draw(&key.counts, tile) {
                continue;
            }
            next_counts[tile] += 1;
            let s_after = shanten_after_add_incremental(
                &next_counts,
                tile,
                dp_base_k0_m,
                dp_base_k0_p,
                dp_base_k0_s,
                dp_base_k0_z,
                dp_len_div3_next,
                self.variant,
            );
            next_counts[tile] -= 1;
            if s_after < shanten {
                effective[n_eff] = (tile as u8, count);
                n_eff += 1;
                sum_required += count as u32;
            }
        }

        let mut tenpai = [0.0f32; SP_MAX_TURNS];
        let mut win = [0.0f32; SP_MAX_TURNS];
        let mut exp = [0.0f32; SP_MAX_TURNS];
        if n_eff == 0 {
            return Values { tenpai, win, exp };
        }

        // 借用回避のため確率テーブル行を局所コピーする (各 [f32; 17] = 68 バイト).
        let total_idx = (sum_required as usize).min(self.not_tsumo_prob.len() - 1);
        let not_tsumo: [f32; SP_MAX_TURNS] = self.not_tsumo_prob[total_idx];

        // Helper that processes a single (sub_count, akas_in_hand_after) draw
        // sub-branch and accumulates into tenpai/win/exp. We invoke it once
        // per sub-branch when splitting a 5x draw into normal/red.
        #[allow(unused_mut)]
        let process_branch = |dp: &mut Self,
                              tile: u8,
                              sub_count: u8,
                              next_in_hand: [bool; 3],
                              next_in_wall: [bool; 3],
                              tenpai: &mut [f32; SP_MAX_TURNS],
                              win: &mut [f32; SP_MAX_TURNS],
                              exp: &mut [f32; SP_MAX_TURNS]| {
            if sub_count == 0 {
                return;
            }
            let tsumo_row: [f32; SP_MAX_TURNS] = dp.tsumo_prob[(sub_count as usize - 1).min(3)];

            if shanten > 0 {
                let mut next_counts = key.counts;
                next_counts[tile as usize] += 1;
                let mut next_remaining = key.remaining;
                next_remaining[tile as usize] -= 1;
                let next_key = DpKey {
                    counts: next_counts,
                    remaining: next_remaining,
                    akas_in_hand: next_in_hand,
                    akas_in_wall: next_in_wall,
                };
                let next_v = dp.discard_dp(next_key, shanten - 1);

                for i in 0..horizon {
                    let m = not_tsumo[i];
                    if m == 0.0 {
                        break;
                    }
                    let m_inv = 1.0 / m;
                    for j in i..horizon {
                        let n = not_tsumo[j];
                        if n == 0.0 {
                            break;
                        }
                        let prob = tsumo_row[j] * n * m_inv;
                        if shanten == 1 {
                            tenpai[i] += prob;
                        }
                        if j + 1 < horizon {
                            let nj = j + 1;
                            if shanten > 1 {
                                tenpai[i] += prob * next_v.tenpai[nj];
                            }
                            win[i] += prob * next_v.win[nj];
                            exp[i] += prob * next_v.exp[nj];
                        }
                    }
                }
            } else {
                // shanten == 0: leaf. Score the agari with the post-draw aka
                // state so a "drew red 5x" branch picks up the extra dora.
                // Branch-free inner loop with post-hoc corrections at the
                // ippatsu/haitei boundary positions (mirrors the same pattern
                // in `tenpai_series_from_waits`).
                let scores =
                    dp.score_vector_for_win(&key.counts, &key.remaining, tile, next_in_hand);
                let Some(scores) = scores else {
                    return;
                };
                for i in 0..horizon {
                    let m = not_tsumo[i];
                    if m == 0.0 {
                        break;
                    }
                    let m_inv = 1.0 / m;
                    let dbl_at_i = if calc_double_riichi && i == 0 {
                        1usize
                    } else {
                        0
                    };
                    let s_base = scores[dbl_at_i.min(3)];

                    let mut sum_prob = 0.0f32;
                    let mut sum_exp = 0.0f32;
                    let mut max_j_seen = i;
                    for j in i..horizon {
                        let n = not_tsumo[j];
                        if n == 0.0 {
                            break;
                        }
                        let prob = tsumo_row[j] * n * m_inv;
                        sum_prob += prob;
                        sum_exp += prob;
                        max_j_seen = j;
                    }
                    win[i] += sum_prob;
                    exp[i] += sum_exp * s_base;

                    let n_at_i = not_tsumo[i];
                    if n_at_i != 0.0 && i <= max_j_seen {
                        let ipp = assume_riichi as usize;
                        let hai = (i == last_turn_idx) as usize;
                        let bonus = ipp + hai;
                        if bonus > 0 {
                            let prob = tsumo_row[i] * n_at_i * m_inv;
                            let han_plus = (dbl_at_i + bonus).min(3);
                            exp[i] += prob * (scores[han_plus] - s_base);
                        }
                    }
                    if last_turn_idx > i && last_turn_idx <= max_j_seen {
                        let n_last = not_tsumo[last_turn_idx];
                        if n_last != 0.0 {
                            let prob = tsumo_row[last_turn_idx] * n_last * m_inv;
                            let han_plus = (dbl_at_i + 1).min(3);
                            exp[i] += prob * (scores[han_plus] - s_base);
                        }
                    }
                }
            }
        };

        for k in 0..n_eff {
            let (tile, count) = effective[k];
            // Mortal-style draw-side aka branching: when a 5m/5p/5s is drawn
            // and the red is still in the wall, split into "drew normal" vs
            // "drew red". The two branches re-converge at the same tile_type
            // but differ in akas_in_hand for downstream scoring.
            let red_idx = match tile {
                4 => Some(0),
                13 => Some(1),
                22 => Some(2),
                _ => None,
            };
            let split = red_idx
                .map(|i| key.akas_in_wall[i] && !key.akas_in_hand[i])
                .unwrap_or(false);
            if split {
                let i = red_idx.unwrap();
                let mut next_in_hand_red = key.akas_in_hand;
                next_in_hand_red[i] = true;
                let mut next_in_wall_red = key.akas_in_wall;
                next_in_wall_red[i] = false;
                if count >= 2 {
                    process_branch(
                        self,
                        tile,
                        count - 1,
                        key.akas_in_hand,
                        key.akas_in_wall,
                        &mut tenpai,
                        &mut win,
                        &mut exp,
                    );
                }
                process_branch(
                    self,
                    tile,
                    1,
                    next_in_hand_red,
                    next_in_wall_red,
                    &mut tenpai,
                    &mut win,
                    &mut exp,
                );
            } else {
                process_branch(
                    self,
                    tile,
                    count,
                    key.akas_in_hand,
                    key.akas_in_wall,
                    &mut tenpai,
                    &mut win,
                    &mut exp,
                );
            }
        }

        for i in 0..horizon {
            tenpai[i] = tenpai[i].clamp(0.0, 1.0);
            win[i] = win[i].clamp(0.0, 1.0);
            if exp[i] < 0.0 {
                exp[i] = 0.0;
            }
        }

        Values { tenpai, win, exp }
    }

    /// 和了牌 `win_tile` を引いて上がる場合の点数を、追加役 (timing han) 0..3 ごとに計算した配列。
    /// `counts_13` は引く前の手牌、`remaining_pre` は引く前の残り枚数。
    /// `akas_in_hand` は DP path 上で更新済の赤 5 in-hand フラグ。
    fn score_vector_for_win(
        &mut self,
        counts_13: &[u8; TILE_MAX],
        remaining_pre: &[u8; TILE_MAX],
        win_tile: u8,
        akas_in_hand: [bool; 3],
    ) -> Option<[f32; 4]> {
        let mut counts_14 = *counts_13;
        counts_14[win_tile as usize] += 1;
        let mut remaining_after_win = *remaining_pre;
        remaining_after_win[win_tile as usize] =
            remaining_after_win[win_tile as usize].saturating_sub(1);

        let key = ScoreVecKey {
            counts_14,
            remaining_after_win,
            win_tile,
            akas_in_hand,
        };
        if let Some(&cached) = self.score_vec_cache.get(&key) {
            return cached;
        }

        let result =
            self.score_vector_for_win_compute(counts_13, remaining_pre, win_tile, akas_in_hand);
        self.score_vec_cache.insert(key, result);
        result
    }

    fn score_vector_for_win_compute(
        &mut self,
        counts_13: &[u8; TILE_MAX],
        remaining_pre: &[u8; TILE_MAX],
        win_tile: u8,
        akas_in_hand: [bool; 3],
    ) -> Option<[f32; 4]> {
        let base = self.base_score_tsumo(counts_13, win_tile, akas_in_hand)?;

        // SP DP always builds `BaseScore` with `honba=0` (both lean and legacy
        // paths bake this in), so we can use the inlined `sp_tsumo_total_fast`
        // instead of going through `score::calculate_score` + the `Score`
        // struct allocation.
        debug_assert_eq!(base.honba, 0, "SP base score must have honba=0");
        let calc_with_han = |delta: u32| -> f32 {
            let han = (base.han + delta).min(13);
            sp_tsumo_total_lut_for_variant(han, base.fu, base.is_oya, self.variant) as f32
        };

        let mut out = [0.0f32; 4];
        if !base.apply_ura {
            for k in 0..4 {
                out[k] = calc_with_han(k as u32);
            }
            return Some(out);
        }

        // 裏ドラ分布を一度だけ計算
        let mut full_counts = *counts_13;
        full_counts[win_tile as usize] += 1;
        for meld in &self.input.melds {
            for &tile in &meld.tiles {
                let tt = (tile / 4) as usize;
                if tt < TILE_MAX {
                    full_counts[tt] += 1;
                }
            }
        }
        let mut remaining_after_win = *remaining_pre;
        remaining_after_win[win_tile as usize] =
            remaining_after_win[win_tile as usize].saturating_sub(1);
        let ura_dist = ura_distribution(
            &full_counts,
            &remaining_after_win,
            self.input.dora_indicators.len(),
            self.variant,
        );

        for k in 0..4 {
            let mut expected = 0.0f32;
            for (u, &p) in ura_dist.iter().enumerate() {
                if p <= 0.0 {
                    continue;
                }
                expected += p * calc_with_han(k as u32 + u as u32);
            }
            let no_bonus = calc_with_han(k as u32);
            out[k] = expected.max(no_bonus);
        }
        Some(out)
    }
}

/// `tsumo_prob[c][j]`: 残り c+1 枚の特定種別を巡目 j で引く確率。
/// 山の総枚数は巡目 j ごとに 1 ずつ減る前提 (denominator = n_left - j)。
fn build_tsumo_prob_table(n_left: u32, horizon: usize) -> [[f32; SP_MAX_TURNS]; 4] {
    let mut table = [[0.0f32; SP_MAX_TURNS]; 4];
    if n_left == 0 {
        return table;
    }
    let max_j = horizon.min(SP_MAX_TURNS);
    for c in 0..4 {
        let count = (c + 1) as f32;
        for j in 0..max_j {
            let denom = n_left as i64 - j as i64;
            if denom <= 0 {
                break;
            }
            table[c][j] = count / denom as f32;
        }
    }
    table
}

/// `not_tsumo_prob[total][j]`: 有効牌が合計 `total` 枚あるとき、巡目 j-1 までに
/// 一度も引けなかった確率。`[*][0] = 1.0`、`[total > n_left]` 行は `[0]` のみ 1.
fn build_not_tsumo_prob_table(n_left: u32, horizon: usize) -> Vec<[f32; SP_MAX_TURNS]> {
    let rows = MAX_TILES_LEFT + 1;
    let mut table = vec![[0.0f32; SP_MAX_TURNS]; rows];
    let max_j = horizon.min(SP_MAX_TURNS);
    let n_left_i = n_left as i64;
    for (i, row) in table.iter_mut().enumerate() {
        if (i as u32) > n_left {
            row[0] = 1.0;
            continue;
        }
        row[0] = 1.0;
        let useful = i as i64;
        let last = (max_j - 1).min((n_left_i - useful).max(0) as usize);
        for j in 0..last {
            let useless_remaining = n_left_i - useful - j as i64;
            let total_remaining = n_left_i - j as i64;
            if useless_remaining <= 0 || total_remaining <= 0 {
                break;
            }
            row[j + 1] = row[j] * (useless_remaining as f32 / total_remaining as f32);
        }
    }
    table
}

fn at_least_one_prob(success_count: f32, total_count: f32, turns: usize) -> f32 {
    if success_count <= 0.0 || total_count <= 0.0 || turns == 0 {
        return 0.0;
    }
    let mut fail = 1.0f32;
    for i in 0..turns {
        let denom = total_count - i as f32;
        if denom <= 0.0 {
            break;
        }
        fail *= (1.0 - success_count / denom).clamp(0.0, 1.0);
    }
    (1.0 - fail).clamp(0.0, 1.0)
}

fn improve_then_win_prob(
    improve_count: f32,
    wait_count: f32,
    total_count: f32,
    turns: usize,
) -> f32 {
    if improve_count <= 0.0 || wait_count <= 0.0 || turns < 2 {
        return 0.0;
    }
    let mut prob = 0.0f32;
    let mut fail_before = 1.0f32;
    for first_success_turn in 1..turns {
        let denom = total_count - (first_success_turn - 1) as f32;
        if denom <= 0.0 {
            break;
        }
        let first_success = fail_before * (improve_count / denom).clamp(0.0, 1.0);
        let rest_total = (denom - 1.0).max(1.0);
        prob +=
            first_success * at_least_one_prob(wait_count, rest_total, turns - first_success_turn);
        fail_before *= (1.0 - improve_count / denom).clamp(0.0, 1.0);
    }
    prob.clamp(0.0, 1.0)
}

fn binomial_at_least(trials: usize, successes: usize, p: f32) -> f32 {
    if successes == 0 {
        return 1.0;
    }
    if p <= 0.0 || trials < successes {
        return 0.0;
    }
    let mut total = 0.0f32;
    for k in successes..=trials {
        total += combination(trials, k) * p.powi(k as i32) * (1.0 - p).powi((trials - k) as i32);
    }
    total.clamp(0.0, 1.0)
}

fn combination(n: usize, k: usize) -> f32 {
    let k = k.min(n - k);
    let mut out = 1.0f32;
    for i in 0..k {
        out *= (n - i) as f32 / (i + 1) as f32;
    }
    out
}

#[cfg(test)]
fn score_tsumo(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    win_tile: u8,
) -> Option<f32> {
    score_tsumo_with_mods(input, counts_13, remaining, win_tile, ScoreMods::default())
}

#[cfg(test)]
fn score_tsumo_with_mods(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    win_tile: u8,
    mods: ScoreMods,
) -> Option<f32> {
    base_score_tsumo(
        input,
        counts_13,
        win_tile,
        input.akas_in_hand,
        SpVariant::FourPlayer,
    )
    .map(|base| {
        score_from_base(
            input,
            base,
            counts_13,
            remaining,
            win_tile,
            mods,
            SpVariant::FourPlayer,
        )
    })
    .or_else(|| {
        mods.haitei
            .then(|| exact_score_tsumo(input, counts_13, win_tile, mods, SpVariant::FourPlayer))
            .flatten()
    })
}

/// Discard-priority tie-break: honors > terminals > inner numbered. Used in
/// `discard_dp_slow` when two candidate discards score equal EV at integer-yen
/// precision.
///
/// Layout: honors (7) > terminals (6) > 2/8 (5) > 3/7 (4) > 4/6 (3) > 5 (2).
/// Takes 0..34 tile_type indices, not 136-form ids.
#[inline]
fn discard_priority(tile: u8) -> u8 {
    if tile >= 27 {
        return 7;
    }
    let pos = tile % 9;
    match pos {
        0 | 8 => 6,
        1 | 7 => 5,
        2 | 6 => 4,
        3 | 5 => 3,
        4 => 2,
        _ => 0,
    }
}

fn potentially_effective_for_draw(counts: &[u8; TILE_MAX], tile: usize) -> bool {
    // Kokushi-eligibility check: when the hand has no melds (= len_div3 >= 4),
    // every yaocchi tile is a potential kokushi-progression draw, regardless
    // of whether the player currently has a copy. The standard "adjacency or
    // count >= 1" filter would skip missing yaocchi like a third honor, but
    // those are precisely the kokushi wait tiles. We bias the heuristic
    // toward false-positives on yaocchi (paying the shanten lookup) so the
    // DP doesn't silently drop kokushi paths.
    let total: u8 = counts.iter().sum();
    let len_div3 = total / 3;
    let is_yaocchi = if tile < 27 {
        matches!(tile % 9, 0 | 8)
    } else {
        true
    };
    if len_div3 >= 4 && is_yaocchi {
        return true;
    }
    if tile >= 27 {
        return counts[tile] >= 1;
    }
    if counts[tile] >= 1 {
        return true;
    }
    let suit_base = (tile / 9) * 9;
    let pos = tile - suit_base;
    let lo = pos.saturating_sub(2);
    let hi = (pos + 2).min(8);
    for p in lo..=hi {
        if p == pos {
            continue;
        }
        if counts[suit_base + p] >= 1 {
            return true;
        }
    }
    false
}

/// Fast inline equivalent of `score::calculate_score(han, fu, is_oya, tsumo=true,
/// honba=0, 4-player) → winner total`. Avoids the `Score` struct
/// allocation and the honba/ron branches.
///
/// Hot path callers should prefer `sp_tsumo_total_lut` (table lookup, ~1ns)
/// instead of this branchy O(N) implementation. This function remains as the
/// authoritative source of truth and is used to populate the LUT itself.
#[inline]
fn sp_tsumo_total_fast(han: u32, fu: u32, is_oya: bool) -> u32 {
    sp_tsumo_total_fast_for_variant(han, fu, is_oya, SpVariant::FourPlayer)
}

#[inline]
fn sp_tsumo_total_fast_for_variant(han: u32, fu: u32, is_oya: bool, variant: SpVariant) -> u32 {
    let base_points: u32 = if han >= 5 {
        match han {
            5 => 2000,
            6 | 7 => 3000,
            8..=10 => 4000,
            11 | 12 => 6000,
            _ => 8000 * (han / 13).max(1),
        }
    } else {
        let fu_rounded = if fu == 25 { 25 } else { fu.div_ceil(10) * 10 };
        let bp = fu_rounded * (1u32 << (2 + han));
        bp.min(2000)
    };
    // pay_tsumo_oya / pay_tsumo_ko via 100-yen rounding.
    let (pay_oya, pay_ko) = if is_oya {
        (0u32, ((base_points * 2).div_ceil(100)) * 100)
    } else {
        (
            ((base_points * 2).div_ceil(100)) * 100,
            (base_points.div_ceil(100)) * 100,
        )
    };
    if pay_oya == 0 {
        // Dealer tsumo: every opponent pays the ko share.
        pay_ko.saturating_mul(match variant {
            SpVariant::FourPlayer => 3,
            SpVariant::ThreePlayer => 2,
        })
    } else {
        // Non-dealer tsumo: the dealer pays the oya share and remaining
        // non-dealers pay the ko share.
        pay_oya
            + pay_ko.saturating_mul(match variant {
                SpVariant::FourPlayer => 2,
                SpVariant::ThreePlayer => 1,
            })
    }
}

// ─────────────── Tsumo score lookup table ────────────────────────────
// `score_vector_for_win_compute` and `score_from_base` evaluate
// `sp_tsumo_total_fast` 4..(4 × ura_dist.len()) times **per wait tile**.
// At tenpai with ~65 unique waits per sample × ~8 invocations ≈ 520 calls
// per sample, the branchy arithmetic shows up in the profile. A static
// LUT indexed by (han, fu_idx, is_oya) collapses each call into a single
// array load — ~1.3KB total, fits in L1 alongside the agari_table working
// set and the SP DP scratch state.

/// Maximum han we need to index. Han caps at 13 (kazoe yakuman) but the
/// caller saturates upstream; we add 1 slot for safety.
const TSUMO_LUT_HAN: usize = 14;

/// Distinct fu values that appear in 4-player riichi: 20, 25, 30, 40, 50,
/// 60, 70, 80, 90, 100, 110. We add a 12-th slot for 120 just for safety
/// (some non-standard rule sets) and `fu_to_lut_idx` returns 11 for any
/// out-of-range value so the LUT lookup is always in-bounds.
const TSUMO_LUT_FU: usize = 12;

const FU_FOR_LUT_IDX: [u32; TSUMO_LUT_FU] = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

#[inline]
fn fu_to_lut_idx(fu: u32) -> usize {
    // Tight match — `match` lowers to a small jump table or branch chain
    // (LLVM emits a bit-test sequence for this dense set), faster than
    // searching `FU_FOR_LUT_IDX` linearly.
    match fu {
        20 => 0,
        25 => 1,
        30 => 2,
        40 => 3,
        50 => 4,
        60 => 5,
        70 => 6,
        80 => 7,
        90 => 8,
        100 => 9,
        110 => 10,
        _ => 11,
    }
}

/// LUT[han][fu_idx][is_oya]. Indexed by absolute han 0..=13 (caller must
/// `.min(13)` first), fu_idx via `fu_to_lut_idx`, and `is_oya as usize`.
/// Built once at first access via `LazyLock` — ~1.3KB resident in L1.
static TSUMO_SCORE_LUT: std::sync::LazyLock<[[[u32; 2]; TSUMO_LUT_FU]; TSUMO_LUT_HAN]> =
    std::sync::LazyLock::new(|| {
        let mut t = [[[0u32; 2]; TSUMO_LUT_FU]; TSUMO_LUT_HAN];
        for han in 0..TSUMO_LUT_HAN {
            for fu_idx in 0..TSUMO_LUT_FU {
                let fu = FU_FOR_LUT_IDX[fu_idx];
                t[han][fu_idx][0] = sp_tsumo_total_fast(han as u32, fu, false);
                t[han][fu_idx][1] = sp_tsumo_total_fast(han as u32, fu, true);
            }
        }
        t
    });

static TSUMO_SCORE_LUT_3P: std::sync::LazyLock<[[[u32; 2]; TSUMO_LUT_FU]; TSUMO_LUT_HAN]> =
    std::sync::LazyLock::new(|| {
        let mut table = [[[0u32; 2]; TSUMO_LUT_FU]; TSUMO_LUT_HAN];
        for han in 0..TSUMO_LUT_HAN {
            for (fu_idx, &fu) in FU_FOR_LUT_IDX.iter().enumerate() {
                table[han][fu_idx][0] =
                    sp_tsumo_total_fast_for_variant(han as u32, fu, false, SpVariant::ThreePlayer);
                table[han][fu_idx][1] =
                    sp_tsumo_total_fast_for_variant(han as u32, fu, true, SpVariant::ThreePlayer);
            }
        }
        table
    });

/// Hot-path tsumo score query. Replaces direct `sp_tsumo_total_fast` calls
/// in score_vector_for_win_compute / score_from_base / base_score_tsumo.
/// Caller is responsible for passing valid `fu` (one of the canonical values)
/// — non-canonical fu falls through to the slot-11 default which holds 120-fu
/// scores; this is unreachable in correct inputs.
#[inline]
fn sp_tsumo_total_lut(han: u32, fu: u32, is_oya: bool) -> u32 {
    sp_tsumo_total_lut_for_variant(han, fu, is_oya, SpVariant::FourPlayer)
}

#[inline]
fn sp_tsumo_total_lut_for_variant(han: u32, fu: u32, is_oya: bool, variant: SpVariant) -> u32 {
    let han_idx = (han as usize).min(TSUMO_LUT_HAN - 1);
    let fu_idx = fu_to_lut_idx(fu);
    match variant {
        SpVariant::FourPlayer => TSUMO_SCORE_LUT[han_idx][fu_idx][is_oya as usize],
        SpVariant::ThreePlayer => TSUMO_SCORE_LUT_3P[han_idx][fu_idx][is_oya as usize],
    }
}

fn base_score_tsumo(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    win_tile: u8,
    akas_in_hand: [bool; 3],
    variant: SpVariant,
) -> Option<BaseScore> {
    if counts_13[win_tile as usize] >= 4 {
        return None;
    }

    // Phase 3 lean fast path: bypasses HandEvaluator + yaku::calculate_yaku for
    // standard 14-tile tsumo wins. Verified equivalent to the legacy path on
    // every leaf of every real-replay sample. Falls back to the legacy path
    // for shapes the lean path doesn't handle (yakuman, chitoitsu, kokushi,
    // hands with kans).
    let is_oya = wind_from_tile(input.jikaze) == Wind::East;
    let assume_riichi = input.is_menzen && input.can_riichi;
    // Skip the lean attempt entirely if the hand contains kans — lean rejects
    // these and the failed attempt would just cost extra work.
    let has_kan = input.melds.iter().any(|m| {
        matches!(
            m.meld_type,
            MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan
        )
    });
    if variant == SpVariant::FourPlayer
        && !has_kan
        && let Some(lean) =
            crate::sp_yaku::compute_for_sp_tsumo(input, counts_13, win_tile, akas_in_hand)
        && lean.han > 0
    {
        // LUT-driven score (SP hot path: tsumo, 4 players, honba=0, no ura).
        let base_total = sp_tsumo_total_lut(lean.han, lean.fu, is_oya);
        return Some(BaseScore {
            total: base_total,
            han: lean.han,
            fu: lean.fu,
            is_oya,
            riichi: assume_riichi,
            apply_ura: assume_riichi && !lean.yakuman && !input.dora_indicators.is_empty(),
            honba: 0,
        });
    }

    // Legacy fallback path (allocates HandEvaluator + Vec).
    // Aka-as-win-tile fix: if `akas_in_hand[i]` is true but counts_13 has 0
    // of that 5x, the aka must be the win tile itself. counts_to_136_stack
    // can't see the win tile, so we'd lose the aka attribution. Pass the win
    // tile as `red=true` in that case so HandEvaluator gets the correct dora.
    let (tiles, tlen) = counts_to_136_stack(counts_13, akas_in_hand);
    let win_is_aka = match win_tile {
        4 => akas_in_hand[0] && counts_13[4] == 0,
        13 => akas_in_hand[1] && counts_13[13] == 0,
        22 => akas_in_hand[2] && counts_13[22] == 0,
        _ => false,
    };
    let conditions = Conditions {
        tsumo: true,
        riichi: assume_riichi,
        player_wind: wind_from_tile(input.jikaze),
        round_wind: wind_from_tile(input.bakaze),
        ..Conditions::default()
    };
    let win_tile_136 = tile_type_to_136(win_tile, win_is_aka);
    let result = match variant {
        SpVariant::FourPlayer => HandEvaluator::new_borrowed(&tiles[..tlen], &input.melds)
            .calc_borrowed(
                win_tile_136,
                &input.dora_indicators,
                &[],
                Some(conditions.clone()),
            ),
        SpVariant::ThreePlayer => HandEvaluator3P::new_borrowed(&tiles[..tlen], &input.melds)
            .calc_borrowed(
                win_tile_136,
                &input.dora_indicators,
                &[],
                Some(conditions.clone()),
            ),
    };
    if !result.is_win {
        return None;
    }

    let base_total =
        tsumo_total_for_variant(result.tsumo_agari_oya, result.tsumo_agari_ko, variant) as f32;
    Some(BaseScore {
        total: base_total as u32,
        han: result.han,
        fu: result.fu,
        is_oya: conditions.player_wind == Wind::East,
        riichi: conditions.riichi,
        apply_ura: conditions.riichi && !result.yakuman && !input.dora_indicators.is_empty(),
        honba: conditions.honba,
    })
}

fn score_from_base(
    input: &SpInput,
    base: BaseScore,
    counts_13: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    win_tile: u8,
    mods: ScoreMods,
    variant: SpVariant,
) -> f32 {
    let extra_han = timing_extra_han(base, mods);
    debug_assert_eq!(base.honba, 0, "SP base score must have honba=0");
    let base_total = if extra_han == 0 {
        base.total as f32
    } else {
        let han = base.han.saturating_add(extra_han).min(13);
        sp_tsumo_total_lut_for_variant(han, base.fu, base.is_oya, variant) as f32
    };
    if !base.apply_ura {
        return base_total;
    }

    let mut full_counts = *counts_13;
    full_counts[win_tile as usize] += 1;
    for meld in &input.melds {
        for &tile in &meld.tiles {
            let tile_type = (tile / 4) as usize;
            if tile_type < TILE_MAX {
                full_counts[tile_type] += 1;
            }
        }
    }

    let mut remaining_after_win = *remaining;
    remaining_after_win[win_tile as usize] =
        remaining_after_win[win_tile as usize].saturating_sub(1);
    let ura_dist = ura_distribution(
        &full_counts,
        &remaining_after_win,
        input.dora_indicators.len(),
        variant,
    );
    let mut expected = 0.0f32;
    for (ura_count, &prob) in ura_dist.iter().enumerate() {
        if prob <= 0.0 {
            continue;
        }
        let han = base
            .han
            .saturating_add(extra_han)
            .saturating_add(ura_count as u32)
            .min(13);
        expected +=
            prob * sp_tsumo_total_lut_for_variant(han, base.fu, base.is_oya, variant) as f32;
    }
    expected.max(base_total)
}

fn timing_extra_han(base: BaseScore, mods: ScoreMods) -> u32 {
    let mut han = 0;
    if base.riichi {
        if mods.ippatsu {
            han += 1;
        }
        if mods.double_riichi {
            han += 1;
        }
    }
    if mods.haitei {
        han += 1;
    }
    han
}

fn exact_score_tsumo(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    win_tile: u8,
    mods: ScoreMods,
    variant: SpVariant,
) -> Option<f32> {
    if counts_13[win_tile as usize] >= 4 {
        return None;
    }
    let (tiles, tlen) = counts_to_136_stack(counts_13, input.akas_in_hand);
    let conditions = Conditions {
        tsumo: true,
        riichi: input.is_menzen && input.can_riichi,
        double_riichi: input.is_menzen && input.can_riichi && mods.double_riichi,
        ippatsu: input.is_menzen && input.can_riichi && mods.ippatsu,
        haitei: mods.haitei,
        player_wind: wind_from_tile(input.jikaze),
        round_wind: wind_from_tile(input.bakaze),
        ..Conditions::default()
    };
    let result = match variant {
        SpVariant::FourPlayer => HandEvaluator::new_borrowed(&tiles[..tlen], &input.melds)
            .calc_borrowed(
                tile_type_to_136(win_tile, false),
                &input.dora_indicators,
                &[],
                Some(conditions),
            ),
        SpVariant::ThreePlayer => HandEvaluator3P::new_borrowed(&tiles[..tlen], &input.melds)
            .calc_borrowed(
                tile_type_to_136(win_tile, false),
                &input.dora_indicators,
                &[],
                Some(conditions),
            ),
    };
    result.is_win.then_some(tsumo_total_for_variant(
        result.tsumo_agari_oya,
        result.tsumo_agari_ko,
        variant,
    ) as f32)
}

/// 裏ドラ枚数の確率分布。返り値は `[f32; URA_DIST_LEN]` で、
/// `dist[u]` が「裏ドラがちょうど u 枚乗る確率」。
/// インデックスは 0..=20 (= D=5 表示牌すべて4枚乗りの最大ケース) をカバーする。
///
/// D=0: dist[0] = 1.0
/// D=1 (= 通常リーチの 83%): O(34) のクローズドフォーム。`gain_counts[u] / N` で終わり。
/// D≥2:   stack-only な反復畳み込み (state[d][u] を gain bucket ごとに更新)。
///        以前は再帰だったが、再帰呼び出しのオーバヘッドと `combination_f64` の
///        インライン化を兼ねて多項式畳み込み形式に書き直した。
const URA_DIST_LEN: usize = 21;

fn ura_distribution(
    full_counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    num_indicators: usize,
    variant: SpVariant,
) -> [f32; URA_DIST_LEN] {
    let mut dist = [0.0f32; URA_DIST_LEN];
    let total_remaining: usize = remaining.iter().map(|&c| c as usize).sum();
    let draws = num_indicators.min(5).min(total_remaining);
    if draws == 0 {
        dist[0] = 1.0;
        return dist;
    }

    // 各表示牌候補について、それを引いたとき対応する裏ドラ牌が手牌に
    // 何枚あるか (= gain) を集計し、gain 値ごとの「該当する表示牌の山残数」を
    // 5 要素のバケットに詰める。
    let mut gain_counts = [0u32; 5];
    for indicator in 0..TILE_MAX {
        let dora_tile = next_dora_tile_for_variant(indicator as u8, variant) as usize;
        let gain = full_counts[dora_tile] as usize;
        if gain < 5 {
            gain_counts[gain] += remaining[indicator] as u32;
        }
    }

    // Fast path: D=1 はバケットに 1 枚引くだけなので分布は自明。
    if draws == 1 {
        let inv_n = 1.0 / total_remaining as f32;
        for u in 0..5 {
            dist[u] = gain_counts[u] as f32 * inv_n;
        }
        return dist;
    }

    // 一般ケース (D = 2..=5): state[d][u] を gain bucket ごとに更新する。
    // state[d][u] = 「これまでに d 枚の表示牌を選び、対応する手牌枚数の
    //               総和が u になる場合の重み (= 各バケットからの組合せ数の積)」
    let mut state = [[0f64; URA_DIST_LEN]; 6];
    state[0][0] = 1.0;
    for g in 0..5usize {
        let bucket = gain_counts[g] as usize;
        if bucket == 0 {
            continue;
        }
        let mut new_state = [[0f64; URA_DIST_LEN]; 6];
        for d in 0..=draws {
            for u in 0..URA_DIST_LEN {
                let w = state[d][u];
                if w == 0.0 {
                    continue;
                }
                let max_take = bucket.min(draws - d);
                // C(bucket, take) を逐次更新: C(b, t+1) = C(b, t) * (b-t) / (t+1)
                let mut comb = 1.0f64;
                for take in 0..=max_take {
                    let new_u = u + g * take;
                    if new_u >= URA_DIST_LEN {
                        break;
                    }
                    new_state[d + take][new_u] += w * comb;
                    if take < max_take {
                        comb = comb * (bucket - take) as f64 / (take + 1) as f64;
                    }
                }
            }
        }
        state = new_state;
    }

    // 正規化。分母 C(N, D) でスケールしたあと、丸め誤差の補正に sum で再正規化。
    let denom = combination_f64(total_remaining, draws);
    if denom <= 0.0 {
        dist[0] = 1.0;
        return dist;
    }
    let denom_inv = 1.0 / denom;
    let max_ura = (draws * 4).min(URA_DIST_LEN - 1);
    let mut sum = 0.0f32;
    for u in 0..=max_ura {
        let p = (state[draws][u] * denom_inv) as f32;
        dist[u] = p;
        sum += p;
    }
    if sum > 0.0 && (sum - 1.0).abs() > 1e-6 {
        let inv_sum = 1.0 / sum;
        for p in &mut dist[..=max_ura] {
            *p *= inv_sum;
        }
    }
    dist
}

fn combination_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut out = 1.0f64;
    for i in 0..k {
        out *= (n - i) as f64 / (i + 1) as f64;
    }
    out
}

/// Total points received by the winner on tsumo (4-player rules).
///
/// `Score::pay_tsumo_oya` / `pay_tsumo_ko` semantics from `score::calculate_score`:
///   - ko (non-dealer) tsumo: oya pays `pay_tsumo_oya` (= 2x ko share), each ko pays `pay_tsumo_ko`,
///     total = pay_oya + 2 * pay_ko.
///   - oya (dealer)    tsumo: every ko pays `pay_tsumo_ko`, oya entry is 0,
///     total = 3 * pay_ko.
fn tsumo_total_for_variant(pay_tsumo_oya: u32, pay_tsumo_ko: u32, variant: SpVariant) -> u32 {
    if pay_tsumo_oya == 0 {
        pay_tsumo_ko.saturating_mul(match variant {
            SpVariant::FourPlayer => 3,
            SpVariant::ThreePlayer => 2,
        })
    } else {
        pay_tsumo_oya
            + pay_tsumo_ko.saturating_mul(match variant {
                SpVariant::FourPlayer => 2,
                SpVariant::ThreePlayer => 1,
            })
    }
}

fn rough_point_estimate(input: &SpInput, counts: &[u8; TILE_MAX], variant: SpVariant) -> f32 {
    let mut dora = input.akas_in_hand.iter().filter(|&&x| x).count() as f32;
    for &indicator in &input.dora_indicators {
        let dora_tile = next_dora_tile_for_variant(indicator / 4, variant) as usize;
        dora += counts[dora_tile] as f32;
    }
    let base = if input.is_menzen && input.can_riichi {
        2000.0
    } else {
        1000.0
    };
    base + dora * 1000.0
}

/// Stack-allocated 14-tile buffer (mahjong hands cap at 14 tiles).
/// Returns `(tiles, len)` so the caller can pass `&tiles[..len]` to
/// `HandEvaluator::new_borrowed` with zero heap allocation.
#[inline]
fn counts_to_136_stack(counts: &[u8; TILE_MAX], akas_in_hand: [bool; 3]) -> ([u8; 14], usize) {
    let mut tiles = [0u8; 14];
    let mut len = 0usize;
    for (tile, &count) in counts.iter().enumerate() {
        let red_index = match tile {
            4 => Some(0),
            13 => Some(1),
            22 => Some(2),
            _ => None,
        };
        let mut emitted = 0u8;
        if let Some(red_index) = red_index
            && akas_in_hand[red_index]
            && count > 0
        {
            tiles[len] = tile_type_to_136(tile as u8, true);
            len += 1;
            emitted = 1;
        }
        for copy in emitted..count {
            tiles[len] = (tile as u8) * 4 + copy + if red_index.is_some() { 1 } else { 0 };
            len += 1;
        }
    }
    (tiles, len)
}

fn tile_type_to_136(tile: u8, red: bool) -> u8 {
    if red && matches!(tile, 4 | 13 | 22) {
        tile * 4
    } else if matches!(tile, 4 | 13 | 22) {
        tile * 4 + 1
    } else {
        tile * 4
    }
}

fn wind_from_tile(tile: u8) -> Wind {
    let wind = if (27..=30).contains(&tile) {
        tile - 27
    } else {
        tile % 4
    };
    Wind::from(wind)
}

pub(crate) fn next_dora_tile(tile_type: u8) -> u8 {
    match tile_type {
        0..=7 | 9..=16 | 18..=25 => tile_type + 1,
        8 => 0,
        17 => 9,
        26 => 18,
        27..=30 => 27 + ((tile_type - 27 + 1) % 4),
        31..=33 => 31 + ((tile_type - 31 + 1) % 3),
        _ => tile_type,
    }
}

fn next_dora_tile_for_variant(tile_type: u8, variant: SpVariant) -> u8 {
    if variant == SpVariant::ThreePlayer {
        match tile_type {
            0 => 8,
            8 => 0,
            _ => next_dora_tile(tile_type),
        }
    } else {
        next_dora_tile(tile_type)
    }
}

fn broadcast_layout(buf: &mut [f32], ch_offset: usize, ch: usize, val: f32, tile_types: usize) {
    let start = (ch_offset + ch) * tile_types;
    for tile in 0..tile_types {
        buf[start + tile] = val;
    }
}

fn set_layout(
    buf: &mut [f32],
    ch_offset: usize,
    ch: usize,
    tile: usize,
    val: f32,
    tile_types: usize,
) {
    buf[(ch_offset + ch) * tile_types + tile] = val;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded_at(buf: &[f32], ch: usize, tile: usize) -> f32 {
        buf[ch * TILE_MAX + tile]
    }

    fn input_from_tiles(tile_types: &[u8], tsumos_left: u8) -> SpInput {
        let mut tehai = [0u8; TILE_MAX];
        let mut seen = [0u8; TILE_MAX];
        for &tile in tile_types {
            tehai[tile as usize] += 1;
            seen[tile as usize] += 1;
        }
        SpInput {
            tehai,
            akas_in_hand: [false; 3],
            tiles_seen: seen,
            akas_seen: [false; 3],
            dora_indicators: vec![],
            melds: vec![],
            bakaze: 27,
            jikaze: 27,
            is_menzen: true,
            can_riichi: true,
            can_double_riichi: false,
            tsumos_left,
            discard_candidates: vec![],
        }
    }

    fn tenpai_fixture(tsumos_left: u8) -> SpInput {
        // 123456789m 12p 11s + extra 5s. Discarding 5s leaves a 3p wait.
        input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18, 22], tsumos_left)
    }

    fn assert_series_monotonic(series: &[f32; SP_MAX_TURNS], horizon: usize) {
        for pair in series[..horizon.min(SP_MAX_TURNS)].windows(2) {
            assert!(
                pair[1] + 1e-6 >= pair[0],
                "series must be non-decreasing: {series:?}"
            );
        }
    }

    #[test]
    fn sp_generates_candidates_and_178_channels() {
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        assert!(!result.candidates.is_empty());
        let encoded = encode_sp(&result);
        assert_eq!(encoded.len(), SP_CHANNELS * TILE_MAX);
        assert_eq!(SP_CHANNELS, 178);
    }

    #[test]
    fn sanma_sp_excludes_middle_manzu_and_encodes_27_columns() {
        let common = input_from_tiles(&[0, 0, 0, 8, 8, 8, 9, 10, 11, 18, 19, 20, 27, 27], 10);
        let input = SpInput3P {
            tehai: common.tehai,
            akas_in_hand: common.akas_in_hand,
            tiles_seen: common.tiles_seen,
            akas_seen: common.akas_seen,
            dora_indicators: common.dora_indicators,
            melds: common.melds,
            bakaze: common.bakaze,
            jikaze: common.jikaze,
            is_menzen: common.is_menzen,
            can_riichi: common.can_riichi,
            can_double_riichi: common.can_double_riichi,
            tsumos_left: common.tsumos_left,
            discard_candidates: common.discard_candidates,
        };

        let result = calculate_sp_3p(&input);
        assert!(!result.candidates.is_empty());
        for candidate in &result.candidates {
            assert!(!(1..=7).contains(&candidate.tile));
            for tile in 1..=7 {
                assert_eq!(candidate.required_tiles[tile], 0.0);
                assert_eq!(candidate.yaku_progress_tiles[tile], 0.0);
                assert_eq!(candidate.future_wait_tiles[tile], 0.0);
            }
        }
        let encoded = encode_sp_3p(&result);
        assert_eq!(encoded.len(), SP_CHANNELS * 27);
        assert!(encoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn sanma_dora_cycles_between_one_and_nine_manzu() {
        assert_eq!(next_dora_tile_for_variant(0, SpVariant::ThreePlayer), 8);
        assert_eq!(next_dora_tile_for_variant(8, SpVariant::ThreePlayer), 0);
        assert_eq!(next_dora_tile_for_variant(9, SpVariant::ThreePlayer), 10);
    }

    #[test]
    fn sanma_tenpai_score_matches_the_three_player_evaluator() {
        let common = input_from_tiles(&[0, 0, 0, 8, 8, 8, 9, 10, 11, 18, 19, 20, 27, 27], 10);
        let input = SpInput3P {
            tehai: common.tehai,
            akas_in_hand: common.akas_in_hand,
            tiles_seen: common.tiles_seen,
            akas_seen: common.akas_seen,
            dora_indicators: common.dora_indicators,
            melds: common.melds,
            bakaze: common.bakaze,
            jikaze: common.jikaze,
            is_menzen: common.is_menzen,
            can_riichi: common.can_riichi,
            can_double_riichi: common.can_double_riichi,
            tsumos_left: common.tsumos_left,
            discard_candidates: vec![27],
        };
        let result = calculate_sp_3p(&input);
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.tile == 27)
            .unwrap();

        let mut counts_13 = input.tehai;
        counts_13[27] -= 1;
        let (tiles, tile_count) = counts_to_136_stack(&counts_13, input.akas_in_hand);
        let conditions = Conditions {
            tsumo: true,
            riichi: true,
            player_wind: Wind::East,
            round_wind: Wind::East,
            ..Conditions::default()
        };
        let score = HandEvaluator3P::new_borrowed(&tiles[..tile_count], &[]).calc_borrowed(
            tile_type_to_136(27, false),
            &[],
            &[],
            Some(conditions),
        );
        assert!(score.is_win);
        let expected = tsumo_total_for_variant(
            score.tsumo_agari_oya,
            score.tsumo_agari_ko,
            SpVariant::ThreePlayer,
        ) as f32;
        assert_eq!(candidate.min_point, expected);
        assert_eq!(candidate.mean_point, expected);
        assert_eq!(candidate.max_point, expected);
    }

    #[test]
    fn sp_target_point_probs_are_monotone_non_increasing() {
        // P(score ≥ T_k) must be non-increasing in T_k since the targets are
        // sorted ascending.
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        let mut saw_signal = false;
        for c in &result.candidates {
            for k in 1..target_points::N {
                let prev = c.point_achievement_probs[k - 1];
                let cur = c.point_achievement_probs[k];
                assert!(
                    prev + 1e-6 >= cur,
                    "discard={} k={}: prob {} → {} should be non-increasing",
                    c.tile,
                    k,
                    prev,
                    cur,
                );
                if prev > 0.0 {
                    saw_signal = true;
                }
            }
        }
        assert!(
            saw_signal,
            "expected at least one candidate with non-zero target-prob"
        );
    }

    #[test]
    fn sp_future_wait_is_superset_of_required_tiles_at_tenpai() {
        // For tenpai-maintaining candidates, `future_wait_tiles` should
        // include every tile in `required_tiles` (the wait set is a subset of
        // structurally-progressive draws). Verifies the new ch 138..172 block
        // is consistent with the legacy ch 2..36 block.
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        for c in &result.candidates {
            for tile in 0..TILE_MAX {
                if c.required_tiles[tile] > 0.0 {
                    assert!(
                        c.future_wait_tiles[tile] > 0.0,
                        "discard={} tile={}: required_tiles set but future_wait_tiles unset",
                        c.tile,
                        tile,
                    );
                }
            }
        }
    }

    #[test]
    fn sp_yaku_mask_riichi_for_menzen_riichi_eligible() {
        // tenpai_fixture uses menzen + can_riichi=true. Every optimal
        // candidate should set the RIICHI and MENZEN_TSUMO bits.
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        let optimal_count = result
            .candidates
            .iter()
            .filter(|c| c.yaku_mask & yaku_mask_bits::RIICHI != 0)
            .count();
        assert!(
            optimal_count > 0,
            "expected at least one riichi-eligible candidate"
        );
        for c in &result.candidates {
            if c.yaku_mask & yaku_mask_bits::RIICHI != 0 {
                assert!(
                    c.yaku_mask & yaku_mask_bits::MENZEN_TSUMO != 0,
                    "menzen+riichi candidate must also set MENZEN_TSUMO"
                );
            }
        }
    }

    #[test]
    fn sp_min_mean_max_point_populated_for_optimal_tenpai() {
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        // At least one optimal candidate (= tenpai-maintaining discard) must
        // have populated min/mean/max scoring.
        let any = result
            .candidates
            .iter()
            .any(|c| c.max_point > 0.0 && c.mean_point > 0.0 && c.min_point > 0.0);
        assert!(
            any,
            "expected scoring stats for at least one tenpai candidate"
        );
        // Sanity: min ≤ mean ≤ max within each candidate.
        for c in &result.candidates {
            if c.max_point > 0.0 {
                assert!(c.min_point <= c.mean_point + 1e-3);
                assert!(c.mean_point <= c.max_point + 1e-3);
            }
        }
    }

    #[test]
    fn tenpai_candidate_has_win_probability() {
        // Discarding 5s leaves 123456789m 12p 11s, waiting 3p by riichi tsumo.
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.tile == 22)
            .expect("5s discard candidate");
        assert!(candidate.tenpai_probs[0] > 0.0);
        assert!(candidate.win_probs[1] > 0.0);
        assert!(candidate.exp_values[1] > 0.0);
    }

    #[test]
    fn encode_sp_layout_matches_candidate_data() {
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        let encoded = encode_sp(&result);
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.tile == 22)
            .expect("5s discard candidate");

        assert!(
            candidate.required_tiles[11] > 0.0,
            "discarding 5s should mark 3p as a required/winning tile"
        );
        assert_eq!(encoded_at(&encoded, 2 + 22, 11), 1.0);
        assert_eq!(
            encoded_at(&encoded, 36 + 22, 11),
            if candidate.yaku_progress_tiles[11] > 0.0 {
                1.0
            } else {
                0.0
            }
        );

        let max_ev = result.candidates[0].exp_values[0].max(0.0);
        let ev_scale = if max_ev >= 1.0 { 1.0 / max_ev } else { 0.0 };
        for turn in 0..SP_MAX_TURNS {
            assert_eq!(
                encoded_at(&encoded, 72 + turn, 22),
                candidate.tenpai_probs[turn]
            );
            assert_eq!(
                encoded_at(&encoded, 89 + turn, 22),
                candidate.win_probs[turn]
            );
            assert_eq!(
                encoded_at(&encoded, 106 + turn, 22),
                (candidate.exp_values[turn] * ev_scale).clamp(0.0, 1.0)
            );
        }

        let expected_ch0 = max_ev.min(100_000.0) / 100_000.0;
        let expected_ch1 = max_ev.min(30_000.0) / 30_000.0;
        for tile in 0..TILE_MAX {
            assert_eq!(encoded_at(&encoded, 0, tile), expected_ch0);
            assert_eq!(encoded_at(&encoded, 1, tile), expected_ch1);
        }

        let best_required = result
            .candidates
            .iter()
            .max_by(|a, b| {
                a.num_required_tiles
                    .partial_cmp(&b.num_required_tiles)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.tile.cmp(&a.tile))
            })
            .expect("best required candidate");
        for tile in 0..TILE_MAX {
            let expected = if tile == best_required.tile as usize {
                1.0
            } else {
                0.0
            };
            assert_eq!(encoded_at(&encoded, 70, tile), expected);
        }
    }

    #[test]
    fn encoded_values_are_finite_bounded_and_monotonic() {
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        let encoded = encode_sp(&result);

        assert!(encoded.iter().all(|value| value.is_finite()));
        for ch in 72..106 {
            for tile in 0..TILE_MAX {
                let value = encoded_at(&encoded, ch, tile);
                assert!(
                    (0.0..=1.0).contains(&value),
                    "probability channel {ch}, tile {tile} out of range: {value}"
                );
            }
        }
        for ch in 106..123 {
            for tile in 0..TILE_MAX {
                let value = encoded_at(&encoded, ch, tile);
                assert!(
                    (0.0..=1.0).contains(&value),
                    "EV channel {ch}, tile {tile} out of range: {value}"
                );
            }
        }
        for candidate in &result.candidates {
            assert_series_monotonic(&candidate.tenpai_probs, input.tsumos_left as usize);
            assert_series_monotonic(&candidate.win_probs, input.tsumos_left as usize);
            assert_series_monotonic(&candidate.exp_values, input.tsumos_left as usize);
        }
    }

    #[test]
    fn encode_sp_is_deterministic() {
        let input = tenpai_fixture(10);
        let first = encode_sp(&calculate_sp(&input));
        let second = encode_sp(&calculate_sp(&input));
        assert_eq!(first, second);
    }

    #[test]
    fn visible_zero_tile_is_not_encoded_as_required_or_yaku_wait() {
        let mut input = tenpai_fixture(10);
        input.tiles_seen[11] = 4;
        let result = calculate_sp(&input);
        let encoded = encode_sp(&result);
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.tile == 22)
            .expect("5s discard candidate");

        assert_eq!(candidate.required_tiles[11], 0.0);
        assert_eq!(candidate.yaku_progress_tiles[11], 0.0);
        assert_eq!(encoded_at(&encoded, 2 + 22, 11), 0.0);
        assert_eq!(encoded_at(&encoded, 36 + 22, 11), 0.0);
    }

    #[test]
    fn four_plus_shanten_skips_yaku_progress_map() {
        let input = input_from_tiles(&[0, 2, 5, 8, 9, 12, 15, 18, 21, 24, 27, 29, 31, 33], 10);
        let result = calculate_sp(&input);
        let encoded = encode_sp(&result);

        assert!(!result.candidates.is_empty());
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.num_yaku_progress_tiles == 0.0)
        );
        for ch in 36..70 {
            for tile in 0..TILE_MAX {
                assert_eq!(encoded_at(&encoded, ch, tile), 0.0);
            }
        }
    }

    #[test]
    fn dora_indicator_increases_tenpai_expected_value() {
        let base_input = tenpai_fixture(10);
        let mut dora_input = tenpai_fixture(10);
        dora_input.dora_indicators = vec![40]; // 2p indicator makes 3p the dora.
        dora_input.tiles_seen[10] += 1;

        let base = calculate_sp(&base_input);
        let with_dora = calculate_sp(&dora_input);
        let base_candidate = base
            .candidates
            .iter()
            .find(|candidate| candidate.tile == 22)
            .expect("base 5s discard candidate");
        let dora_candidate = with_dora
            .candidates
            .iter()
            .find(|candidate| candidate.tile == 22)
            .expect("dora 5s discard candidate");

        assert!(
            dora_candidate.exp_values[1] > base_candidate.exp_values[1],
            "dora wait should increase EV: base={}, dora={}",
            base_candidate.exp_values[1],
            dora_candidate.exp_values[1]
        );
    }

    #[test]
    fn riichi_ura_dora_increases_expected_tsumo_score() {
        let base_input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 1);
        let base_remaining = remaining_counts(&base_input);
        let base_score = score_tsumo(&base_input, &base_input.tehai, &base_remaining, 11)
            .expect("3p tsumo should win");

        let mut ura_input = base_input.clone();
        ura_input.dora_indicators = vec![132]; // C indicator makes P the dora, absent from hand.
        ura_input.tiles_seen[33] += 1;
        let ura_remaining = remaining_counts(&ura_input);
        let ura_score = score_tsumo(&ura_input, &ura_input.tehai, &ura_remaining, 11)
            .expect("3p tsumo should win with ura expectation");

        assert!(
            ura_score > base_score,
            "riichi ura expectation should increase score: base={base_score}, ura={ura_score}"
        );
    }

    #[test]
    fn riichi_timing_yaku_increase_expected_tsumo_score() {
        let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 3);
        let remaining = remaining_counts(&input);
        let base =
            score_tsumo_with_mods(&input, &input.tehai, &remaining, 11, ScoreMods::default())
                .expect("3p tsumo should win");
        let ippatsu = score_tsumo_with_mods(
            &input,
            &input.tehai,
            &remaining,
            11,
            ScoreMods {
                ippatsu: true,
                ..ScoreMods::default()
            },
        )
        .expect("ippatsu 3p tsumo should win");
        let haitei = score_tsumo_with_mods(
            &input,
            &input.tehai,
            &remaining,
            11,
            ScoreMods {
                haitei: true,
                ..ScoreMods::default()
            },
        )
        .expect("haitei 3p tsumo should win");

        assert!(ippatsu > base, "ippatsu should increase score");
        assert!(haitei > base, "haitei should increase score");
    }

    #[test]
    fn double_riichi_increases_first_tenpai_score() {
        let mut input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 3);
        input.can_double_riichi = true;
        let remaining = remaining_counts(&input);
        let regular_riichi =
            score_tsumo_with_mods(&input, &input.tehai, &remaining, 11, ScoreMods::default())
                .expect("regular riichi 3p tsumo should win");
        let double_riichi = score_tsumo_with_mods(
            &input,
            &input.tehai,
            &remaining,
            11,
            ScoreMods {
                double_riichi: true,
                ..ScoreMods::default()
            },
        )
        .expect("double riichi 3p tsumo should win");

        assert!(
            double_riichi > regular_riichi,
            "double riichi should increase score"
        );
    }

    #[test]
    fn ura_dora_is_ignored_without_riichi() {
        let mut base_input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 1);
        base_input.can_riichi = false;
        let base_remaining = remaining_counts(&base_input);
        let base_score = score_tsumo(&base_input, &base_input.tehai, &base_remaining, 11)
            .expect("3p tsumo should win without riichi");

        let mut ura_input = base_input.clone();
        ura_input.dora_indicators = vec![132]; // C indicator makes P the dora, absent from hand.
        ura_input.tiles_seen[33] += 1;
        let ura_remaining = remaining_counts(&ura_input);
        let ura_score = score_tsumo(&ura_input, &ura_input.tehai, &ura_remaining, 11)
            .expect("3p tsumo should win without riichi and without ura");

        assert_eq!(ura_score, base_score);
    }

    #[test]
    fn ura_distribution_for_multiple_indicators_is_normalized() {
        let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 1);
        let mut full_counts = input.tehai;
        full_counts[11] += 1;
        let mut remaining = remaining_counts(&input);
        remaining[11] -= 1;

        let dist = ura_distribution(&full_counts, &remaining, 3, SpVariant::FourPlayer);
        let sum = dist.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}, dist={dist:?}");
        assert!(
            dist.iter()
                .enumerate()
                .any(|(ura, &prob)| ura > 0 && prob > 0.0),
            "some positive ura count should be possible: {dist:?}"
        );
    }

    #[test]
    fn observation_input_allows_future_riichi_when_closed_and_has_points() {
        let hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18]
            .into_iter()
            .map(|tile| tile_type_to_136(tile, false))
            .collect::<Vec<_>>();
        let obs = Observation::new(
            0,
            [hand, vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            Default::default(),
            vec![132],
            [1000, 25000, 25000, 25000],
            [false; 4],
            vec![],
            vec![],
            0,
            0,
            27,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        );

        let input = SpInput::from_observation(&obs);
        assert!(input.is_menzen);
        assert!(
            input.can_riichi,
            "closed hands with at least 1000 points should evaluate future tenpai as riichi-capable"
        );
        assert!(
            input.can_double_riichi,
            "first-discard closed hands with at least 1000 points should allow double riichi"
        );
    }

    #[test]
    fn tenpai_series_uses_actual_remaining_wait_count() {
        // 123456789m 12p 11s waits on 3p. If the only unknown tile is 3p,
        // the first draw wins with probability 1.
        let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 1);
        let counts = input.tehai;
        let mut remaining = [0u8; TILE_MAX];
        remaining[11] = 1;

        let mut dp = DpContext::new(&input, SpVariant::FourPlayer);
        let (tenpai, win, ev) = dp.series(&counts, &remaining, 1);

        assert_eq!(tenpai[0], 1.0);
        assert_eq!(win[0], 1.0);
        assert!(ev[0] > 0.0);
    }

    #[test]
    fn one_shanten_series_branches_by_first_effective_tile() {
        // 123456789m 1p 11s 8s is one-shanten. 2p first makes 3p a win,
        // so it cannot win on the first draw but can win by the second.
        let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25], 2);
        let counts = input.tehai;
        let mut remaining = [0u8; TILE_MAX];
        remaining[10] = 1;
        remaining[11] = 1;
        // 有効牌だけの山だと「2巡目開始 = 1巡目が必ず有効牌」という確率0条件付けが
        // 発生して退化するため、無関係な牌を山に足して条件付きが定義されるようにする。
        remaining[20] = 4;
        remaining[21] = 4;

        let mut dp = DpContext::new(&input, SpVariant::FourPlayer);
        let (tenpai, win, ev) = dp.series(&counts, &remaining, 2);

        assert!(tenpai[0] > 0.0);
        assert_eq!(win[0], 0.0);
        assert!(win[1] > 0.0);
        assert!(ev[1] > 0.0);
    }

    #[test]
    fn tenpai_series_matches_without_replacement_probability_oracle() {
        // Two winning 3p remain among ten unknown tiles. Once tenpai, missing
        // on each draw leaves a smaller without-replacement population, so
        // P(win by h) = 1 - C(8, h) / C(10, h).
        let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 18], 3);
        let counts = input.tehai;
        let mut remaining = [0u8; TILE_MAX];
        remaining[11] = 2;
        remaining[20] = 4;
        remaining[21] = 4;

        let mut dp = DpContext::new(&input, SpVariant::FourPlayer);
        let (tenpai, win, ev) = dp.series(&counts, &remaining, 3);

        let mut miss = 1.0f32;
        for turn in 0..3 {
            miss *= (8 - turn) as f32 / (10 - turn) as f32;
            let expected_win = 1.0 - miss;
            assert!((tenpai[turn] - 1.0).abs() < 1e-6);
            assert!(
                (win[turn] - expected_win).abs() < 1e-6,
                "turn {turn}: got {}, expected {expected_win}",
                win[turn]
            );
            assert!(ev[turn] > 0.0);
        }
    }

    #[test]
    fn one_shanten_two_draw_series_matches_exact_path_enumeration() {
        // 123456789m 1p 11s 8s needs both 2p and 3p. The two winning
        // two-draw orders are [2p, 3p] and [3p, 2p] among 10 * 9 ordered
        // draws, hence P(win by draw 2) = 2 / 90 = 1 / 45.
        let input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25], 2);
        let counts = input.tehai;
        let mut remaining = [0u8; TILE_MAX];
        remaining[10] = 1;
        remaining[11] = 1;
        remaining[20] = 4;
        remaining[21] = 4;

        let mut dp = DpContext::new(&input, SpVariant::FourPlayer);
        let (_tenpai, win, ev) = dp.series(&counts, &remaining, 2);

        assert_eq!(win[0], 0.0);
        assert!(
            (win[1] - 1.0 / 45.0).abs() < 1e-6,
            "got {}, expected {}",
            win[1],
            1.0 / 45.0
        );
        assert!(ev[1] > 0.0);
    }

    // ----- Semantic correctness tests -----

    /// 全候補・全巡目で win_prob ≤ tenpai_prob を満たすこと。
    /// 聴牌に達さなければ和了できないという条件は SP の設計不変条件。
    #[test]
    fn win_prob_never_exceeds_tenpai_prob_per_turn() {
        let inputs = [
            tenpai_fixture(10),
            input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10),
            input_from_tiles(&[0, 1, 4, 5, 8, 9, 10, 13, 18, 19, 22, 27, 28, 31], 10),
        ];
        for input in &inputs {
            let result = calculate_sp(input);
            for c in &result.candidates {
                for turn in 0..SP_MAX_TURNS {
                    assert!(
                        c.win_probs[turn] <= c.tenpai_probs[turn] + 1e-5,
                        "candidate tile {} turn {}: win {} > tenpai {}",
                        c.tile,
                        turn,
                        c.win_probs[turn],
                        c.tenpai_probs[turn],
                    );
                }
            }
        }
    }

    /// shanten s 状態の手牌からは min s 回の自摸を要するため、
    /// `series[0..s-1]` の win_prob は 0 でなければならない。
    #[test]
    fn shanten_lower_bounds_first_winning_turn() {
        // tehai sums to 14; for each candidate discard, the remaining 13-tile hand has shanten s_d.
        // At every turn t in 0..s_d the win_prob must be 0.
        let inputs = [
            tenpai_fixture(10),
            input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10),
            input_from_tiles(&[0, 1, 4, 5, 8, 9, 10, 13, 18, 19, 22, 27, 28, 31], 10),
        ];
        for input in &inputs {
            let result = calculate_sp(input);
            for c in &result.candidates {
                let mut post = input.tehai;
                if post[c.tile as usize] == 0 {
                    continue;
                }
                post[c.tile as usize] -= 1;
                let shanten = shanten_of_counts(&post);
                if shanten <= 0 {
                    continue;
                }
                for turn in 0..(shanten as usize).min(SP_MAX_TURNS) {
                    assert!(
                        c.win_probs[turn] < 1e-6,
                        "candidate tile {} (post-shanten {}): win_probs[{}]={} should be ~0",
                        c.tile,
                        shanten,
                        turn,
                        c.win_probs[turn],
                    );
                }
            }
        }
    }

    /// 聴牌維持の打牌候補は、ホライズン内の全巡目で tenpai_prob = 1 でなければならない
    /// (聴牌後は自摸の有無に関わらず聴牌状態が保たれる)。
    #[test]
    fn tenpai_candidate_has_full_tenpai_probability_in_horizon() {
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        let horizon = (input.tsumos_left as usize).min(SP_MAX_TURNS);
        for c in &result.candidates {
            let mut post = input.tehai;
            if post[c.tile as usize] == 0 {
                continue;
            }
            post[c.tile as usize] -= 1;
            let s = shanten_of_counts(&post);
            if s != 0 {
                continue;
            }
            for turn in 0..horizon {
                assert!(
                    (c.tenpai_probs[turn] - 1.0).abs() < 1e-5,
                    "tenpai discard tile {} turn {}: tenpai_prob={}",
                    c.tile,
                    turn,
                    c.tenpai_probs[turn],
                );
            }
        }
    }

    /// `encode_sp` の best required-tile marker (channel 70) は、required-tile column
    /// (channel 2 + d) に有効値を持つ候補 d を指していなければならない。
    #[test]
    fn best_required_marker_points_to_a_real_candidate() {
        let inputs = [
            tenpai_fixture(10),
            input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10),
        ];
        for input in &inputs {
            let result = calculate_sp(input);
            let encoded = encode_sp(&result);
            let mut marker_tile: Option<usize> = None;
            for tile in 0..TILE_MAX {
                if encoded[70 * TILE_MAX + tile] > 0.5 {
                    assert!(
                        marker_tile.is_none(),
                        "best required-tile marker is not one-hot"
                    );
                    marker_tile = Some(tile);
                }
            }
            if let Some(tile) = marker_tile {
                let any = (0..TILE_MAX).any(|t| encoded[(2 + tile) * TILE_MAX + t] > 0.5);
                assert!(
                    any,
                    "marker at tile {} but channel 2+{} has no required tiles",
                    tile, tile,
                );
            }
        }
    }

    /// 全候補の tenpai/win/ev 系列はホライズン内で単調非減少 (ターン0..horizon-1)。
    #[test]
    fn series_are_monotonic_within_horizon() {
        let inputs = [
            tenpai_fixture(10),
            input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10),
            input_from_tiles(&[0, 1, 4, 5, 8, 9, 10, 13, 18, 19, 22, 27, 28, 31], 10),
        ];
        for input in &inputs {
            let result = calculate_sp(input);
            let horizon = (input.tsumos_left as usize).min(SP_MAX_TURNS);
            for c in &result.candidates {
                for series in [&c.tenpai_probs, &c.win_probs, &c.exp_values] {
                    for turn in 1..horizon {
                        assert!(
                            series[turn] + 1e-5 >= series[turn - 1],
                            "candidate tile {}: series not monotonic at turn {}: {:?}",
                            c.tile,
                            turn,
                            series,
                        );
                    }
                }
            }
        }
    }

    /// 残り牌の合計枚数 = sum(tiles_in_wall) で、required_tiles[t] > 0 なら remaining[t] > 0。
    /// SP の required_tiles は実際に山から引ける牌でなければならない。
    #[test]
    fn required_tiles_are_still_drawable() {
        let inputs = [
            tenpai_fixture(10),
            input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25, 26], 10),
        ];
        for input in &inputs {
            let remaining = remaining_counts(input);
            let result = calculate_sp(input);
            for c in &result.candidates {
                for t in 0..TILE_MAX {
                    if c.required_tiles[t] > 0.0 {
                        assert!(
                            remaining[t] > 0,
                            "required tile {} for discard {} has 0 remaining",
                            t,
                            c.tile,
                        );
                    }
                    if c.yaku_progress_tiles[t] > 0.0 {
                        assert!(
                            remaining[t] > 0,
                            "yaku-progress tile {} for discard {} has 0 remaining",
                            t,
                            c.tile,
                        );
                    }
                }
            }
        }
    }

    /// horizon を増やすと win_prob[最後の有効ターン] は単調非減少 (ホライズンが長いほうが
    /// 和了機会が多い)。
    #[test]
    fn longer_horizon_does_not_decrease_terminal_win_prob() {
        // Construct a 1-shanten with several useless tiles in the wall so the conditional
        // probabilities are well defined.
        let mut input = input_from_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 18, 25], 1);
        // augment tiles_seen so n_left_tiles is realistic
        for t in 13..30 {
            input.tiles_seen[t] = 0;
        }
        let mut prev_win: Option<f32> = None;
        for h in [3u8, 5, 8, 12, 17] {
            input.tsumos_left = h;
            let mut dp = DpContext::new(&input, SpVariant::FourPlayer);
            let counts = input.tehai;
            let remaining = remaining_counts(&input);
            let (_t, win, _e) = dp.series(&counts, &remaining, h as usize);
            let terminal = win[(h as usize) - 1];
            if let Some(prev) = prev_win {
                assert!(
                    terminal + 1e-5 >= prev,
                    "longer horizon h={} gave smaller terminal win {} < prev {}",
                    h,
                    terminal,
                    prev,
                );
            }
            prev_win = Some(terminal);
        }
    }
}
