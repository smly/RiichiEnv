use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::rc::Rc;

use crate::action::ActionType;
use crate::hand_evaluator::HandEvaluator;
use crate::observation::Observation;
use crate::score;
use crate::shanten;
use crate::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};

pub const SP_MAX_TURNS: usize = 17;
pub const SP_CHANNELS: usize = 123;

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

#[derive(Debug, Clone)]
pub struct SpCandidate {
    pub tile: u8,
    pub tenpai_probs: [f32; SP_MAX_TURNS],
    pub win_probs: [f32; SP_MAX_TURNS],
    pub exp_values: [f32; SP_MAX_TURNS],
    pub required_tiles: [f32; TILE_MAX],
    pub yaku_progress_tiles: [f32; TILE_MAX],
    pub min_point: f32,
    pub mean_point: f32,
    pub high_point: f32,
    pub num_required_tiles: f32,
    pub num_yaku_progress_tiles: f32,
}

#[derive(Debug, Clone)]
pub struct SpResult {
    pub candidates: Vec<SpCandidate>,
}

impl SpInput {
    pub fn from_observation(obs: &Observation) -> Self {
        let player_idx = obs.player_id as usize;
        let mut tehai = [0u8; TILE_MAX];
        let mut akas_in_hand = [false; 3];
        for &tile in &obs.hands[player_idx] {
            let tile_type = (tile / 4) as usize;
            if tile_type < TILE_MAX {
                tehai[tile_type] = tehai[tile_type].saturating_add(1);
            }
            match tile {
                16 => akas_in_hand[0] = true,
                52 => akas_in_hand[1] = true,
                88 => akas_in_hand[2] = true,
                _ => {}
            }
        }

        let mut tiles_seen = [0u8; TILE_MAX];
        let mut akas_seen = akas_in_hand;
        let mut mark_aka = |t: u32| match t {
            16 => akas_seen[0] = true,
            52 => akas_seen[1] = true,
            88 => akas_seen[2] = true,
            _ => {}
        };
        for &tile in &obs.hands[player_idx] {
            add_seen(&mut tiles_seen, tile);
        }
        for melds in &obs.melds {
            for meld in melds {
                for &tile in &meld.tiles {
                    add_seen(&mut tiles_seen, tile as u32);
                    mark_aka(tile as u32);
                }
            }
        }
        for discards in &obs.discards {
            for &tile in discards {
                add_seen(&mut tiles_seen, tile);
                mark_aka(tile);
            }
        }
        for &tile in &obs.dora_indicators {
            add_seen(&mut tiles_seen, tile);
            mark_aka(tile);
        }
        drop(mark_aka);

        let mut discard_candidates = Vec::new();
        for action in &obs._legal_actions {
            if matches!(action.action_type, ActionType::Discard)
                && let Some(tile) = action.tile
            {
                let tile_type = tile / 4;
                if !discard_candidates.contains(&tile_type) {
                    discard_candidates.push(tile_type);
                }
            }
        }
        discard_candidates.sort_unstable();

        let rel_seat = (obs.player_id + 4 - obs.oya) % 4;
        let can_riichi = obs.riichi_declared[player_idx] || obs.scores[player_idx] >= 1000;
        let can_double_riichi = can_riichi
            && obs.discards.iter().all(Vec::is_empty)
            && obs.melds.iter().all(Vec::is_empty);

        Self {
            tehai,
            akas_in_hand,
            tiles_seen,
            akas_seen,
            dora_indicators: obs.dora_indicators.iter().map(|&x| x as u8).collect(),
            melds: obs.melds[player_idx].clone(),
            bakaze: obs.round_wind,
            jikaze: 27 + rel_seat,
            is_menzen: obs.melds[player_idx].iter().all(|m| !m.opened),
            can_riichi,
            can_double_riichi,
            tsumos_left: remaining_self_draws(obs),
            discard_candidates,
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
    let base = base_score_tsumo(input, tehai_13, win_tile, input.akas_in_hand)?;
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
fn aka_after_discard(
    akas: [bool; 3],
    counts: &[u8; TILE_MAX],
    tile: u8,
) -> [bool; 3] {
    let red_idx = match tile {
        4 => 0,
        13 => 1,
        22 => 2,
        _ => return akas,
    };
    if akas[red_idx] && counts[tile as usize] == 1 {
        let mut next = akas;
        next[red_idx] = false;
        #[cfg(feature = "debug_mortal_sp")]
        debug_aka_log::record_aka_drop(tile, red_idx);
        next
    } else {
        akas
    }
}

#[cfg(feature = "debug_mortal_sp")]
pub mod debug_aka_log {
    use std::cell::Cell;
    thread_local! {
        static DROPS: Cell<u32> = const { Cell::new(0) };
    }
    pub fn record_aka_drop(_tile: u8, _red_idx: usize) {
        DROPS.with(|c| c.set(c.get() + 1));
    }
    pub fn drain_drops() -> u32 {
        DROPS.with(|c| {
            let v = c.get();
            c.set(0);
            v
        })
    }
}

/// DEBUG-ONLY: thread-local log of all (counts_13, win_tile, han, fu, total)
/// triples observed at DP leaves during a single calculate_sp call. The
/// Mortal-comparison harness drains this to localize EV diffs.
#[cfg(feature = "debug_mortal_sp")]
pub mod debug_leaf_log {
    use super::TILE_MAX;
    use std::cell::RefCell;

    #[derive(Clone)]
    pub struct LeafEntry {
        pub counts_13: [u8; TILE_MAX],
        pub win_tile: u8,
        pub han: u32,
        pub fu: u32,
        pub total: u32,
    }

    thread_local! {
        static LOG: RefCell<Vec<LeafEntry>> = const { RefCell::new(Vec::new()) };
    }

    pub fn record(
        counts_13: &[u8; TILE_MAX],
        win_tile: u8,
        han: u32,
        fu: u32,
        total: u32,
    ) {
        LOG.with(|l| {
            l.borrow_mut().push(LeafEntry {
                counts_13: *counts_13,
                win_tile,
                han,
                fu,
                total,
            })
        });
    }

    thread_local! {
        static AKA_LOG: RefCell<Vec<([u8; TILE_MAX], u8, [bool; 3])>> =
            const { RefCell::new(Vec::new()) };
    }
    pub fn record_with_akas(
        counts_13: &[u8; TILE_MAX],
        win_tile: u8,
        akas: [bool; 3],
        han: u32,
        fu: u32,
        total: u32,
    ) {
        AKA_LOG.with(|l| l.borrow_mut().push((*counts_13, win_tile, akas)));
        record(counts_13, win_tile, han, fu, total);
    }
    pub fn drain_akas() -> Vec<([u8; TILE_MAX], u8, [bool; 3])> {
        AKA_LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    pub fn drain() -> Vec<LeafEntry> {
        LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }
}

pub fn calculate_sp(input: &SpInput) -> SpResult {
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

    let remaining = remaining_counts(input);
    let total_remaining: f32 = remaining.iter().map(|&x| x as f32).sum::<f32>().max(1.0);

    let mut dp = DpContext::new(input);

    // First pass: compute the post-discard shanten for every candidate so we
    // know which discards are "shanten-maintaining" (= optimal). Mortal-style
    // pre-filtering: only the optimal-shanten discards run the full DP; the
    // shanten-down ones use the cheap probability_series approximation.
    let mut prepared: Vec<(u8, [u8; TILE_MAX], i8)> =
        Vec::with_capacity(raw_discard_tiles.len());
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
            let req = required_tiles(&after_discard, &remaining, shanten_after);
            let sc = score_waits(&mut dp, &after_discard, &remaining);
            let yp = yaku_progress_tiles(&mut dp, &after_discard, &remaining, shanten_after);
            (req, sc, yp)
        };
        let num_required_tiles = required_tiles.iter().sum::<f32>();
        let num_yaku_progress_tiles = yaku_progress_tiles.iter().sum::<f32>();

        if scoring.mean_point <= 0.0 {
            scoring.mean_point = rough_point_estimate(input, &after_discard);
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

        candidates.push(SpCandidate {
            tile,
            tenpai_probs,
            win_probs,
            exp_values,
            required_tiles,
            yaku_progress_tiles,
            min_point: scoring.min_point,
            mean_point: scoring.mean_point,
            high_point: scoring.high_point,
            num_required_tiles,
            num_yaku_progress_tiles,
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
    if buf.len() < (ch_offset + SP_CHANNELS) * TILE_MAX {
        return;
    }

    let Some(best) = result.candidates.first() else {
        return;
    };

    let max_ev = best.exp_values[0].max(0.0);
    broadcast(buf, ch_offset, 0, (max_ev.min(100_000.0)) / 100_000.0);
    broadcast(buf, ch_offset, 1, (max_ev.min(30_000.0)) / 30_000.0);

    for candidate in &result.candidates {
        let discard = candidate.tile as usize;
        if discard >= TILE_MAX {
            continue;
        }
        for tile in 0..TILE_MAX {
            if candidate.required_tiles[tile] > 0.0 {
                set(buf, ch_offset, 2 + discard, tile, 1.0);
            }
            if candidate.yaku_progress_tiles[tile] > 0.0 {
                set(buf, ch_offset, 2 + TILE_MAX + discard, tile, 1.0);
            }
        }
    }

    if let Some(best_required) = result.candidates.iter().max_by(|a, b| {
        a.num_required_tiles
            .partial_cmp(&b.num_required_tiles)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.tile.cmp(&a.tile))
    }) {
        set(buf, ch_offset, 70, best_required.tile as usize, 1.0);
    }
    if let Some(best_yaku) = result.candidates.iter().max_by(|a, b| {
        a.num_yaku_progress_tiles
            .partial_cmp(&b.num_yaku_progress_tiles)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.tile.cmp(&a.tile))
    }) && best_yaku.num_yaku_progress_tiles > 0.0
    {
        set(buf, ch_offset, 71, best_yaku.tile as usize, 1.0);
    }

    let ev_scale = if max_ev >= 1.0 { 1.0 / max_ev } else { 0.0 };
    for candidate in &result.candidates {
        let discard = candidate.tile as usize;
        if discard >= TILE_MAX {
            continue;
        }
        for turn in 0..SP_MAX_TURNS {
            set(
                buf,
                ch_offset,
                72 + turn,
                discard,
                candidate.tenpai_probs[turn],
            );
            set(
                buf,
                ch_offset,
                72 + SP_MAX_TURNS + turn,
                discard,
                candidate.win_probs[turn],
            );
            set(
                buf,
                ch_offset,
                72 + SP_MAX_TURNS * 2 + turn,
                discard,
                (candidate.exp_values[turn] * ev_scale).clamp(0.0, 1.0),
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoringSummary {
    wait_count: f32,
    min_point: f32,
    mean_point: f32,
    high_point: f32,
}

fn add_seen(tiles_seen: &mut [u8; TILE_MAX], tile: u32) {
    let tile_type = (tile / 4) as usize;
    if tile_type < TILE_MAX {
        tiles_seen[tile_type] = tiles_seen[tile_type].saturating_add(1).min(4);
    }
}

fn remaining_self_draws(obs: &Observation) -> u8 {
    let self_discards = obs.discards[obs.player_id as usize].len();
    17usize.saturating_sub(self_discards).min(SP_MAX_TURNS) as u8
}

fn remaining_counts(input: &SpInput) -> [u8; TILE_MAX] {
    let mut remaining = [0u8; TILE_MAX];
    for (tile, out) in remaining.iter_mut().enumerate() {
        *out = 4u8.saturating_sub(input.tiles_seen[tile].min(4));
    }
    remaining
}

fn shanten_of_counts(counts: &[u8; TILE_MAX]) -> i8 {
    let len_div3 = counts.iter().sum::<u8>() / 3;
    shanten::calc_shanten_from_counts(counts, len_div3)
}

fn required_tiles(
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    current_shanten: i8,
) -> [f32; TILE_MAX] {
    let mut out = [0.0; TILE_MAX];
    if current_shanten < 0 {
        return out;
    }
    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }
        let mut next = *counts;
        next[tile] += 1;
        if shanten_of_counts(&next) < current_shanten {
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
    let mut scoring = ScoringSummary {
        wait_count: 0.0,
        min_point: 0.0,
        mean_point: 0.0,
        high_point: 0.0,
    };
    let mut next = *counts;
    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }
        next[tile] += 1;
        let s_after = dp.shanten(&next);
        next[tile] -= 1;
        if s_after >= 0 {
            // Not a wait — drawing this tile does not complete the hand.
            continue;
        }
        required[tile] = remaining[tile] as f32;
        if let Some(point) =
            dp.score_tsumo(counts, remaining, tile as u8, ScoreMods::default())
        {
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

    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
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

        let mut drawn = *counts;
        drawn[tile] += 1;
        let mut next_remaining = *remaining;
        next_remaining[tile] -= 1;
        if shanten_of_counts(&drawn) >= current_shanten {
            continue;
        }
        if has_yaku_tenpai_after_best_discard(dp, &drawn, &next_remaining) {
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
    let mut scoring = ScoringSummary {
        wait_count: 0.0,
        min_point: 0.0,
        mean_point: 0.0,
        high_point: 0.0,
    };
    if shanten_of_counts(counts) != 0 {
        return scoring;
    }

    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
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
    if scoring.min_point == 0.0 || point < scoring.min_point {
        scoring.min_point = point;
    }
    scoring.high_point = scoring.high_point.max(point);
    scoring.wait_count += weight;
    if scoring.wait_count > 0.0 {
        scoring.mean_point = (old_weighted_sum + point * weight) / scoring.wait_count;
    }
}

fn has_yaku_tenpai_after_best_discard(
    dp: &mut DpContext<'_>,
    counts_14: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
) -> bool {
    let key = YakuTenpaiKey {
        counts: *counts_14,
        remaining: *remaining,
        elapsed_turns: 0,
        turns_left: 0,
    };
    if let Some(&cached) = dp.yaku_tenpai_cache.get(&key) {
        return cached;
    }

    let mut best_shanten = i8::MAX;
    let mut tenpai_counts = Vec::new();
    for discard in 0..TILE_MAX {
        if counts_14[discard] == 0 {
            continue;
        }
        let mut next = *counts_14;
        next[discard] -= 1;
        let s = shanten_of_counts(&next);
        if s < best_shanten {
            best_shanten = s;
            tenpai_counts.clear();
        }
        if s == best_shanten && s == 0 {
            tenpai_counts.push(next);
        }
    }

    let result = tenpai_counts.iter().any(|counts| {
        (0..TILE_MAX).any(|tile| {
            remaining[tile] > 0
                && counts[tile] < 4
                && dp
                    .score_tsumo(counts, remaining, tile as u8, ScoreMods::default())
                    .is_some()
        })
    });
    dp.yaku_tenpai_cache.insert(key, result);
    result
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

    shanten_cache: FxHashMap<[u8; TILE_MAX], i8>,
    score_cache: FxHashMap<ScoreKey, Option<u32>>,
    base_score_cache: FxHashMap<BaseScoreKey, Option<BaseScore>>,
    yaku_tenpai_cache: FxHashMap<YakuTenpaiKey, bool>,
}

impl<'a> DpContext<'a> {
    fn new(input: &'a SpInput) -> Self {
        let horizon = (input.tsumos_left as usize).min(SP_MAX_TURNS).max(1);
        let remaining = remaining_counts(input);
        let n_left_tiles = remaining.iter().map(|&v| v as u32).sum::<u32>();
        let tsumo_prob = build_tsumo_prob_table(n_left_tiles, horizon);
        let not_tsumo_prob = build_not_tsumo_prob_table(n_left_tiles, horizon);
        Self {
            input,
            horizon,
            n_left_tiles,
            tsumo_prob,
            not_tsumo_prob,
            discard_cache: Default::default(),
            draw_cache: Default::default(),
            score_vec_cache: FxHashMap::default(),
            shanten_cache: FxHashMap::default(),
            score_cache: FxHashMap::default(),
            base_score_cache: FxHashMap::default(),
            yaku_tenpai_cache: FxHashMap::default(),
        }
    }

    fn shanten(&mut self, counts: &[u8; TILE_MAX]) -> i8 {
        if let Some(&cached) = self.shanten_cache.get(counts) {
            return cached;
        }
        let shanten = shanten_of_counts(counts);
        self.shanten_cache.insert(*counts, shanten);
        shanten
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
                .map(|base| score_from_base(self.input, base, counts, remaining, win_tile, mods))
                .or_else(|| {
                    mods.haitei
                        .then(|| exact_score_tsumo(self.input, counts, win_tile, mods))
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
            .map(|base| score_from_base(self.input, base, counts, remaining, win_tile, mods) as u32)
            .or_else(|| {
                mods.haitei
                    .then(|| exact_score_tsumo(self.input, counts, win_tile, mods))
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

        let score = base_score_tsumo(self.input, counts, win_tile, akas_in_hand);
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
    /// 内部では Mortal 流の「ターン位置 i に対する Values<MAX_TSUMOS>」を一回だけ計算し、
    /// `series[i] = Values[N - 1 - i]` として読み出す。
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
        if s == 0 {
            // すでに聴牌している場合は tenpai_probs を 1 で埋める
            for i in 0..horizon {
                tenpai[i] = 1.0;
            }
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

        // Internal accumulators are absolute-turn indexed (matches `draw_dp_slow`'s
        // `Values`); we reverse-map to horizon-distance at the end.
        let mut win_abs = [0.0f32; SP_MAX_TURNS];
        let mut exp_abs = [0.0f32; SP_MAX_TURNS];

        let mut accumulate = |dp: &mut Self,
                              tile: u8,
                              sub_count: u32,
                              scoring_akas: [bool; 3],
                              win_abs: &mut [f32; SP_MAX_TURNS],
                              exp_abs: &mut [f32; SP_MAX_TURNS]| {
            if sub_count == 0 {
                return;
            }
            let scores = dp.score_vector_for_win(counts_13, remaining, tile, scoring_akas);
            let Some(scores) = scores else { return };
            let tsumo_row: [f32; SP_MAX_TURNS] =
                dp.tsumo_prob[(sub_count as usize - 1).min(3)];
            for i in 0..horizon {
                let m = not_tsumo[i];
                if m == 0.0 {
                    break;
                }
                let m_inv = 1.0 / m;
                let dbl_at_i = if calc_double_riichi && i == 0 { 1 } else { 0 };
                for j in i..horizon {
                    let n = not_tsumo[j];
                    if n == 0.0 {
                        break;
                    }
                    let prob = tsumo_row[j] * n * m_inv;
                    let ipp = if assume_riichi && j == i { 1 } else { 0 };
                    let hai = if j == last_turn_idx { 1 } else { 0 };
                    let han_plus = (dbl_at_i + ipp + hai).min(3);
                    win_abs[i] += prob;
                    exp_abs[i] += prob * scores[han_plus];
                }
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
                    accumulate(self, tile, count - 1, akas_in_hand, &mut win_abs, &mut exp_abs);
                }
                accumulate(self, tile, 1, akas_red, &mut win_abs, &mut exp_abs);
            } else {
                accumulate(self, tile, count, akas_in_hand, &mut win_abs, &mut exp_abs);
            }
        }

        // Map absolute turn → horizon distance:  ours[i] = abs[N - 1 - i].
        for i in 0..horizon {
            let v_idx = horizon - 1 - i;
            win_out[i] = win_abs[v_idx].clamp(0.0, 1.0);
            exp_out[i] = exp_abs[v_idx].max(0.0);
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
        let mut any_valid = false;

        for tile in 0..TILE_MAX {
            if key.counts[tile] == 0 {
                continue;
            }
            let mut next_counts = key.counts;
            next_counts[tile] -= 1;
            let s = self.shanten(&next_counts);
            if s != shanten {
                // 向聴維持のみ。向聴落としは現状サポートしない。
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
                if !any_valid || v.exp[i] > best_exp[i] {
                    best_tenpai[i] = v.tenpai[i];
                    best_win[i] = v.win[i];
                    best_exp[i] = v.exp[i];
                }
            }
            any_valid = true;
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
        let mut effective: [(u8, u8); 34] = [(0, 0); 34];
        let mut n_eff: usize = 0;
        let mut sum_required: u32 = 0;
        for tile in 0..TILE_MAX {
            let count = key.remaining[tile];
            if count == 0 || key.counts[tile] >= 4 {
                continue;
            }
            let mut next_counts = key.counts;
            next_counts[tile] += 1;
            let s_after = self.shanten(&next_counts);
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
            let tsumo_row: [f32; SP_MAX_TURNS] =
                dp.tsumo_prob[(sub_count as usize - 1).min(3)];

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
                    let dbl_at_i = if calc_double_riichi && i == 0 { 1 } else { 0 };
                    for j in i..horizon {
                        let n = not_tsumo[j];
                        if n == 0.0 {
                            break;
                        }
                        let prob = tsumo_row[j] * n * m_inv;
                        let ipp = if assume_riichi && j == i { 1 } else { 0 };
                        let hai = if j == last_turn_idx { 1 } else { 0 };
                        let han_plus = (dbl_at_i + ipp + hai).min(3);
                        win[i] += prob;
                        exp[i] += prob * scores[han_plus];
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
        #[cfg(feature = "debug_mortal_sp")]
        debug_leaf_log::record_with_akas(
            counts_13, win_tile, akas_in_hand, base.han, base.fu, base.total,
        );

        let calc_with_han = |delta: u32| -> f32 {
            let han = (base.han + delta).min(13) as u8;
            let s = score::calculate_score(han, base.fu as u8, base.is_oya, true, base.honba, 4);
            tsumo_total_for_sp(s.pay_tsumo_oya, s.pay_tsumo_ko) as f32
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
    base_score_tsumo(input, counts_13, win_tile, input.akas_in_hand)
        .map(|base| score_from_base(input, base, counts_13, remaining, win_tile, mods))
        .or_else(|| {
            mods.haitei
                .then(|| exact_score_tsumo(input, counts_13, win_tile, mods))
                .flatten()
        })
}

#[cfg(feature = "debug_mortal_sp")]
pub mod base_score_calls {
    use std::cell::Cell;
    thread_local! {
        static N: Cell<u64> = const { Cell::new(0) };
    }
    pub fn bump() {
        N.with(|c| c.set(c.get() + 1));
    }
    pub fn drain() -> u64 {
        N.with(|c| {
            let v = c.get();
            c.set(0);
            v
        })
    }
}

fn base_score_tsumo(
    input: &SpInput,
    counts_13: &[u8; TILE_MAX],
    win_tile: u8,
    akas_in_hand: [bool; 3],
) -> Option<BaseScore> {
    if counts_13[win_tile as usize] >= 4 {
        return None;
    }
    #[cfg(feature = "debug_mortal_sp")]
    base_score_calls::bump();

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
    if !has_kan
        && let Some(lean) =
            crate::sp_yaku::compute_for_sp_tsumo(input, counts_13, win_tile, akas_in_hand)
        && lean.han > 0
    {
        let s = score::calculate_score(
            lean.han.min(13) as u8,
            lean.fu as u8,
            is_oya,
            true,
            0,
            4,
        );
        let base_total = tsumo_total_for_sp(s.pay_tsumo_oya, s.pay_tsumo_ko);
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
    let (tiles, tlen) = counts_to_136_stack(counts_13, akas_in_hand);
    let evaluator = HandEvaluator::new_borrowed(&tiles[..tlen], &input.melds);
    let conditions = Conditions {
        tsumo: true,
        riichi: assume_riichi,
        player_wind: wind_from_tile(input.jikaze),
        round_wind: wind_from_tile(input.bakaze),
        ..Conditions::default()
    };
    let result = evaluator.calc_borrowed(
        tile_type_to_136(win_tile, false),
        &input.dora_indicators,
        &[],
        Some(conditions.clone()),
    );
    if !result.is_win {
        return None;
    }

    let base_total = tsumo_total_for_sp(result.tsumo_agari_oya, result.tsumo_agari_ko) as f32;
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
) -> f32 {
    let extra_han = timing_extra_han(base, mods);
    let base_total = if extra_han == 0 {
        base.total as f32
    } else {
        let han = base.han.saturating_add(extra_han).min(13) as u8;
        let score = score::calculate_score(han, base.fu as u8, base.is_oya, true, base.honba, 4);
        tsumo_total_for_sp(score.pay_tsumo_oya, score.pay_tsumo_ko) as f32
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
            .min(13) as u8;
        let score = score::calculate_score(han, base.fu as u8, base.is_oya, true, base.honba, 4);
        expected += prob * tsumo_total_for_sp(score.pay_tsumo_oya, score.pay_tsumo_ko) as f32;
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
) -> Option<f32> {
    if counts_13[win_tile as usize] >= 4 {
        return None;
    }
    let (tiles, tlen) = counts_to_136_stack(counts_13, input.akas_in_hand);
    let evaluator = HandEvaluator::new_borrowed(&tiles[..tlen], &input.melds);
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
    let result = evaluator.calc_borrowed(
        tile_type_to_136(win_tile, false),
        &input.dora_indicators,
        &[],
        Some(conditions),
    );
    result
        .is_win
        .then_some(tsumo_total_for_sp(result.tsumo_agari_oya, result.tsumo_agari_ko) as f32)
}

fn ura_distribution(
    full_counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    num_indicators: usize,
) -> Vec<f32> {
    let total_remaining = remaining.iter().map(|&count| count as usize).sum::<usize>();
    let draws = num_indicators.min(5).min(total_remaining);
    let max_ura = draws * 4;
    let mut dist = vec![0.0f32; max_ura + 1];
    if draws == 0 {
        dist[0] = 1.0;
        return dist;
    }

    let mut gain_counts = [0usize; 5];
    for indicator in 0..TILE_MAX {
        let dora_tile = next_dora_tile(indicator as u8) as usize;
        let gain = full_counts[dora_tile] as usize;
        gain_counts[gain] += remaining[indicator] as usize;
    }

    let denom = combination_f64(total_remaining, draws);
    if denom <= 0.0 {
        dist[0] = 1.0;
        return dist;
    }

    fn visit(
        gain: usize,
        gain_counts: &[usize; 5],
        draws_left: usize,
        ura_sum: usize,
        weight: f64,
        denom: f64,
        dist: &mut [f32],
    ) {
        if gain == gain_counts.len() {
            if draws_left == 0 {
                dist[ura_sum] += (weight / denom) as f32;
            }
            return;
        }

        let max_take = gain_counts[gain].min(draws_left);
        for take in 0..=max_take {
            visit(
                gain + 1,
                gain_counts,
                draws_left - take,
                ura_sum + gain * take,
                weight * combination_f64(gain_counts[gain], take),
                denom,
                dist,
            );
        }
    }

    visit(0, &gain_counts, draws, 0, 1.0, denom, &mut dist);
    let sum = dist.iter().sum::<f32>();
    if sum > 0.0 {
        for prob in &mut dist {
            *prob /= sum;
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
fn tsumo_total_for_sp(pay_tsumo_oya: u32, pay_tsumo_ko: u32) -> u32 {
    if pay_tsumo_oya == 0 {
        pay_tsumo_ko.saturating_mul(3)
    } else {
        pay_tsumo_oya + pay_tsumo_ko.saturating_mul(2)
    }
}

fn rough_point_estimate(input: &SpInput, counts: &[u8; TILE_MAX]) -> f32 {
    let mut dora = input.akas_in_hand.iter().filter(|&&x| x).count() as f32;
    for &indicator in &input.dora_indicators {
        let dora_tile = next_dora_tile(indicator / 4) as usize;
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

fn broadcast(buf: &mut [f32], ch_offset: usize, ch: usize, val: f32) {
    let start = (ch_offset + ch) * TILE_MAX;
    for tile in 0..TILE_MAX {
        buf[start + tile] = val;
    }
}

fn set(buf: &mut [f32], ch_offset: usize, ch: usize, tile: usize, val: f32) {
    buf[(ch_offset + ch) * TILE_MAX + tile] = val;
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
    fn sp_generates_candidates_and_123_channels() {
        let input = tenpai_fixture(10);
        let result = calculate_sp(&input);
        assert!(!result.candidates.is_empty());
        let encoded = encode_sp(&result);
        assert_eq!(encoded.len(), SP_CHANNELS * TILE_MAX);
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

        let dist = ura_distribution(&full_counts, &remaining, 3);
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

        let mut dp = DpContext::new(&input);
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

        let mut dp = DpContext::new(&input);
        let (tenpai, win, ev) = dp.series(&counts, &remaining, 2);

        assert!(tenpai[0] > 0.0);
        assert_eq!(win[0], 0.0);
        assert!(win[1] > 0.0);
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
                let any = (0..TILE_MAX)
                    .any(|t| encoded[(2 + tile) * TILE_MAX + t] > 0.5);
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
            let mut dp = DpContext::new(&input);
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
