use std::collections::HashMap;

use crate::action::ActionType;
use crate::hand_evaluator::HandEvaluator;
use crate::observation::Observation;
use crate::shanten;
use crate::types::{Conditions, Meld, TILE_MAX, Wind};

pub const SP_MAX_TURNS: usize = 17;
pub const SP_CHANNELS: usize = 123;

#[derive(Debug, Clone)]
pub struct SpInput {
    pub tehai: [u8; TILE_MAX],
    pub akas_in_hand: [bool; 3],
    pub tiles_seen: [u8; TILE_MAX],
    pub dora_indicators: Vec<u8>,
    pub melds: Vec<Meld>,
    pub bakaze: u8,
    pub jikaze: u8,
    pub is_menzen: bool,
    pub can_riichi: bool,
    pub tsumos_left: u8,
    pub discard_candidates: Vec<u8>,
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
        for &tile in &obs.hands[player_idx] {
            add_seen(&mut tiles_seen, tile);
        }
        for melds in &obs.melds {
            for meld in melds {
                for &tile in &meld.tiles {
                    add_seen(&mut tiles_seen, tile as u32);
                }
            }
        }
        for discards in &obs.discards {
            for &tile in discards {
                add_seen(&mut tiles_seen, tile);
            }
        }
        for &tile in &obs.dora_indicators {
            add_seen(&mut tiles_seen, tile);
        }

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
        let can_riichi = obs.riichi_declared[player_idx]
            || obs
                ._legal_actions
                .iter()
                .any(|a| matches!(a.action_type, ActionType::Riichi));

        Self {
            tehai,
            akas_in_hand,
            tiles_seen,
            dora_indicators: obs.dora_indicators.iter().map(|&x| x as u8).collect(),
            melds: obs.melds[player_idx].clone(),
            bakaze: obs.round_wind,
            jikaze: 27 + rel_seat,
            is_menzen: obs.melds[player_idx].iter().all(|m| !m.opened),
            can_riichi,
            tsumos_left: remaining_self_draws(obs),
            discard_candidates,
        }
    }
}

pub fn calculate_sp(input: &SpInput) -> SpResult {
    let mut candidates = Vec::new();
    let discard_tiles = if input.discard_candidates.is_empty() {
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

    for tile in discard_tiles {
        let tile_idx = tile as usize;
        if tile_idx >= TILE_MAX || input.tehai[tile_idx] == 0 {
            continue;
        }

        let mut after_discard = input.tehai;
        after_discard[tile_idx] -= 1;
        let shanten_after = shanten_of_counts(&after_discard);
        let required_tiles = required_tiles(&after_discard, &remaining, shanten_after);
        let num_required_tiles = required_tiles.iter().sum::<f32>();

        let mut scoring = score_waits(&mut dp, &after_discard, &remaining);
        let yaku_progress_tiles =
            yaku_progress_tiles(&mut dp, &after_discard, &remaining, shanten_after);
        let num_yaku_progress_tiles = yaku_progress_tiles.iter().sum::<f32>();

        if scoring.mean_point <= 0.0 {
            scoring.mean_point = rough_point_estimate(input, &after_discard);
        }

        let (tenpai_probs, win_probs, exp_values) = series_for_candidate(
            &mut dp,
            &after_discard,
            &remaining,
            shanten_after,
            num_required_tiles,
            scoring.wait_count,
            scoring.mean_point,
            total_remaining,
        );

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

fn yaku_progress_tiles(
    dp: &mut DpContext<'_>,
    counts: &[u8; TILE_MAX],
    remaining: &[u8; TILE_MAX],
    current_shanten: i8,
) -> [f32; TILE_MAX] {
    let mut out = [0.0; TILE_MAX];
    if current_shanten > 3 {
        return out;
    }

    for tile in 0..TILE_MAX {
        if remaining[tile] == 0 || counts[tile] >= 4 {
            continue;
        }

        if current_shanten == 0 {
            if dp.score_tsumo(counts, tile as u8).is_some() {
                out[tile] = remaining[tile] as f32;
            }
            continue;
        }

        let mut drawn = *counts;
        drawn[tile] += 1;
        if shanten_of_counts(&drawn) >= current_shanten {
            continue;
        }
        if has_yaku_tenpai_after_best_discard(dp, &drawn) {
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
        if let Some(point) = dp.score_tsumo(counts, tile as u8) {
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

fn has_yaku_tenpai_after_best_discard(dp: &mut DpContext<'_>, counts_14: &[u8; TILE_MAX]) -> bool {
    if let Some(&cached) = dp.yaku_tenpai_cache.get(counts_14) {
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

    let result = tenpai_counts
        .iter()
        .any(|counts| (0..TILE_MAX).any(|tile| dp.score_tsumo(counts, tile as u8).is_some()));
    dp.yaku_tenpai_cache.insert(*counts_14, result);
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
) -> (
    [f32; SP_MAX_TURNS],
    [f32; SP_MAX_TURNS],
    [f32; SP_MAX_TURNS],
) {
    let horizon = dp.input.tsumos_left as usize;
    if shanten_after_discard <= 3 {
        return dp.series(counts, remaining, horizon);
    }

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

#[derive(Debug, Clone, Copy, Default)]
struct DpOutcome {
    tenpai_prob: f32,
    win_prob: f32,
    exp_value: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DpKey {
    counts: [u8; TILE_MAX],
    remaining: [u8; TILE_MAX],
    turns_left: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ScoreKey {
    counts: [u8; TILE_MAX],
    win_tile: u8,
}

struct DpContext<'a> {
    input: &'a SpInput,
    memo: HashMap<DpKey, DpOutcome>,
    score_cache: HashMap<ScoreKey, Option<u32>>,
    yaku_tenpai_cache: HashMap<[u8; TILE_MAX], bool>,
}

impl<'a> DpContext<'a> {
    fn new(input: &'a SpInput) -> Self {
        Self {
            input,
            memo: HashMap::new(),
            score_cache: HashMap::new(),
            yaku_tenpai_cache: HashMap::new(),
        }
    }

    fn score_tsumo(&mut self, counts: &[u8; TILE_MAX], win_tile: u8) -> Option<f32> {
        let key = ScoreKey {
            counts: *counts,
            win_tile,
        };
        if let Some(&cached) = self.score_cache.get(&key) {
            return cached.map(|point| point as f32);
        }

        let score = score_tsumo(self.input, counts, win_tile).map(|point| point as u32);
        self.score_cache.insert(key, score);
        score.map(|point| point as f32)
    }

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
        let mut tenpai = [0.0; SP_MAX_TURNS];
        let mut win = [0.0; SP_MAX_TURNS];
        let mut ev = [0.0; SP_MAX_TURNS];
        let horizon = tsumos_left.min(SP_MAX_TURNS);
        for turns in 1..=horizon {
            let outcome = self.eval(counts, remaining, turns as u8);
            tenpai[turns - 1] = outcome.tenpai_prob;
            win[turns - 1] = outcome.win_prob;
            ev[turns - 1] = outcome.exp_value;
        }
        (tenpai, win, ev)
    }

    fn eval(
        &mut self,
        counts: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        turns_left: u8,
    ) -> DpOutcome {
        let key = DpKey {
            counts: *counts,
            remaining: *remaining,
            turns_left,
        };
        if let Some(&cached) = self.memo.get(&key) {
            return cached;
        }

        let current = DpOutcome {
            tenpai_prob: if self.has_yaku_wait(counts, remaining) {
                1.0
            } else {
                0.0
            },
            win_prob: 0.0,
            exp_value: 0.0,
        };
        if turns_left == 0 {
            self.memo.insert(key, current);
            return current;
        }

        if self.has_yaku_wait(counts, remaining) {
            let outcome = self.tenpai_wait_outcome(counts, remaining, turns_left);
            self.memo.insert(key, outcome);
            return outcome;
        }

        let current_shanten = shanten_of_counts(counts);
        if !(1..=3).contains(&current_shanten) {
            self.memo.insert(key, current);
            return current;
        }

        let total = remaining.iter().map(|&x| x as f32).sum::<f32>();
        if total <= 0.0 {
            self.memo.insert(key, current);
            return current;
        }

        let effective = required_tiles(counts, remaining, current_shanten);
        let effective_total = effective.iter().sum::<f32>();
        if effective_total <= 0.0 {
            self.memo.insert(key, current);
            return current;
        }

        let mut outcome = DpOutcome::default();
        let mut no_effective_before = 1.0f32;
        for first_effective_turn in 1..=turns_left {
            let denom = total - (first_effective_turn - 1) as f32;
            if denom <= 0.0 {
                break;
            }

            for draw in 0..TILE_MAX {
                if effective[draw] <= 0.0 {
                    continue;
                }
                let p_draw = no_effective_before * (remaining[draw] as f32 / denom).clamp(0.0, 1.0);
                if p_draw <= 0.0 {
                    continue;
                }

                let mut next_remaining = *remaining;
                next_remaining[draw] -= 1;
                let mut counts_14 = *counts;
                counts_14[draw] += 1;
                let branch = self.best_after_discard(
                    &counts_14,
                    &next_remaining,
                    turns_left - first_effective_turn,
                );
                outcome.tenpai_prob += p_draw * branch.tenpai_prob;
                outcome.win_prob += p_draw * branch.win_prob;
                outcome.exp_value += p_draw * branch.exp_value;
            }

            no_effective_before *= (1.0 - effective_total / denom).clamp(0.0, 1.0);
        }

        outcome.tenpai_prob = outcome.tenpai_prob.clamp(0.0, 1.0);
        outcome.win_prob = outcome.win_prob.clamp(0.0, 1.0);
        self.memo.insert(key, outcome);
        outcome
    }

    fn tenpai_wait_outcome(
        &mut self,
        counts: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        turns_left: u8,
    ) -> DpOutcome {
        let mut wait_points = [0.0f32; TILE_MAX];
        let mut wait_total = 0.0f32;
        for tile in 0..TILE_MAX {
            if remaining[tile] == 0 || counts[tile] >= 4 {
                continue;
            }
            if let Some(point) = self.score_tsumo(counts, tile as u8) {
                wait_points[tile] = point;
                wait_total += remaining[tile] as f32;
            }
        }

        let total = remaining.iter().map(|&x| x as f32).sum::<f32>();
        let mut outcome = DpOutcome {
            tenpai_prob: 1.0,
            ..DpOutcome::default()
        };
        if wait_total <= 0.0 || total <= 0.0 {
            return outcome;
        }

        let mut no_wait_before = 1.0f32;
        for turn in 1..=turns_left {
            let denom = total - (turn - 1) as f32;
            if denom <= 0.0 {
                break;
            }
            for tile in 0..TILE_MAX {
                if wait_points[tile] <= 0.0 {
                    continue;
                }
                let p = no_wait_before * (remaining[tile] as f32 / denom).clamp(0.0, 1.0);
                outcome.win_prob += p;
                outcome.exp_value += p * wait_points[tile];
            }
            no_wait_before *= (1.0 - wait_total / denom).clamp(0.0, 1.0);
        }

        outcome.win_prob = outcome.win_prob.clamp(0.0, 1.0);
        outcome
    }

    fn best_after_discard(
        &mut self,
        counts_14: &[u8; TILE_MAX],
        remaining: &[u8; TILE_MAX],
        turns_left: u8,
    ) -> DpOutcome {
        let best_shanten = (0..TILE_MAX)
            .filter(|&discard| counts_14[discard] > 0)
            .map(|discard| {
                let mut next_counts = *counts_14;
                next_counts[discard] -= 1;
                shanten_of_counts(&next_counts)
            })
            .min();
        let Some(best_shanten) = best_shanten else {
            return DpOutcome::default();
        };

        let mut best: Option<DpOutcome> = None;
        for discard in 0..TILE_MAX {
            if counts_14[discard] == 0 {
                continue;
            }
            let mut next_counts = *counts_14;
            next_counts[discard] -= 1;
            let shanten = shanten_of_counts(&next_counts);
            if shanten != best_shanten || shanten > 3 {
                continue;
            }
            let outcome = self.eval(&next_counts, remaining, turns_left);
            if best.is_none_or(|current| is_better_dp(outcome, current)) {
                best = Some(outcome);
            }
        }
        best.unwrap_or_default()
    }

    fn has_yaku_wait(&mut self, counts: &[u8; TILE_MAX], remaining: &[u8; TILE_MAX]) -> bool {
        if shanten_of_counts(counts) != 0 {
            return false;
        }
        (0..TILE_MAX).any(|tile| {
            remaining[tile] > 0
                && counts[tile] < 4
                && self.score_tsumo(counts, tile as u8).is_some()
        })
    }
}

fn is_better_dp(candidate: DpOutcome, current: DpOutcome) -> bool {
    candidate
        .exp_value
        .partial_cmp(&current.exp_value)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            candidate
                .win_prob
                .partial_cmp(&current.win_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            candidate
                .tenpai_prob
                .partial_cmp(&current.tenpai_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .is_gt()
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

fn score_tsumo(input: &SpInput, counts_13: &[u8; TILE_MAX], win_tile: u8) -> Option<f32> {
    if counts_13[win_tile as usize] >= 4 {
        return None;
    }
    let tiles = counts_to_136(counts_13, input.akas_in_hand);
    let evaluator = HandEvaluator::new(tiles, input.melds.clone());
    let conditions = Conditions {
        tsumo: true,
        riichi: input.is_menzen && input.can_riichi,
        player_wind: wind_from_tile(input.jikaze),
        round_wind: wind_from_tile(input.bakaze),
        ..Conditions::default()
    };
    let result = evaluator.calc(
        tile_type_to_136(win_tile, false),
        input.dora_indicators.clone(),
        vec![],
        Some(conditions),
    );
    result
        .is_win
        .then_some((result.tsumo_agari_oya + result.tsumo_agari_ko.saturating_mul(2)) as f32)
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

fn counts_to_136(counts: &[u8; TILE_MAX], akas_in_hand: [bool; 3]) -> Vec<u8> {
    let mut tiles = Vec::new();
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
            tiles.push(tile_type_to_136(tile as u8, true));
            emitted = 1;
        }
        for copy in emitted..count {
            tiles.push((tile as u8) * 4 + copy + if red_index.is_some() { 1 } else { 0 });
        }
    }
    tiles
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

fn next_dora_tile(tile_type: u8) -> u8 {
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
            dora_indicators: vec![],
            melds: vec![],
            bakaze: 27,
            jikaze: 27,
            is_menzen: true,
            can_riichi: true,
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

        let mut dp = DpContext::new(&input);
        let (tenpai, win, ev) = dp.series(&counts, &remaining, 2);

        assert!(tenpai[0] > 0.0);
        assert_eq!(win[0], 0.0);
        assert!(win[1] > 0.0);
        assert!(ev[1] > 0.0);
    }
}
