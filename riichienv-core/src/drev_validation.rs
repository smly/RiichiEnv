//! Replay-backed validation for the public-history DREV-v2 feature.
//!
//! The encoded feature sees only a player's public observation.  This module
//! deliberately keeps the replay's authoritative concealed state beside that
//! observation and uses it only after the feature has been calculated.  That
//! separation lets validators prove exact claims such as `hard_safe_zero`
//! without accidentally feeding hidden information into the feature itself.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::action::{Action as EnvAction, ActionType};
use crate::drev_v2::{
    DREV_V2_BASE_CHANNELS, DREV_V2_CHANNELS, DREV_V2_SCHEMA_ID, DREV_V2_YAKU_CHANNELS_PER_OPPONENT,
    DrevV2OpponentResult, YAKU_EVIDENCE_NAMES, calculate_drev_3p_v2, calculate_drev_v2,
};
use crate::errors::{RiichiError, RiichiResult};
use crate::hand_evaluator::HandEvaluator;
use crate::hand_evaluator_3p::HandEvaluator3P;
use crate::replay::{Action as LogAction, LogKyoku, ReplayLog};
use crate::state::GameState;
use crate::state::legal_actions::GameStateLegalActions;
use crate::state_3p::GameState3P;
use crate::state_3p::legal_actions::GameState3PLegalActions;
use crate::types::{Conditions, MeldType, WinResult, Wind};

const REPORT_SCHEMA_VERSION: u16 = 1;
const EPSILON: f32 = 1.0e-6;
const LOG_EPSILON: f64 = 1.0e-7;

/// A single exact-contract violation found while replaying a fixture.
#[derive(Debug, Clone, Serialize)]
pub struct DrevValidationViolation {
    pub round: usize,
    pub action: usize,
    pub actor: u8,
    pub opponent: Option<u8>,
    pub tile_type: Option<u8>,
    pub code: String,
    pub message: String,
}

/// Descriptive quality metrics for DREV's deliberately uncalibrated heads.
///
/// These values are reported rather than used as hard correctness thresholds.
/// Exact-zero and feature-semantic failures are carried by the counters on
/// [`DrevReplayValidationReport`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct DrevValidationMetrics {
    pub wait_brier: f64,
    pub wait_log_loss: f64,
    pub wait_average_precision: f64,
    pub wait_ece_10: f64,
    pub ron_brier: f64,
    pub ron_log_loss: f64,
    pub ron_average_precision: f64,
    pub ron_ece_10: f64,
    pub legal_ron_loss_mae: f64,
    pub legal_ron_loss_bias: f64,
}

/// Aggregate report produced from one typed MJAI replay.
#[derive(Debug, Clone, Serialize)]
pub struct DrevReplayValidationReport {
    pub schema_version: u16,
    pub drev_schema_id: String,
    pub variant: String,
    pub rounds: usize,
    pub decisions: usize,
    pub candidate_tile_types: usize,
    pub physical_discard_candidates: usize,
    pub opponent_candidate_cells: usize,
    pub shape_wait_cells: usize,
    pub yaku_valid_win_cells: usize,
    pub legal_ron_cells: usize,
    pub hard_safe_cells: usize,
    pub hard_safe_false_positives: usize,
    pub legal_ron_zero_probability: usize,
    pub yaku_impossible_conflicts: usize,
    pub yaku_confirmed_conflicts: usize,
    pub loss_lower_bound_violations: usize,
    pub feature_invariant_failures: usize,
    pub semantic_sha256: String,
    pub metrics: DrevValidationMetrics,
    pub yaku_support: BTreeMap<String, usize>,
    pub violations: Vec<DrevValidationViolation>,
}

impl Default for DrevReplayValidationReport {
    fn default() -> Self {
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            drev_schema_id: DREV_V2_SCHEMA_ID.to_string(),
            variant: String::new(),
            rounds: 0,
            decisions: 0,
            candidate_tile_types: 0,
            physical_discard_candidates: 0,
            opponent_candidate_cells: 0,
            shape_wait_cells: 0,
            yaku_valid_win_cells: 0,
            legal_ron_cells: 0,
            hard_safe_cells: 0,
            hard_safe_false_positives: 0,
            legal_ron_zero_probability: 0,
            yaku_impossible_conflicts: 0,
            yaku_confirmed_conflicts: 0,
            loss_lower_bound_violations: 0,
            feature_invariant_failures: 0,
            semantic_sha256: String::new(),
            metrics: DrevValidationMetrics::default(),
            yaku_support: BTreeMap::new(),
            violations: Vec::new(),
        }
    }
}

impl DrevReplayValidationReport {
    /// Count failures covered by exact, deterministic correctness contracts.
    pub fn accuracy_failures(&self) -> usize {
        self.hard_safe_false_positives
            + self.legal_ron_zero_probability
            + self.yaku_impossible_conflicts
            + self.yaku_confirmed_conflicts
            + self.loss_lower_bound_violations
            + self.feature_invariant_failures
    }

    pub fn is_valid(&self) -> bool {
        self.accuracy_failures() == 0 && self.violations.is_empty()
    }

    pub fn ensure_valid(&self) -> RiichiResult<()> {
        if self.is_valid() {
            return Ok(());
        }
        Err(RiichiError::InvalidState {
            message: format!(
                "DREV replay validation failed with {} accuracy failures and {} violations",
                self.accuracy_failures(),
                self.violations.len()
            ),
        })
    }
}

#[derive(Default)]
struct ValidationAccumulator {
    report: DrevReplayValidationReport,
    variants: BTreeSet<usize>,
    wait_samples: Vec<(f32, bool)>,
    ron_samples: Vec<(f32, bool)>,
    loss_samples: Vec<(f32, f32)>,
    digest: Sha256,
}

#[derive(Debug)]
struct OracleLabel {
    shape_wait: bool,
    legal_ron: bool,
    result: WinResult,
}

/// Validate every complete round in a typed replay.
pub fn validate_drev_replay(replay: &ReplayLog) -> RiichiResult<DrevReplayValidationReport> {
    if replay.is_empty() {
        return Err(RiichiError::InvalidState {
            message: "DREV validation requires at least one complete round".to_string(),
        });
    }

    let mut acc = ValidationAccumulator::default();
    for (round_index, kyoku) in replay.rounds().iter().enumerate() {
        match kyoku.scores.len() {
            4 => validate_round_4p(kyoku, round_index, &mut acc)?,
            3 => validate_round_3p(kyoku, round_index, &mut acc)?,
            count => {
                return Err(RiichiError::InvalidState {
                    message: format!("DREV validation cannot replay a {count}-player round"),
                });
            }
        }
        acc.report.rounds += 1;
        acc.variants.insert(kyoku.scores.len());
    }

    if acc.report.decisions == 0 || acc.report.opponent_candidate_cells == 0 {
        return Err(RiichiError::InvalidState {
            message: "DREV validation replay contains no discard decisions to validate".to_string(),
        });
    }

    acc.report.variant = if acc.variants.len() == 1 && acc.variants.contains(&4) {
        "4p".to_string()
    } else if acc.variants.len() == 1 && acc.variants.contains(&3) {
        "3p".to_string()
    } else {
        "mixed".to_string()
    };
    acc.report.semantic_sha256 = hex::encode(acc.digest.finalize());
    acc.report.metrics = calculate_metrics(&acc.wait_samples, &acc.ron_samples, &acc.loss_samples);
    Ok(acc.report)
}

fn validate_round_4p(
    kyoku: &LogKyoku,
    round_index: usize,
    acc: &mut ValidationAccumulator,
) -> RiichiResult<()> {
    let mut state = initial_state_4p(kyoku)?;
    for (action_index, action) in kyoku.actions.iter().enumerate() {
        if let LogAction::DiscardTile {
            seat,
            tile,
            is_liqi,
            is_wliqi,
            ..
        } = action
        {
            let actor = checked_actor(*seat, 4, "4P discard")?;
            let env_action = EnvAction::new(ActionType::Discard, Some(*tile), Vec::new(), None);
            let staged = (*is_liqi || *is_wliqi)
                && !state.players[actor as usize].riichi_declared
                && !state.players[actor as usize].riichi_stage;
            if staged {
                state.players[actor as usize].riichi_stage = true;
            }
            let observation =
                state.get_observation_for_replay(actor, &env_action, &format!("{action:?}"));
            if staged {
                state.players[actor as usize].riichi_stage = false;
            }
            let observation = observation?;
            let result = calculate_drev_v2(&observation)?;
            validate_decision_4p(
                &state,
                &observation,
                result.opponents(),
                &result.encode(),
                round_index,
                action_index,
                acc,
            );
        }
        apply_action_4p(&mut state, kyoku, action_index, action);
    }
    if !state.public_history_valid {
        return Err(RiichiError::InvalidState {
            message: format!("round {round_index} ended with incomplete DREV public history"),
        });
    }
    Ok(())
}

fn validate_round_3p(
    kyoku: &LogKyoku,
    round_index: usize,
    acc: &mut ValidationAccumulator,
) -> RiichiResult<()> {
    let mut state = initial_state_3p(kyoku)?;
    for (action_index, action) in kyoku.actions.iter().enumerate() {
        if let LogAction::DiscardTile {
            seat,
            tile,
            is_liqi,
            is_wliqi,
            ..
        } = action
        {
            let actor = checked_actor(*seat, 3, "3P discard")?;
            let env_action = EnvAction::new(ActionType::Discard, Some(*tile), Vec::new(), None);
            let staged = (*is_liqi || *is_wliqi)
                && !state.players[actor as usize].riichi_declared
                && !state.players[actor as usize].riichi_stage;
            if staged {
                state.players[actor as usize].riichi_stage = true;
            }
            let observation =
                state.get_observation_for_replay(actor, &env_action, &format!("{action:?}"));
            if staged {
                state.players[actor as usize].riichi_stage = false;
            }
            let observation = observation?;
            let result = calculate_drev_3p_v2(&observation)?;
            validate_decision_3p(
                &state,
                &observation,
                result.opponents(),
                &result.encode(),
                round_index,
                action_index,
                acc,
            );
        }
        apply_action_3p(&mut state, kyoku, action_index, action);
    }
    if !state.public_history_valid {
        return Err(RiichiError::InvalidState {
            message: format!("round {round_index} ended with incomplete DREV public history"),
        });
    }
    Ok(())
}

fn validate_decision_4p(
    state: &GameState,
    observation: &crate::observation::Observation,
    opponents: &[DrevV2OpponentResult; 3],
    encoded: &[f32],
    round_index: usize,
    action_index: usize,
    acc: &mut ValidationAccumulator,
) {
    let actor = observation.player_id;
    let candidates = discard_candidates(&observation._legal_actions);
    begin_decision(
        actor,
        &candidates,
        opponents,
        encoded,
        34,
        3,
        round_index,
        action_index,
        acc,
    );
    for &tile in &candidates {
        for (slot, prediction) in opponents.iter().enumerate() {
            let opponent = (actor + slot as u8 + 1) % 4;
            let label = oracle_4p(state, opponent, actor, tile);
            validate_cell(
                prediction,
                &label,
                round_index,
                action_index,
                actor,
                opponent,
                slot,
                tile,
                acc,
            );
        }
    }
}

fn validate_decision_3p(
    state: &GameState3P,
    observation: &crate::observation_3p::Observation3P,
    opponents: &[DrevV2OpponentResult; 3],
    encoded: &[f32],
    round_index: usize,
    action_index: usize,
    acc: &mut ValidationAccumulator,
) {
    let actor = observation.player_id;
    let candidates = discard_candidates_3p(&observation._legal_actions);
    begin_decision(
        actor,
        &candidates,
        opponents,
        encoded,
        27,
        2,
        round_index,
        action_index,
        acc,
    );
    validate_inactive_3p_slot(&opponents[2], round_index, action_index, actor, acc);
    for &tile in &candidates {
        for (slot, prediction) in opponents.iter().enumerate().take(2) {
            let opponent = (actor + slot as u8 + 1) % 3;
            let label = oracle_3p(state, opponent, actor, tile);
            validate_cell(
                prediction,
                &label,
                round_index,
                action_index,
                actor,
                opponent,
                slot,
                tile,
                acc,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn begin_decision(
    actor: u8,
    candidates: &[u8],
    opponents: &[DrevV2OpponentResult; 3],
    encoded: &[f32],
    tile_types: usize,
    active_opponents: usize,
    round_index: usize,
    action_index: usize,
    acc: &mut ValidationAccumulator,
) {
    acc.report.decisions += 1;
    acc.report.physical_discard_candidates += candidates.len();
    acc.report.candidate_tile_types += candidates
        .iter()
        .map(|tile| tile / 4)
        .collect::<BTreeSet<_>>()
        .len();
    validate_encoded_output(
        actor,
        candidates,
        opponents,
        encoded,
        tile_types,
        active_opponents,
        round_index,
        action_index,
        acc,
    );
    update_encoded_digest(acc, round_index, action_index, actor, encoded);
}

#[allow(clippy::too_many_arguments)]
fn validate_cell(
    prediction: &DrevV2OpponentResult,
    label: &OracleLabel,
    round_index: usize,
    action_index: usize,
    actor: u8,
    opponent: u8,
    slot: usize,
    tile: u8,
    acc: &mut ValidationAccumulator,
) {
    let tile_type = tile as usize / 4;
    let hard = prediction.hard_safe_zero[tile_type];
    let wait = prediction.wait_prob[tile_type];
    let ron = prediction.ron_prob[tile_type];
    let loss = prediction.mean_loss_points[tile_type];
    let uncertainty = prediction.uncertainty[tile_type];
    acc.report.opponent_candidate_cells += 1;
    acc.report.shape_wait_cells += usize::from(label.shape_wait);
    acc.report.yaku_valid_win_cells += usize::from(label.result.is_win);
    acc.report.legal_ron_cells += usize::from(label.legal_ron);
    acc.report.hard_safe_cells += usize::from(hard == 1.0);
    acc.wait_samples.push((wait, label.shape_wait));
    acc.ron_samples.push((ron, label.legal_ron));
    if label.legal_ron {
        acc.loss_samples.push((loss, label.result.ron_agari as f32));
    }

    let context = CellContext {
        round: round_index,
        action: action_index,
        actor,
        opponent,
        tile_type: tile_type as u8,
    };
    if !is_binary(hard)
        || !is_probability(wait)
        || !is_probability(ron)
        || !is_probability(uncertainty)
        || !loss.is_finite()
        || loss < 0.0
        || ron > wait + EPSILON
        || (hard == 1.0 && ron != 0.0)
    {
        acc.report.feature_invariant_failures += 1;
        push_cell_violation(
            acc,
            &context,
            "feature_invariant",
            format!("hard={hard}, wait={wait}, ron={ron}, loss={loss}, uncertainty={uncertainty}"),
        );
    }

    if hard == 1.0 && label.legal_ron {
        acc.report.hard_safe_false_positives += 1;
        push_cell_violation(
            acc,
            &context,
            "hard_safe_false_positive",
            "engine offers a legal Ron for a rule-proven safe tile".to_string(),
        );
    }
    if label.legal_ron && ron <= 0.0 {
        acc.report.legal_ron_zero_probability += 1;
        push_cell_violation(
            acc,
            &context,
            "legal_ron_zero_probability",
            "engine offers a legal Ron but DREV assigns zero Ron probability".to_string(),
        );
    }
    if label.legal_ron && !label.result.is_win {
        acc.report.feature_invariant_failures += 1;
        push_cell_violation(
            acc,
            &context,
            "oracle_disagreement",
            "legal-action oracle and hand evaluator disagree".to_string(),
        );
    }
    if label.result.is_win && !label.shape_wait {
        acc.report.feature_invariant_failures += 1;
        push_cell_violation(
            acc,
            &context,
            "oracle_wait_disagreement",
            "hand evaluator reports a win outside its structural wait set".to_string(),
        );
    }
    if label.result.is_win && loss > label.result.ron_agari as f32 + EPSILON {
        acc.report.loss_lower_bound_violations += 1;
        push_cell_violation(
            acc,
            &context,
            "loss_lower_bound_violation",
            format!(
                "public-information loss floor {loss} exceeds evaluated Ron points {}",
                label.result.ron_agari
            ),
        );
    }

    let actual_families = yaku_families(&label.result.yaku);
    if label.result.is_win {
        for (index, &present) in actual_families.iter().enumerate() {
            if present {
                *acc.report
                    .yaku_support
                    .entry(YAKU_EVIDENCE_NAMES[index].to_string())
                    .or_default() += 1;
            }
            if index == 9 {
                continue;
            }
            let evidence = prediction.yaku.values[index];
            if present && evidence == 0.0 {
                acc.report.yaku_impossible_conflicts += 1;
                push_cell_violation(
                    acc,
                    &context,
                    "yaku_impossible_conflict",
                    format!(
                        "{} is present in the hidden win but public evidence says impossible",
                        YAKU_EVIDENCE_NAMES[index]
                    ),
                );
            }
            // The hand evaluator deliberately emits only yakuman IDs once a
            // yakuman is found.  Ordinary yaku which remain logically true
            // (for example Riichi, Yakuhai in Daisangen, or Toitoi in
            // Suukantsu) are consequently absent from `result.yaku`.  Keep
            // exact confirmed-absence checks for yakuman families, but do not
            // turn that scoring-list suppression into a validator failure.
            if confirmed_yaku_absence_is_conflict(index, evidence, present, label.result.yakuman) {
                acc.report.yaku_confirmed_conflicts += 1;
                push_cell_violation(
                    acc,
                    &context,
                    "yaku_confirmed_conflict",
                    format!(
                        "{} is publicly confirmed but absent from the evaluated win",
                        YAKU_EVIDENCE_NAMES[index]
                    ),
                );
            }
        }
    }

    for (index, &evidence) in prediction.yaku.values.iter().enumerate() {
        let valid = if index == 9 {
            is_probability(evidence)
        } else {
            [0.0, 0.5, 0.75, 1.0].contains(&evidence)
        };
        if !valid {
            acc.report.feature_invariant_failures += 1;
            push_cell_violation(
                acc,
                &context,
                "invalid_yaku_evidence",
                format!(
                    "{} has invalid value {evidence}",
                    YAKU_EVIDENCE_NAMES[index]
                ),
            );
        }
    }

    update_digest(acc, &context, slot, tile, prediction, label);
}

fn confirmed_yaku_absence_is_conflict(
    index: usize,
    evidence: f32,
    present: bool,
    scoring_result_is_yakuman: bool,
) -> bool {
    evidence == 1.0 && !present && !(scoring_result_is_yakuman && index < 10)
}

struct CellContext {
    round: usize,
    action: usize,
    actor: u8,
    opponent: u8,
    tile_type: u8,
}

fn push_cell_violation(
    acc: &mut ValidationAccumulator,
    context: &CellContext,
    code: &str,
    message: String,
) {
    acc.report.violations.push(DrevValidationViolation {
        round: context.round,
        action: context.action,
        actor: context.actor,
        opponent: Some(context.opponent),
        tile_type: Some(context.tile_type),
        code: code.to_string(),
        message,
    });
}

#[allow(clippy::too_many_arguments)]
fn validate_encoded_output(
    actor: u8,
    candidates: &[u8],
    opponents: &[DrevV2OpponentResult; 3],
    encoded: &[f32],
    tile_types: usize,
    active: usize,
    round: usize,
    action: usize,
    acc: &mut ValidationAccumulator,
) {
    if encoded.len() != DREV_V2_CHANNELS * tile_types {
        acc.report.feature_invariant_failures += 1;
        acc.report.violations.push(DrevValidationViolation {
            round,
            action,
            actor,
            opponent: None,
            tile_type: None,
            code: "encoded_shape".to_string(),
            message: format!(
                "encoded DREV length is {}; expected {}",
                encoded.len(),
                DREV_V2_CHANNELS * tile_types
            ),
        });
        return;
    }
    let is_sanma = tile_types == 27;
    let candidate_types = candidates
        .iter()
        .map(|tile| *tile as usize / 4)
        .collect::<BTreeSet<_>>();
    if candidate_types.is_empty() {
        acc.report.feature_invariant_failures += 1;
        acc.report.violations.push(DrevValidationViolation {
            round,
            action,
            actor,
            opponent: None,
            tile_type: None,
            code: "discard_candidates".to_string(),
            message: "discard decision has no representable candidate tile type".to_string(),
        });
    }
    for &tile in &candidate_types {
        if output_column(tile, is_sanma).is_none() {
            acc.report.feature_invariant_failures += 1;
            acc.report.violations.push(DrevValidationViolation {
                round,
                action,
                actor,
                opponent: None,
                tile_type: Some(tile as u8),
                code: "discard_candidate_column".to_string(),
                message: "discard candidate has no encoded DREV column".to_string(),
            });
        }
    }

    let aggregate_loss = |tile: usize| {
        (0..active)
            .map(|slot| opponents[slot].ron_prob[tile] * opponents[slot].mean_loss_points[tile])
            .sum::<f32>()
    };
    let best_candidate = candidate_types.iter().copied().min_by(|&left, &right| {
        aggregate_loss(left)
            .total_cmp(&aggregate_loss(right))
            .then_with(|| left.cmp(&right))
    });

    for tile in 0..crate::types::TILE_MAX {
        let Some(column) = output_column(tile, is_sanma) else {
            continue;
        };
        for (slot, opponent) in opponents.iter().enumerate() {
            let base = slot * 6;
            let expected_risk = [
                opponent.hard_safe_zero[tile],
                opponent.wait_prob[tile],
                opponent.ron_prob[tile],
                (opponent.mean_loss_points[tile] / 100_000.0).clamp(0.0, 1.0),
                (opponent.mean_loss_points[tile] / 30_000.0).clamp(0.0, 1.0),
                opponent.uncertainty[tile],
            ];
            for (offset, expected) in expected_risk.into_iter().enumerate() {
                validate_encoded_value(
                    encoded,
                    tile_types,
                    base + offset,
                    column,
                    tile,
                    expected,
                    round,
                    action,
                    actor,
                    acc,
                );
            }
            for (evidence, expected) in opponent.yaku.values.into_iter().enumerate() {
                let channel =
                    DREV_V2_BASE_CHANNELS + slot * DREV_V2_YAKU_CHANNELS_PER_OPPONENT + evidence;
                validate_encoded_value(
                    encoded, tile_types, channel, column, tile, expected, round, action, actor, acc,
                );
            }
        }

        let all_safe =
            (0..active).all(|slot| opponents[slot].hard_safe_zero[tile] == 1.0) as u8 as f32;
        let max_ron = (0..active)
            .map(|slot| opponents[slot].ron_prob[tile])
            .fold(0.0, f32::max);
        let expected_loss = (0..active)
            .map(|slot| opponents[slot].ron_prob[tile] * opponents[slot].mean_loss_points[tile])
            .sum::<f32>();
        let max_uncertainty = (0..active)
            .map(|slot| opponents[slot].uncertainty[tile])
            .fold(0.0, f32::max);
        let expected = [
            all_safe,
            max_ron,
            (expected_loss / 100_000.0).clamp(0.0, 1.0),
            (expected_loss / 30_000.0).clamp(0.0, 1.0),
            usize::from(best_candidate == Some(tile)) as f32,
            max_uncertainty,
        ];
        for (offset, expected) in expected.into_iter().enumerate() {
            validate_encoded_value(
                encoded,
                tile_types,
                18 + offset,
                column,
                tile,
                expected,
                round,
                action,
                actor,
                acc,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_encoded_value(
    encoded: &[f32],
    tile_types: usize,
    channel: usize,
    column: usize,
    tile: usize,
    expected: f32,
    round: usize,
    action: usize,
    actor: u8,
    acc: &mut ValidationAccumulator,
) {
    let actual = encoded[channel * tile_types + column];
    let matches_raw =
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= EPSILON;
    let valid = encoded_channel_value_is_valid(channel, actual);
    if !matches_raw || !valid {
        acc.report.feature_invariant_failures += 1;
        acc.report.violations.push(DrevValidationViolation {
            round,
            action,
            actor,
            opponent: None,
            tile_type: Some(tile as u8),
            code: "encoded_channel_mismatch".to_string(),
            message: format!(
                "channel={channel}, column={column}, actual={actual}, expected={expected}, valid={valid}"
            ),
        });
    }
}

fn encoded_channel_value_is_valid(channel: usize, value: f32) -> bool {
    if matches!(channel, 0 | 6 | 12 | 18 | 22) {
        return is_binary(value);
    }
    if channel >= DREV_V2_BASE_CHANNELS {
        let evidence = (channel - DREV_V2_BASE_CHANNELS) % DREV_V2_YAKU_CHANNELS_PER_OPPONENT;
        return if evidence == 9 {
            is_probability(value)
        } else {
            [0.0, 0.5, 0.75, 1.0].contains(&value)
        };
    }
    is_probability(value)
}

fn validate_inactive_3p_slot(
    opponent: &DrevV2OpponentResult,
    round: usize,
    action: usize,
    actor: u8,
    acc: &mut ValidationAccumulator,
) {
    let nonzero = opponent
        .hard_safe_zero
        .iter()
        .chain(&opponent.wait_prob)
        .chain(&opponent.ron_prob)
        .chain(&opponent.mean_loss_points)
        .chain(&opponent.uncertainty)
        .chain(&opponent.yaku.values)
        .any(|&value| value != 0.0);
    if nonzero {
        acc.report.feature_invariant_failures += 1;
        acc.report.violations.push(DrevValidationViolation {
            round,
            action,
            actor,
            opponent: None,
            tile_type: None,
            code: "inactive_3p_slot".to_string(),
            message: "the third sanma opponent slot must be all zero".to_string(),
        });
    }
}

fn discard_candidates(actions: &[EnvAction]) -> Vec<u8> {
    let mut physical = BTreeSet::new();
    for action in actions {
        if matches!(action.action_type, ActionType::Discard | ActionType::Riichi)
            && let Some(tile) = action.tile
        {
            physical.insert(tile);
        }
    }
    physical.into_iter().collect()
}

fn discard_candidates_3p(actions: &[crate::action::Action3P]) -> Vec<u8> {
    discard_candidates(
        &actions
            .iter()
            .map(|action| action.0.clone())
            .collect::<Vec<_>>(),
    )
}

fn oracle_4p(state: &GameState, opponent: u8, actor: u8, tile: u8) -> OracleLabel {
    let player = &state.players[opponent as usize];
    let evaluator = HandEvaluator::new_borrowed(&player.hand, &player.melds);
    let shape_wait = evaluator.get_waits_u8().contains(&(tile / 4));
    let conditions = Conditions {
        tsumo: false,
        riichi: player.riichi_declared,
        double_riichi: player.double_riichi_declared,
        ippatsu: player.ippatsu_cycle,
        player_wind: Wind::from((opponent + 4 - state.oya) % 4),
        round_wind: Wind::from(state.round_wind),
        houtei: state.wall.drawable_count == 0 && !state.is_rinshan_flag,
        riichi_sticks: state.riichi_sticks,
        honba: state.honba as u32,
        ..Default::default()
    };
    let result = evaluator.calc_borrowed(tile, &state.wall.dora_indicators, &[], Some(conditions));
    let (claims, _) = state._get_claim_actions_for_player(opponent, actor, tile);
    let legal_ron = claims
        .iter()
        .any(|action| action.action_type == ActionType::Ron);
    OracleLabel {
        shape_wait,
        legal_ron,
        result,
    }
}

fn oracle_3p(state: &GameState3P, opponent: u8, actor: u8, tile: u8) -> OracleLabel {
    let player = &state.players[opponent as usize];
    let evaluator = HandEvaluator3P::new_borrowed(&player.hand, &player.melds);
    let shape_wait = evaluator.get_waits_u8().contains(&(tile / 4));
    let conditions = Conditions {
        tsumo: false,
        riichi: player.riichi_declared,
        double_riichi: player.double_riichi_declared,
        ippatsu: player.ippatsu_cycle,
        player_wind: Wind::from((opponent + 3 - state.oya) % 3),
        round_wind: Wind::from(state.round_wind),
        houtei: state.wall.drawable_count == 0 && !state.is_rinshan_flag,
        riichi_sticks: state.riichi_sticks,
        honba: state.honba as u32,
        kita_count: player.kita_tiles.len() as u8,
        is_sanma: true,
        num_players: 3,
        ..Default::default()
    };
    let result = evaluator.calc_borrowed(tile, &state.wall.dora_indicators, &[], Some(conditions));
    let (claims, _) = state._get_claim_actions_for_player(opponent, actor, tile);
    let legal_ron = claims
        .iter()
        .any(|action| action.action_type == ActionType::Ron);
    OracleLabel {
        shape_wait,
        legal_ron,
        result,
    }
}

fn initial_state_4p(kyoku: &LogKyoku) -> RiichiResult<GameState> {
    let scores: [i32; 4] =
        kyoku
            .scores
            .clone()
            .try_into()
            .map_err(|_| RiichiError::InvalidState {
                message: "4P replay round must contain four scores".to_string(),
            })?;
    if kyoku.hands.len() != 4 {
        return Err(RiichiError::InvalidState {
            message: "4P replay round must contain four hands".to_string(),
        });
    }
    let bakaze = checked_wind(kyoku.chang)?;
    let oya = inferred_oya(kyoku, 4)?;
    let mut wall = decoded_wall(kyoku)?;
    let mut state = GameState::new(0, true, None, bakaze, kyoku.rule);
    state._initialize_round(
        oya,
        bakaze,
        kyoku.ben,
        kyoku.liqibang as u32,
        wall.take(),
        Some(scores.to_vec()),
    );
    for (player, hand) in state.players.iter_mut().zip(&kyoku.hands) {
        player.hand = hand.clone();
        player.hand.sort_unstable();
    }
    set_initial_draw_4p(&mut state, kyoku, oya as usize);
    state.wall.dora_indicators = initial_doras(kyoku);
    if state.wall.tiles.len() == 136 {
        let total: usize = state.players.iter().map(|player| player.hand.len()).sum();
        state
            .wall
            .tiles
            .truncate(state.wall.tiles.len().saturating_sub(total));
    }
    Ok(state)
}

fn initial_state_3p(kyoku: &LogKyoku) -> RiichiResult<GameState3P> {
    let scores: [i32; 3] =
        kyoku
            .scores
            .clone()
            .try_into()
            .map_err(|_| RiichiError::InvalidState {
                message: "3P replay round must contain three scores".to_string(),
            })?;
    if kyoku.hands.len() != 3 {
        return Err(RiichiError::InvalidState {
            message: "3P replay round must contain three hands".to_string(),
        });
    }
    let bakaze = checked_wind(kyoku.chang)?;
    let oya = inferred_oya(kyoku, 3)?;
    let mut wall = decoded_wall(kyoku)?;
    let mut state = GameState3P::new(0, true, None, bakaze, kyoku.rule);
    state._initialize_round(
        oya,
        bakaze,
        kyoku.ben,
        kyoku.liqibang as u32,
        wall.take(),
        Some(scores.to_vec()),
    );
    for (player, hand) in state.players.iter_mut().zip(&kyoku.hands) {
        player.hand = hand.clone();
        player.hand.sort_unstable();
    }
    set_initial_draw_3p(&mut state, kyoku, oya as usize);
    state.wall.dora_indicators = initial_doras(kyoku);
    Ok(state)
}

fn set_initial_draw_4p(state: &mut GameState, kyoku: &LogKyoku, oya: usize) {
    if state.players[oya].hand.len() == 14 {
        state.drawn_tile =
            inferred_initial_draw(kyoku, oya).or_else(|| state.players[oya].hand.last().copied());
        state.needs_tsumo = false;
    } else {
        if let Some(tile) = state.drawn_tile {
            state.wall.tiles.push(tile);
            state.wall.drawable_count += 1;
        }
        state.drawn_tile = None;
        state.needs_tsumo = true;
    }
}

fn set_initial_draw_3p(state: &mut GameState3P, kyoku: &LogKyoku, oya: usize) {
    if state.players[oya].hand.len() == 14 {
        state.drawn_tile =
            inferred_initial_draw(kyoku, oya).or_else(|| state.players[oya].hand.last().copied());
        state.needs_tsumo = false;
    } else {
        if let Some(tile) = state.drawn_tile {
            state.wall.tiles.push(tile);
            state.wall.drawable_count += 1;
        }
        state.drawn_tile = None;
        state.needs_tsumo = true;
    }
}

fn inferred_initial_draw(kyoku: &LogKyoku, oya: usize) -> Option<u8> {
    match kyoku.actions.first()? {
        LogAction::Hule { hules } => hules
            .iter()
            .find(|hule| hule.seat == oya && hule.zimo)
            .map(|hule| hule.hu_tile),
        LogAction::DiscardTile { seat, tile, .. } if *seat == oya => Some(*tile),
        LogAction::AnGangAddGang { seat, tiles, .. } if *seat == oya => tiles.first().copied(),
        LogAction::ChiPengGang {
            seat,
            meld_type: MeldType::Ankan,
            tiles,
            ..
        } if *seat == oya => tiles.first().copied(),
        _ => None,
    }
}

fn inferred_oya(kyoku: &LogKyoku, players: usize) -> RiichiResult<u8> {
    let mut oya = usize::from(kyoku.ju);
    for (seat, hand) in kyoku.hands.iter().enumerate() {
        if hand.len() == 14 {
            oya = seat;
            break;
        }
    }
    if oya >= players {
        return Err(RiichiError::InvalidState {
            message: format!("replay dealer {oya} is out of range for {players} players"),
        });
    }
    Ok(oya as u8)
}

fn decoded_wall(kyoku: &LogKyoku) -> RiichiResult<Option<Vec<u8>>> {
    let Some(hex_wall) = &kyoku.paishan else {
        return Ok(None);
    };
    let wall = hex::decode(hex_wall).map_err(|error| RiichiError::Parse {
        input: "replay wall".to_string(),
        message: error.to_string(),
    })?;
    Ok(Some(wall))
}

fn initial_doras(kyoku: &LogKyoku) -> Vec<u8> {
    let has_updates = kyoku.action_dora_snapshots.iter().any(Option::is_some)
        || kyoku.actions.iter().any(|action| match action {
            LogAction::Dora { .. } => true,
            LogAction::DealTile { doras, .. }
            | LogAction::DiscardTile { doras, .. }
            | LogAction::AnGangAddGang { doras, .. } => doras.is_some(),
            _ => false,
        });
    if has_updates {
        kyoku.doras.first().copied().into_iter().collect()
    } else {
        kyoku.doras.clone()
    }
}

fn apply_action_4p(state: &mut GameState, kyoku: &LogKyoku, index: usize, action: &LogAction) {
    if matches!(action, LogAction::Other(_)) {
        return;
    }
    state.apply_log_action_with_metadata(
        action,
        kyoku.action_tsumogiri.get(index).copied().flatten(),
        kyoku
            .action_dora_snapshots
            .get(index)
            .and_then(Option::as_deref),
    );
}

fn apply_action_3p(state: &mut GameState3P, kyoku: &LogKyoku, index: usize, action: &LogAction) {
    if matches!(action, LogAction::Other(_)) {
        return;
    }
    state.apply_log_action_with_metadata(
        action,
        kyoku.action_tsumogiri.get(index).copied().flatten(),
        kyoku
            .action_dora_snapshots
            .get(index)
            .and_then(Option::as_deref),
    );
}

fn checked_actor(actor: usize, players: usize, field: &str) -> RiichiResult<u8> {
    if actor < players {
        Ok(actor as u8)
    } else {
        Err(RiichiError::InvalidState {
            message: format!("{field} actor {actor} is out of range"),
        })
    }
}

fn checked_wind(wind: u8) -> RiichiResult<u8> {
    if wind <= 3 {
        Ok(wind)
    } else {
        Err(RiichiError::InvalidState {
            message: format!("replay round wind {wind} is out of range"),
        })
    }
}

fn is_binary(value: f32) -> bool {
    matches!(value, 0.0 | 1.0)
}

fn is_probability(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn output_column(tile: usize, sanma: bool) -> Option<usize> {
    if !sanma {
        return (tile < 34).then_some(tile);
    }
    match tile {
        0 => Some(0),
        1..=7 => None,
        8..=33 => Some(tile - 7),
        _ => None,
    }
}

fn yaku_families(ids: &[u32]) -> [bool; DREV_V2_YAKU_CHANNELS_PER_OPPONENT] {
    let mut families = [false; DREV_V2_YAKU_CHANNELS_PER_OPPONENT];
    for &id in ids {
        match id {
            2 | 18 => families[0] = true,
            12 => families[1] = true,
            7..=11 => families[2] = true,
            27 => families[3] = true,
            29 => families[4] = true,
            21 => families[5] = true,
            25 => families[6] = true,
            15 | 24 | 26 => families[7] = true,
            13 | 14 | 16 | 17 | 28 => families[8] = true,
            42 | 49 => families[10] = true,
            37 => families[11] = true,
            43 | 50 => families[12] = true,
            38 | 48 => families[13] = true,
            39 => families[14] = true,
            40 => families[15] = true,
            41 => families[16] = true,
            45 | 47 => families[17] = true,
            44 => families[18] = true,
            _ => {}
        }
    }
    families
}

fn update_encoded_digest(
    acc: &mut ValidationAccumulator,
    round: usize,
    action: usize,
    actor: u8,
    encoded: &[f32],
) {
    // Domain-separate the model-facing wire tensor from the candidate/label
    // rows appended by `update_digest`.  This makes the golden sensitive to
    // channel order, non-candidate columns, and the inactive sanma slot.
    acc.digest.update(b"DREV-v2-encoded");
    acc.digest.update((round as u64).to_le_bytes());
    acc.digest.update((action as u64).to_le_bytes());
    acc.digest.update([actor]);
    acc.digest.update((encoded.len() as u64).to_le_bytes());
    for &value in encoded {
        acc.digest.update(quantized_feature(value).to_le_bytes());
    }
}

fn update_digest(
    acc: &mut ValidationAccumulator,
    context: &CellContext,
    slot: usize,
    tile: u8,
    prediction: &DrevV2OpponentResult,
    label: &OracleLabel,
) {
    acc.digest.update((context.round as u64).to_le_bytes());
    acc.digest.update((context.action as u64).to_le_bytes());
    acc.digest
        .update([context.actor, context.opponent, slot as u8, tile]);
    acc.digest
        .update([u8::from(label.shape_wait), u8::from(label.legal_ron)]);
    let tile_type = tile as usize / 4;
    for value in [
        prediction.hard_safe_zero[tile_type],
        prediction.wait_prob[tile_type],
        prediction.ron_prob[tile_type],
        prediction.mean_loss_points[tile_type],
        prediction.uncertainty[tile_type],
    ] {
        acc.digest.update(quantized_feature(value).to_le_bytes());
    }
    for value in prediction.yaku.values {
        acc.digest.update(quantized_feature(value).to_le_bytes());
    }
    acc.digest.update([u8::from(label.result.is_win)]);
    acc.digest.update(label.result.han.to_le_bytes());
    acc.digest.update(label.result.fu.to_le_bytes());
    acc.digest.update(label.result.ron_agari.to_le_bytes());
    let mut yaku = label.result.yaku.clone();
    yaku.sort_unstable();
    acc.digest.update((yaku.len() as u64).to_le_bytes());
    for id in yaku {
        acc.digest.update(id.to_le_bytes());
    }
}

fn quantized_feature(value: f32) -> i64 {
    // Raw float bits can differ across target architectures even when the
    // public feature is numerically equivalent. A one-millionth resolution
    // is much finer than the feature contract while keeping corpus digests
    // reproducible across CI targets.
    (f64::from(value) * 1_000_000.0).round() as i64
}

fn calculate_metrics(
    wait: &[(f32, bool)],
    ron: &[(f32, bool)],
    loss: &[(f32, f32)],
) -> DrevValidationMetrics {
    let (wait_brier, wait_log_loss, wait_average_precision, wait_ece_10) = binary_metrics(wait);
    let (ron_brier, ron_log_loss, ron_average_precision, ron_ece_10) = binary_metrics(ron);
    let (legal_ron_loss_mae, legal_ron_loss_bias) = if loss.is_empty() {
        (0.0, 0.0)
    } else {
        let count = loss.len() as f64;
        (
            loss.iter()
                .map(|&(prediction, actual)| (prediction - actual).abs() as f64)
                .sum::<f64>()
                / count,
            loss.iter()
                .map(|&(prediction, actual)| (prediction - actual) as f64)
                .sum::<f64>()
                / count,
        )
    };
    DrevValidationMetrics {
        wait_brier,
        wait_log_loss,
        wait_average_precision,
        wait_ece_10,
        ron_brier,
        ron_log_loss,
        ron_average_precision,
        ron_ece_10,
        legal_ron_loss_mae,
        legal_ron_loss_bias,
    }
}

fn binary_metrics(samples: &[(f32, bool)]) -> (f64, f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let count = samples.len() as f64;
    let brier = samples
        .iter()
        .map(|&(prediction, label)| {
            let error = prediction as f64 - f64::from(u8::from(label));
            error * error
        })
        .sum::<f64>()
        / count;
    let log_loss = samples
        .iter()
        .map(|&(prediction, label)| {
            let probability = (prediction as f64).clamp(LOG_EPSILON, 1.0 - LOG_EPSILON);
            if label {
                -probability.ln()
            } else {
                -(1.0 - probability).ln()
            }
        })
        .sum::<f64>()
        / count;
    let mut ranked = samples.to_vec();
    ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
    let positives = ranked.iter().filter(|(_, label)| *label).count();
    let average_precision = if positives == 0 {
        0.0
    } else {
        let mut found = 0usize;
        let mut precision_sum = 0.0;
        for (index, (_, label)) in ranked.iter().enumerate() {
            if *label {
                found += 1;
                precision_sum += found as f64 / (index + 1) as f64;
            }
        }
        precision_sum / positives as f64
    };
    let mut bins = [(0usize, 0.0f64, 0usize); 10];
    for &(prediction, label) in samples {
        let index = ((prediction.clamp(0.0, 1.0) * 10.0) as usize).min(9);
        bins[index].0 += 1;
        bins[index].1 += prediction as f64;
        bins[index].2 += usize::from(label);
    }
    let ece = bins
        .iter()
        .filter(|(size, _, _)| *size > 0)
        .map(|&(size, prediction_sum, positive_count)| {
            let weight = size as f64 / count;
            let mean_prediction = prediction_sum / size as f64;
            let empirical = positive_count as f64 / size as f64;
            weight * (mean_prediction - empirical).abs()
        })
        .sum();
    (brier, log_loss, average_precision, ece)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct AgariCorpus {
        cases: Vec<AgariCase>,
    }

    #[derive(Deserialize)]
    struct AgariCase {
        tiles_136: Vec<u8>,
        melds: Vec<CorpusMeld>,
        win_tile_136: u8,
        conditions: CorpusConditions,
        expected: CorpusExpected,
    }

    #[derive(Deserialize)]
    struct CorpusMeld {
        meld_type: String,
        tiles: Vec<u8>,
        opened: bool,
        from_who: i8,
    }

    #[derive(Deserialize)]
    struct CorpusConditions {
        riichi: bool,
        double_riichi: bool,
    }

    #[derive(Deserialize)]
    struct CorpusExpected {
        is_win: bool,
        yaku: Vec<u32>,
    }

    fn corpus_meld(meld: &CorpusMeld) -> crate::types::Meld {
        let meld_type = match meld.meld_type.as_str() {
            "chi" => MeldType::Chi,
            "pon" => MeldType::Pon,
            "daiminkan" => MeldType::Daiminkan,
            "ankan" => MeldType::Ankan,
            "kakan" => MeldType::Kakan,
            other => panic!("unknown corpus meld type {other}"),
        };
        crate::types::Meld::new(
            meld_type,
            meld.tiles.clone(),
            meld.opened,
            meld.from_who,
            None,
        )
    }

    fn public_projection(case: &AgariCase) -> crate::observation::Observation {
        let mut occupied = [false; crate::types::TILES_4P];
        for &tile in case
            .tiles_136
            .iter()
            .chain(std::iter::once(&case.win_tile_136))
            .chain(case.melds.iter().flat_map(|meld| meld.tiles.iter()))
        {
            occupied[tile as usize] = true;
        }
        let available = (0..crate::types::TILES_4P)
            .filter(|&tile| !occupied[tile])
            .map(|tile| tile as u8)
            .collect::<Vec<_>>();
        let observer_hand = available[..crate::types::INITIAL_HAND_SIZE].to_vec();
        assert_eq!(observer_hand.len(), crate::types::INITIAL_HAND_SIZE);

        let mut riichi = [false; 4];
        riichi[1] = case.conditions.riichi || case.conditions.double_riichi;
        let riichi_discard = riichi[1].then_some(available[crate::types::INITIAL_HAND_SIZE]);
        let mut discards: [Vec<u8>; 4] = Default::default();
        if let Some(tile) = riichi_discard {
            discards[1].push(tile);
        }
        let mut observation = crate::observation::Observation::new(
            0,
            [observer_hand, Vec::new(), Vec::new(), Vec::new()],
            [
                Vec::new(),
                case.melds.iter().map(corpus_meld).collect(),
                Vec::new(),
                Vec::new(),
            ],
            discards,
            Vec::new(),
            [25_000; 4],
            riichi,
            vec![EnvAction::new(
                ActionType::Discard,
                Some(0),
                Vec::new(),
                Some(0),
            )],
            Vec::new(),
            0,
            0,
            0,
            0,
            0,
            Vec::new(),
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        );
        if riichi_discard.is_some() {
            observation.public_tsumogiri_history[1].push(false);
            observation.discard_is_riichi[1].push(true);
            observation.discard_actor_history.push(1);
            observation.resolved_discard_count = 1;
        }
        observation.public_history_complete = true;
        observation
    }

    #[test]
    fn yaku_family_mapping_covers_all_public_evidence_heads() {
        let families = yaku_families(&[
            2, 12, 7, 27, 29, 21, 25, 24, 16, 42, 37, 50, 48, 39, 40, 41, 47, 44,
        ]);
        for (index, &present) in families.iter().enumerate() {
            if index == 9 {
                assert!(!present, "public bonus is not a yaku family");
            } else {
                assert!(present, "missing family {}", YAKU_EVIDENCE_NAMES[index]);
            }
        }
    }

    #[test]
    fn yakuman_scoring_list_may_suppress_confirmed_ordinary_yaku() {
        // HandEvaluator intentionally returns only yakuman IDs for a yakuman
        // result.  Riichi/Yakuhai/Toitoi may still be logically confirmed and
        // must not become confirmed-absence failures merely because they were
        // suppressed from the scoring list.
        for ordinary_family in [0, 2, 5] {
            assert!(!confirmed_yaku_absence_is_conflict(
                ordinary_family,
                1.0,
                false,
                true,
            ));
        }
        // Yakuman evidence remains exact: a confirmed Daisangen family may
        // not disappear merely because a different yakuman was scored.
        assert!(confirmed_yaku_absence_is_conflict(11, 1.0, false, true));
        // Ordinary wins still require every confirmed ordinary family.
        assert!(confirmed_yaku_absence_is_conflict(2, 1.0, false, false));

        let mut prediction = DrevV2OpponentResult::default();
        prediction.yaku.values[0] = 1.0; // Riichi
        prediction.yaku.values[2] = 1.0; // Yakuhai
        prediction.yaku.values[5] = 1.0; // Toitoi
        prediction.yaku.values[11] = 1.0; // Daisangen
        let label = OracleLabel {
            shape_wait: true,
            legal_ron: false,
            result: WinResult::new(true, true, 32_000, 0, 0, vec![37], 13, 0, None, true),
        };
        let mut acc = ValidationAccumulator::default();
        validate_cell(&prediction, &label, 0, 0, 0, 1, 0, 0, &mut acc);
        assert_eq!(acc.report.yaku_confirmed_conflicts, 0);
        assert_eq!(acc.report.yaku_impossible_conflicts, 0);
        assert!(acc.report.violations.is_empty());
    }

    fn zero_encoded_fixture(tile_types: usize, candidate_column: usize) -> Vec<f32> {
        let mut encoded = vec![0.0; DREV_V2_CHANNELS * tile_types];
        encoded[22 * tile_types + candidate_column] = 1.0;
        encoded
    }

    #[test]
    fn encoded_validator_checks_non_candidate_channels_and_all_marker_columns() {
        let opponents: [DrevV2OpponentResult; 3] = std::array::from_fn(|_| Default::default());
        let encoded = zero_encoded_fixture(34, 0);
        let mut valid = ValidationAccumulator::default();
        validate_encoded_output(0, &[0], &opponents, &encoded, 34, 3, 0, 0, &mut valid);
        assert_eq!(valid.report.feature_invariant_failures, 0);

        let mut corrupt_non_candidate = encoded.clone();
        corrupt_non_candidate[DREV_V2_BASE_CHANNELS * 34 + 33] = 0.5;
        let mut non_candidate = ValidationAccumulator::default();
        validate_encoded_output(
            0,
            &[0],
            &opponents,
            &corrupt_non_candidate,
            34,
            3,
            0,
            0,
            &mut non_candidate,
        );
        assert!(non_candidate.report.feature_invariant_failures > 0);

        let mut extra_marker = encoded;
        extra_marker[22 * 34 + 33] = 1.0;
        let mut marker = ValidationAccumulator::default();
        validate_encoded_output(0, &[0], &opponents, &extra_marker, 34, 3, 0, 0, &mut marker);
        assert!(marker.report.feature_invariant_failures > 0);
    }

    #[test]
    fn encoded_validator_checks_inactive_sanma_slot() {
        let opponents: [DrevV2OpponentResult; 3] = std::array::from_fn(|_| Default::default());
        let encoded = zero_encoded_fixture(27, 0);
        let mut valid = ValidationAccumulator::default();
        validate_encoded_output(0, &[0], &opponents, &encoded, 27, 2, 0, 0, &mut valid);
        assert_eq!(valid.report.feature_invariant_failures, 0);

        let mut corrupt = encoded;
        corrupt[12 * 27] = 1.0;
        let mut invalid = ValidationAccumulator::default();
        validate_encoded_output(0, &[0], &opponents, &corrupt, 27, 2, 0, 0, &mut invalid);
        assert!(invalid.report.feature_invariant_failures > 0);
    }

    #[test]
    fn binary_metrics_are_well_defined_without_positive_labels() {
        let metrics = binary_metrics(&[(0.1, false), (0.2, false)]);
        assert!(metrics.0 > 0.0);
        assert!(metrics.1 > 0.0);
        assert_eq!(metrics.2, 0.0);
        assert!(metrics.3 > 0.0);
    }

    #[test]
    fn agari_corpus_never_marks_an_actual_major_yaku_family_impossible() {
        let corpus: AgariCorpus =
            serde_json::from_str(include_str!("../benches/data/agari_4p.json"))
                .expect("agari corpus must be valid JSON");
        let mut covered = [false; DREV_V2_YAKU_CHANNELS_PER_OPPONENT];

        for (case_index, case) in corpus.cases.iter().enumerate() {
            if !case.expected.is_win {
                continue;
            }
            let actual = yaku_families(&case.expected.yaku);
            if !actual.iter().any(|present| *present) {
                continue;
            }
            let result = calculate_drev_v2(&public_projection(case))
                .unwrap_or_else(|error| panic!("corpus case {case_index}: {error}"));
            let evidence = &result.opponents()[0].yaku.values;
            for (family, &present) in actual.iter().enumerate() {
                if present {
                    covered[family] = true;
                    assert_ne!(
                        evidence[family], 0.0,
                        "corpus case {case_index} actual family {} was marked impossible",
                        YAKU_EVIDENCE_NAMES[family]
                    );
                }
            }
        }

        for (family, &present) in covered.iter().enumerate() {
            if matches!(family, 9 | 18) {
                // Public bonus is not a yaku family; the agari corpus has no
                // Suukantsu case, which is covered by the four-kan unit fixture.
                continue;
            }
            assert!(
                present,
                "agari corpus does not cover {}",
                YAKU_EVIDENCE_NAMES[family]
            );
        }
    }
}
