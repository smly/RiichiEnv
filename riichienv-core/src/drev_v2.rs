//! Versioned public-information deal-in risk evidence.
//!
//! DREV v1 is a frozen nine-plane heuristic.  V2 deliberately separates:
//!
//! * rule-proven zero ron risk (`hard_safe_zero`),
//! * soft wait/ron likelihood estimates, and
//! * public evidence for common yaku and yakuman.
//!
//! Only the observing player's concealed hand and public table state are read.
//! Opponent hands, the wall, ura indicators, and legal-action oracles are not
//! inputs.  The probability-like values are deterministic priors rather than
//! calibrated model outputs; callers should use the dedicated uncertainty and
//! evidence planes instead of interpreting every small number as "safe".

use crate::action::ActionType;
use crate::errors::{RiichiError, RiichiResult};
use crate::observation::{OBS_TILE_TYPES, Observation};
use crate::observation_3p::{OBS_3P_TILE_TYPES, Observation3P};
use crate::score::calculate_score;
use crate::types::{Meld, MeldType, TILE_MAX};

/// Six per-opponent DREV heads followed by six aggregate heads.
pub const DREV_V2_BASE_CHANNELS: usize = 24;
/// Number of public yaku/value/yakuman evidence heads per opponent.
pub const DREV_V2_YAKU_CHANNELS_PER_OPPONENT: usize = 19;
/// Complete DREV-v2 layout: 24 risk heads plus 3 x 19 evidence heads.
pub const DREV_V2_CHANNELS: usize = DREV_V2_BASE_CHANNELS + 3 * DREV_V2_YAKU_CHANNELS_PER_OPPONENT;
/// Stable machine-readable identifier for the 81-channel DREV-v2 wire layout.
pub const DREV_V2_SCHEMA_ID: &str = "riichienv.drev_v2.81ch.v1";
/// Relative-opponent slot order used by risk and evidence channels.
///
/// Absolute seat is `(observer + slot + 1) % num_players`. In 3P,
/// `relative_3` is the all-zero inactive compatibility slot.
pub const OPPONENT_SLOT_ORDER: [&str; 3] = ["relative_1", "relative_2", "relative_3"];
/// Ordered names for the first 24 risk channels.
pub const RISK_CHANNEL_NAMES: [&str; DREV_V2_BASE_CHANNELS] = [
    "relative_1.hard_safe_zero",
    "relative_1.wait_prob",
    "relative_1.ron_prob",
    "relative_1.mean_loss_points_100k",
    "relative_1.mean_loss_points_30k",
    "relative_1.uncertainty",
    "relative_2.hard_safe_zero",
    "relative_2.wait_prob",
    "relative_2.ron_prob",
    "relative_2.mean_loss_points_100k",
    "relative_2.mean_loss_points_30k",
    "relative_2.uncertainty",
    "relative_3.hard_safe_zero",
    "relative_3.wait_prob",
    "relative_3.ron_prob",
    "relative_3.mean_loss_points_100k",
    "relative_3.mean_loss_points_30k",
    "relative_3.uncertainty",
    "aggregate.all_safe_zero",
    "aggregate.max_ron_prob",
    "aggregate.sum_expected_loss_points_100k",
    "aggregate.sum_expected_loss_points_30k",
    "aggregate.min_loss_discard_marker",
    "aggregate.max_uncertainty",
];

pub const YAKU_EVIDENCE_NAMES: [&str; DREV_V2_YAKU_CHANNELS_PER_OPPONENT] = [
    "riichi",
    "tanyao",
    "yakuhai",
    "honitsu",
    "chinitsu",
    "toitoi",
    "chiitoitsu",
    "terminal_honor",
    "sequence",
    "public_bonus",
    "kokushi",
    "daisangen",
    "wind_yakuman",
    "suuankou",
    "tsuuiisou",
    "ryuuiisou",
    "chinroutou",
    "chuuren",
    "suukantsu",
];

/// Ordered names for every encoded DREV-v2 channel.
pub fn channel_names() -> Vec<String> {
    let mut names = RISK_CHANNEL_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    for slot in OPPONENT_SLOT_ORDER {
        names.extend(
            YAKU_EVIDENCE_NAMES
                .iter()
                .map(|evidence| format!("{slot}.yaku.{evidence}")),
        );
    }
    names
}

const EVIDENCE_IMPOSSIBLE: f32 = 0.0;
const EVIDENCE_POSSIBLE: f32 = 0.5;
const EVIDENCE_STRONG: f32 = 0.75;
const EVIDENCE_CONFIRMED: f32 = 1.0;

#[derive(Debug, Clone)]
struct OpponentInput {
    absolute_seat: u8,
    melds: Vec<Meld>,
    discards: Vec<u32>,
    tsumogiri: Vec<bool>,
    riichi: bool,
    kita_count: u8,
    river_mask: u64,
    post_riichi_pass_mask: u64,
    temporary_safe_mask: u64,
}

struct OpponentPublicView<'a> {
    absolute_seat: usize,
    melds: &'a [Meld],
    discards: &'a [u32],
    tsumogiri: &'a [bool],
    riichi: bool,
    kita_count: u8,
    post_riichi_pass_mask: u64,
    temporary_safe_mask: u64,
}

#[derive(Debug, Clone)]
struct DrevV2Input {
    num_players: u8,
    oya: u8,
    round_wind: u8,
    honba: u8,
    dora_indicators: Vec<u8>,
    visible_counts: [u8; TILE_MAX],
    opponents: [Option<OpponentInput>; 3],
    discard_candidates: Vec<u8>,
}

/// Public evidence for normal yaku, bonus value, and yakuman families.
///
/// Except for `public_bonus`, values use the following ordinal convention:
/// `0.0=impossible`, `0.5=not ruled out`, `0.75=positive public evidence`,
/// `1.0=confirmed by public state`.  They are not probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrevYakuEvidence {
    pub values: [f32; DREV_V2_YAKU_CHANNELS_PER_OPPONENT],
}

impl Default for DrevYakuEvidence {
    fn default() -> Self {
        Self {
            // `Default` represents an inactive opponent slot (the third slot
            // in sanma), not an active opponent with unknown composition.
            // Active opponents are explicitly initialized to POSSIBLE in
            // `yaku_evidence` below.
            values: [EVIDENCE_IMPOSSIBLE; DREV_V2_YAKU_CHANNELS_PER_OPPONENT],
        }
    }
}

impl DrevYakuEvidence {
    pub fn value(&self, name: &str) -> Option<f32> {
        YAKU_EVIDENCE_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| self.values[index])
    }

    pub fn max_yakuman_evidence(&self) -> f32 {
        self.values[10..].iter().copied().fold(0.0, f32::max)
    }
}

/// One relative opponent's tile-conditioned risk evidence.
#[derive(Debug, Clone)]
pub struct DrevV2OpponentResult {
    /// Rule-proven inability to ron this tile type at the current decision.
    pub hard_safe_zero: [f32; TILE_MAX],
    /// Deterministic public-information wait prior. Not calibrated.
    pub wait_prob: [f32; TILE_MAX],
    /// Wait prior after public yaku viability; this is zero for hard-safe tiles.
    pub ron_prob: [f32; TILE_MAX],
    /// Conditional heuristic ron loss in points.
    pub mean_loss_points: [f32; TILE_MAX],
    /// Soft-estimation uncertainty; hard-safe proof is carried separately.
    pub uncertainty: [f32; TILE_MAX],
    /// Major normal-yaku and yakuman evidence.
    pub yaku: DrevYakuEvidence,
}

impl Default for DrevV2OpponentResult {
    fn default() -> Self {
        Self {
            hard_safe_zero: [0.0; TILE_MAX],
            wait_prob: [0.0; TILE_MAX],
            ron_prob: [0.0; TILE_MAX],
            mean_loss_points: [0.0; TILE_MAX],
            uncertainty: [0.0; TILE_MAX],
            yaku: DrevYakuEvidence::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct DrevV2Result {
    opponents: [DrevV2OpponentResult; 3],
    all_safe_zero: [f32; TILE_MAX],
    max_ron_prob: [f32; TILE_MAX],
    sum_expected_loss_points: [f32; TILE_MAX],
    min_loss_discard_marker: [f32; TILE_MAX],
    max_uncertainty: [f32; TILE_MAX],
}

/// Version-tagged 4P DREV-v2 result.
#[derive(Debug, Clone)]
pub struct Drev4PV2Result(DrevV2Result);

impl Drev4PV2Result {
    pub const VERSION: u16 = 2;

    pub fn opponents(&self) -> &[DrevV2OpponentResult; 3] {
        &self.0.opponents
    }

    pub fn encode(&self) -> Vec<f32> {
        encode_v2(&self.0, OBS_TILE_TYPES, false)
    }

    pub fn encode_into(&self, output: &mut [f32]) -> RiichiResult<()> {
        encode_v2_into(&self.0, output, OBS_TILE_TYPES, false)
    }
}

/// Version-tagged 3P DREV-v2 result. Relative opponent slot 2 stays zero.
#[derive(Debug, Clone)]
pub struct Drev3PV2Result(DrevV2Result);

impl Drev3PV2Result {
    pub const VERSION: u16 = 2;

    pub fn opponents(&self) -> &[DrevV2OpponentResult; 3] {
        &self.0.opponents
    }

    pub fn encode(&self) -> Vec<f32> {
        encode_v2(&self.0, OBS_3P_TILE_TYPES, true)
    }

    pub fn encode_into(&self, output: &mut [f32]) -> RiichiResult<()> {
        encode_v2_into(&self.0, output, OBS_3P_TILE_TYPES, true)
    }
}

/// Calculate 4P DREV v2 from a complete engine-produced public history.
pub fn calculate_drev_v2(observation: &Observation) -> RiichiResult<Drev4PV2Result> {
    validate_drev_v2_observation(observation)?;
    calculate_drev_v2_prevalidated(observation)
}

/// Calculate after [`validate_drev_v2_observation`] has succeeded.
pub(crate) fn calculate_drev_v2_prevalidated(
    observation: &Observation,
) -> RiichiResult<Drev4PV2Result> {
    let input = input_4p(observation)?;
    Ok(Drev4PV2Result(calculate(&input)))
}

/// Calculate 3P DREV v2 from a complete engine-produced public history.
pub fn calculate_drev_3p_v2(observation: &Observation3P) -> RiichiResult<Drev3PV2Result> {
    validate_drev_3p_v2_observation(observation)?;
    calculate_drev_3p_v2_prevalidated(observation)
}

/// Calculate after [`validate_drev_3p_v2_observation`] has succeeded.
pub(crate) fn calculate_drev_3p_v2_prevalidated(
    observation: &Observation3P,
) -> RiichiResult<Drev3PV2Result> {
    let input = input_3p(observation)?;
    Ok(Drev3PV2Result(calculate(&input)))
}

/// Preflight a 4P observation for DREV-v2 batch calculation.
pub(crate) fn validate_drev_v2_observation(observation: &Observation) -> RiichiResult<()> {
    observation.validate()?;
    observation.validate_public_discard_history()?;
    require_complete_history(observation.public_history_complete)
}

/// Preflight a 3P observation for DREV-v2 batch calculation.
pub(crate) fn validate_drev_3p_v2_observation(observation: &Observation3P) -> RiichiResult<()> {
    observation.validate()?;
    observation.validate_public_discard_history()?;
    require_complete_history(observation.public_history_complete)
}

fn require_complete_history(complete: bool) -> RiichiResult<()> {
    if complete {
        Ok(())
    } else {
        Err(RiichiError::InvalidState {
            message: "DREV v2 requires the complete runtime public-history sidecar; frozen base64 Observation payloads carry only DREV v1 state".to_string(),
        })
    }
}

fn input_4p(observation: &Observation) -> RiichiResult<DrevV2Input> {
    let pass_masks = post_riichi_pass_masks(
        &observation.discards,
        &observation.discard_is_riichi,
        &observation.discard_actor_history,
        observation.resolved_discard_count,
    )?;
    let opponents = std::array::from_fn(|slot| {
        let seat = (observation.player_id as usize + slot + 1) % 4;
        Some(opponent_input(OpponentPublicView {
            absolute_seat: seat,
            melds: &observation.melds[seat],
            discards: &observation.discards[seat],
            tsumogiri: &observation.public_tsumogiri_history[seat],
            riichi: observation.riichi_declared[seat],
            kita_count: 0,
            post_riichi_pass_mask: pass_masks[seat],
            temporary_safe_mask: observation.public_temporary_safe_masks[seat],
        }))
    });
    Ok(DrevV2Input {
        num_players: 4,
        oya: observation.oya,
        round_wind: observation.round_wind,
        honba: observation.honba,
        dora_indicators: observation
            .dora_indicators
            .iter()
            .map(|&tile| tile as u8)
            .collect(),
        visible_counts: corrected_visible_counts_4p(observation),
        opponents,
        discard_candidates: discard_candidates_4p(observation),
    })
}

fn input_3p(observation: &Observation3P) -> RiichiResult<DrevV2Input> {
    let pass_masks = post_riichi_pass_masks(
        &observation.discards,
        &observation.discard_is_riichi,
        &observation.discard_actor_history,
        observation.resolved_discard_count,
    )?;
    let opponents = std::array::from_fn(|slot| {
        if slot >= 2 {
            return None;
        }
        let seat = (observation.player_id as usize + slot + 1) % 3;
        Some(opponent_input(OpponentPublicView {
            absolute_seat: seat,
            melds: &observation.melds[seat],
            discards: &observation.discards[seat],
            tsumogiri: &observation.public_tsumogiri_history[seat],
            riichi: observation.riichi_declared[seat],
            kita_count: observation.kita_counts[seat],
            post_riichi_pass_mask: pass_masks[seat],
            temporary_safe_mask: observation.public_temporary_safe_masks[seat],
        }))
    });
    Ok(DrevV2Input {
        num_players: 3,
        oya: observation.oya,
        round_wind: observation.round_wind,
        honba: observation.honba,
        dora_indicators: observation
            .dora_indicators
            .iter()
            .map(|&tile| tile as u8)
            .collect(),
        visible_counts: corrected_visible_counts_3p(observation),
        opponents,
        discard_candidates: discard_candidates_3p(observation),
    })
}

fn opponent_input(view: OpponentPublicView<'_>) -> OpponentInput {
    OpponentInput {
        absolute_seat: view.absolute_seat as u8,
        melds: view.melds.to_vec(),
        discards: view.discards.to_vec(),
        tsumogiri: view.tsumogiri.to_vec(),
        riichi: view.riichi,
        kita_count: view.kita_count,
        river_mask: tile_mask(view.discards),
        post_riichi_pass_mask: view.post_riichi_pass_mask,
        temporary_safe_mask: view.temporary_safe_mask,
    }
}

fn tile_mask(tiles: &[u32]) -> u64 {
    tiles.iter().fold(0u64, |mask, &tile| {
        let tile_type = tile as usize / 4;
        if tile_type < TILE_MAX {
            mask | (1u64 << tile_type)
        } else {
            mask
        }
    })
}

fn post_riichi_pass_masks<const N: usize>(
    discards: &[Vec<u32>; N],
    riichi_flags: &[Vec<bool>; N],
    actors: &[u8],
    resolved_count: usize,
) -> RiichiResult<[u64; N]> {
    if resolved_count > actors.len() {
        return Err(RiichiError::InvalidState {
            message: "resolved discard cursor exceeds public history".to_string(),
        });
    }
    let mut masks = [0u64; N];
    let mut local_index = [0usize; N];
    let mut fixed_wait = [false; N];
    for &actor in actors.iter().take(resolved_count) {
        let actor = actor as usize;
        let Some(tiles) = discards.get(actor) else {
            return Err(RiichiError::InvalidState {
                message: "public discard history contains an invalid actor".to_string(),
            });
        };
        let index = local_index[actor];
        let Some(&physical_tile) = tiles.get(index) else {
            return Err(RiichiError::InvalidState {
                message: "public discard history is not aligned with per-seat rivers".to_string(),
            });
        };
        let tile_type = physical_tile as usize / 4;
        if tile_type < TILE_MAX {
            let bit = 1u64 << tile_type;
            for seat in 0..N {
                if fixed_wait[seat] && seat != actor {
                    masks[seat] |= bit;
                }
            }
        }
        if riichi_flags[actor].get(index).copied().unwrap_or(false) {
            fixed_wait[actor] = true;
        }
        local_index[actor] += 1;
    }
    Ok(masks)
}

fn corrected_visible_counts_4p(observation: &Observation) -> [u8; TILE_MAX] {
    corrected_visible_counts(
        &observation.hands[observation.player_id as usize],
        &observation.melds,
        &observation.discards,
        &observation.dora_indicators,
        0,
    )
}

fn corrected_visible_counts_3p(observation: &Observation3P) -> [u8; TILE_MAX] {
    let kita = observation
        .kita_counts
        .iter()
        .copied()
        .fold(0u8, u8::saturating_add);
    corrected_visible_counts(
        &observation.hands[observation.player_id as usize],
        &observation.melds,
        &observation.discards,
        &observation.dora_indicators,
        kita,
    )
}

fn corrected_visible_counts<const N: usize>(
    own_hand: &[u32],
    melds: &[Vec<Meld>; N],
    discards: &[Vec<u32>; N],
    dora_indicators: &[u32],
    kita_count: u8,
) -> [u8; TILE_MAX] {
    let mut counts = [0u8; TILE_MAX];
    for &tile in own_hand
        .iter()
        .chain(discards.iter().flatten())
        .chain(dora_indicators.iter())
    {
        bump_count(&mut counts, tile as usize / 4);
    }
    for meld in melds.iter().flatten() {
        let mut skipped_called = false;
        for &tile in &meld.tiles {
            if !skipped_called && meld.called_tile == Some(tile) {
                // The claimed physical tile remains in the source river.
                skipped_called = true;
                continue;
            }
            bump_count(&mut counts, tile as usize / 4);
        }
    }
    counts[30] = counts[30].saturating_add(kita_count).min(4);
    counts
}

fn bump_count(counts: &mut [u8; TILE_MAX], tile_type: usize) {
    if let Some(count) = counts.get_mut(tile_type) {
        *count = count.saturating_add(1).min(4);
    }
}

fn discard_candidates_4p(observation: &Observation) -> Vec<u8> {
    let mut candidates = Vec::new();
    for action in &observation._legal_actions {
        if matches!(action.action_type, ActionType::Discard | ActionType::Riichi)
            && let Some(tile) = action.tile
        {
            let tile_type = tile / 4;
            if !candidates.contains(&tile_type) {
                candidates.push(tile_type);
            }
        }
    }
    candidates.sort_unstable();
    candidates
}

fn discard_candidates_3p(observation: &Observation3P) -> Vec<u8> {
    let mut candidates = Vec::new();
    for action in &observation._legal_actions {
        if matches!(
            action.0.action_type,
            ActionType::Discard | ActionType::Riichi
        ) && let Some(tile) = action.0.tile
        {
            let tile_type = tile / 4;
            if !candidates.contains(&tile_type) {
                candidates.push(tile_type);
            }
        }
    }
    candidates.sort_unstable();
    candidates
}

fn calculate(input: &DrevV2Input) -> DrevV2Result {
    let mut opponents: [DrevV2OpponentResult; 3] = std::array::from_fn(|_| Default::default());
    for (slot, opponent) in input.opponents.iter().enumerate() {
        let Some(opponent) = opponent else {
            continue;
        };
        opponents[slot] = calculate_opponent(input, opponent);
    }

    let mut result = DrevV2Result {
        opponents,
        all_safe_zero: [0.0; TILE_MAX],
        max_ron_prob: [0.0; TILE_MAX],
        sum_expected_loss_points: [0.0; TILE_MAX],
        min_loss_discard_marker: [0.0; TILE_MAX],
        max_uncertainty: [0.0; TILE_MAX],
    };
    let active = input.num_players.saturating_sub(1) as usize;
    for tile in 0..TILE_MAX {
        result.all_safe_zero[tile] =
            (0..active).all(|slot| result.opponents[slot].hard_safe_zero[tile] == 1.0) as u8 as f32;
        for slot in 0..active {
            let opponent = &result.opponents[slot];
            result.max_ron_prob[tile] = result.max_ron_prob[tile].max(opponent.ron_prob[tile]);
            result.sum_expected_loss_points[tile] +=
                opponent.ron_prob[tile] * opponent.mean_loss_points[tile];
            result.max_uncertainty[tile] =
                result.max_uncertainty[tile].max(opponent.uncertainty[tile]);
        }
    }
    if let Some(&best) = input.discard_candidates.iter().min_by(|&&left, &&right| {
        result.sum_expected_loss_points[left as usize]
            .total_cmp(&result.sum_expected_loss_points[right as usize])
            .then_with(|| left.cmp(&right))
    }) {
        result.min_loss_discard_marker[best as usize] = 1.0;
    }
    result
}

fn calculate_opponent(input: &DrevV2Input, opponent: &OpponentInput) -> DrevV2OpponentResult {
    let yaku = yaku_evidence(input, opponent);
    let mut exposed_set_safe_mask = opponent
        .melds
        .iter()
        .filter_map(|meld| {
            let tile = meld_triplet_type(meld)?;
            let is_kan = meld.tiles.len() == 4;
            let cannot_form_sequence =
                tile >= 27 || (input.num_players == 3 && matches!(tile, 0 | 8));
            (is_kan || cannot_form_sequence || opponent.melds.len() == 4).then_some(tile)
        })
        .fold(0u64, |mask, tile| mask | (1u64 << tile));
    if opponent.melds.len() == 4 {
        for tile in 0..TILE_MAX {
            if visible_excluding_possible_winning_discard(input, tile) >= 3 {
                exposed_set_safe_mask |= 1u64 << tile;
            }
        }
    }
    let hard_mask = opponent.river_mask
        | opponent.post_riichi_pass_mask
        | opponent.temporary_safe_mask
        | exposed_set_safe_mask;
    let tenpai = tenpai_prior(opponent);
    let uncertainty = uncertainty_prior(opponent);
    let mut weights = [0.0f32; TILE_MAX];

    for (tile, weight_slot) in weights.iter_mut().enumerate() {
        if input.num_players == 3 && (1..=7).contains(&tile) {
            continue;
        }
        let mut weight = if tile >= 27 { 1.05 } else { 1.0 };
        weight *= suji_multiplier(opponent.river_mask, tile, input.num_players == 3);
        if kabe_no_ryanmen(&input.visible_counts, tile, input.num_players == 3) {
            // Kabe excludes ryanmen shapes only. Tanki, shanpon and kanchan
            // remain possible, so this must never become a hard zero.
            weight *= 0.35;
        }
        weight *= tedashi_shape_multiplier(opponent, tile);
        *weight_slot = weight;
    }

    let weight_sum: f32 = weights.iter().sum();
    let expected_wait_count = if opponent.riichi { 1.65 } else { 1.35 };
    let yaku_factor = ron_yaku_factor(opponent, &yaku);
    let confirmed_base_han = confirmed_normal_han(input, opponent, &yaku);
    let public_bonus = public_bonus_count(input, opponent);
    let broadcast_yakuman_units = confirmed_yakuman_units(&yaku);
    let candidate_yakuman = candidate_yakuman_context(opponent);
    let is_dealer = opponent.absolute_seat == input.oya;
    // One unit per publicly confirmed family is a lower bound on hand value;
    // rule-dependent double-yakuman variants are deliberately omitted. The
    // payment uses ordinary ron settlement because this observation has no
    // rule/pao policy; it is not a guaranteed lower bound on the discarder's
    // own liability in a pao hand.
    let mut out = DrevV2OpponentResult {
        yaku,
        ..Default::default()
    };
    for (tile, &weight) in weights.iter().enumerate() {
        if weight == 0.0 || weight_sum == 0.0 {
            continue;
        }
        let wait = (tenpai * expected_wait_count * weight / weight_sum).clamp(0.0, 0.45);
        out.wait_prob[tile] = wait;
        out.uncertainty[tile] = uncertainty;

        let candidate_yakuman_units =
            candidate_confirmed_yakuman_units(broadcast_yakuman_units, candidate_yakuman, tile);
        let candidate_normal_han = candidate_confirmed_normal_han(candidate_yakuman, tile);
        let hard = hard_mask & (1u64 << tile) != 0;
        if hard {
            out.hard_safe_zero[tile] = 1.0;
        } else {
            let candidate_yaku_factor = if candidate_yakuman_units > 0 || candidate_normal_han > 0 {
                1.0
            } else {
                yaku_factor
            };
            out.ron_prob[tile] = (wait * candidate_yaku_factor).min(wait);
        }

        let candidate_han = confirmed_base_han
            .saturating_add(candidate_normal_han)
            .max(1) as f32
            + public_bonus as f32
            + dora_multiplicity(tile, &input.dora_indicators, input.num_players == 3) as f32;
        // Exact public/candidate dora can reach kazoe yakuman. Match the hand
        // evaluator's single-kazoe cap instead of clipping the loss path at
        // the encoded six-bonus visualization scale.
        let normal_han = candidate_han.round().clamp(1.0, 13.0) as u8;
        let normal_points = calculate_score(
            normal_han,
            30,
            is_dealer,
            false,
            input.honba as u32,
            input.num_players,
        )
        .pay_ron as f32;
        out.mean_loss_points[tile] = if candidate_yakuman_units > 0 {
            calculate_score(
                candidate_yakuman_units.saturating_mul(13),
                0,
                is_dealer,
                false,
                input.honba as u32,
                input.num_players,
            )
            .pay_ron as f32
        } else {
            normal_points
        };
    }
    out
}

fn tenpai_prior(opponent: &OpponentInput) -> f32 {
    if opponent.riichi {
        return 1.0;
    }
    let open_melds = opponent.melds.iter().filter(|meld| meld.opened).count() as f32;
    let discards = opponent.discards.len() as f32;
    (0.05 + 0.045 * discards.min(18.0) + 0.14 * open_melds.min(4.0)).clamp(0.05, 0.82)
}

fn uncertainty_prior(opponent: &OpponentInput) -> f32 {
    if opponent.riichi {
        return 0.25;
    }
    let tail_tsumogiri = opponent
        .tsumogiri
        .iter()
        .rev()
        .take_while(|&&flag| flag)
        .count() as f32;
    let public_structure = opponent.melds.len() as f32 * 0.10;
    (0.90 - public_structure - tail_tsumogiri.min(6.0) * 0.035).clamp(0.35, 0.95)
}

fn suji_multiplier(river_mask: u64, tile: usize, is_sanma: bool) -> f32 {
    if tile >= 27 || (is_sanma && tile < 9) {
        return 1.0;
    }
    let base = tile / 9 * 9;
    let rank = tile % 9;
    let low = rank.checked_sub(3).map(|value| base + value);
    let high = (rank + 3 <= 8).then_some(base + rank + 3);
    let mut hits = 0;
    let mut blocker_count = 0;
    for blocker in [low, high].into_iter().flatten() {
        blocker_count += 1;
        hits += usize::from(river_mask & (1u64 << blocker) != 0);
    }
    match (hits, blocker_count) {
        (0, _) => 1.0,
        (1, 2) => 0.78,
        _ => 0.58,
    }
}

fn kabe_no_ryanmen(visible: &[u8; TILE_MAX], tile: usize, is_sanma: bool) -> bool {
    if tile >= 27 || (is_sanma && tile < 9) {
        return false;
    }
    let base = tile / 9 * 9;
    let rank = tile % 9;
    let high_blocked =
        rank + 2 > 8 || visible[base + rank + 1] >= 4 || visible[base + rank + 2] >= 4;
    let low_blocked = rank < 2 || visible[base + rank - 2] >= 4 || visible[base + rank - 1] >= 4;
    high_blocked && low_blocked
}

fn tedashi_shape_multiplier(opponent: &OpponentInput, candidate: usize) -> f32 {
    if candidate >= 27 || opponent.discards.is_empty() {
        return 1.0;
    }
    let mut lift = 0.0f32;
    let len = opponent.discards.len() as f32;
    for (index, (&physical, &tsumogiri)) in opponent
        .discards
        .iter()
        .zip(opponent.tsumogiri.iter())
        .enumerate()
    {
        if tsumogiri {
            continue;
        }
        let tile = physical as usize / 4;
        if tile >= 27 || tile / 9 != candidate / 9 {
            continue;
        }
        let distance = tile.abs_diff(candidate);
        if matches!(distance, 1 | 2) {
            let recency = (index as f32 + 1.0) / len;
            lift = lift.max((0.08 + 0.12 * recency) / distance as f32);
        }
    }
    1.0 + lift
}

fn ron_yaku_factor(opponent: &OpponentInput, yaku: &DrevYakuEvidence) -> f32 {
    let confirmed_normal = yaku.values[..9]
        .iter()
        .copied()
        .any(|value| value >= EVIDENCE_CONFIRMED);
    if confirmed_normal || confirmed_yakuman_units(yaku) > 0 {
        1.0
    } else if opponent.melds.iter().any(|meld| meld.opened) {
        0.72
    } else {
        0.82
    }
}

fn confirmed_normal_han(
    input: &DrevV2Input,
    opponent: &OpponentInput,
    yaku: &DrevYakuEvidence,
) -> usize {
    let mut confirmed_yaku_han = usize::from(opponent.riichi);
    if yaku.values[2] == EVIDENCE_CONFIRMED {
        let seat_wind =
            27 + (opponent.absolute_seat + input.num_players - input.oya) % input.num_players;
        confirmed_yaku_han += value_triplet_han(opponent, seat_wind, input.round_wind);
    }
    if yaku.values[5] == EVIDENCE_CONFIRMED {
        confirmed_yaku_han += 2;
    }
    if yaku.values[8] == EVIDENCE_CONFIRMED {
        // A fully public Ittsu or Sanshoku doujun contains an open chi and
        // therefore has a conservative one-han lower bound.
        confirmed_yaku_han += 1;
    }
    let concealed_kans = opponent
        .melds
        .iter()
        .filter(|meld| meld.meld_type == MeldType::Ankan)
        .count();
    if concealed_kans >= 3 {
        confirmed_yaku_han += 2;
    }
    let kans = opponent
        .melds
        .iter()
        .filter(|meld| {
            matches!(
                meld.meld_type,
                MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan
            )
        })
        .count();
    if kans >= 3 {
        confirmed_yaku_han += 2;
    }
    if public_sanshoku_doukou_confirmed(&opponent.melds) {
        confirmed_yaku_han += 2;
    }
    confirmed_yaku_han
}

#[cfg(test)]
fn estimated_han(input: &DrevV2Input, opponent: &OpponentInput, yaku: &DrevYakuEvidence) -> f32 {
    let confirmed_yaku_han = confirmed_normal_han(input, opponent, yaku);

    // Evidence ordinals are categorical, not calibrated probabilities.
    // Conditional on a legal ron there must be at least one ordinary yaku;
    // only publicly confirmed yaku and exact visible bonus value can raise
    // that floor.
    confirmed_yaku_han.max(1) as f32 + public_bonus_count(input, opponent) as f32
}

fn confirmed_yakuman_units(yaku: &DrevYakuEvidence) -> u8 {
    yaku.values[10..]
        .iter()
        .filter(|&&value| value == EVIDENCE_CONFIRMED)
        .count()
        .min(u8::MAX as usize) as u8
}

#[derive(Clone, Copy, Default)]
struct CandidateYakumanContext {
    missing_wind: Option<usize>,
    missing_dragon: Option<usize>,
    flush_suit: Option<usize>,
    has_honor: bool,
    all_groups_terminal_or_honor: bool,
    four_meld_all_simples: bool,
    has_chi: bool,
    four_meld_all_honors: bool,
    four_meld_all_green: bool,
    four_meld_all_terminals: bool,
}

fn candidate_yakuman_context(opponent: &OpponentInput) -> CandidateYakumanContext {
    let mut public_winds = [false; 4];
    for wind in opponent
        .melds
        .iter()
        .filter_map(meld_triplet_type)
        .filter(|tile| (27..=30).contains(tile))
    {
        public_winds[wind - 27] = true;
    }
    let missing_wind = (public_winds.into_iter().filter(|present| *present).count() == 3)
        .then(|| (0..4).find(|&index| !public_winds[index]))
        .flatten()
        .map(|index| index + 27);

    let four_public_melds = opponent.melds.len() == 4;
    let mut suits = [false; 3];
    let mut has_honor = false;
    for tile in opponent
        .melds
        .iter()
        .flat_map(|meld| meld.tiles.iter().map(|tile| *tile as usize / 4))
    {
        if tile < 27 {
            suits[tile / 9] = true;
        } else {
            has_honor = true;
        }
    }
    let flush_suit = (four_public_melds
        && suits.into_iter().filter(|present| *present).count() == 1)
        .then(|| (0..3).find(|&suit| suits[suit]))
        .flatten();
    let mut public_dragons = [false; 3];
    for dragon in opponent
        .melds
        .iter()
        .filter_map(meld_triplet_type)
        .filter(|tile| (31..=33).contains(tile))
    {
        public_dragons[dragon - 31] = true;
    }
    let missing_dragon = (public_dragons
        .into_iter()
        .filter(|present| *present)
        .count()
        == 2)
        .then(|| (0..3).find(|&index| !public_dragons[index]))
        .flatten()
        .map(|index| index + 31);
    let all_tiles_match = |allows: fn(usize) -> bool| {
        four_public_melds
            && opponent
                .melds
                .iter()
                .all(|meld| meld.tiles.iter().all(|tile| allows(*tile as usize / 4)))
    };
    CandidateYakumanContext {
        missing_wind,
        missing_dragon,
        flush_suit,
        has_honor,
        all_groups_terminal_or_honor: four_public_melds
            && opponent.melds.iter().all(meld_contains_terminal_or_honor),
        four_meld_all_simples: all_tiles_match(is_simple),
        has_chi: opponent
            .melds
            .iter()
            .any(|meld| meld.meld_type == MeldType::Chi),
        four_meld_all_honors: all_tiles_match(|tile| tile >= 27),
        four_meld_all_green: all_tiles_match(is_green),
        four_meld_all_terminals: all_tiles_match(is_terminal),
    }
}

fn candidate_confirmed_normal_han(context: CandidateYakumanContext, candidate: usize) -> usize {
    let mut han = 0;
    if let Some(suit) = context.flush_suit {
        if candidate >= 27 {
            han += 2;
        } else if candidate / 9 == suit {
            han += if context.has_honor { 2 } else { 5 };
        }
    } else if context.four_meld_all_honors && candidate < 27 {
        // Four honor melds plus a numbered pair contain exactly one numbered
        // suit and honors, which confirms open Honitsu.
        han += 2;
    }
    if context.four_meld_all_simples && is_simple(candidate) {
        han += 1;
    }
    if context.all_groups_terminal_or_honor && is_terminal_or_honor(candidate) {
        han += if context.has_chi {
            if context.has_honor || candidate >= 27 {
                1
            } else {
                2
            }
        } else {
            2
        };
    }
    if context.missing_dragon == Some(candidate) {
        han += 2;
    }
    han
}

fn candidate_confirmed_yakuman_units(
    broadcast_units: u8,
    context: CandidateYakumanContext,
    candidate: usize,
) -> u8 {
    // With three public wind triplets, a ron on the only missing wind must
    // complete either its pair (Shousuushii) or its triplet (Daisuushii).
    // The family is therefore candidate-confirmed even though its broadcast
    // evidence remains STRONG for all other tile columns.
    let mut units = broadcast_units;
    if context.missing_wind == Some(candidate) {
        units = units.saturating_add(1);
    }
    // Four public melds leave only the pair. Conditional on a legal ron, a
    // candidate in the same composition set therefore confirms the grouped
    // yakuman even though the tile-broadcast evidence remains ordinal.
    if context.four_meld_all_honors && candidate >= 27 {
        units = units.saturating_add(1);
    }
    if context.four_meld_all_green && is_green(candidate) {
        units = units.saturating_add(1);
    }
    if context.four_meld_all_terminals && is_terminal(candidate) {
        units = units.saturating_add(1);
    }
    units
}

fn yaku_evidence(input: &DrevV2Input, opponent: &OpponentInput) -> DrevYakuEvidence {
    let mut values = [EVIDENCE_POSSIBLE; DREV_V2_YAKU_CHANNELS_PER_OPPONENT];
    values[0] = if opponent.riichi {
        EVIDENCE_CONFIRMED
    } else {
        EVIDENCE_IMPOSSIBLE
    };

    let mut meld_type_storage = [0usize; 16];
    let mut meld_type_len = 0;
    for tile in opponent
        .melds
        .iter()
        .flat_map(|meld| meld.tiles.iter().map(|tile| *tile as usize / 4))
    {
        debug_assert!(meld_type_len < meld_type_storage.len());
        if meld_type_len == meld_type_storage.len() {
            break;
        }
        meld_type_storage[meld_type_len] = tile;
        meld_type_len += 1;
    }
    let meld_types = &meld_type_storage[..meld_type_len];
    let all_meld_tiles_simple = meld_types.iter().all(|&tile| is_simple(tile));
    values[1] = if meld_types.iter().any(|&tile| !is_simple(tile)) {
        EVIDENCE_IMPOSSIBLE
    } else if !meld_types.is_empty() && all_meld_tiles_simple {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };

    let seat_wind =
        27 + (opponent.absolute_seat + input.num_players - input.oya) % input.num_players;
    let yakuhai_han = value_triplet_han(opponent, seat_wind, input.round_wind);
    values[2] = if yakuhai_han > 0 {
        EVIDENCE_CONFIRMED
    } else {
        EVIDENCE_POSSIBLE
    };

    let (suit_count, has_honor) = meld_suit_summary(meld_types);
    values[3] = match suit_count {
        0 => EVIDENCE_POSSIBLE,
        1 => EVIDENCE_STRONG,
        _ => EVIDENCE_IMPOSSIBLE,
    };
    let sanma_manzu_chinitsu_impossible = input.num_players == 3
        && suit_count == 1
        && meld_types
            .iter()
            .copied()
            .filter(|&tile| tile < 27)
            .all(|tile| tile < 9);
    values[4] = if suit_count > 1 || has_honor || sanma_manzu_chinitsu_impossible {
        EVIDENCE_IMPOSSIBLE
    } else if suit_count == 1 && !meld_types.is_empty() {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };

    let chi_count = opponent
        .melds
        .iter()
        .filter(|meld| meld.meld_type == MeldType::Chi)
        .count();
    let triplet_count = opponent
        .melds
        .iter()
        .filter(|meld| meld.meld_type != MeldType::Chi)
        .count();
    values[5] = if chi_count > 0 {
        EVIDENCE_IMPOSSIBLE
    } else if triplet_count == 4 {
        EVIDENCE_CONFIRMED
    } else if triplet_count > 0 {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };
    values[6] = if opponent.melds.is_empty() {
        EVIDENCE_POSSIBLE
    } else {
        EVIDENCE_IMPOSSIBLE
    };

    values[7] = if opponent.melds.is_empty() {
        EVIDENCE_POSSIBLE
    } else if opponent.melds.iter().all(meld_contains_terminal_or_honor) {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_IMPOSSIBLE
    };
    values[8] = if public_sequence_family_confirmed(&opponent.melds) {
        // Three public chi already prove an open Ittsu or Sanshoku; the
        // concealed fourth group cannot undo that yaku.
        EVIDENCE_CONFIRMED
    } else if opponent.melds.len() == 4 {
        EVIDENCE_IMPOSSIBLE
    } else {
        EVIDENCE_POSSIBLE
    };
    values[9] = public_bonus_value(input, opponent);

    values[10] = if opponent.melds.is_empty() {
        EVIDENCE_POSSIBLE
    } else {
        EVIDENCE_IMPOSSIBLE
    };

    let dragon_triplets = honor_triplet_count(opponent, 31..=33);
    values[11] = if dragon_triplets == 3 {
        EVIDENCE_CONFIRMED
    } else if dragon_yakuman_impossible(input, opponent) {
        EVIDENCE_IMPOSSIBLE
    } else if dragon_triplets == 2 {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };

    let wind_triplets = honor_triplet_count(opponent, 27..=30);
    values[12] = if wind_triplets == 4 {
        EVIDENCE_CONFIRMED
    } else if wind_yakuman_impossible(input, opponent) {
        EVIDENCE_IMPOSSIBLE
    } else if wind_triplets == 3 {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };

    let concealed_kans = opponent
        .melds
        .iter()
        .filter(|meld| meld.meld_type == MeldType::Ankan)
        .count();
    values[13] = if opponent.melds.iter().any(|meld| meld.opened) {
        EVIDENCE_IMPOSSIBLE
    } else if concealed_kans == 4 {
        // Four concealed kans leave only a pair wait, so Suuankou tanki is
        // publicly certain if the opponent can ron.
        EVIDENCE_CONFIRMED
    } else if concealed_kans == 3 {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };

    values[14] = public_composition_evidence(opponent, meld_types, |tile| tile >= 27);
    values[15] = public_composition_evidence(opponent, meld_types, |tile| {
        matches!(tile, 19 | 20 | 21 | 23 | 25 | 32)
    });
    values[16] = public_composition_evidence(opponent, meld_types, |tile| {
        tile < 27 && matches!(tile % 9, 0 | 8)
    });
    // Chuuren has no public positive proof before the concealed hand is
    // revealed. Any meld, including an ankan, makes the shape impossible.
    values[17] = if opponent.melds.is_empty() {
        EVIDENCE_POSSIBLE
    } else {
        EVIDENCE_IMPOSSIBLE
    };
    let kans = opponent
        .melds
        .iter()
        .filter(|meld| {
            matches!(
                meld.meld_type,
                MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan
            )
        })
        .count();
    values[18] = if opponent
        .melds
        .iter()
        .any(|meld| meld.meld_type == MeldType::Chi)
    {
        EVIDENCE_IMPOSSIBLE
    } else if kans == 4 {
        EVIDENCE_CONFIRMED
    } else if kans == 3 {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_POSSIBLE
    };

    DrevYakuEvidence { values }
}

fn value_triplet_han(opponent: &OpponentInput, seat_wind: u8, round_wind: u8) -> usize {
    opponent
        .melds
        .iter()
        .filter_map(meld_triplet_type)
        .map(|tile| {
            usize::from(matches!(tile, 31..=33))
                + usize::from(tile == seat_wind as usize)
                + usize::from(tile == 27 + round_wind as usize)
        })
        .sum()
}

fn honor_triplet_count(opponent: &OpponentInput, range: std::ops::RangeInclusive<usize>) -> usize {
    opponent
        .melds
        .iter()
        .filter_map(meld_triplet_type)
        .filter(|tile| range.contains(tile))
        .count()
}

fn meld_triplet_type(meld: &Meld) -> Option<usize> {
    if meld.meld_type == MeldType::Chi || meld.tiles.len() < 3 {
        return None;
    }
    let tile = meld.tiles[0] as usize / 4;
    meld.tiles
        .iter()
        .all(|candidate| *candidate as usize / 4 == tile)
        .then_some(tile)
}

fn chi_start_type(meld: &Meld) -> Option<usize> {
    if meld.meld_type != MeldType::Chi || meld.tiles.len() != 3 {
        return None;
    }
    let mut tiles = [
        meld.tiles[0] as usize / 4,
        meld.tiles[1] as usize / 4,
        meld.tiles[2] as usize / 4,
    ];
    tiles.sort_unstable();
    let start = tiles[0];
    (start < 27 && tiles[1] == start + 1 && tiles[2] == start + 2 && tiles[2] / 9 == start / 9)
        .then_some(start)
}

fn public_sequence_family_confirmed(melds: &[Meld]) -> bool {
    let mut starts = [false; 27];
    let mut chi_count = 0;
    for start in melds.iter().filter_map(chi_start_type) {
        starts[start] = true;
        chi_count += 1;
    }
    if chi_count < 3 {
        return false;
    }
    let ittsu = (0..3).any(|suit| {
        let base = suit * 9;
        [base, base + 3, base + 6]
            .iter()
            .all(|&start| starts[start])
    });
    let sanshoku = (0..=6).any(|rank| {
        [rank, rank + 9, rank + 18]
            .iter()
            .all(|&start| starts[start])
    });
    ittsu || sanshoku
}

fn public_sanshoku_doukou_confirmed(melds: &[Meld]) -> bool {
    let mut triplets = [[false; 9]; 3];
    for tile in melds.iter().filter_map(meld_triplet_type) {
        if tile < 27 {
            triplets[tile / 9][tile % 9] = true;
        }
    }
    (0..9).any(|rank| (0..3).all(|suit| triplets[suit][rank]))
}

fn meld_suit_summary(tiles: &[usize]) -> (usize, bool) {
    let mut suits = [false; 3];
    let mut honor = false;
    for &tile in tiles {
        if tile < 27 {
            suits[tile / 9] = true;
        } else {
            honor = true;
        }
    }
    (suits.into_iter().filter(|present| *present).count(), honor)
}

fn is_simple(tile: usize) -> bool {
    tile < 27 && !matches!(tile % 9, 0 | 8)
}

fn is_green(tile: usize) -> bool {
    matches!(tile, 19 | 20 | 21 | 23 | 25 | 32)
}

fn is_terminal(tile: usize) -> bool {
    tile < 27 && matches!(tile % 9, 0 | 8)
}

fn is_terminal_or_honor(tile: usize) -> bool {
    tile >= 27 || is_terminal(tile)
}

fn meld_contains_terminal_or_honor(meld: &Meld) -> bool {
    meld.tiles
        .iter()
        .map(|tile| *tile as usize / 4)
        .any(is_terminal_or_honor)
}

fn dragon_yakuman_impossible(input: &DrevV2Input, opponent: &OpponentInput) -> bool {
    (31..=33).any(|dragon| {
        visible_excluding_possible_winning_discard(input, dragon) >= 2
            && !opponent
                .melds
                .iter()
                .filter_map(meld_triplet_type)
                .any(|tile| tile == dragon)
    })
}

fn wind_yakuman_impossible(input: &DrevV2Input, opponent: &OpponentInput) -> bool {
    let mut public_triplet = [false; 4];
    for wind in opponent
        .melds
        .iter()
        .filter_map(meld_triplet_type)
        .filter(|tile| (27..=30).contains(tile))
    {
        public_triplet[wind - 27] = true;
    }

    // Shousuushii is the least restrictive member of the grouped wind-yakuman
    // family: one missing wind needs a pair and every other missing wind needs
    // a triplet. If no pair assignment is feasible, neither small nor big
    // four winds can remain possible.
    !(27..=30).any(|pair| {
        !public_triplet[pair - 27]
            && (27..=30).all(|wind| {
                public_triplet[wind - 27]
                    || 4usize.saturating_sub(
                        visible_excluding_possible_winning_discard(input, wind) as usize,
                    ) >= if wind == pair { 2 } else { 3 }
            })
    })
}

fn visible_excluding_possible_winning_discard(input: &DrevV2Input, tile: usize) -> u8 {
    input.visible_counts[tile].saturating_sub(u8::from(
        input
            .discard_candidates
            .iter()
            .any(|&candidate| candidate as usize == tile),
    ))
}

fn public_composition_evidence(
    opponent: &OpponentInput,
    meld_tiles: &[usize],
    allows: impl Fn(usize) -> bool,
) -> f32 {
    if opponent.melds.is_empty() {
        return EVIDENCE_POSSIBLE;
    }
    if meld_tiles.iter().copied().all(allows) {
        EVIDENCE_STRONG
    } else {
        EVIDENCE_IMPOSSIBLE
    }
}

fn public_bonus_count(input: &DrevV2Input, opponent: &OpponentInput) -> usize {
    let mut bonus = opponent.kita_count as usize;
    let is_sanma = input.num_players == 3;
    // A set-aside North is always one nukidora and can additionally be a
    // normal visible dora when a West indicator points to North.
    bonus += opponent.kita_count as usize
        * input
            .dora_indicators
            .iter()
            .filter(|&&indicator| dora_type(indicator as usize / 4, is_sanma) == 30)
            .count();
    for meld in &opponent.melds {
        for &physical in &meld.tiles {
            if matches!(physical, 16 | 52 | 88) {
                bonus += 1;
            }
            let tile = physical as usize / 4;
            bonus += input
                .dora_indicators
                .iter()
                .filter(|&&indicator| dora_type(indicator as usize / 4, is_sanma) == tile)
                .count();
        }
    }
    bonus
}

fn public_bonus_value(input: &DrevV2Input, opponent: &OpponentInput) -> f32 {
    (public_bonus_count(input, opponent) as f32 / 6.0).clamp(0.0, 1.0)
}

fn dora_multiplicity(candidate: usize, indicators: &[u8], is_sanma: bool) -> u8 {
    indicators
        .iter()
        .filter(|&&indicator| dora_type(indicator as usize / 4, is_sanma) == candidate)
        .count()
        .min(u8::MAX as usize) as u8
}

fn dora_type(indicator: usize, is_sanma: bool) -> usize {
    match indicator {
        0 if is_sanma => 8,
        8 if is_sanma => 0,
        1..=7 if is_sanma => indicator,
        0..=26 => {
            let base = indicator / 9 * 9;
            base + (indicator % 9 + 1) % 9
        }
        27..=30 => 27 + (indicator - 27 + 1) % 4,
        31..=33 => 31 + (indicator - 31 + 1) % 3,
        _ => indicator,
    }
}

fn encode_v2(result: &DrevV2Result, tile_types: usize, is_sanma: bool) -> Vec<f32> {
    let mut output = vec![0.0; DREV_V2_CHANNELS * tile_types];
    encode_v2_into(result, &mut output, tile_types, is_sanma)
        .expect("fresh DREV-v2 output has the canonical shape");
    output
}

fn encode_v2_into(
    result: &DrevV2Result,
    output: &mut [f32],
    tile_types: usize,
    is_sanma: bool,
) -> RiichiResult<()> {
    let expected = DREV_V2_CHANNELS * tile_types;
    if output.len() != expected {
        return Err(RiichiError::InvalidState {
            message: format!(
                "DREV v2 output has length {}; expected {expected}",
                output.len()
            ),
        });
    }
    output.fill(0.0);
    for tile in 0..TILE_MAX {
        let Some(column) = output_column(tile, is_sanma) else {
            continue;
        };
        for slot in 0..3 {
            let opponent = &result.opponents[slot];
            let base = slot * 6;
            set(
                output,
                tile_types,
                base,
                column,
                opponent.hard_safe_zero[tile],
            );
            set(
                output,
                tile_types,
                base + 1,
                column,
                opponent.wait_prob[tile],
            );
            set(
                output,
                tile_types,
                base + 2,
                column,
                opponent.ron_prob[tile],
            );
            set(
                output,
                tile_types,
                base + 3,
                column,
                (opponent.mean_loss_points[tile] / 100_000.0).clamp(0.0, 1.0),
            );
            set(
                output,
                tile_types,
                base + 4,
                column,
                (opponent.mean_loss_points[tile] / 30_000.0).clamp(0.0, 1.0),
            );
            set(
                output,
                tile_types,
                base + 5,
                column,
                opponent.uncertainty[tile],
            );
        }
        set(output, tile_types, 18, column, result.all_safe_zero[tile]);
        set(output, tile_types, 19, column, result.max_ron_prob[tile]);
        set(
            output,
            tile_types,
            20,
            column,
            (result.sum_expected_loss_points[tile] / 100_000.0).clamp(0.0, 1.0),
        );
        set(
            output,
            tile_types,
            21,
            column,
            (result.sum_expected_loss_points[tile] / 30_000.0).clamp(0.0, 1.0),
        );
        set(
            output,
            tile_types,
            22,
            column,
            result.min_loss_discard_marker[tile],
        );
        set(output, tile_types, 23, column, result.max_uncertainty[tile]);
        for slot in 0..3 {
            for evidence in 0..DREV_V2_YAKU_CHANNELS_PER_OPPONENT {
                let channel =
                    DREV_V2_BASE_CHANNELS + slot * DREV_V2_YAKU_CHANNELS_PER_OPPONENT + evidence;
                set(
                    output,
                    tile_types,
                    channel,
                    column,
                    result.opponents[slot].yaku.values[evidence],
                );
            }
        }
    }
    Ok(())
}

fn output_column(tile: usize, is_sanma: bool) -> Option<usize> {
    if !is_sanma {
        return Some(tile);
    }
    match tile {
        0 => Some(0),
        1..=7 => None,
        8..=33 => Some(tile - 7),
        _ => None,
    }
}

fn set(output: &mut [f32], tile_types: usize, channel: usize, tile: usize, value: f32) {
    output[channel * tile_types + tile] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;

    fn empty_observation() -> Observation {
        let mut observation = Observation::new(
            0,
            [
                vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
                vec![],
                vec![],
                vec![],
            ],
            Default::default(),
            Default::default(),
            vec![52],
            [25_000; 4],
            [false; 4],
            vec![Action::new(ActionType::Discard, Some(0), vec![], Some(0))],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 4],
            [None; 4],
            None,
            None,
        );
        observation.public_history_complete = true;
        observation
    }

    fn empty_observation_3p() -> Observation3P {
        let mut observation = Observation3P::new(
            0,
            [
                vec![0, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76],
                vec![],
                vec![],
            ],
            Default::default(),
            Default::default(),
            vec![80],
            [35_000; 3],
            [false; 3],
            vec![Action::new(ActionType::Discard, Some(0), vec![], Some(0))],
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None; 3],
            [None; 3],
            None,
            None,
        );
        observation.public_history_complete = true;
        observation
    }

    #[test]
    fn v2_requires_runtime_complete_history() {
        let mut observation = empty_observation();
        observation.public_history_complete = false;
        assert!(calculate_drev_v2(&observation).is_err());
    }

    #[test]
    fn v2_rejects_an_opponent_hidden_hand_instead_of_leaking_it() {
        let mut observation = empty_observation();
        observation.hands[1] = vec![0; 13];
        assert!(calculate_drev_v2(&observation).is_err());
    }

    #[test]
    fn resolved_post_riichi_pass_is_hard_safe_but_unresolved_is_not() {
        let mut observation = empty_observation();
        observation.riichi_declared[1] = true;
        observation.discards[1] = vec![56];
        observation.discard_is_riichi[1] = vec![true];
        observation.public_tsumogiri_history[1] = vec![false];
        observation.discards[2] = vec![36];
        observation.discard_is_riichi[2] = vec![false];
        observation.public_tsumogiri_history[2] = vec![false];
        observation.discard_actor_history = vec![1, 2];
        observation.resolved_discard_count = 1;

        let unresolved = calculate_drev_v2(&observation).unwrap();
        assert_eq!(unresolved.opponents()[0].hard_safe_zero[9], 0.0);

        observation.resolved_discard_count = 2;
        let resolved = calculate_drev_v2(&observation).unwrap();
        assert_eq!(resolved.opponents()[0].hard_safe_zero[9], 1.0);
        assert!(resolved.opponents()[0].wait_prob[9] > 0.0);
        assert_eq!(resolved.opponents()[0].ron_prob[9], 0.0);
    }

    #[test]
    fn hard_safe_masks_do_not_renormalize_the_wait_prior() {
        let baseline = calculate_drev_v2(&empty_observation()).unwrap();
        let mut observation = empty_observation();
        observation.public_temporary_safe_masks[1] = 1u64 << 31;
        let safe = calculate_drev_v2(&observation).unwrap();

        let baseline_opponent = &baseline.opponents()[0];
        let safe_opponent = &safe.opponents()[0];
        assert_eq!(safe_opponent.wait_prob, baseline_opponent.wait_prob);
        assert!(safe_opponent.wait_prob[31] > 0.0);
        assert_eq!(safe_opponent.hard_safe_zero[31], 1.0);
        assert_eq!(safe_opponent.ron_prob[31], 0.0);
        assert!(baseline_opponent.ron_prob[31] > 0.0);
    }

    #[test]
    fn tedashi_history_changes_only_soft_wait_shape_and_uncertainty() {
        let mut tedashi = empty_observation();
        tedashi.discards[1] = vec![12, 108]; // 4m tedashi, then East tsumogiri.
        tedashi.discard_is_riichi[1] = vec![false, false];
        tedashi.public_tsumogiri_history[1] = vec![false, true];
        tedashi.discard_actor_history = vec![1, 1];
        tedashi.resolved_discard_count = 2;

        let mut all_tsumogiri = tedashi.clone();
        all_tsumogiri.public_tsumogiri_history[1] = vec![true, true];

        let with_tedashi = calculate_drev_v2(&tedashi).unwrap();
        let without_tedashi = calculate_drev_v2(&all_tsumogiri).unwrap();
        let with_tedashi = &with_tedashi.opponents()[0];
        let without_tedashi = &without_tedashi.opponents()[0];

        // 5m is adjacent to the historical 4m tedashi, so its soft shape
        // prior reacts. The trailing run also changes uncertainty, while the
        // hard rules and value evidence below remain invariant.
        assert_eq!(with_tedashi.hard_safe_zero[4], 0.0);
        assert!(with_tedashi.wait_prob[4] > without_tedashi.wait_prob[4]);
        assert!(with_tedashi.uncertainty[4] > without_tedashi.uncertainty[4]);
        assert_eq!(with_tedashi.hard_safe_zero, without_tedashi.hard_safe_zero);
        assert_eq!(with_tedashi.yaku, without_tedashi.yaku);
        assert_eq!(
            with_tedashi.mean_loss_points,
            without_tedashi.mean_loss_points
        );
    }

    #[test]
    fn aggregate_heads_are_exact_reductions_and_ties_choose_lowest_candidate() {
        let mut input = input_4p(&empty_observation()).unwrap();
        input.opponents[0].as_mut().unwrap().riichi = true;
        input.opponents[1].as_mut().unwrap().melds.push(Meld::new(
            MeldType::Chi,
            vec![36, 40, 44],
            true,
            0,
            Some(36),
        ));
        input.opponents[2].as_mut().unwrap().tsumogiri = vec![true, true];
        for opponent in input.opponents.iter_mut().flatten() {
            opponent.temporary_safe_mask |= 1u64 << 31;
        }
        input.opponents[0].as_mut().unwrap().temporary_safe_mask |= 1u64 << 30;

        let result = calculate(&input);
        for tile in 0..TILE_MAX {
            let mut expected_max_ron = 0.0f32;
            let mut expected_loss = 0.0f32;
            let mut expected_max_uncertainty = 0.0f32;
            for opponent in result.opponents.iter().take(3) {
                expected_max_ron = expected_max_ron.max(opponent.ron_prob[tile]);
                expected_loss += opponent.ron_prob[tile] * opponent.mean_loss_points[tile];
                expected_max_uncertainty = expected_max_uncertainty.max(opponent.uncertainty[tile]);
            }
            let expected_all_safe = result
                .opponents
                .iter()
                .take(3)
                .all(|opponent| opponent.hard_safe_zero[tile] == 1.0)
                as u8 as f32;

            assert_eq!(result.all_safe_zero[tile], expected_all_safe);
            assert_eq!(result.max_ron_prob[tile], expected_max_ron);
            assert_eq!(result.sum_expected_loss_points[tile], expected_loss);
            assert_eq!(result.max_uncertainty[tile], expected_max_uncertainty);
        }
        assert_eq!(result.all_safe_zero[31], 1.0);
        assert_eq!(result.all_safe_zero[30], 0.0);

        let mut tied = input_4p(&empty_observation()).unwrap();
        tied.opponents = [None, None, None];
        tied.discard_candidates = vec![12, 4, 8];
        let tied = calculate(&tied);
        assert_eq!(tied.sum_expected_loss_points, [0.0; TILE_MAX]);
        assert_eq!(tied.min_loss_discard_marker[4], 1.0);
        assert_eq!(tied.min_loss_discard_marker.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn own_river_is_hard_safe_while_suji_and_kabe_remain_soft() {
        let mut observation = empty_observation();
        observation.discards[1] = vec![12]; // 4m: own river and suji blocker.
        observation.discard_is_riichi[1] = vec![false];
        observation.public_tsumogiri_history[1] = vec![false];
        observation.discard_actor_history = vec![1];
        observation.resolved_discard_count = 1;
        // A genuine 2m wall; 1m still has tanki/shanpon/kanchan risk.
        observation.hands[0] = vec![4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 36, 40];

        let result = calculate_drev_v2(&observation).unwrap();
        let opponent = &result.opponents()[0];
        assert_eq!(opponent.hard_safe_zero[3], 1.0);
        assert!(opponent.wait_prob[3] > 0.0);
        assert_eq!(opponent.ron_prob[3], 0.0);
        assert_eq!(opponent.hard_safe_zero[0], 0.0);
        assert!(opponent.ron_prob[0] > 0.0);
    }

    #[test]
    fn completed_public_sets_only_mark_rule_proven_tile_types_hard_safe() {
        for mut input in [
            input_4p(&empty_observation()).unwrap(),
            input_3p(&empty_observation_3p()).unwrap(),
        ] {
            let mut opponent = input.opponents[0].take().unwrap();
            opponent.melds.push(Meld::new(
                MeldType::Pon,
                vec![124, 125, 126],
                true,
                0,
                Some(124),
            ));
            let result = calculate_opponent(&input, &opponent);
            assert_eq!(result.hard_safe_zero[31], 1.0);
            assert!(result.wait_prob[31] > 0.0);
            assert_eq!(result.ron_prob[31], 0.0);

            opponent.melds[0] = Meld::new(MeldType::Pon, vec![52, 53, 54], true, 0, Some(52));
            assert_eq!(
                calculate_opponent(&input, &opponent).hard_safe_zero[13],
                0.0
            );

            opponent.melds[0] = Meld::new(MeldType::Ankan, vec![52, 53, 54, 55], false, -1, None);
            let result = calculate_opponent(&input, &opponent);
            assert_eq!(result.hard_safe_zero[13], 1.0);
            assert_eq!(result.ron_prob[13], 0.0);

            opponent.melds[0] = Meld::new(MeldType::Pon, vec![0, 1, 2], true, 0, Some(0));
            assert_eq!(
                calculate_opponent(&input, &opponent).hard_safe_zero[0],
                f32::from(input.num_players == 3)
            );

            opponent.melds = vec![
                Meld::new(MeldType::Chi, vec![36, 40, 44], true, 0, Some(36)),
                Meld::new(MeldType::Chi, vec![48, 52, 56], true, 0, Some(48)),
                Meld::new(MeldType::Chi, vec![72, 76, 80], true, 0, Some(72)),
                Meld::new(MeldType::Chi, vec![84, 88, 92], true, 0, Some(84)),
            ];
            input.discard_candidates.push(15);
            input.visible_counts[15] = 4;
            assert_eq!(
                calculate_opponent(&input, &opponent).hard_safe_zero[15],
                1.0
            );
            input.visible_counts[15] = 3;
            assert_eq!(
                calculate_opponent(&input, &opponent).hard_safe_zero[15],
                0.0
            );
        }
    }

    #[test]
    fn called_tile_is_not_double_counted_into_a_false_kabe() {
        let mut observation = empty_observation();
        observation.discards[1] = vec![56];
        observation.discard_is_riichi[1] = vec![false];
        observation.public_tsumogiri_history[1] = vec![false];
        observation.discard_actor_history = vec![1];
        observation.resolved_discard_count = 1;
        observation.melds[2].push(Meld::new(
            MeldType::Pon,
            vec![56, 57, 58],
            true,
            1,
            Some(56),
        ));
        let counts = corrected_visible_counts_4p(&observation);
        assert_eq!(counts[14], 3);
    }

    #[test]
    fn public_yaku_and_yakuman_evidence_covers_major_patterns() {
        let mut observation = empty_observation();
        observation.riichi_declared[1] = true;
        observation.discards[1] = vec![56];
        observation.discard_is_riichi[1] = vec![true];
        observation.public_tsumogiri_history[1] = vec![false];
        observation.discard_actor_history = vec![1];
        observation.resolved_discard_count = 1;
        observation.melds[2] = vec![
            Meld::new(MeldType::Pon, vec![124, 125, 126], true, 1, Some(124)),
            Meld::new(MeldType::Pon, vec![128, 129, 130], true, 1, Some(128)),
            Meld::new(MeldType::Pon, vec![132, 133, 134], true, 1, Some(132)),
        ];
        observation.melds[3] = vec![
            Meld::new(
                MeldType::Daiminkan,
                vec![108, 109, 110, 111],
                true,
                1,
                Some(108),
            ),
            Meld::new(
                MeldType::Daiminkan,
                vec![112, 113, 114, 115],
                true,
                1,
                Some(112),
            ),
            Meld::new(
                MeldType::Daiminkan,
                vec![116, 117, 118, 119],
                true,
                1,
                Some(116),
            ),
            Meld::new(
                MeldType::Daiminkan,
                vec![120, 121, 122, 123],
                true,
                1,
                Some(120),
            ),
        ];

        let result = calculate_drev_v2(&observation).unwrap();
        assert_eq!(result.opponents()[0].yaku.value("riichi"), Some(1.0));
        assert_eq!(result.opponents()[1].yaku.value("daisangen"), Some(1.0));
        assert_eq!(result.opponents()[2].yaku.value("wind_yakuman"), Some(1.0));
        assert_eq!(result.opponents()[2].yaku.value("suukantsu"), Some(1.0));
        assert!(result.opponents()[1].mean_loss_points[0] >= 32_000.0);
    }

    #[test]
    fn normal_yaku_evidence_distinguishes_open_hand_structure() {
        let mut observation = empty_observation();
        // 234p chi: all-simple, one-suit and sequence evidence. It excludes
        // toitoi, chiitoitsu, kokushi and suuankou.
        observation.melds[1].push(Meld::new(
            MeldType::Chi,
            vec![40, 44, 48],
            true,
            0,
            Some(40),
        ));
        // White pon is a publicly confirmed yakuhai and triplet tendency.
        observation.melds[2].push(Meld::new(
            MeldType::Pon,
            vec![124, 125, 126],
            true,
            0,
            Some(124),
        ));

        let result = calculate_drev_v2(&observation).unwrap();
        let chi = &result.opponents()[0].yaku;
        assert_eq!(chi.value("tanyao"), Some(EVIDENCE_STRONG));
        assert_eq!(chi.value("honitsu"), Some(EVIDENCE_STRONG));
        assert_eq!(chi.value("chinitsu"), Some(EVIDENCE_STRONG));
        assert_eq!(chi.value("sequence"), Some(EVIDENCE_POSSIBLE));
        assert_eq!(chi.value("toitoi"), Some(EVIDENCE_IMPOSSIBLE));
        assert_eq!(chi.value("chiitoitsu"), Some(EVIDENCE_IMPOSSIBLE));
        assert_eq!(chi.value("kokushi"), Some(EVIDENCE_IMPOSSIBLE));
        assert_eq!(chi.value("suuankou"), Some(EVIDENCE_IMPOSSIBLE));

        let pon = &result.opponents()[1].yaku;
        assert_eq!(pon.value("yakuhai"), Some(EVIDENCE_CONFIRMED));
        assert_eq!(pon.value("toitoi"), Some(EVIDENCE_STRONG));
    }

    #[test]
    fn sequence_evidence_requires_a_complete_public_ittsu_or_sanshoku() {
        let chi = |start: usize| {
            Meld::new(
                MeldType::Chi,
                vec![
                    (start * 4 + 1) as u8,
                    ((start + 1) * 4 + 1) as u8,
                    ((start + 2) * 4 + 1) as u8,
                ],
                true,
                0,
                Some((start * 4 + 1) as u8),
            )
        };
        let extra = Meld::new(MeldType::Pon, vec![124, 125, 126], true, 0, Some(124));

        let ittsu = vec![chi(9), chi(12), chi(15)];
        let sanshoku = vec![chi(0), chi(9), chi(18)];
        let incomplete = vec![chi(9), chi(12), extra.clone()];
        let broken = vec![chi(9), chi(10), chi(13), extra];
        assert!(public_sequence_family_confirmed(&ittsu));
        assert!(public_sequence_family_confirmed(&sanshoku));
        assert!(!public_sequence_family_confirmed(&broken));

        let mut input = input_4p(&empty_observation()).unwrap();
        input.dora_indicators.clear();
        let mut opponent = input.opponents[0].clone().unwrap();
        opponent.melds = ittsu;
        let evidence = yaku_evidence(&input, &opponent);
        assert_eq!(evidence.value("sequence"), Some(EVIDENCE_CONFIRMED));
        assert_eq!(estimated_han(&input, &opponent, &evidence), 1.0);
        assert_eq!(ron_yaku_factor(&opponent, &evidence), 1.0);

        let sanma = input_3p(&empty_observation_3p()).unwrap();
        let mut sanma_opponent = sanma.opponents[0].clone().unwrap();
        sanma_opponent.melds = vec![chi(9), chi(12), chi(15)];
        assert_eq!(
            yaku_evidence(&sanma, &sanma_opponent).value("sequence"),
            Some(EVIDENCE_CONFIRMED)
        );
        opponent.melds = incomplete;
        assert_eq!(
            yaku_evidence(&input, &opponent).value("sequence"),
            Some(EVIDENCE_POSSIBLE)
        );
        opponent.melds = broken;
        assert_eq!(
            yaku_evidence(&input, &opponent).value("sequence"),
            Some(EVIDENCE_IMPOSSIBLE)
        );
    }

    #[test]
    fn public_sanshoku_doukou_adds_confirmed_loss_floor() {
        let melds = vec![
            Meld::new(MeldType::Pon, vec![0, 1, 2], true, 0, Some(0)),
            Meld::new(MeldType::Pon, vec![36, 37, 38], true, 0, Some(36)),
            Meld::new(MeldType::Pon, vec![72, 73, 74], true, 0, Some(72)),
        ];
        assert!(public_sanshoku_doukou_confirmed(&melds));
        for mut input in [
            input_4p(&empty_observation()).unwrap(),
            input_3p(&empty_observation_3p()).unwrap(),
        ] {
            input.dora_indicators.clear();
            let mut opponent = input.opponents[0].clone().unwrap();
            opponent.melds = melds.clone();
            let evidence = yaku_evidence(&input, &opponent);
            assert_eq!(confirmed_normal_han(&input, &opponent, &evidence), 2);
            let result = calculate_opponent(&input, &opponent);
            let expected = calculate_score(
                2,
                30,
                opponent.absolute_seat == input.oya,
                false,
                input.honba as u32,
                input.num_players,
            )
            .pay_ron as f32;
            assert_eq!(result.mean_loss_points[0], expected);
        }
    }

    #[test]
    fn yakuman_families_have_separate_fail_closed_evidence() {
        let mut observation = empty_observation();
        // Public green sequence: Ryuu iisou remains possible with positive
        // evidence, while Tsuu iisou and Chinroutou are impossible.
        observation.melds[1].push(Meld::new(
            MeldType::Chi,
            vec![76, 80, 84], // 2s 3s 4s
            true,
            0,
            Some(76),
        ));
        // Three concealed triplets are positive evidence for Suuankou.
        observation.melds[2] = vec![
            Meld::new(MeldType::Ankan, vec![56, 57, 58, 59], false, -1, None),
            Meld::new(MeldType::Ankan, vec![60, 61, 62, 63], false, -1, None),
            Meld::new(MeldType::Ankan, vec![64, 65, 66, 67], false, -1, None),
        ];
        // Public terminal triplet supports Chinroutou only among the three
        // composition families.
        observation.melds[3].push(Meld::new(
            MeldType::Pon,
            vec![32, 33, 34],
            true,
            0,
            Some(32),
        ));

        let result = calculate_drev_v2(&observation).unwrap();
        let green = &result.opponents()[0].yaku;
        assert_eq!(green.value("ryuuiisou"), Some(EVIDENCE_STRONG));
        assert_eq!(green.value("tsuuiisou"), Some(EVIDENCE_IMPOSSIBLE));
        assert_eq!(green.value("chinroutou"), Some(EVIDENCE_IMPOSSIBLE));
        assert_eq!(green.value("chuuren"), Some(EVIDENCE_IMPOSSIBLE));

        let concealed = &result.opponents()[1].yaku;
        assert_eq!(concealed.value("suuankou"), Some(EVIDENCE_STRONG));
        assert_eq!(concealed.value("suukantsu"), Some(EVIDENCE_STRONG));
        assert_eq!(concealed.value("chuuren"), Some(EVIDENCE_IMPOSSIBLE));

        let terminals = &result.opponents()[2].yaku;
        assert_eq!(terminals.value("chinroutou"), Some(EVIDENCE_STRONG));
        assert_eq!(terminals.value("tsuuiisou"), Some(EVIDENCE_IMPOSSIBLE));
        assert_eq!(terminals.value("ryuuiisou"), Some(EVIDENCE_IMPOSSIBLE));

        let closed = calculate_drev_v2(&empty_observation()).unwrap();
        assert_eq!(
            closed.opponents()[0].yaku.value("kokushi"),
            Some(EVIDENCE_POSSIBLE)
        );
        assert_eq!(
            closed.opponents()[0].yaku.value("chuuren"),
            Some(EVIDENCE_POSSIBLE)
        );
    }

    #[test]
    fn yakuman_evidence_distinguishes_progress_from_public_blockers() {
        let mut input = input_4p(&empty_observation()).unwrap();
        let mut opponent = input.opponents[0].clone().unwrap();

        opponent.melds = vec![
            Meld::new(MeldType::Ankan, vec![124, 125, 126, 127], false, -1, None),
            Meld::new(MeldType::Ankan, vec![128, 129, 130, 131], false, -1, None),
        ];
        assert_eq!(
            yaku_evidence(&input, &opponent).value("suukantsu"),
            Some(EVIDENCE_POSSIBLE)
        );
        // With two White/Green kans exposed, two publicly unavailable Red
        // dragons are enough to prove Daisangen impossible.
        input.visible_counts[33] = 2;
        assert_eq!(
            yaku_evidence(&input, &opponent).value("daisangen"),
            Some(EVIDENCE_IMPOSSIBLE)
        );
        // A tile held by the observer is unavailable to the opponent now, but
        // becomes the winning tile when it is discarded.  Blocker evidence
        // must therefore put one candidate copy back before declaring the
        // yakuman impossible.
        input.discard_candidates.push(33);
        assert_eq!(
            yaku_evidence(&input, &opponent).value("daisangen"),
            Some(EVIDENCE_STRONG)
        );
        input.discard_candidates.retain(|&tile| tile != 33);

        opponent.melds.push(Meld::new(
            MeldType::Ankan,
            vec![132, 133, 134, 135],
            false,
            -1,
            None,
        ));
        assert_eq!(
            yaku_evidence(&input, &opponent).value("suukantsu"),
            Some(EVIDENCE_STRONG)
        );

        opponent.melds = vec![
            Meld::new(MeldType::Pon, vec![108, 109, 110], true, 1, Some(108)),
            Meld::new(MeldType::Pon, vec![112, 113, 114], true, 1, Some(112)),
            Meld::new(MeldType::Pon, vec![116, 117, 118], true, 1, Some(116)),
        ];
        input.visible_counts[30] = 3;
        assert_eq!(
            yaku_evidence(&input, &opponent).value("wind_yakuman"),
            Some(EVIDENCE_IMPOSSIBLE)
        );
        input.discard_candidates.push(30);
        assert_eq!(
            yaku_evidence(&input, &opponent).value("wind_yakuman"),
            Some(EVIDENCE_STRONG)
        );
        input.discard_candidates.retain(|&tile| tile != 30);

        opponent.melds = vec![Meld::new(
            MeldType::Chi,
            vec![36, 40, 44],
            true,
            1,
            Some(36),
        )];
        assert_eq!(
            yaku_evidence(&input, &opponent).value("suukantsu"),
            Some(EVIDENCE_IMPOSSIBLE)
        );

        // The same candidate-aware blocker semantics apply in sanma.
        let mut sanma = input_3p(&empty_observation_3p()).unwrap();
        let mut sanma_opponent = sanma.opponents[0].clone().unwrap();
        sanma_opponent.melds = vec![
            Meld::new(MeldType::Ankan, vec![124, 125, 126, 127], false, -1, None),
            Meld::new(MeldType::Ankan, vec![128, 129, 130, 131], false, -1, None),
        ];
        sanma.visible_counts[33] = 2;
        sanma.discard_candidates.push(33);
        assert_eq!(
            yaku_evidence(&sanma, &sanma_opponent).value("daisangen"),
            Some(EVIDENCE_STRONG)
        );

        sanma_opponent.melds = vec![
            Meld::new(MeldType::Pon, vec![108, 109, 110], true, 1, Some(108)),
            Meld::new(MeldType::Pon, vec![112, 113, 114], true, 1, Some(112)),
            Meld::new(MeldType::Pon, vec![116, 117, 118], true, 1, Some(116)),
        ];
        sanma.visible_counts[30] = 3;
        sanma.discard_candidates.push(30);
        assert_eq!(
            yaku_evidence(&sanma, &sanma_opponent).value("wind_yakuman"),
            Some(EVIDENCE_STRONG)
        );
    }

    #[test]
    fn four_ankan_confirms_suuankou_and_stacks_confirmed_yakuman_families() {
        let input = input_4p(&empty_observation()).unwrap();
        let mut opponent = input.opponents[0].clone().unwrap();
        opponent.melds = vec![
            Meld::new(MeldType::Ankan, vec![124, 125, 126, 127], false, -1, None),
            Meld::new(MeldType::Ankan, vec![128, 129, 130, 131], false, -1, None),
            Meld::new(MeldType::Ankan, vec![132, 133, 134, 135], false, -1, None),
            Meld::new(MeldType::Ankan, vec![108, 109, 110, 111], false, -1, None),
        ];

        let evidence = yaku_evidence(&input, &opponent);
        assert_eq!(evidence.value("daisangen"), Some(EVIDENCE_CONFIRMED));
        assert_eq!(evidence.value("suuankou"), Some(EVIDENCE_CONFIRMED));
        assert_eq!(evidence.value("suukantsu"), Some(EVIDENCE_CONFIRMED));
        assert_eq!(confirmed_yakuman_units(&evidence), 3);
        assert_eq!(ron_yaku_factor(&opponent, &evidence), 1.0);

        let result = calculate_opponent(&input, &opponent);
        let expected = calculate_score(39, 0, false, false, 0, 4).pay_ron as f32;
        assert_eq!(result.mean_loss_points[0], expected);
    }

    #[test]
    fn missing_fourth_wind_candidate_has_confirmed_yakuman_loss() {
        let wind_pons = vec![
            Meld::new(MeldType::Pon, vec![108, 109, 110], true, 0, Some(108)),
            Meld::new(MeldType::Pon, vec![112, 113, 114], true, 0, Some(112)),
            Meld::new(MeldType::Pon, vec![116, 117, 118], true, 0, Some(116)),
        ];

        for mut input in [
            input_4p(&empty_observation()).unwrap(),
            input_3p(&empty_observation_3p()).unwrap(),
        ] {
            input.dora_indicators.clear();
            let mut opponent = input.opponents[0].clone().unwrap();
            opponent.melds = wind_pons.clone();
            let evidence = yaku_evidence(&input, &opponent);
            assert_eq!(evidence.value("wind_yakuman"), Some(EVIDENCE_STRONG));

            let result = calculate_opponent(&input, &opponent);
            let is_dealer = opponent.absolute_seat == input.oya;
            let expected = calculate_score(
                13,
                0,
                is_dealer,
                false,
                input.honba as u32,
                input.num_players,
            )
            .pay_ron as f32;
            assert_eq!(result.mean_loss_points[30], expected);
            assert_eq!(result.ron_prob[30], result.wait_prob[30]);
            assert!(result.mean_loss_points[30] > result.mean_loss_points[31]);
        }
    }

    #[test]
    fn four_public_meld_compositions_confirm_candidate_yakuman_loss() {
        let cases = [
            (
                vec![
                    Meld::new(MeldType::Pon, vec![108, 109, 110], true, 0, Some(108)),
                    Meld::new(MeldType::Pon, vec![112, 113, 114], true, 0, Some(112)),
                    Meld::new(MeldType::Pon, vec![124, 125, 126], true, 0, Some(124)),
                    Meld::new(MeldType::Pon, vec![128, 129, 130], true, 0, Some(128)),
                ],
                33usize, // Red dragon pair completes Tsuuiisou.
            ),
            (
                vec![
                    Meld::new(MeldType::Chi, vec![76, 80, 84], true, 0, Some(76)),
                    Meld::new(MeldType::Pon, vec![92, 93, 94], true, 0, Some(92)),
                    Meld::new(MeldType::Pon, vec![100, 101, 102], true, 0, Some(100)),
                    Meld::new(MeldType::Pon, vec![128, 129, 130], true, 0, Some(128)),
                ],
                19usize, // 2s pair completes Ryuuiisou.
            ),
            (
                vec![
                    Meld::new(MeldType::Pon, vec![0, 1, 2], true, 0, Some(0)),
                    Meld::new(MeldType::Pon, vec![32, 33, 34], true, 0, Some(32)),
                    Meld::new(MeldType::Pon, vec![36, 37, 38], true, 0, Some(36)),
                    Meld::new(MeldType::Pon, vec![68, 69, 70], true, 0, Some(68)),
                ],
                18usize, // 1s pair completes Chinroutou.
            ),
        ];

        for input in [
            input_4p(&empty_observation()).unwrap(),
            input_3p(&empty_observation_3p()).unwrap(),
        ] {
            for (melds, candidate) in &cases {
                let mut opponent = input.opponents[0].clone().unwrap();
                opponent.melds = melds.clone();
                let context = candidate_yakuman_context(&opponent);
                assert_eq!(candidate_confirmed_yakuman_units(0, context, *candidate), 1);

                let result = calculate_opponent(&input, &opponent);
                let expected = calculate_score(
                    13,
                    0,
                    opponent.absolute_seat == input.oya,
                    false,
                    input.honba as u32,
                    input.num_players,
                )
                .pay_ron as f32;
                assert_eq!(result.mean_loss_points[*candidate], expected);
                assert_eq!(result.ron_prob[*candidate], result.wait_prob[*candidate]);
            }
        }
    }

    #[test]
    fn four_public_melds_add_candidate_conditioned_normal_yaku_floor() {
        let cases = [
            (
                vec![
                    Meld::new(MeldType::Chi, vec![36, 40, 44], true, 0, Some(36)),
                    Meld::new(MeldType::Chi, vec![48, 52, 56], true, 0, Some(48)),
                    Meld::new(MeldType::Chi, vec![60, 64, 68], true, 0, Some(60)),
                    Meld::new(MeldType::Pon, vec![37, 38, 39], true, 0, Some(37)),
                ],
                10usize,
                5usize, // Open Chinitsu.
            ),
            (
                vec![
                    Meld::new(MeldType::Chi, vec![36, 40, 44], true, 0, Some(36)),
                    Meld::new(MeldType::Chi, vec![60, 64, 68], true, 0, Some(60)),
                    Meld::new(MeldType::Pon, vec![72, 73, 74], true, 0, Some(72)),
                    Meld::new(MeldType::Pon, vec![104, 105, 106], true, 0, Some(104)),
                ],
                0usize,
                2usize, // Open Junchan.
            ),
            (
                vec![
                    Meld::new(MeldType::Pon, vec![124, 125, 126], true, 0, Some(124)),
                    Meld::new(MeldType::Pon, vec![128, 129, 130], true, 0, Some(128)),
                    Meld::new(MeldType::Pon, vec![40, 41, 42], true, 0, Some(40)),
                    Meld::new(MeldType::Pon, vec![76, 77, 78], true, 0, Some(76)),
                ],
                33usize,
                2usize, // Shousangen beyond the two confirmed Yakuhai.
            ),
            (
                vec![
                    Meld::new(MeldType::Pon, vec![108, 109, 110], true, 0, Some(108)),
                    Meld::new(MeldType::Pon, vec![112, 113, 114], true, 0, Some(112)),
                    Meld::new(MeldType::Pon, vec![124, 125, 126], true, 0, Some(124)),
                    Meld::new(MeldType::Pon, vec![128, 129, 130], true, 0, Some(128)),
                ],
                10usize,
                2usize, // Four honor melds plus 2p pair confirm Honitsu.
            ),
            (
                vec![
                    Meld::new(MeldType::Pon, vec![40, 41, 42], true, 0, Some(40)),
                    Meld::new(MeldType::Pon, vec![44, 45, 46], true, 0, Some(44)),
                    Meld::new(MeldType::Pon, vec![76, 77, 78], true, 0, Some(76)),
                    Meld::new(MeldType::Pon, vec![80, 81, 82], true, 0, Some(80)),
                ],
                12usize,
                1usize, // All-simple pair confirms Tanyao.
            ),
        ];

        for mut input in [
            input_4p(&empty_observation()).unwrap(),
            input_3p(&empty_observation_3p()).unwrap(),
        ] {
            input.dora_indicators.clear();
            for (melds, candidate, expected_extra) in &cases {
                let mut opponent = input.opponents[0].clone().unwrap();
                opponent.melds = melds.clone();
                let context = candidate_yakuman_context(&opponent);
                assert_eq!(
                    candidate_confirmed_normal_han(context, *candidate),
                    *expected_extra
                );
                let result = calculate_opponent(&input, &opponent);
                assert_eq!(result.ron_prob[*candidate], result.wait_prob[*candidate]);
                let evidence = yaku_evidence(&input, &opponent);
                let total_han = confirmed_normal_han(&input, &opponent, &evidence)
                    .saturating_add(*expected_extra)
                    .max(1);
                let expected_loss = calculate_score(
                    total_han.min(13) as u8,
                    30,
                    opponent.absolute_seat == input.oya,
                    false,
                    input.honba as u32,
                    input.num_players,
                )
                .pay_ron as f32;
                assert_eq!(result.mean_loss_points[*candidate], expected_loss);
            }
        }
    }

    #[test]
    fn strong_ordinal_evidence_does_not_lift_loss() {
        let input = input_4p(&empty_observation()).unwrap();
        let mut opponent = input.opponents[0].clone().unwrap();
        opponent.melds = vec![
            Meld::new(MeldType::Ankan, vec![36, 37, 38, 39], false, -1, None),
            Meld::new(MeldType::Ankan, vec![40, 41, 42, 43], false, -1, None),
            Meld::new(MeldType::Ankan, vec![44, 45, 46, 47], false, -1, None),
        ];

        let evidence = yaku_evidence(&input, &opponent);
        assert_eq!(evidence.value("honitsu"), Some(EVIDENCE_STRONG));
        assert_eq!(evidence.value("chinitsu"), Some(EVIDENCE_STRONG));
        assert_eq!(evidence.value("toitoi"), Some(EVIDENCE_STRONG));
        assert_eq!(evidence.value("suuankou"), Some(EVIDENCE_STRONG));
        assert_eq!(confirmed_yakuman_units(&evidence), 0);
        // The three concealed kans prove Sanankou and Sankantsu (four han),
        // while the merely STRONG flush/Toitoi/Suuankou ordinals add nothing.
        assert_eq!(estimated_han(&input, &opponent, &evidence), 4.0);
        assert_eq!(
            calculate_opponent(&input, &opponent).mean_loss_points[0],
            calculate_score(4, 30, false, false, 0, 4).pay_ron as f32
        );
    }

    #[test]
    fn loss_estimate_uses_only_confirmed_yaku_and_exact_public_bonus() {
        let mut observation = empty_observation();
        observation.oya = 1;
        observation.round_wind = 0;
        observation.melds[1].push(Meld::new(
            MeldType::Pon,
            vec![108, 109, 110], // East, both seat and round wind.
            true,
            0,
            Some(108),
        ));
        let input = input_4p(&observation).unwrap();
        let opponent = input.opponents[0].as_ref().unwrap();
        assert_eq!(value_triplet_han(opponent, 27, 0), 2);
        let evidence = yaku_evidence(&input, opponent);
        assert_eq!(estimated_han(&input, opponent, &evidence), 2.0);
        assert_eq!(ron_yaku_factor(opponent, &evidence), 1.0);

        let mut flush_observation = empty_observation();
        flush_observation.melds[1].push(Meld::new(
            MeldType::Chi,
            vec![40, 44, 48],
            true,
            0,
            Some(40),
        ));
        let flush_input = input_4p(&flush_observation).unwrap();
        let flush_opponent = flush_input.opponents[0].as_ref().unwrap();
        let evidence = yaku_evidence(&flush_input, flush_opponent);
        assert_eq!(evidence.value("honitsu"), Some(EVIDENCE_STRONG));
        assert_eq!(evidence.value("chinitsu"), Some(EVIDENCE_STRONG));
        // STRONG is ordinal evidence, not a probability or a han estimate.
        assert_eq!(estimated_han(&flush_input, flush_opponent, &evidence), 1.0);
    }

    #[test]
    fn visible_bonus_and_candidate_dora_set_exact_dealer_honba_loss_floor() {
        let mut observation = empty_observation();
        observation.oya = 1;
        observation.honba = 2;
        observation.dora_indicators = vec![45]; // 3p -> 4p.
        observation.discards[0] = vec![57];
        observation.discard_is_riichi[0] = vec![false];
        observation.public_tsumogiri_history[0] = vec![false];
        observation.discard_actor_history = vec![0];
        observation.resolved_discard_count = 1;
        observation.melds[1].push(Meld::new(
            MeldType::Chi,
            vec![49, 52, 57], // 4p dora, red 5p, 6p.
            true,
            0,
            Some(57),
        ));
        observation._legal_actions = vec![
            Action::new(ActionType::Discard, Some(48), vec![], Some(0)),
            Action::new(ActionType::Discard, Some(0), vec![], Some(0)),
        ];

        let result = calculate_drev_v2(&observation).unwrap();
        let opponent_result = &result.opponents()[0];
        let input = input_4p(&observation).unwrap();
        let opponent = input.opponents[0].as_ref().unwrap();

        // The public meld contributes one visible dora and one aka dora.
        assert_eq!(opponent.absolute_seat, input.oya);
        assert_eq!(input.honba, 2);
        assert_eq!(public_bonus_count(&input, opponent), 2);
        assert_eq!(public_bonus_value(&input, opponent), 2.0 / 6.0);
        assert_eq!(opponent_result.yaku.value("public_bonus"), Some(2.0 / 6.0));
        assert_eq!(dora_multiplicity(0, &input.dora_indicators, false), 0);
        assert_eq!(dora_multiplicity(12, &input.dora_indicators, false), 1);

        // A legal ron has a one-han floor. Public bonuses lift it to three;
        // discarding the 4p dora lifts it once more. Dealer and honba are exact.
        let expected_without_candidate_dora = calculate_score(3, 30, true, false, 2, 4).pay_ron;
        let expected_with_candidate_dora = calculate_score(4, 30, true, false, 2, 4).pay_ron;
        assert_eq!(expected_without_candidate_dora, 6_400);
        assert_eq!(expected_with_candidate_dora, 12_200);
        assert_eq!(
            opponent_result.mean_loss_points[0],
            expected_without_candidate_dora as f32
        );
        assert_eq!(
            opponent_result.mean_loss_points[12],
            expected_with_candidate_dora as f32
        );
        assert!(expected_with_candidate_dora > expected_without_candidate_dora);
    }

    #[test]
    fn sanma_dora_mapping_skips_removed_manzu_and_counts_north_twice() {
        assert_eq!(dora_type(0, true), 8);
        assert_eq!(dora_type(8, true), 0);
        assert_eq!(dora_type(0, false), 1);
        assert_eq!(dora_multiplicity(8, &[0], true), 1);
        assert_eq!(dora_multiplicity(1, &[0], true), 0);

        let mut observation = empty_observation_3p();
        observation.kita_counts[1] = 4;
        observation.dora_indicators = vec![116]; // West -> North.
        let input = input_3p(&observation).unwrap();
        let opponent = input.opponents[0].as_ref().unwrap();
        // Each set-aside North is one nukidora plus one ordinary dora.
        assert_eq!(public_bonus_count(&input, opponent), 8);
        assert_eq!(public_bonus_value(&input, opponent), 1.0);
        let evidence = yaku_evidence(&input, opponent);
        assert_eq!(estimated_han(&input, opponent, &evidence), 9.0);

        let mut high_bonus_input = input.clone();
        high_bonus_input.dora_indicators = vec![116; 3];
        let evidence = yaku_evidence(&high_bonus_input, opponent);
        assert_eq!(public_bonus_count(&high_bonus_input, opponent), 16);
        assert_eq!(estimated_han(&high_bonus_input, opponent, &evidence), 17.0);
        assert_eq!(
            calculate_opponent(&high_bonus_input, opponent).mean_loss_points[0],
            calculate_score(13, 30, false, false, 0, 3).pay_ron as f32
        );
    }

    #[test]
    fn sanma_manzu_chinitsu_is_impossible() {
        let input = input_3p(&empty_observation_3p()).unwrap();
        let mut opponent = input.opponents[0].clone().unwrap();
        opponent.melds = vec![Meld::new(MeldType::Pon, vec![0, 1, 2], true, 0, Some(0))];
        let evidence = yaku_evidence(&input, &opponent);
        assert_eq!(evidence.value("honitsu"), Some(EVIDENCE_STRONG));
        assert_eq!(evidence.value("chinitsu"), Some(EVIDENCE_IMPOSSIBLE));
    }

    #[test]
    fn v2_layout_is_finite_and_versioned() {
        let result = calculate_drev_v2(&empty_observation()).unwrap();
        let encoded = result.encode();
        assert_eq!(Drev4PV2Result::VERSION, 2);
        assert_eq!(DREV_V2_SCHEMA_ID, "riichienv.drev_v2.81ch.v1");
        assert_eq!(
            OPPONENT_SLOT_ORDER,
            ["relative_1", "relative_2", "relative_3"]
        );
        assert_eq!(RISK_CHANNEL_NAMES.len(), DREV_V2_BASE_CHANNELS);
        assert_eq!(DREV_V2_CHANNELS, 81);
        let names = channel_names();
        assert_eq!(names.len(), DREV_V2_CHANNELS);
        assert_eq!(names[0], "relative_1.hard_safe_zero");
        assert_eq!(names[5], "relative_1.uncertainty");
        assert_eq!(names[18], "aggregate.all_safe_zero");
        assert_eq!(names[23], "aggregate.max_uncertainty");
        assert_eq!(names[24], "relative_1.yaku.riichi");
        assert_eq!(names[42], "relative_1.yaku.suukantsu");
        assert_eq!(names[43], "relative_2.yaku.riichi");
        assert_eq!(names[62], "relative_3.yaku.riichi");
        assert_eq!(names[80], "relative_3.yaku.suukantsu");
        assert_eq!(encoded.len(), DREV_V2_CHANNELS * TILE_MAX);
        assert!(encoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn sanma_v2_uses_compact_axis_zero_slot_and_kita_value() {
        let mut observation = empty_observation_3p();
        observation.kita_counts[1] = 3;
        observation.dora_indicators = vec![116]; // West -> North dora.
        let result = calculate_drev_3p_v2(&observation).unwrap();
        assert_eq!(result.opponents()[0].yaku.value("public_bonus"), Some(1.0));
        assert!(
            result.opponents()[2]
                .ron_prob
                .iter()
                .all(|value| *value == 0.0)
        );
        let encoded = result.encode();
        assert_eq!(Drev3PV2Result::VERSION, 2);
        assert_eq!(encoded.len(), DREV_V2_CHANNELS * OBS_3P_TILE_TYPES);
        assert!(encoded.iter().all(|value| value.is_finite()));
        let inactive_yaku_start =
            (DREV_V2_BASE_CHANNELS + 2 * DREV_V2_YAKU_CHANNELS_PER_OPPONENT) * OBS_3P_TILE_TYPES;
        assert!(
            encoded[inactive_yaku_start..]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
}
