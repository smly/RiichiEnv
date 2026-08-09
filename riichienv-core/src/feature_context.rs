//! Shared, borrowed preprocessing for one player observation.
//!
//! Feature planes are a frozen model ABI, but several encoders need the same
//! hand counts, visible-tile counts, red-five flags, and discard candidates.
//! `FeatureContext` computes those values once without taking ownership of the
//! observation.  Standalone encoders may construct a short-lived context;
//! combined/batch encoders reuse one context across extended, SP, and DREV.

use crate::action::ActionType;
use crate::errors::RiichiResult;
use crate::observation::Observation;
use crate::observation_3p::Observation3P;
use crate::types::TILE_MAX;

pub struct FeatureContext<'a> {
    observation: &'a Observation,
    player_index: usize,
    hand_counts: [u8; TILE_MAX],
    /// Own hand plus public zones, preserving duplicate called tiles for the
    /// frozen base-v0 channel 63 behavior.
    visible_counts_v0: [u8; TILE_MAX],
    /// Same zones capped at four copies for SP remaining-tile arithmetic.
    visible_counts_capped: [u8; TILE_MAX],
    /// All hands present in the DTO plus public zones, matching legacy DREV.
    all_observed_counts_capped: [u8; TILE_MAX],
    akas_in_hand: [bool; 3],
    akas_seen: [bool; 3],
    discard_candidates: Vec<u8>,
}

/// Sanma counterpart of [`FeatureContext`]. Counts stay on the canonical
/// 34-tile axis for shanten/scoring code; 2m through 8m remain zero and are
/// compressed away only when the final feature tensor is written.
pub struct FeatureContext3P<'a> {
    observation: &'a Observation3P,
    player_index: usize,
    hand_counts: [u8; TILE_MAX],
    visible_counts_capped: [u8; TILE_MAX],
    all_observed_counts_capped: [u8; TILE_MAX],
    akas_in_hand: [bool; 3],
    akas_seen: [bool; 3],
    discard_candidates: Vec<u8>,
}

impl<'a> FeatureContext<'a> {
    pub fn new(observation: &'a Observation) -> RiichiResult<Self> {
        observation.validate()?;
        Ok(Self::new_unchecked(observation))
    }

    pub(crate) fn new_unchecked(observation: &'a Observation) -> Self {
        let player_index = observation.player_id as usize;
        let mut hand_counts = [0u8; TILE_MAX];
        let mut visible_counts_v0 = [0u8; TILE_MAX];
        let mut visible_counts_capped = [0u8; TILE_MAX];
        let mut all_observed_counts_capped = [0u8; TILE_MAX];
        let mut akas_in_hand = [false; 3];
        let mut akas_seen = [false; 3];

        for (seat, hand) in observation.hands.iter().enumerate() {
            for &tile in hand {
                bump_capped(&mut all_observed_counts_capped, tile);
                if seat == player_index {
                    let tile_type = (tile / 4) as usize;
                    if tile_type < TILE_MAX {
                        hand_counts[tile_type] = hand_counts[tile_type].saturating_add(1);
                        visible_counts_v0[tile_type] =
                            visible_counts_v0[tile_type].saturating_add(1);
                    }
                    bump_capped(&mut visible_counts_capped, tile);
                    mark_aka(&mut akas_in_hand, tile);
                    mark_aka(&mut akas_seen, tile);
                }
            }
        }

        for melds in &observation.melds {
            for meld in melds {
                for &tile in &meld.tiles {
                    let tile = tile as u32;
                    bump_v0(&mut visible_counts_v0, tile);
                    bump_capped(&mut visible_counts_capped, tile);
                    bump_capped(&mut all_observed_counts_capped, tile);
                    mark_aka(&mut akas_seen, tile);
                }
            }
        }
        for discards in &observation.discards {
            for &tile in discards {
                bump_v0(&mut visible_counts_v0, tile);
                bump_capped(&mut visible_counts_capped, tile);
                bump_capped(&mut all_observed_counts_capped, tile);
                mark_aka(&mut akas_seen, tile);
            }
        }
        for &tile in &observation.dora_indicators {
            bump_v0(&mut visible_counts_v0, tile);
            bump_capped(&mut visible_counts_capped, tile);
            bump_capped(&mut all_observed_counts_capped, tile);
            mark_aka(&mut akas_seen, tile);
        }

        let mut discard_candidates = Vec::new();
        for action in &observation._legal_actions {
            if action.action_type == ActionType::Discard
                && let Some(tile) = action.tile
            {
                let tile_type = tile / 4;
                if !discard_candidates.contains(&tile_type) {
                    discard_candidates.push(tile_type);
                }
            }
        }
        discard_candidates.sort_unstable();

        Self {
            observation,
            player_index,
            hand_counts,
            visible_counts_v0,
            visible_counts_capped,
            all_observed_counts_capped,
            akas_in_hand,
            akas_seen,
            discard_candidates,
        }
    }

    pub fn observation(&self) -> &'a Observation {
        self.observation
    }

    pub fn player_index(&self) -> usize {
        self.player_index
    }

    pub fn hand_counts(&self) -> &[u8; TILE_MAX] {
        &self.hand_counts
    }

    pub fn visible_counts_v0(&self) -> &[u8; TILE_MAX] {
        &self.visible_counts_v0
    }

    pub fn visible_counts_capped(&self) -> &[u8; TILE_MAX] {
        &self.visible_counts_capped
    }

    pub fn all_observed_counts_capped(&self) -> &[u8; TILE_MAX] {
        &self.all_observed_counts_capped
    }

    pub fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }

    pub fn akas_seen(&self) -> [bool; 3] {
        self.akas_seen
    }

    pub fn discard_candidates(&self) -> &[u8] {
        &self.discard_candidates
    }
}

impl<'a> FeatureContext3P<'a> {
    pub fn new(observation: &'a Observation3P) -> RiichiResult<Self> {
        observation.validate()?;
        Ok(Self::new_unchecked(observation))
    }

    pub(crate) fn new_unchecked(observation: &'a Observation3P) -> Self {
        let player_index = observation.player_id as usize;
        let mut hand_counts = [0u8; TILE_MAX];
        let mut visible_counts_capped = [0u8; TILE_MAX];
        let mut all_observed_counts_capped = [0u8; TILE_MAX];
        let mut akas_in_hand = [false; 3];
        let mut akas_seen = [false; 3];

        for (seat, hand) in observation.hands.iter().enumerate() {
            for &tile in hand {
                bump_capped(&mut all_observed_counts_capped, tile);
                if seat == player_index {
                    let tile_type = (tile / 4) as usize;
                    if tile_type < TILE_MAX {
                        hand_counts[tile_type] = hand_counts[tile_type].saturating_add(1);
                    }
                    bump_capped(&mut visible_counts_capped, tile);
                    mark_aka(&mut akas_in_hand, tile);
                    mark_aka(&mut akas_seen, tile);
                }
            }
        }

        for melds in &observation.melds {
            for meld in melds {
                for &tile in &meld.tiles {
                    bump_capped(&mut visible_counts_capped, u32::from(tile));
                    bump_capped(&mut all_observed_counts_capped, u32::from(tile));
                    mark_aka(&mut akas_seen, u32::from(tile));
                }
            }
        }
        for discards in &observation.discards {
            for &tile in discards {
                bump_capped(&mut visible_counts_capped, tile);
                bump_capped(&mut all_observed_counts_capped, tile);
                mark_aka(&mut akas_seen, tile);
            }
        }
        for &tile in &observation.dora_indicators {
            bump_capped(&mut visible_counts_capped, tile);
            bump_capped(&mut all_observed_counts_capped, tile);
            mark_aka(&mut akas_seen, tile);
        }

        // Kita tiles are public, removed from the declaring player's hand,
        // and therefore must be restored to visible/unseen-wall accounting.
        let kita_count = observation
            .kita_counts
            .iter()
            .copied()
            .fold(0u8, u8::saturating_add)
            .min(4);
        visible_counts_capped[30] = visible_counts_capped[30].saturating_add(kita_count).min(4);
        all_observed_counts_capped[30] = all_observed_counts_capped[30]
            .saturating_add(kita_count)
            .min(4);

        let mut discard_candidates = Vec::new();
        for action in &observation._legal_actions {
            if action.0.action_type == ActionType::Discard
                && let Some(tile) = action.0.tile
            {
                let tile_type = tile / 4;
                if !discard_candidates.contains(&tile_type) {
                    discard_candidates.push(tile_type);
                }
            }
        }
        discard_candidates.sort_unstable();

        Self {
            observation,
            player_index,
            hand_counts,
            visible_counts_capped,
            all_observed_counts_capped,
            akas_in_hand,
            akas_seen,
            discard_candidates,
        }
    }

    pub fn observation(&self) -> &'a Observation3P {
        self.observation
    }

    pub fn player_index(&self) -> usize {
        self.player_index
    }

    pub fn hand_counts(&self) -> &[u8; TILE_MAX] {
        &self.hand_counts
    }

    pub fn visible_counts_capped(&self) -> &[u8; TILE_MAX] {
        &self.visible_counts_capped
    }

    pub fn all_observed_counts_capped(&self) -> &[u8; TILE_MAX] {
        &self.all_observed_counts_capped
    }

    pub fn akas_in_hand(&self) -> [bool; 3] {
        self.akas_in_hand
    }

    pub fn akas_seen(&self) -> [bool; 3] {
        self.akas_seen
    }

    pub fn discard_candidates(&self) -> &[u8] {
        &self.discard_candidates
    }
}

fn bump_v0(counts: &mut [u8; TILE_MAX], tile: u32) {
    let tile_type = (tile / 4) as usize;
    if let Some(count) = counts.get_mut(tile_type) {
        *count = count.saturating_add(1);
    }
}

fn bump_capped(counts: &mut [u8; TILE_MAX], tile: u32) {
    let tile_type = (tile / 4) as usize;
    if let Some(count) = counts.get_mut(tile_type) {
        *count = count.saturating_add(1).min(4);
    }
}

fn mark_aka(akas: &mut [bool; 3], tile: u32) {
    match tile {
        16 => akas[0] = true,
        52 => akas[1] = true,
        88 => akas[2] = true,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drev::DrevInput;
    use crate::engine::{EngineConfig, GameEngine, GameMode, ObservationVariant};
    use crate::sp::{SpInput, SpInput3P};

    fn observation() -> Observation {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        match engine.decisions().remove(0).observation {
            ObservationVariant::FourPlayer(observation) => observation,
            ObservationVariant::ThreePlayer(_) => unreachable!(),
        }
    }

    #[test]
    fn shared_inputs_match_standalone_constructors() {
        let observation = observation();
        let standalone_sp = SpInput::from_observation(&observation);
        let standalone_drev = DrevInput::from_observation(&observation);
        let context = FeatureContext::new(&observation).unwrap();
        let shared_sp = SpInput::from_feature_context(&context);
        let shared_drev = DrevInput::from_feature_context(&context);

        assert_eq!(shared_sp.tehai, standalone_sp.tehai);
        assert_eq!(shared_sp.tiles_seen, standalone_sp.tiles_seen);
        assert_eq!(shared_sp.akas_seen, standalone_sp.akas_seen);
        assert_eq!(
            shared_sp.discard_candidates,
            standalone_sp.discard_candidates
        );
        assert_eq!(shared_drev.tiles_seen, standalone_drev.tiles_seen);
        assert_eq!(shared_drev.opp_safe_mask, standalone_drev.opp_safe_mask);
    }

    #[test]
    fn public_constructor_rejects_an_invalid_observation() {
        let mut observation = observation();
        observation.player_id = 4;
        assert!(FeatureContext::new(&observation).is_err());
    }

    #[test]
    fn shared_three_player_inputs_match_standalone_constructors() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::ThreePlayerSingle).with_seed(42)).unwrap();
        let observation = match engine.decisions().remove(0).observation {
            ObservationVariant::ThreePlayer(observation) => observation,
            ObservationVariant::FourPlayer(_) => unreachable!(),
        };
        let standalone_sp = SpInput3P::from_observation(&observation);
        let standalone_drev = DrevInput::from_observation_3p(&observation);
        let context = FeatureContext3P::new(&observation).unwrap();
        let shared_sp = SpInput3P::from_feature_context(&context);
        let shared_drev = DrevInput::from_feature_context_3p(&context);

        assert_eq!(shared_sp.tehai, standalone_sp.tehai);
        assert_eq!(shared_sp.tiles_seen, standalone_sp.tiles_seen);
        assert_eq!(shared_sp.akas_seen, standalone_sp.akas_seen);
        assert_eq!(
            shared_sp.discard_candidates,
            standalone_sp.discard_candidates
        );
        assert_eq!(shared_drev.tiles_seen, standalone_drev.tiles_seen);
        assert_eq!(shared_drev.opp_safe_mask, standalone_drev.opp_safe_mask);
        assert_eq!(shared_drev.n_active_opponents, 2);
    }

    #[test]
    fn sanma_kita_is_visible_and_reaches_sp_scoring_input() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::ThreePlayerSingle).with_seed(42)).unwrap();
        let mut observation = match engine.decisions().remove(0).observation {
            ObservationVariant::ThreePlayer(observation) => observation,
            ObservationVariant::FourPlayer(_) => unreachable!(),
        };
        let baseline = FeatureContext3P::new(&observation)
            .unwrap()
            .all_observed_counts_capped()[30];
        let player = observation.player_id as usize;
        observation.kita_counts[player] = 1;

        let context = FeatureContext3P::new(&observation).unwrap();
        assert_eq!(
            context.all_observed_counts_capped()[30],
            baseline.saturating_add(1).min(4)
        );
        assert_eq!(SpInput3P::from_feature_context(&context).kita_count, 1);
    }
}
