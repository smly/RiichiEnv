//! Versioned, allocation-conscious feature encoding entry points.
//!
//! Feature values and channel order are a model ABI.  This module makes that
//! boundary explicit and provides batch writers shared by Rust and language
//! bindings.  The current layouts are named `v0`; changing their meaning must
//! introduce a new version instead of silently reusing these specifications.

use crate::drev::DREV_CHANNELS;
use crate::errors::{RiichiError, RiichiResult};
use crate::observation::{OBS_BASE_CHANNELS, OBS_EXTENDED_CHANNELS, OBS_TILE_TYPES, Observation};
use crate::observation_3p::{
    OBS_3P_BASE_CHANNELS, OBS_3P_EXTENDED_CHANNELS, OBS_3P_TILE_TYPES, Observation3P,
};
use crate::sp::SP_CHANNELS;

/// Immutable description of a channel-major feature tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSpec {
    pub name: &'static str,
    pub version: u16,
    pub channels: usize,
    pub tile_types: usize,
}

impl FeatureSpec {
    pub const fn values_per_observation(self) -> usize {
        self.channels * self.tile_types
    }

    pub const fn values_for_batch(self, batch_size: usize) -> usize {
        self.values_per_observation() * batch_size
    }
}

pub const BASE_4P_V0: FeatureSpec = FeatureSpec {
    name: "base-4p",
    version: 0,
    channels: OBS_BASE_CHANNELS,
    tile_types: OBS_TILE_TYPES,
};

pub const EXTENDED_4P_V0: FeatureSpec = FeatureSpec {
    name: "extended-4p",
    version: 0,
    channels: OBS_EXTENDED_CHANNELS,
    tile_types: OBS_TILE_TYPES,
};

pub const SP_4P_V0: FeatureSpec = FeatureSpec {
    name: "sp-4p",
    version: 0,
    channels: SP_CHANNELS,
    tile_types: OBS_TILE_TYPES,
};

pub const DREV_4P_V0: FeatureSpec = FeatureSpec {
    name: "drev-4p",
    version: 0,
    channels: DREV_CHANNELS,
    tile_types: OBS_TILE_TYPES,
};

pub const EXTENDED_SP_DREV_4P_V0: FeatureSpec = FeatureSpec {
    name: "extended-sp-drev-4p",
    version: 0,
    channels: OBS_EXTENDED_CHANNELS + SP_CHANNELS + DREV_CHANNELS,
    tile_types: OBS_TILE_TYPES,
};

pub const BASE_3P_V0: FeatureSpec = FeatureSpec {
    name: "base-3p",
    version: 0,
    channels: OBS_3P_BASE_CHANNELS,
    tile_types: OBS_3P_TILE_TYPES,
};

pub const EXTENDED_3P_V0: FeatureSpec = FeatureSpec {
    name: "extended-3p",
    version: 0,
    channels: OBS_3P_EXTENDED_CHANNELS,
    tile_types: OBS_3P_TILE_TYPES,
};

fn validate_output_len(
    spec: FeatureSpec,
    batch_size: usize,
    output_len: usize,
) -> RiichiResult<()> {
    let expected = spec.values_for_batch(batch_size);
    if output_len != expected {
        return Err(RiichiError::InvalidState {
            message: format!(
                "{} v{} batch output has length {output_len}; expected {expected} for batch size {batch_size}",
                spec.name, spec.version
            ),
        });
    }
    Ok(())
}

fn validate_4p_observations(observations: &[Observation]) -> RiichiResult<()> {
    observations.iter().try_for_each(Observation::validate)
}

fn validate_3p_observations(observations: &[Observation3P]) -> RiichiResult<()> {
    observations.iter().try_for_each(Observation3P::validate)
}

pub fn encode_base_4p_batch(observations: &[Observation]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; BASE_4P_V0.values_for_batch(observations.len())];
    encode_base_4p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_base_4p_batch_into(
    observations: &[Observation],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(BASE_4P_V0, observations.len(), output.len())?;
    validate_4p_observations(observations)?;
    let row_len = BASE_4P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        observation.encode_base_features_into_unchecked(row);
    }
    Ok(())
}

pub fn encode_extended_4p_batch(observations: &[Observation]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; EXTENDED_4P_V0.values_for_batch(observations.len())];
    encode_extended_4p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_extended_4p_batch_into(
    observations: &[Observation],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(EXTENDED_4P_V0, observations.len(), output.len())?;
    validate_4p_observations(observations)?;
    let row_len = EXTENDED_4P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        observation.encode_extended_features_into_unchecked(row);
    }
    Ok(())
}

pub fn encode_sp_4p_batch(observations: &[Observation]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; SP_4P_V0.values_for_batch(observations.len())];
    encode_sp_4p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_sp_4p_batch_into(
    observations: &[Observation],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(SP_4P_V0, observations.len(), output.len())?;
    validate_4p_observations(observations)?;
    let row_len = SP_4P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        row.fill(0.0);
        observation.encode_sp_into(row, 0);
    }
    Ok(())
}

pub fn encode_drev_4p_batch(observations: &[Observation]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; DREV_4P_V0.values_for_batch(observations.len())];
    encode_drev_4p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_drev_4p_batch_into(
    observations: &[Observation],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(DREV_4P_V0, observations.len(), output.len())?;
    validate_4p_observations(observations)?;
    let row_len = DREV_4P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        row.fill(0.0);
        observation.encode_drev_into(row, 0);
    }
    Ok(())
}

pub fn encode_extended_sp_drev_4p_batch(observations: &[Observation]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; EXTENDED_SP_DREV_4P_V0.values_for_batch(observations.len())];
    encode_extended_sp_drev_4p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_extended_sp_drev_4p_batch_into(
    observations: &[Observation],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(EXTENDED_SP_DREV_4P_V0, observations.len(), output.len())?;
    validate_4p_observations(observations)?;
    let row_len = EXTENDED_SP_DREV_4P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        observation.encode_extended_with_sp_features_into_unchecked(row);
    }
    Ok(())
}

pub fn encode_base_3p_batch(observations: &[Observation3P]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; BASE_3P_V0.values_for_batch(observations.len())];
    encode_base_3p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_base_3p_batch_into(
    observations: &[Observation3P],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(BASE_3P_V0, observations.len(), output.len())?;
    validate_3p_observations(observations)?;
    let row_len = BASE_3P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        observation.encode_base_features_into_unchecked(row);
    }
    Ok(())
}

pub fn encode_extended_3p_batch(observations: &[Observation3P]) -> RiichiResult<Vec<f32>> {
    let mut output = vec![0.0; EXTENDED_3P_V0.values_for_batch(observations.len())];
    encode_extended_3p_batch_into(observations, &mut output)?;
    Ok(output)
}

pub fn encode_extended_3p_batch_into(
    observations: &[Observation3P],
    output: &mut [f32],
) -> RiichiResult<()> {
    validate_output_len(EXTENDED_3P_V0, observations.len(), output.len())?;
    validate_3p_observations(observations)?;
    let row_len = EXTENDED_3P_V0.values_per_observation();
    for (observation, row) in observations.iter().zip(output.chunks_exact_mut(row_len)) {
        observation.encode_extended_features_into_unchecked(row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, GameEngine, GameMode, ObservationVariant};

    fn initial_observations(mode: GameMode, count: usize) -> Vec<ObservationVariant> {
        (0..count)
            .map(|index| {
                let mut engine =
                    GameEngine::new(EngineConfig::new(mode).with_seed(1_000 + index as u64))
                        .unwrap();
                engine.decisions().remove(0).observation
            })
            .collect()
    }

    #[test]
    fn four_player_batch_rows_match_single_encoders() {
        let observations = initial_observations(GameMode::FourPlayerSingle, 3)
            .into_iter()
            .map(|observation| match observation {
                ObservationVariant::FourPlayer(observation) => observation,
                ObservationVariant::ThreePlayer(_) => unreachable!(),
            })
            .collect::<Vec<_>>();

        let base = encode_base_4p_batch(&observations).unwrap();
        let extended = encode_extended_4p_batch(&observations).unwrap();
        for (index, observation) in observations.iter().enumerate() {
            let base_start = index * BASE_4P_V0.values_per_observation();
            assert_eq!(
                &base[base_start..base_start + BASE_4P_V0.values_per_observation()],
                observation.encode_base_features().unwrap()
            );
            let extended_start = index * EXTENDED_4P_V0.values_per_observation();
            assert_eq!(
                &extended[extended_start..extended_start + EXTENDED_4P_V0.values_per_observation()],
                observation.encode_extended_features().unwrap()
            );
        }
    }

    #[test]
    fn three_player_batch_rows_match_single_encoders() {
        let observations = initial_observations(GameMode::ThreePlayerSingle, 2)
            .into_iter()
            .map(|observation| match observation {
                ObservationVariant::ThreePlayer(observation) => observation,
                ObservationVariant::FourPlayer(_) => unreachable!(),
            })
            .collect::<Vec<_>>();

        let extended = encode_extended_3p_batch(&observations).unwrap();
        for (index, observation) in observations.iter().enumerate() {
            let start = index * EXTENDED_3P_V0.values_per_observation();
            assert_eq!(
                &extended[start..start + EXTENDED_3P_V0.values_per_observation()],
                observation.encode_extended_features().unwrap()
            );
        }
    }

    #[test]
    fn caller_buffer_length_is_validated() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(7)).unwrap();
        let observation = match engine.decisions().remove(0).observation {
            ObservationVariant::FourPlayer(observation) => observation,
            ObservationVariant::ThreePlayer(_) => unreachable!(),
        };
        let mut too_short = vec![0.0; EXTENDED_4P_V0.values_per_observation() - 1];
        assert!(encode_extended_4p_batch_into(&[observation], &mut too_short).is_err());
    }

    #[test]
    fn empty_batch_has_empty_output() {
        assert!(encode_base_4p_batch(&[]).unwrap().is_empty());
        assert!(encode_extended_3p_batch(&[]).unwrap().is_empty());
    }

    #[test]
    fn malformed_observations_are_rejected_before_output_mutation() {
        let mut observations = initial_observations(GameMode::FourPlayerSingle, 1);
        let mut four_player = match observations.remove(0) {
            ObservationVariant::FourPlayer(observation) => observation,
            ObservationVariant::ThreePlayer(_) => unreachable!(),
        };
        four_player.player_id = 4;
        let mut output = vec![7.0; BASE_4P_V0.values_per_observation()];
        assert!(encode_base_4p_batch_into(&[four_player], &mut output).is_err());
        assert!(output.iter().all(|&value| value == 7.0));

        let mut observations = initial_observations(GameMode::ThreePlayerSingle, 1);
        let mut three_player = match observations.remove(0) {
            ObservationVariant::ThreePlayer(observation) => observation,
            ObservationVariant::FourPlayer(_) => unreachable!(),
        };
        three_player.hands[three_player.player_id as usize].clear();
        let mut output = vec![7.0; EXTENDED_3P_V0.values_per_observation()];
        assert!(encode_extended_3p_batch_into(&[three_player], &mut output).is_err());
        assert!(output.iter().all(|&value| value == 7.0));
    }
}
