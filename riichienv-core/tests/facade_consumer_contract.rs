//! Downstream compile/runtime contract for the additive facade APIs.

use std::collections::HashMap;

use riichienv_core::drev::DrevInput;
use riichienv_core::engine::{EngineConfig, GameEngine, GameMode, ObservationVariant};
use riichienv_core::feature_context::{FeatureContext, FeatureContext3P};
use riichienv_core::features::{
    EXTENDED_SP_DREV_3P_V0, EXTENDED_SP_DREV_4P_V0, encode_extended_sp_drev_3p_batch_into,
    encode_extended_sp_drev_4p_batch_into,
};
use riichienv_core::replay::{EventJournal, ReplayLog};
use riichienv_core::rule::GameRule;
use riichienv_core::sp::{SpInput, SpInput3P};

#[test]
fn four_player_facade_and_borrowed_features_are_public() {
    let mut engine =
        GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
    let decision = engine.decisions().remove(0);
    let observation = match &decision.observation {
        ObservationVariant::FourPlayer(observation) => observation,
        ObservationVariant::ThreePlayer(_) => unreachable!(),
    };

    let context = FeatureContext::new(observation).unwrap();
    let _sp = SpInput::from_feature_context(&context);
    let _drev = DrevInput::from_feature_context(&context);
    let mut output = vec![0.0; EXTENDED_SP_DREV_4P_V0.values_per_observation()];
    encode_extended_sp_drev_4p_batch_into(std::slice::from_ref(observation), &mut output).unwrap();

    let action = decision.observation.legal_actions().remove(0);
    let outcome = engine.step(&HashMap::from([(decision.player_id, action)]));
    assert!(outcome.error.is_none());
}

#[test]
fn replay_and_live_journal_are_available_without_python() {
    let jsonl = include_str!("../../tests/data/126_204_0_mjai.jsonl");
    let replay = ReplayLog::from_jsonl(jsonl, GameRule::default_tenhou()).unwrap();
    assert_eq!(replay.cursor().count(), 12);
    assert!(!replay.rounds()[0].actions().is_empty());

    let journal = EventJournal::from_jsonl(jsonl).unwrap();
    assert_eq!(journal.completed_kyokus().len(), replay.len());
}

#[test]
fn three_player_sp_and_drev_features_are_public() {
    let mut engine =
        GameEngine::new(EngineConfig::new(GameMode::ThreePlayerSingle).with_seed(42)).unwrap();
    let decision = engine.decisions().remove(0);
    let observation = match &decision.observation {
        ObservationVariant::ThreePlayer(observation) => observation,
        ObservationVariant::FourPlayer(_) => unreachable!(),
    };

    let context = FeatureContext3P::new(observation).unwrap();
    let _sp = SpInput3P::from_feature_context(&context);
    let _drev = DrevInput::from_feature_context_3p(&context);
    let mut output = vec![0.0; EXTENDED_SP_DREV_3P_V0.values_per_observation()];
    encode_extended_sp_drev_3p_batch_into(std::slice::from_ref(observation), &mut output).unwrap();
    assert_eq!(decision.observation.sp_feature_shape(), (178, 27));
    assert_eq!(decision.observation.drev_feature_shape(), (9, 27));
    assert_eq!(
        decision.observation.extended_with_sp_feature_shape(),
        (402, 27)
    );
}
