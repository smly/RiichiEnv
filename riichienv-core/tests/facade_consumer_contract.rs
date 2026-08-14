//! Downstream compile/runtime contract for the additive facade APIs.

use std::collections::HashMap;

use riichienv_core::action::Action as EnvAction;
use riichienv_core::drev::DrevInput;
use riichienv_core::engine::{EngineConfig, GameEngine, GameMode, ObservationVariant};
use riichienv_core::feature_context::{FeatureContext, FeatureContext3P};
use riichienv_core::features::{
    DREV_3P_V2, DREV_4P_V2, EXTENDED_SP_DREV_3P_V0, EXTENDED_SP_DREV_3P_V2, EXTENDED_SP_DREV_4P_V0,
    EXTENDED_SP_DREV_4P_V2, encode_drev_v2_3p_batch_into, encode_drev_v2_4p_batch_into,
    encode_extended_sp_drev_3p_batch_into, encode_extended_sp_drev_4p_batch_into,
    encode_extended_sp_drev_v2_3p_batch_into, encode_extended_sp_drev_v2_4p_batch_into,
};
use riichienv_core::replay::mjsoul_replay::RawAction;
use riichienv_core::replay::{Action as ReplayAction, EventJournal, ReplayLog};
use riichienv_core::rule::GameRule;
use riichienv_core::sp::{SpInput, SpInput3P};
use riichienv_core::state::GameState;
use riichienv_core::state_3p::GameState3P;

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
    let mut drev_v2 = vec![0.0; DREV_4P_V2.values_per_observation()];
    encode_drev_v2_4p_batch_into(std::slice::from_ref(observation), &mut drev_v2).unwrap();
    assert_eq!(decision.observation.drev_v2_feature_shape(), (81, 34));
    let mut combined_v2 = vec![0.0; EXTENDED_SP_DREV_4P_V2.values_per_observation()];
    encode_extended_sp_drev_v2_4p_batch_into(std::slice::from_ref(observation), &mut combined_v2)
        .unwrap();
    assert_eq!(
        decision
            .observation
            .extended_with_sp_drev_v2_feature_shape(),
        (474, 34)
    );

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
fn replay_discard_dtos_keep_the_v048_public_source_shape() {
    let _: fn(&mut GameState, u8, EnvAction) = GameState::_resolve_kan;
    let _: fn(&mut GameState3P, u8, EnvAction) = GameState3P::_resolve_kan;

    let action = ReplayAction::DiscardTile {
        seat: 1,
        tile: 52,
        is_liqi: false,
        is_wliqi: false,
        doras: None,
    };
    let ReplayAction::DiscardTile {
        seat,
        tile,
        is_liqi,
        is_wliqi,
        doras,
    } = action
    else {
        unreachable!()
    };
    assert_eq!(
        (seat, tile, is_liqi, is_wliqi, doras),
        (1, 52, false, false, None)
    );

    let raw = RawAction::DiscardTile {
        seat: 1,
        tile: "5p".to_string(),
        is_liqi: false,
        is_wliqi: false,
        doras: Vec::new(),
    };
    let RawAction::DiscardTile {
        seat,
        tile,
        is_liqi,
        is_wliqi,
        doras,
    } = raw
    else {
        unreachable!()
    };
    assert_eq!(
        (seat, tile.as_str(), is_liqi, is_wliqi, doras.len()),
        (1, "5p", false, false, 0)
    );

    // Newer Mahjong Soul wire data may still contain `moqie`; serde must
    // accept it even though it is no longer part of the public Rust DTO.
    let parsed: RawAction = serde_json::from_value(serde_json::json!({
        "name": "DiscardTile",
        "data": {
            "seat": 1,
            "tile": "5p",
            "is_liqi": false,
            "is_wliqi": false,
            "moqie": true,
            "doras": []
        }
    }))
    .unwrap();
    assert!(matches!(parsed, RawAction::DiscardTile { seat: 1, .. }));
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
    let mut drev_v2 = vec![0.0; DREV_3P_V2.values_per_observation()];
    encode_drev_v2_3p_batch_into(std::slice::from_ref(observation), &mut drev_v2).unwrap();
    let mut combined_v2 = vec![0.0; EXTENDED_SP_DREV_3P_V2.values_per_observation()];
    encode_extended_sp_drev_v2_3p_batch_into(std::slice::from_ref(observation), &mut combined_v2)
        .unwrap();
    assert_eq!(decision.observation.sp_feature_shape(), (178, 27));
    assert_eq!(decision.observation.drev_feature_shape(), (9, 27));
    assert_eq!(decision.observation.drev_v2_feature_shape(), (81, 27));
    assert_eq!(
        decision.observation.extended_with_sp_feature_shape(),
        (402, 27)
    );
    assert_eq!(
        decision
            .observation
            .extended_with_sp_drev_v2_feature_shape(),
        (474, 27)
    );
}
