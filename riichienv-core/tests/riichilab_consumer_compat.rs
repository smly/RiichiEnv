//! Compile/runtime contract for the public surface used by riichilab 0.4.8.
//!
//! This deliberately imports the crate as an external consumer and touches
//! legacy public fields.  The new engine facade is additive; these contracts
//! cannot be made private until riichilab has migrated.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::errors::RiichiResult;
use riichienv_core::observation::Observation;
use riichienv_core::observation_3p::Observation3P;
use riichienv_core::parser::mjai_to_tid;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use riichienv_core::state_3p::GameState3P;

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::WaitAct => "wait_act",
        Phase::WaitResponse => "wait_response",
    }
}

fn consume_observation(observation: &Observation) -> RiichiResult<()> {
    let _events: Vec<String> = observation.new_events();
    let legal: Vec<Action> = observation.legal_actions_method();
    let _wire: String = observation.serialize_to_base64()?;
    for action in legal {
        let _: serde_json::Value = serde_json::from_str(&action.to_mjai()).unwrap();
    }
    Ok(())
}

fn consume_observation_3p(observation: &Observation3P) -> RiichiResult<()> {
    let _events: Vec<String> = observation.new_events();
    let _wire: String = observation.serialize_to_base64()?;
    for action in observation.legal_actions_method() {
        let _: serde_json::Value = serde_json::from_str(&action.to_mjai()).unwrap();
    }
    Ok(())
}

#[test]
fn four_player_legacy_surface_remains_available() {
    let mut state = GameState::new(0, false, Some(42), 0, GameRule::default_tenhou());

    let _status = (
        state.is_done,
        phase_name(state.phase),
        state.current_player,
        state.active_players.clone(),
        state.drawn_tile,
        state.players[0].score,
        state.mjai_log.len(),
        state.round_wind,
        state.kyoku_idx,
        state.honba,
        state.wall.tiles.len(),
    );

    let player_id = state.active_players[0];
    let observation = state.get_observation(player_id);
    consume_observation(&observation).unwrap();
    assert!(state.get_observation(player_id).new_events().is_empty());

    let action = observation.legal_actions_method().remove(0);
    state.step(&HashMap::from([(player_id, action)]));
}

#[test]
fn three_player_legacy_surface_remains_available() {
    let mut state = GameState3P::new(3, false, Some(42), 0, GameRule::default_tenhou());

    let _status = (
        state.is_done,
        phase_name(state.phase),
        state.current_player,
        state.active_players.clone(),
        state.drawn_tile,
        state.players[0].score,
        state.mjai_log.len(),
        state.round_wind,
        state.kyoku_idx,
        state.honba,
        state.wall.tiles.len(),
    );

    let player_id = state.active_players[0];
    let observation = state.get_observation(player_id);
    consume_observation_3p(&observation).unwrap();
    assert!(state.get_observation(player_id).new_events().is_empty());

    let action = observation.legal_actions_method().remove(0).0;
    state.step(&HashMap::from([(player_id, action)]));
}

#[test]
fn action_and_parser_wire_contract_remains_available() {
    assert_eq!(mjai_to_tid("5mr"), Some(16));
    let action = Action::new(ActionType::Riichi, Some(16), vec![], Some(2));
    let wire: serde_json::Value = serde_json::from_str(&action.to_mjai()).unwrap();
    assert_eq!(wire, serde_json::json!({"type": "reach", "actor": 2}));
}
