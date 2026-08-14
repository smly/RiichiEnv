//! Hidden-state integration oracle for DREV-v2's exact-zero contract.
//!
//! DREV itself only receives public observations.  The test deliberately
//! retains the authoritative engine state and checks every legal physical
//! discard candidate against each opponent's real Ron legality.  A public
//! hard-safe proof may be incomplete, but it must never be a false positive.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::drev_v2::{calculate_drev_3p_v2, calculate_drev_v2};
use riichienv_core::engine::{
    EngineConfig, EventLogPolicy, GameEngine, GameMode, ObservationVariant,
};
use riichienv_core::game_variant::GameStateVariant;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::state_3p::legal_actions::GameState3PLegalActions;

fn choose_action(actions: &[Action], phase: Phase, turn: usize) -> Action {
    let find = |kind| {
        actions
            .iter()
            .find(|action| action.action_type == kind)
            .cloned()
    };

    if let Some(action) = find(ActionType::Ron).or_else(|| find(ActionType::Tsumo)) {
        return action;
    }
    if phase == Phase::WaitResponse {
        return find(ActionType::Pass).expect("every declined response has Pass");
    }
    for kind in [
        ActionType::Ankan,
        ActionType::Kakan,
        ActionType::Kita,
        ActionType::Riichi,
    ] {
        if let Some(action) = find(kind) {
            return action;
        }
    }

    let discards: Vec<_> = actions
        .iter()
        .filter(|action| action.action_type == ActionType::Discard)
        .cloned()
        .collect();
    if !discards.is_empty() {
        return discards[turn % discards.len()].clone();
    }
    actions
        .first()
        .expect("active player has a legal action")
        .clone()
}

fn check_decision(engine: &GameEngine, observation: &ObservationVariant) -> (usize, usize) {
    let actor = observation.player_id();
    let candidates: Vec<u8> = observation
        .legal_actions()
        .into_iter()
        .filter(|action| action.action_type == ActionType::Discard)
        .filter_map(|action| action.tile)
        .collect();
    if candidates.is_empty() {
        return (0, 0);
    }

    let mut checked = 0;
    let mut hard_safe = 0;
    match (engine.state(), observation) {
        (GameStateVariant::FourPlayer(state), ObservationVariant::FourPlayer(obs)) => {
            let result = calculate_drev_v2(obs).expect("engine observation has complete history");
            for tile in candidates {
                for slot in 0..3 {
                    checked += 1;
                    if result.opponents()[slot].hard_safe_zero[tile as usize / 4] != 1.0 {
                        continue;
                    }
                    hard_safe += 1;
                    let opponent = (actor + slot as u8 + 1) % 4;
                    let (claims, _) = state._get_claim_actions_for_player(opponent, actor, tile);
                    assert!(
                        claims
                            .iter()
                            .all(|action| action.action_type != ActionType::Ron),
                        "false hard-safe proof: 4P actor={actor}, opponent={opponent}, tile={tile}"
                    );
                }
            }
        }
        (GameStateVariant::ThreePlayer(state), ObservationVariant::ThreePlayer(obs)) => {
            let result =
                calculate_drev_3p_v2(obs).expect("engine observation has complete history");
            for tile in candidates {
                for slot in 0..2 {
                    checked += 1;
                    if result.opponents()[slot].hard_safe_zero[tile as usize / 4] != 1.0 {
                        continue;
                    }
                    hard_safe += 1;
                    let opponent = (actor + slot as u8 + 1) % 3;
                    let (claims, _) = state._get_claim_actions_for_player(opponent, actor, tile);
                    assert!(
                        claims
                            .iter()
                            .all(|action| action.action_type != ActionType::Ron),
                        "false hard-safe proof: 3P actor={actor}, opponent={opponent}, tile={tile}"
                    );
                }
            }
        }
        _ => panic!("engine and observation variants disagree"),
    }
    (checked, hard_safe)
}

fn run_seed(mode: GameMode, seed: u64) -> (usize, usize) {
    let config = EngineConfig::new(mode)
        .with_seed(seed)
        .with_event_log(EventLogPolicy::Off);
    let mut engine = GameEngine::new(config).unwrap();
    let mut checked = 0;
    let mut hard_safe = 0;

    for turn in 0..512 {
        if engine.is_done() {
            break;
        }
        let decisions = engine.decisions();
        assert!(!decisions.is_empty(), "unfinished engine has no decision");
        let phase = engine.snapshot().phase;
        for decision in &decisions {
            let counts = check_decision(&engine, &decision.observation);
            checked += counts.0;
            hard_safe += counts.1;
        }
        let actions: HashMap<_, _> = decisions
            .iter()
            .map(|decision| {
                (
                    decision.player_id,
                    choose_action(&decision.observation.legal_actions(), phase, turn),
                )
            })
            .collect();
        let outcome = engine.step(&actions);
        assert!(
            outcome.error.is_none(),
            "oracle policy produced an engine error: {:?}",
            outcome.error
        );
    }
    assert!(engine.is_done(), "single-hand oracle exceeded step budget");
    (checked, hard_safe)
}

#[test]
fn hard_safe_zero_has_no_hidden_state_false_positives() {
    let mut checked = 0;
    let mut hard_safe = 0;
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for seed in [3, 17, 41, 73] {
            let counts = run_seed(mode, seed);
            checked += counts.0;
            hard_safe += counts.1;
        }
    }
    eprintln!("DREV-v2 hidden oracle: checked={checked}, hard_safe={hard_safe}, false_positive=0");
    assert!(checked > 1_000, "oracle corpus was unexpectedly small");
    assert!(
        hard_safe > 100,
        "oracle did not exercise enough exact-zero cells"
    );
}
