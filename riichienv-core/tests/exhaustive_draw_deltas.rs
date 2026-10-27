//! Regression coverage for https://github.com/smly/RiichiEnv/issues/247.
//! Draw deltas must exclude deposits already recorded by reach_accepted.

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use serde_json::{Value, json};
use std::collections::HashMap;

const START_SCORES: [i32; 4] = [32600, 23500, 17400, 26500];

macro_rules! draw_fixture {
    ($name:ident, $state:ty, $evaluator:ty, $legal:path, $np:literal, $mode:literal, $scores:expr) => {
        fn $name(
            tenpai: [bool; $np],
            riichi: &[u8],
            nagashi_winner: Option<u8>,
            expected_deltas: [i32; $np],
        ) {
            use $legal;

            let mut state = <$state>::new($mode, false, Some(247), 0, GameRule::default_tenhou());
            state._initialize_round(0, 0, 2, 0, None, Some(($scores).to_vec()));
            for (pid, player) in state.players.iter_mut().enumerate() {
                // 123456789p1234s waits on 1s/4s; 123456789p124sE is noten.
                // Seats use distinct physical copies of each tile.
                let tiles = if tenpai[pid] {
                    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                } else {
                    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 27]
                };
                player.hand = tiles.iter().map(|t| t * 4 + pid as u8).collect();
                player.nagashi_eligible = nagashi_winner == Some(pid as u8);
                assert_eq!(
                    <$evaluator>::new(player.hand.clone(), player.melds.clone()).is_tenpai(),
                    tenpai[pid],
                    "fixture tenpai for seat {pid}"
                );
            }

            // Late-round fixture: eight live tiles remain after the dealer's draw.
            // The live draws are White/Green and cannot complete any fixture hand.
            // All remaining discards are honors, preserving the winner's nagashi eligibility.
            state.wall.tiles = (0..4)
                .chain(32..36)
                .chain(96..102)
                .chain((124..132).rev())
                .collect();
            state.wall.drawable_count = 8;
            state.players[0].hand.push(132);
            state.drawn_tile = Some(132);
            state.current_player = 0;
            state.active_players = vec![0];
            state.phase = Phase::WaitAct;
            state.is_first_turn = false;
            state.mjai_log.clear();

            // Exercise real declarations, acceptance, and exhausted-wall settlement.
            // Do not pre-populate score_delta or synthesize reach_accepted events.
            for _ in 0..9 {
                let pid = state.current_player;
                let tile = state.drawn_tile.expect("each turn must have a drawn tile");
                if riichi.contains(&pid) && !state.players[pid as usize].riichi_declared {
                    let reach = state
                        ._get_legal_actions_internal(pid)
                        .into_iter()
                        .find(|a| a.action_type == ActionType::Riichi)
                        .expect("fixture must allow riichi");
                    state.step(&HashMap::from([(pid, reach)]));
                    assert!(state.last_error.is_none());
                }
                let discard = state
                    ._get_legal_actions_internal(pid)
                    .into_iter()
                    .find(|a| a.action_type == ActionType::Discard && a.tile == Some(tile))
                    .expect("tsumogiri must be legal");
                state.step(&HashMap::from([(pid, discard)]));
                assert!(state.last_error.is_none());
                if state.phase == Phase::WaitResponse {
                    let passes = state
                        .active_players
                        .iter()
                        .map(|&p| (p, Action::new(ActionType::Pass, None, vec![], Some(p))))
                        .collect();
                    state.step(&passes);
                    assert!(state.last_error.is_none());
                }
            }

            let events: Vec<Value> = state
                .mjai_log
                .iter()
                .map(|event| serde_json::from_str(event).unwrap())
                .collect();
            let draw_index = events
                .iter()
                .position(|e| e["type"] == "ryukyoku")
                .expect("exhausted wall must produce a draw");
            let accepted: Vec<u8> = events[..draw_index]
                .iter()
                .filter(|e| e["type"] == "reach_accepted")
                .map(|e| e["actor"].as_u64().unwrap() as u8)
                .collect();
            assert_eq!(
                accepted, riichi,
                "deposits must be accepted before the draw"
            );
            assert_eq!(
                events[draw_index]["reason"],
                if nagashi_winner.is_some() {
                    "nagashimangan"
                } else {
                    "exhaustive_draw"
                }
            );

            // Verify the actual scores and pot before checking the faulty log field.
            let scores: Vec<i32> = (0..$np)
                .map(|pid| {
                    ($scores)[pid] + expected_deltas[pid]
                        - if riichi.contains(&(pid as u8)) {
                            1000
                        } else {
                            0
                        }
                })
                .collect();
            assert_eq!(
                state.players.iter().map(|p| p.score).collect::<Vec<_>>(),
                scores
            );
            assert_eq!(state.riichi_sticks, riichi.len() as u32);
            let next = events[draw_index + 1..]
                .iter()
                .find(|e| e["type"] == "start_kyoku")
                .expect("draw must start the next round");
            assert_eq!(next["scores"], json!(scores));
            assert_eq!(next["kyotaku"], json!(riichi.len()));
            assert_eq!(next["honba"], 3);
            assert_eq!(
                events[draw_index]["deltas"],
                json!(expected_deltas),
                "ryukyoku.deltas must contain only draw payments, excluding accepted riichi deposits"
            );
        }

    };
}

draw_fixture!(
    play_draw,
    riichienv_core::state::GameState,
    riichienv_core::hand_evaluator::HandEvaluator,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    4,
    2,
    START_SCORES
);
draw_fixture!(
    play_draw_3p,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::hand_evaluator_3p::HandEvaluator3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    5,
    [35000; 3]
);

#[test]
fn all_tenpai_after_three_riichi_has_no_draw_payments() {
    play_draw([true; 4], &[0, 2, 3], None, [0; 4]);
}

#[test]
fn all_tenpai_without_riichi_has_no_draw_payments() {
    play_draw([true; 4], &[], None, [0; 4]);
}

#[test]
fn two_tenpai_after_riichi_logs_only_noten_payments() {
    play_draw(
        [true, false, false, true],
        &[0],
        None,
        [1500, -1500, -1500, 1500],
    );
}

#[test]
fn dealer_nagashi_after_winners_riichi_logs_only_mangan_payments() {
    play_draw([true; 4], &[0], Some(0), [12000, -4000, -4000, -4000]);
}

#[test]
fn dealer_nagashi_without_riichi_logs_only_mangan_payments() {
    play_draw([true; 4], &[], Some(0), [12000, -4000, -4000, -4000]);
}

#[test]
fn non_dealer_nagashi_after_payers_riichi_logs_only_mangan_payments() {
    play_draw([true; 4], &[0], Some(1), [-4000, 8000, -2000, -2000]);
}

#[test]
fn non_dealer_nagashi_without_riichi_logs_only_mangan_payments() {
    play_draw([true; 4], &[], Some(1), [-4000, 8000, -2000, -2000]);
}

#[test]
fn three_player_all_tenpai_after_riichi_has_no_draw_payments() {
    play_draw_3p([true; 3], &[0, 2], None, [0; 3]);
}

#[test]
fn three_player_dealer_nagashi_after_winners_riichi_logs_only_mangan_payments() {
    play_draw_3p([true; 3], &[0], Some(0), [8000, -4000, -4000]);
}

#[test]
fn three_player_non_dealer_nagashi_after_payers_riichi_logs_only_mangan_payments() {
    play_draw_3p([true; 3], &[0], Some(1), [-4000, 6000, -2000]);
}
