//! Abortive draws must not trigger agari-yame or tenpai-yame (issue #229).

use std::collections::HashMap;

use riichienv_core::action::ActionType;
use riichienv_core::rule::GameRule;
use serde_json::{Value, json};

fn events(log: &[String]) -> Vec<Value> {
    log.iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn reported_south_4_kyushu_kyuhai_repeats() {
    use riichienv_core::state::{GameState, legal_actions::GameStateLegalActions};

    // Unmodified final round from the reported game:
    // https://logs.riichi.dev/mjai-logs/2026/09/08/4ffa7961-4004-4aae-92f9-afa617f48c69.jsonl.gz
    let reported: Vec<Value> = include_str!("fixtures/issue_229_south_4.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
        let mut state = GameState::new(2, false, Some(229), 0, rule);
        // Restore the opening, then generate the ending through a legal action.
        for event in &reported[..2] {
            state.apply_mjai_event(serde_json::from_value(event.clone()).unwrap());
        }
        let action = state
            ._get_legal_actions_internal(3)
            .into_iter()
            .find(|action| action.action_type == ActionType::KyushuKyuhai)
            .expect("the reported declaration must be legal");
        state.mjai_log.clear();
        state.step(&HashMap::from([(3, action)]));

        assert!(state.last_error.is_none());
        assert!(!state.is_done);
        assert_eq!((state.round_wind, state.oya, state.honba), (1, 3, 1));
        let generated = events(&state.mjai_log);
        assert_eq!(generated[..2], reported[2..4]);
        assert_eq!(generated[2]["type"], "start_kyoku");
        assert_eq!(generated[2]["bakaze"], "S");
        assert_eq!(generated[2]["kyoku"], 4);
        assert_eq!(generated[2]["honba"], 1);
        assert_eq!(generated[2]["scores"], reported[0]["scores"]);
        assert_eq!(generated[3]["type"], "tsumo");
        assert_eq!(generated[3]["actor"], 3);
        assert_eq!(generated.len(), 4);
    }
}

macro_rules! renchan_tests {
    ($module:ident, $state:ty, $legal:path, $evaluator:ty, $np:expr, $single:expr, $east:expr, $half:expr, $start:expr, $target:expr) => {
        mod $module {
            use super::*;
            use $legal;

            fn state(rule: GameRule, mode: u8, wind: u8, dealer: u8) -> $state {
                let mut scores = vec![$start; $np];
                scores[dealer as usize] = $target + 10000;
                scores[(dealer as usize + 1) % $np] -= $target + 10000 - $start + 1000;
                let mut state = <$state>::new(mode, false, Some(229), 0, rule);
                state._initialize_round(dealer, wind, 2, 1, None, Some(scores));
                state.mjai_log.clear();
                state
            }

            fn assert_repeats(mut state: $state, reason: &str) {
                let before = (state.round_wind, state.oya, state.honba);
                let scores: Vec<i32> = state.players.iter().map(|p| p.score).collect();
                state._trigger_ryukyoku(reason);
                assert!(!state.is_done, "{reason} must repeat at {before:?}");
                assert_eq!(
                    (state.round_wind, state.oya, state.honba),
                    (before.0, before.1, before.2 + 1)
                );
                assert_eq!(state.riichi_sticks, 1);
                let generated = events(&state.mjai_log);
                assert_eq!(generated[0]["reason"], reason);
                assert_eq!(generated[0]["deltas"], json!(vec![0; $np]));
                assert_eq!(generated[1]["type"], "end_kyoku");
                assert_eq!(generated[2]["type"], "start_kyoku");
                assert_eq!(generated[2]["scores"], json!(scores));
                assert_eq!(generated[2]["kyotaku"], 1);
                assert_eq!(generated[3]["type"], "tsumo");
                assert_eq!(generated.len(), 4);
            }

            #[test]
            fn abortive_draws_repeat_in_orasu_and_extension() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    let mut reasons = vec!["kyushu_kyuhai", "suukansansen"];
                    if $np == 4 {
                        reasons.extend(["sufuurenta", "suucha_riichi"]);
                        if rule.sanchaho_is_draw {
                            reasons.push("sanchaho");
                        }
                    }
                    for (mode, wind) in [($east, 0), ($half, 1), ($east, 1), ($half, 2)] {
                        for &reason in &reasons {
                            // Test the scheduled last round and the extension's
                            // last dealer; extension must also allow other seats.
                            assert_repeats(state(rule, mode, wind, $np - 1), reason);
                            assert_repeats(state(rule, mode, wind, 0), reason);
                        }
                    }
                }
            }

            #[test]
            fn kyushu_kyuhai_repeats_regardless_of_declarer_and_dealer_rank() {
                for actor in [0, $np - 1] {
                    for dealer_top in [false, true] {
                        let mut state = state(GameRule::default_tenhou(), $half, 1, $np - 1);
                        if !dealer_top {
                            let top = state.players[$np - 1].score;
                            state.players[$np - 1].score = state.players[0].score;
                            state.players[0].score = top;
                        }
                        // Nine distinct terminals/honors, plus five middle tiles.
                        state.players[actor as usize].hand =
                            [0, 8, 9, 17, 18, 26, 27, 28, 29, 10, 11, 12, 13, 14]
                                .iter()
                                .map(|t| t * 4 + actor)
                                .collect();
                        state.current_player = actor;
                        state.active_players = vec![actor];
                        state.drawn_tile = state.players[actor as usize].hand.last().copied();
                        let action = state
                            ._get_legal_actions_internal(actor)
                            .into_iter()
                            .find(|a| a.action_type == ActionType::KyushuKyuhai)
                            .expect("fixture must offer kyushu kyuhai");
                        state.step(&HashMap::from([(actor, action)]));
                        assert!(state.last_error.is_none());
                        assert!(!state.is_done);
                        assert_eq!(state.honba, 3);
                        assert_eq!(state.oya, $np - 1);
                    }
                }
            }

            #[test]
            fn exhaustive_draw_still_ends_with_top_tenpai_dealer() {
                for (mode, wind) in [($east, 0), ($half, 1)] {
                    let mut state = state(GameRule::default_tenhou(), mode, wind, $np - 1);
                    for (pid, player) in state.players.iter_mut().enumerate() {
                        // 123456789p1234s: all players tenpai, so no score changes.
                        player.hand = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                            .iter()
                            .map(|t| t * 4 + pid as u8)
                            .collect();
                        player.nagashi_eligible = false;
                        assert!(<$evaluator>::new(player.hand.clone(), vec![]).is_tenpai());
                    }
                    state.wall.drawable_count = 0;
                    state._deal_next();
                    assert!(state.is_done);
                    let generated = events(&state.mjai_log);
                    assert_eq!(generated[0]["reason"], "exhaustive_draw");
                    assert_eq!(generated[2]["type"], "end_game");
                }
            }

            #[test]
            fn dealer_win_still_ends_in_orasu() {
                for (mode, wind) in [($east, 0), ($half, 1)] {
                    let mut state = state(GameRule::default_tenhou(), mode, wind, $np - 1);
                    let dealer = $np - 1;
                    // 123456789p123sEE: a complete closed hand with an East pair.
                    state.players[dealer as usize].hand =
                        vec![36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 108, 109];
                    state.drawn_tile = Some(109);
                    // Avoid tenhou and bankruptcy masking the agari-yame check.
                    state.is_first_turn = false;
                    let action = state
                        ._get_legal_actions_internal(dealer)
                        .into_iter()
                        .find(|a| a.action_type == ActionType::Tsumo)
                        .expect("fixture must offer tsumo");
                    state.step(&HashMap::from([(dealer, action)]));
                    assert!(state.last_error.is_none());
                    assert!(state.players.iter().all(|p| p.score >= 0));
                    assert!(state.is_done);
                    let generated = events(&state.mjai_log);
                    assert_eq!(generated[0]["type"], "hora");
                    assert_eq!(generated[2]["type"], "end_game");
                }
            }

            #[test]
            fn single_round_and_bankruptcy_still_end() {
                let mut single = state(GameRule::default_tenhou(), $single, 0, 0);
                single._trigger_ryukyoku("kyushu_kyuhai");
                assert!(single.is_done);

                let mut bankrupt = state(GameRule::default_tenhou(), $half, 1, $np - 1);
                bankrupt.players[0].score = -1000;
                bankrupt._trigger_ryukyoku("kyushu_kyuhai");
                assert!(bankrupt.is_done);
            }
        }
    };
}

renchan_tests!(
    four_player,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    riichienv_core::hand_evaluator::HandEvaluator,
    4,
    0,
    1,
    2,
    25000,
    30000
);
renchan_tests!(
    three_player,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    riichienv_core::hand_evaluator_3p::HandEvaluator3P,
    3,
    3,
    4,
    5,
    35000,
    40000
);
