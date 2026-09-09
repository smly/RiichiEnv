//! Nagashi mangan replaces exhaustive-draw payments, not tenpai-based renchan.
//! Regression coverage for https://github.com/smly/RiichiEnv/issues/214.
//! Rule reference: https://tenhou.net/man/#RULE.

use riichienv_core::rule::GameRule;
use serde_json::{Value, json};

// Exercise the same settlement contract through both independent game engines.
macro_rules! draw_tests {
    ($module:ident, $state:ty, $evaluator:ty, $mode:expr, $np:expr, $score:expr) => {
        mod $module {
            use super::*;

            fn state(rule: GameRule, dealer: u8, tenpai: bool, nagashi: &[u8]) -> $state {
                let mut state = <$state>::new($mode, false, Some(214), 0, rule);
                state._initialize_round(dealer, 0, 3, 2, None, Some(vec![$score; $np]));
                for (pid, player) in state.players.iter_mut().enumerate() {
                    // 123456789p124sE is noten; 123456789p1234s waits on 1s/4s.
                    // Each seat uses a different physical copy of every tile.
                    let tiles: &[u8] = if pid == dealer as usize && tenpai {
                        &[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                    } else {
                        &[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 27]
                    };
                    player.hand = tiles.iter().map(|t| t * 4 + pid as u8).collect();
                    player.nagashi_eligible = nagashi.contains(&(pid as u8));
                    assert_eq!(
                        <$evaluator>::new(player.hand.clone(), vec![]).is_tenpai(),
                        pid == dealer as usize && tenpai,
                        "fixture tenpai for seat {pid}"
                    );
                }
                state.drawn_tile = None;
                state.wall.drawable_count = 0;
                state.mjai_log.clear();
                state
            }

            fn settle(
                mut state: $state,
                reason: &str,
                deltas: &[i32],
                next_dealer: u8,
                next_wind: u8,
            ) {
                if matches!(reason, "exhaustive_draw" | "nagashimangan") {
                    // Use the actual exhausted-wall entry point.
                    state._deal_next();
                } else {
                    state._trigger_ryukyoku(reason);
                }
                assert!(!state.is_done);
                assert_eq!(state.oya, next_dealer);
                assert_eq!(state.kyoku_idx, next_dealer);
                assert_eq!(state.round_wind, next_wind);
                assert_eq!(state.honba, 4, "every draw increments honba");
                assert_eq!(state.riichi_sticks, 2, "draws carry the riichi pot");
                let scores: Vec<i32> = deltas.iter().map(|delta| $score + delta).collect();
                assert_eq!(
                    state.players.iter().map(|p| p.score).collect::<Vec<_>>(),
                    scores
                );
                let events: Vec<Value> = state
                    .mjai_log
                    .iter()
                    .map(|event| serde_json::from_str(event).unwrap())
                    .collect();
                assert_eq!(events[0]["type"], "ryukyoku");
                assert_eq!(events[0]["deltas"], json!(deltas));
                assert_eq!(events[0]["reason"], reason);
                assert_eq!(events[1]["type"], "end_kyoku");
                assert_eq!(events[2]["type"], "start_kyoku");
                assert_eq!(events[2]["oya"], next_dealer);
                assert_eq!(events[2]["honba"], 4);
                assert_eq!(events[2]["kyotaku"], 2);
                assert_eq!(events[2]["scores"], json!(scores));
            }

            #[test]
            fn dealer_nagashi_without_tenpai_rotates() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    let mut deltas = vec![-4000; $np];
                    deltas[0] = 4000 * ($np as i32 - 1);
                    settle(state(rule, 0, false, &[0]), "nagashimangan", &deltas, 1, 0);
                }
            }

            #[test]
            fn non_dealer_nagashi_with_tenpai_dealer_repeats() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    let mut deltas = vec![-2000; $np];
                    deltas[0] = -4000;
                    deltas[1] = 4000 + 2000 * ($np as i32 - 2);
                    settle(state(rule, 0, true, &[1]), "nagashimangan", &deltas, 0, 0);
                }
            }

            #[test]
            fn dealer_nagashi_with_tenpai_repeats() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    let mut deltas = vec![-4000; $np];
                    deltas[0] = 4000 * ($np as i32 - 1);
                    settle(state(rule, 0, true, &[0]), "nagashimangan", &deltas, 0, 0);
                }
            }

            #[test]
            fn non_dealer_nagashi_without_tenpai_dealer_rotates() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    let mut deltas = vec![-2000; $np];
                    deltas[0] = -4000;
                    deltas[1] = 4000 + 2000 * ($np as i32 - 2);
                    settle(state(rule, 0, false, &[1]), "nagashimangan", &deltas, 1, 0);
                }
            }

            #[test]
            fn multiple_nagashi_do_not_override_dealer_tenpai() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    let mut deltas = vec![-6000; $np];
                    deltas[0] = 4000 * ($np as i32 - 2);
                    deltas[1] = 2000 * ($np as i32 - 2);
                    for tenpai in [false, true] {
                        settle(
                            state(rule, 0, tenpai, &[0, 1]),
                            "nagashimangan",
                            &deltas,
                            if tenpai { 0 } else { 1 },
                            0,
                        );
                    }
                }
            }

            #[test]
            fn noten_dealer_nagashi_advances_to_next_wind() {
                let dealer = $np as u8 - 1;
                let mut deltas = vec![-4000; $np];
                deltas[dealer as usize] = 4000 * ($np as i32 - 1);
                settle(
                    state(GameRule::default_tenhou(), dealer, false, &[dealer]),
                    "nagashimangan",
                    &deltas,
                    0,
                    1,
                );
            }

            #[test]
            fn ordinary_draw_keeps_tenpai_payments_and_renchan() {
                for tenpai in [false, true] {
                    let mut deltas = vec![0; $np];
                    if tenpai {
                        deltas.fill(-1000);
                        deltas[0] = 1000 * ($np as i32 - 1);
                    }
                    settle(
                        state(GameRule::default_tenhou(), 0, tenpai, &[]),
                        "exhaustive_draw",
                        &deltas,
                        if tenpai { 0 } else { 1 },
                        0,
                    );
                }
            }

            #[test]
            fn abortive_draw_repeats_without_nagashi_payments() {
                settle(
                    state(GameRule::default_tenhou(), 0, false, &[1]),
                    "kyushu_kyuhai",
                    &[0; $np],
                    0,
                    0,
                );
            }
        }
    };
}

draw_tests!(
    four_player,
    riichienv_core::state::GameState,
    riichienv_core::hand_evaluator::HandEvaluator,
    2,
    4,
    25000
);
draw_tests!(
    three_player,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::hand_evaluator_3p::HandEvaluator3P,
    5,
    3,
    35000
);
