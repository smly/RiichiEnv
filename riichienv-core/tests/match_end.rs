//! Match-end regressions found by replaying raw Mahjong Soul records.
//! Also matches the sudden-death and leftover-deposit rules at
//! https://tenhou.net/man/#RULE.

use riichienv_core::rule::GameRule;
use serde_json::{Value, json};

macro_rules! match_end_tests {
    ($module:ident, $state:ty, $np:expr, $single:expr, $east:expr, $half:expr, $start:expr, $target:expr) => {
        mod $module {
            use super::*;

            fn state(mode: u8, wind: u8, dealer: u8, scores: Vec<i32>, pot: u32) -> $state {
                let mut s = <$state>::new(mode, false, Some(230), 0, GameRule::default_mjsoul());
                s._initialize_round(dealer, wind, 2, pot, None, Some(scores));
                s.mjai_log.clear();
                s
            }

            fn exhaust(s: &mut $state, tenpai: bool) {
                for (seat, p) in s.players.iter_mut().enumerate() {
                    let tiles: &[u8] = if tenpai {
                        &[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                    } else {
                        &[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 27]
                    };
                    p.hand = tiles.iter().map(|t| t * 4 + seat as u8).collect();
                    p.nagashi_eligible = false;
                }
                s.drawn_tile = None;
                s.wall.drawable_count = 0;
                s._deal_next();
            }

            fn scores(s: &$state) -> Vec<i32> {
                s.players.iter().map(|p| p.score).collect()
            }

            #[test]
            fn extension_tenpai_dealer_repeats_when_child_is_top() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    for (mode, wind) in [($east, 1), ($half, 2)] {
                        for dealer in [0, $np - 1] {
                            let mut initial = vec![$start; $np];
                            initial[(dealer as usize + 1) % $np] = $target;
                            let mut s = state(mode, wind, dealer, initial.clone(), 2);
                            s.rule = rule;
                            exhaust(&mut s, true);
                            assert!(
                                !s.is_done,
                                "child reaching the target must not cancel renchan"
                            );
                            assert_eq!((s.round_wind, s.oya, s.honba), (wind, dealer, 3));
                            assert_eq!(scores(&s), initial);
                            assert_eq!(s.riichi_sticks, 2);
                        }
                    }
                }
            }

            #[test]
            fn extension_dealer_win_repeats_when_child_is_top() {
                for (mode, wind) in [($east, 1), ($half, 2)] {
                    let mut after_double_ron = vec![$start; $np];
                    after_double_ron[1] = $target + 1000;
                    let mut s = state(mode, wind, 0, after_double_ron.clone(), 0);
                    // The settlement entry point receives renchan=true when
                    // the dealer is among multiple ron winners.
                    s._initialize_next_round(true, false);
                    assert!(!s.is_done);
                    assert_eq!((s.round_wind, s.oya, s.honba), (wind, 0, 3));
                    assert_eq!(scores(&s), after_double_ron);
                }
            }

            #[test]
            fn extension_top_dealer_stops_only_at_target_with_seat_tiebreak() {
                for (mode, wind) in [($east, 1), ($half, 2)] {
                    for dealer in [0, $np - 1] {
                        for score in [$target - 100, $target] {
                            for tied in [false, true] {
                                for draw in [false, true] {
                                    let mut initial = vec![$start - 1000; $np];
                                    initial[dealer as usize] = score;
                                    if tied {
                                        initial[(dealer as usize + 1) % $np] = score;
                                    }
                                    let mut s = state(mode, wind, dealer, initial, 0);
                                    s._initialize_next_round(true, draw);
                                    assert_eq!(
                                        s.is_done,
                                        score >= $target && (!tied || dealer == 0)
                                    );
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn extension_rotating_dealer_obeys_target_and_round_limit() {
                for (mode, wind) in [($east, 1), ($half, 2)] {
                    for dealer in [0, $np - 1] {
                        for reached in [false, true] {
                            let mut initial = vec![$start; $np];
                            if reached {
                                initial[1] = $target;
                            }
                            let mut s = state(mode, wind, dealer, initial, 0);
                            exhaust(&mut s, false);
                            assert_eq!(s.is_done, reached || dealer == $np - 1);
                        }
                    }
                }
            }

            #[test]
            fn final_draw_awards_deposits_once_and_keeps_hand_payments_separate() {
                for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    for quiet in [false, true] {
                        for pot in [0, 1, 4] {
                            let mut initial = vec![$start - 10000; $np];
                            initial[1] = $target + 5000;
                            let mut expected = initial.clone();
                            expected[1] += pot as i32 * 1000;
                            let mut s = state($half, 1, $np - 1, initial, pot);
                            s.rule = rule;
                            s.skip_mjai_logging = quiet;
                            exhaust(&mut s, false);
                            assert!(s.is_done);
                            assert_eq!(scores(&s), expected);
                            assert_eq!(s.riichi_sticks, 0);
                            assert_eq!(s.players[1].score_delta, pot as i32 * 1000);
                            if !quiet {
                                let log: Vec<Value> = s
                                    .mjai_log
                                    .iter()
                                    .map(|line| serde_json::from_str(line).unwrap())
                                    .collect();
                                assert_eq!(log[0]["type"], "ryukyoku");
                                assert_eq!(log[0]["deltas"], json!(vec![0; $np]));
                                assert_eq!(log[1]["type"], "end_kyoku");
                                assert_eq!(log[2]["type"], "end_game");
                            }
                            s._initialize_next_round(false, true);
                            assert_eq!(scores(&s), expected, "settlement must not run twice");
                        }
                    }
                }
            }

            #[test]
            fn tied_top_deposit_goes_to_earlier_seat_not_current_dealer() {
                let mut initial = vec![$start - 10000; $np];
                initial[1] = $target;
                initial[$np - 1] = $target;
                let mut expected = initial.clone();
                expected[1] += 2000;
                let mut s = state($half, 1, $np - 1, initial, 2);
                exhaust(&mut s, false);
                assert!(s.is_done);
                assert_eq!(scores(&s), expected);
                assert_eq!(s.riichi_sticks, 0);
            }

            #[test]
            fn bankruptcy_also_settles_the_pot() {
                let mut initial = vec![$start; $np];
                initial[0] = -1000;
                initial[1] = $target;
                let mut s = state($half, 0, 0, initial, 3);
                exhaust(&mut s, false);
                assert!(s.is_done);
                assert_eq!(s.players[0].score, -1000);
                assert_eq!(s.players[1].score, $target + 3000);
                assert_eq!(s.riichi_sticks, 0);
            }

            #[test]
            fn unclaimed_pot_does_not_count_toward_the_end_threshold() {
                let mut initial = vec![$start - 1000; $np];
                initial[$np - 1] = $target - 100;
                let mut s = state($half, 1, $np - 1, initial.clone(), 4);
                exhaust(&mut s, false);
                assert!(!s.is_done);
                assert_eq!((s.round_wind, s.oya), (2, 0));
                assert_eq!(scores(&s), initial);
                assert_eq!(s.riichi_sticks, 4);
            }

            #[test]
            fn single_round_preserves_unclaimed_deposits() {
                let initial = vec![$start; $np];
                let mut s = state($single, 0, 0, initial.clone(), 2);
                exhaust(&mut s, false);
                assert!(s.is_done);
                assert_eq!(scores(&s), initial);
                assert_eq!(s.riichi_sticks, 2);
            }
        }
    };
}

match_end_tests!(
    four_player,
    riichienv_core::state::GameState,
    4,
    0,
    1,
    2,
    25000,
    30000
);
match_end_tests!(
    three_player,
    riichienv_core::state_3p::GameState3P,
    3,
    3,
    4,
    5,
    35000,
    40000
);
