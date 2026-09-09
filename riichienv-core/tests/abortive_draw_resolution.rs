//! Abortive-draw priority, kan limits, deposits, and first-turn replay state.

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::replay::{Action as LogAction, MjaiEvent};
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};
use serde_json::Value;
use std::collections::HashMap;

fn events(log: &[String]) -> Vec<Value> {
    log.iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

fn kan(t: u8) -> Meld {
    Meld::new(
        MeldType::Ankan,
        (t * 4..t * 4 + 4).collect(),
        false,
        -1,
        None,
    )
}

macro_rules! regression_tests {
    ($module:ident, $state:ty, $legal:path, $np:expr, $mode:expr) => {
        mod $module {
            use super::*;
            use $legal;

            fn fresh() -> $state {
                let mut s = <$state>::new($mode, false, Some(229), 0, GameRule::default_tenhou());
                s._initialize_round(0, 0, 0, 0, None, None);
                for (pid, p) in s.players.iter_mut().enumerate() {
                    p.hand = [13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 29, 30]
                        .iter()
                        .map(|t| t * 4 + pid as u8)
                        .collect();
                    p.nagashi_eligible = false;
                }
                s.current_player = 0;
                s.active_players = vec![0];
                s.phase = Phase::WaitAct;
                s.is_first_turn = false;
                s.mjai_log.clear();
                s
            }

            fn action(s: &$state, pid: u8, kind: ActionType) -> Action {
                s._get_legal_actions_internal(pid)
                    .into_iter()
                    .find(|a| a.action_type == kind)
                    .expect("fixture action must be legal")
            }

            fn pass_all(s: &mut $state) {
                let acts = s
                    .active_players
                    .iter()
                    .map(|&p| (p, Action::new(ActionType::Pass, None, vec![], Some(p))))
                    .collect();
                s.step(&acts);
            }

            fn four_kans() -> $state {
                let mut s = fresh();
                // All three kan types count toward the abortive draw.
                s.players[0].melds = vec![
                    Meld::new(MeldType::Daiminkan, vec![36, 37, 38, 39], true, 2, Some(36)),
                    kan(10),
                ];
                s.players[1].melds = vec![
                    Meld::new(MeldType::Kakan, vec![44, 45, 46, 47], true, 2, Some(44)),
                    kan(12),
                ];
                s.players[0].hand = vec![52, 56, 60, 64, 68, 72, 80, 108];
                s.players[1].hand = vec![53, 57, 61, 65, 69, 73, 81];
                // Player 2 can pon East (109,110), but cannot ron it.
                s.players[2].hand = vec![109, 110, 54, 58, 62, 66, 70, 74, 82, 90, 98, 114, 118];
                s.drawn_tile = Some(108);
                s.wall.rinshan_draw_count = 4;
                s
            }

            #[test]
            fn four_kans_must_abort_after_all_pass() {
                let mut s = four_kans();
                // The discard can be ronned, so abort only after the response.
                s.players[2].hand = [18, 19, 20, 21, 22, 23, 24, 25, 26, 14, 15, 16, 27]
                    .iter()
                    .map(|t| t * 4 + 2)
                    .collect();
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(108), vec![], Some(0)),
                )]));
                assert_eq!(s.phase, Phase::WaitResponse);
                assert_eq!(action(&s, 2, ActionType::Ron).tile, Some(108));
                pass_all(&mut s);
                assert!(s.last_error.is_none());
                assert_eq!(s.honba, 1);
                assert!(
                    events(&s.mjai_log)
                        .iter()
                        .any(|e| e["reason"] == "suukansansen"),
                    "four kans were bypassed after all-pass"
                );
            }

            #[test]
            fn pon_must_not_take_priority_over_four_kans() {
                let mut s = four_kans();
                let (claims, _) = s._get_claim_actions_for_player(2, 0, 108);
                assert!(!claims.iter().any(|a| a.action_type == ActionType::Pon));
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(108), vec![], Some(0)),
                )]));
                assert!(s.last_error.is_none());
                assert_eq!(s.honba, 1);
                assert!(!events(&s.mjai_log).iter().any(|e| e["type"] == "pon"));
                assert!(
                    events(&s.mjai_log)
                        .iter()
                        .any(|e| e["reason"] == "suukansansen"),
                    "pon delayed four-kan abort"
                );
            }

            #[test]
            fn fifth_ankan_must_be_forbidden() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(0), kan(8), kan(9), kan(10)];
                s.players[0].hand = vec![108];
                s.players[1].hand = vec![44, 45, 46, 47, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88];
                s.current_player = 1;
                s.active_players = vec![1];
                s.drawn_tile = Some(47);
                s.wall.rinshan_draw_count = 4;
                assert!(
                    !s._get_legal_actions_internal(1)
                        .iter()
                        .any(|a| a.action_type == ActionType::Ankan)
                );
                s.players[0].melds.pop();
                s.players[0].hand.extend([112, 116, 120]);
                assert_eq!(action(&s, 1, ActionType::Ankan).tile, Some(44));
            }

            #[test]
            fn fifth_kakan_must_be_forbidden() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(0), kan(8), kan(9), kan(10)];
                s.players[0].hand = vec![108];
                s.players[1].melds = vec![Meld::new(
                    MeldType::Pon,
                    vec![44, 45, 46],
                    true,
                    0,
                    Some(44),
                )];
                s.players[1].hand = vec![47, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88];
                s.current_player = 1;
                s.active_players = vec![1];
                s.drawn_tile = Some(47);
                assert!(
                    !s._get_legal_actions_internal(1)
                        .iter()
                        .any(|a| a.action_type == ActionType::Kakan)
                );
                s.players[0].melds.pop();
                s.players[0].hand.extend([112, 116, 120]);
                assert_eq!(action(&s, 1, ActionType::Kakan).tile, Some(47));
            }

            #[test]
            fn fifth_daiminkan_must_be_forbidden() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(0), kan(8), kan(9), kan(10)];
                s.players[0].hand = vec![108, 47];
                s.players[1].hand = vec![44, 45, 46, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88];
                s.drawn_tile = Some(47);
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(47), vec![], Some(0)),
                )]));
                assert!(s.last_error.is_none());
                assert!(
                    !s._get_legal_actions_internal(1)
                        .iter()
                        .any(|a| a.action_type == ActionType::Daiminkan)
                );
                s.players[0].melds.pop();
                s.players[0].hand.extend([112, 116, 120]);
                let (claims, _) = s._get_claim_actions_for_player(1, 0, 47);
                assert!(
                    claims
                        .iter()
                        .any(|a| a.action_type == ActionType::Daiminkan)
                );
            }

            #[test]
            fn fourth_kan_allows_rinshan_win_before_abort() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(0), kan(8)];
                s.players[1].melds = vec![kan(9)];
                s.players[1].hand.truncate(10);
                s.players[0].hand = vec![44, 45, 46, 47, 72, 76, 80, 108];
                s.drawn_tile = Some(47);
                s.wall.rinshan_draw_count = 3;
                s.wall.tiles[0] = 109;
                let ankan = action(&s, 0, ActionType::Ankan);
                s.step(&HashMap::from([(0, ankan)]));
                if s.phase == Phase::WaitResponse {
                    pass_all(&mut s);
                }
                assert!(s.last_error.is_none());
                assert_eq!(s.drawn_tile, Some(109));
                assert_eq!(s.wall.rinshan_draw_count, 4);
                assert_eq!(s.honba, 0);
                assert!(!events(&s.mjai_log).iter().any(|e| e["type"] == "ryukyoku"));

                let mut win = s.clone();
                let tsumo = action(&win, 0, ActionType::Tsumo);
                win.step(&HashMap::from([(0, tsumo)]));
                let log = events(&win.mjai_log);
                assert!(log.iter().any(|e| e["type"] == "hora"));
                assert!(!log.iter().any(|e| e["type"] == "ryukyoku"));

                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(109), vec![], Some(0)),
                )]));
                if s.phase == Phase::WaitResponse {
                    pass_all(&mut s);
                }
                assert!(s.last_error.is_none());
                assert!(
                    events(&s.mjai_log)
                        .iter()
                        .any(|e| e["reason"] == "suukansansen")
                );
            }

            #[test]
            fn single_player_four_kans_should_continue() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(0), kan(8), kan(9), kan(10)];
                s.players[0].hand = vec![108, 47];
                s.drawn_tile = Some(47);
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(47), vec![], Some(0)),
                )]));
                if s.phase == Phase::WaitResponse {
                    pass_all(&mut s);
                }
                assert!(!events(&s.mjai_log).iter().any(|e| e["type"] == "ryukyoku"));
            }

            fn opening() -> $state {
                let mut s = fresh();
                let hand = vec![
                    "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "2p", "3p", "4p", "5p",
                ];
                let start = serde_json::json!({"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":vec![35000;$np],"dora_marker":"8s","tehais":vec![hand;$np]});
                s.apply_mjai_event(serde_json::from_value(start).unwrap());
                s
            }

            #[test]
            fn mjai_second_turn_must_not_offer_kyushu() {
                let mut s = opening();
                for (pid, pai) in ["N", "P", "F", "C"].iter().enumerate().take($np) {
                    s.apply_mjai_event(MjaiEvent::Tsumo {
                        actor: pid,
                        pai: pai.to_string(),
                    });
                    assert!(
                        s._get_legal_actions_internal(pid as u8)
                            .iter()
                            .any(|a| a.action_type == ActionType::KyushuKyuhai)
                    );
                    s.apply_mjai_event(MjaiEvent::Dahai {
                        actor: pid,
                        pai: pai.to_string(),
                        tsumogiri: true,
                    });
                }
                s.apply_mjai_event(MjaiEvent::Tsumo {
                    actor: 0,
                    pai: "N".to_string(),
                });
                assert!(!s.is_first_turn);
                assert_eq!(s.turn_count, $np);
                assert!(
                    !s._get_legal_actions_internal(0)
                        .iter()
                        .any(|a| a.action_type == ActionType::KyushuKyuhai)
                );
            }

            #[test]
            fn log_action_kyushu_only_on_first_draw() {
                let mut s = opening();
                for pid in 0..$np {
                    let tile = 120 + pid as u8 * 4;
                    s.apply_log_action(&LogAction::DealTile {
                        seat: pid,
                        tile,
                        doras: None,
                        left_tile_count: None,
                    });
                    assert!(s.is_first_turn);
                    assert!(
                        s._get_legal_actions_internal(pid as u8)
                            .iter()
                            .any(|a| a.action_type == ActionType::KyushuKyuhai)
                    );
                    s.apply_log_action(&LogAction::DiscardTile {
                        seat: pid,
                        tile,
                        is_liqi: false,
                        is_wliqi: false,
                        doras: None,
                    });
                }
                s.apply_log_action(&LogAction::DealTile {
                    seat: 0,
                    tile: 120,
                    doras: None,
                    left_tile_count: None,
                });
                assert!(!s.is_first_turn);
                assert!(
                    !s._get_legal_actions_internal(0)
                        .iter()
                        .any(|a| a.action_type == ActionType::KyushuKyuhai)
                );
            }

            #[test]
            fn kyushu_after_riichi_has_no_additional_payment() {
                let mut s = fresh();
                s.is_first_turn = true;
                s.players[0].hand = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 31]
                    .iter()
                    .map(|t| t * 4)
                    .collect();
                s.players[1].hand = [0, 8, 9, 17, 18, 26, 27, 28, 29, 10, 11, 12, 13]
                    .iter()
                    .map(|t| t * 4 + 1)
                    .collect();
                s.drawn_tile = Some(124);
                let starting_score = s.players[0].score;
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Riichi, None, vec![], Some(0)),
                )]));
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(124), vec![], Some(0)),
                )]));
                if s.phase == Phase::WaitResponse {
                    pass_all(&mut s);
                }
                assert!(s.last_error.is_none());
                let kyushu = action(&s, 1, ActionType::KyushuKyuhai);
                s.step(&HashMap::from([(1, kyushu)]));
                assert!(s.last_error.is_none());
                assert_eq!(s.riichi_sticks, 1);
                assert_eq!(s.players[0].score, starting_score - 1000);
                let log = events(&s.mjai_log);
                assert_eq!(
                    log.iter().filter(|e| e["type"] == "reach_accepted").count(),
                    1
                );
                let draw = log.iter().find(|e| e["reason"] == "kyushu_kyuhai").unwrap();
                assert_eq!(draw["deltas"], serde_json::json!(vec![0; $np]));
            }

            #[test]
            fn simulator_kyushu_requires_first_turn_and_nine_distinct_types() {
                let mut s = opening();
                for pid in 0..$np {
                    s.players[pid].hand = [0, 8, 9, 17, 18, 26, 27, 28, 29, 10, 11, 12, 13]
                        .iter()
                        .map(|t| t * 4 + pid as u8)
                        .collect();
                }
                s.current_player = 0;
                s.active_players = vec![0];
                s.phase = Phase::WaitAct;
                s.needs_tsumo = false;
                for pid in 0..$np {
                    // Consume the draw made by step(), then use a safe honor.
                    s.players[pid].hand.truncate(13);
                    let tile = 120 + pid as u8 * 4;
                    s.players[pid].hand.push(tile);
                    s.drawn_tile = Some(tile);
                    assert!(
                        s._get_legal_actions_internal(pid as u8)
                            .iter()
                            .any(|a| a.action_type == ActionType::KyushuKyuhai)
                    );
                    s.step(&HashMap::from([(
                        pid as u8,
                        Action::new(ActionType::Discard, Some(tile), vec![], Some(pid as u8)),
                    )]));
                    if s.phase == Phase::WaitResponse {
                        pass_all(&mut s);
                    }
                    assert!(s.last_error.is_none());
                }
                assert!(
                    !s._get_legal_actions_internal(0)
                        .iter()
                        .any(|a| a.action_type == ActionType::KyushuKyuhai)
                );
                // Eight distinct terminals/honors do not qualify even with duplicates.
                s.is_first_turn = true;
                s.players[0].discards.clear();
                s.players[0].hand = vec![0, 1, 32, 36, 68, 72, 104, 108, 112, 40, 44, 48, 52, 56];
                s.drawn_tile = Some(56);
                assert!(
                    !s._get_legal_actions_internal(0)
                        .iter()
                        .any(|a| a.action_type == ActionType::KyushuKyuhai)
                );
            }

            #[test]
            fn four_kans_with_no_claims_abort_without_payments() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(9), kan(10)];
                s.players[1].melds = vec![kan(11), kan(12)];
                s.players[0].hand = vec![52, 56, 60, 64, 68, 72, 80, 108];
                s.players[1].hand.truncate(7);
                s.drawn_tile = Some(108);
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(108), vec![], Some(0)),
                )]));
                assert!(s.last_error.is_none());
                assert_eq!(s.honba, 1);
                assert_eq!(s.oya, 0);
                let ev = events(&s.mjai_log);
                assert!(
                    ev.iter().any(|e| e["reason"] == "suukansansen"
                        && e["deltas"] == serde_json::json!(vec![0; $np]))
                );
            }

            #[test]
            fn ron_takes_priority_over_four_kan_abort() {
                let mut s = fresh();
                s.players[0].melds = vec![kan(9), kan(10)];
                s.players[1].melds = vec![kan(11), kan(12)];
                s.players[0].hand = vec![52, 56, 60, 64, 68, 72, 80, 108];
                s.players[1].hand.truncate(7);
                s.players[2].hand = [18, 19, 20, 21, 22, 23, 24, 25, 26, 14, 15, 16, 27]
                    .iter()
                    .map(|t| t * 4 + 2)
                    .collect();
                s.drawn_tile = Some(108);
                s.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(108), vec![], Some(0)),
                )]));
                let a = action(&s, 2, ActionType::Ron);
                s.step(&HashMap::from([(2, a)]));
                assert!(s.last_error.is_none());
                let ev = events(&s.mjai_log);
                assert!(ev.iter().any(|e| e["type"] == "hora"));
                assert!(!ev.iter().any(|e| e["type"] == "ryukyoku"));
            }
        }
    }
}
regression_tests!(
    four,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    4,
    2
);
regression_tests!(
    three,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    5
);

#[cfg(test)]
mod riichi {
    use super::*;
    use riichienv_core::state::{GameState, legal_actions::GameStateLegalActions};

    fn fourth_riichi(rule: GameRule, ron_on_east: bool) -> GameState {
        let mut s = GameState::new(2, false, Some(229), 0, rule);
        s._initialize_round(0, 0, 0, 0, None, None);
        for pid in 0..3 {
            let p = &mut s.players[pid];
            p.hand = [
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                if ron_on_east { 27 } else { 21 },
            ]
            .iter()
            .map(|t| t * 4 + pid as u8)
            .collect();
            p.riichi_declared = true;
            p.score -= 1000;
            p.score_delta = -1000;
            p.discards = vec![112 + pid as u8];
        }
        s.riichi_sticks = 3;
        s.players[3].hand = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            .iter()
            .map(|t| t * 4 + 3)
            .collect();
        s.players[3].hand.push(111);
        s.drawn_tile = Some(111);
        s.current_player = 3;
        s.active_players = vec![3];
        s.is_first_turn = false;
        s.turn_count = 11;
        s.mjai_log.clear();
        s.step(&HashMap::from([(
            3,
            Action::new(ActionType::Riichi, None, vec![], Some(3)),
        )]));
        s.step(&HashMap::from([(
            3,
            Action::new(ActionType::Discard, Some(111), vec![], Some(3)),
        )]));
        assert!(s.last_error.is_none());
        if ron_on_east {
            assert_eq!(s.phase, Phase::WaitResponse);
            for pid in 0..3 {
                assert!(
                    s._get_legal_actions_internal(pid)
                        .iter()
                        .any(|a| a.action_type == ActionType::Ron)
                );
            }
        }
        s
    }

    #[test]
    fn fourth_riichi_ron_pass_must_abort() {
        let mut s = fourth_riichi(GameRule::default_tenhou(), true);
        s.step(&HashMap::from_iter((0..3).map(|pid| {
            (pid, Action::new(ActionType::Pass, None, vec![], Some(pid)))
        })));
        assert_eq!(s.honba, 1);
        assert_eq!(s.riichi_sticks, 4);
        assert!(
            events(&s.mjai_log)
                .iter()
                .any(|e| e["reason"] == "suucha_riichi")
        );
    }

    #[test]
    fn three_ron_rule_difference() {
        for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
            let mut s = fourth_riichi(rule, true);
            s.step(&HashMap::from_iter((0..3).map(|pid| {
                (
                    pid,
                    Action::new(ActionType::Ron, Some(111), vec![], Some(pid)),
                )
            })));
            let ev = events(&s.mjai_log);
            assert_eq!(
                ev.iter().filter(|e| e["type"] == "hora").count(),
                if rule.sanchaho_is_draw { 0 } else { 3 }
            );
            assert_eq!(
                ev.iter().filter(|e| e["reason"] == "sanchaho").count(),
                usize::from(rule.sanchaho_is_draw)
            );
        }
    }

    #[test]
    fn sanchaho_declaration_discard_does_not_pay_riichi() {
        let mut s = fourth_riichi(GameRule::default_tenhou(), true);
        let before = s.players[3].score;
        s.step(&HashMap::from_iter((0..3).map(|pid| {
            (
                pid,
                Action::new(ActionType::Ron, Some(111), vec![], Some(pid)),
            )
        })));
        assert!(
            !events(&s.mjai_log)
                .iter()
                .any(|e| e["type"] == "reach_accepted")
        );
        assert_eq!(s.players[3].score, before);
        assert_eq!(s.riichi_sticks, 3);
    }

    #[test]
    fn four_riichi_without_ron_repeats_and_keeps_four_sticks() {
        let s = fourth_riichi(GameRule::default_tenhou(), false);
        assert_eq!(s.honba, 1);
        assert_eq!(s.riichi_sticks, 4);
        assert!(s.players.iter().all(|p| p.score == 24000));
        assert!(
            events(&s.mjai_log)
                .iter()
                .any(|e| e["reason"] == "suucha_riichi")
        );
    }

    #[test]
    fn abortive_deltas_must_not_repeat_reach_deductions() {
        let s = fourth_riichi(GameRule::default_tenhou(), false);
        let ev = events(&s.mjai_log);
        let draw = ev.iter().find(|e| e["type"] == "ryukyoku").unwrap();
        assert_eq!(draw["deltas"], serde_json::json!([0, 0, 0, 0]));
    }
}

#[test]
fn four_winds_without_calls_repeats() {
    use riichienv_core::state::GameState;
    let mut s = GameState::new(2, false, Some(229), 0, GameRule::default_tenhou());
    s._initialize_round(0, 0, 0, 0, None, None);
    for pid in 0..4 {
        s.players[pid].hand = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            .iter()
            .map(|t| t * 4 + pid as u8)
            .collect();
    }
    s.mjai_log.clear();
    for pid in 0..4 {
        s.players[pid].hand.truncate(13);
        let tile = 108 + pid as u8;
        s.players[pid].hand.push(tile);
        s.drawn_tile = Some(tile);
        s.step(&HashMap::from([(
            pid as u8,
            Action::new(ActionType::Discard, Some(tile), vec![], Some(pid as u8)),
        )]));
    }
    assert!(s.last_error.is_none());
    assert_eq!(s.honba, 1);
    assert!(
        events(&s.mjai_log)
            .iter()
            .any(|e| e["reason"] == "sufuurenta")
    );
}

#[test]
fn chi_must_not_take_priority_over_four_kans() {
    use riichienv_core::state::{GameState, legal_actions::GameStateLegalActions};
    let mut s = GameState::new(2, false, Some(42), 0, GameRule::default_tenhou());
    s._initialize_round(0, 0, 0, 0, None, None);
    s.players[0].melds = vec![kan(0), kan(8)];
    s.players[1].melds = vec![kan(9), kan(10)];
    s.players[1].hand = vec![56, 60, 64, 68, 72, 80, 108];
    let (claims, _) = s._get_claim_actions_for_player(1, 0, 52);
    assert!(!claims.iter().any(|a| a.action_type == ActionType::Chi));
    s.players[0].melds.pop();
    let (claims, _) = s._get_claim_actions_for_player(1, 0, 52);
    assert!(claims.iter().any(|a| a.action_type == ActionType::Chi));
}

#[test]
fn mjai_kita_must_disable_kyushu() {
    use riichienv_core::state_3p::{GameState3P, legal_actions::GameState3PLegalActions};
    let mut s = GameState3P::new(5, false, Some(229), 0, GameRule::default_tenhou());
    s._initialize_round(0, 0, 0, 0, None, None);
    s.players[0].hand = vec![0, 32, 36, 68, 72, 104, 108, 112, 116, 40, 44, 48, 52, 120];
    let mut simulated = s.clone();
    simulated.drawn_tile = Some(120);
    simulated.step(&HashMap::from([(
        0,
        Action::new(ActionType::Kita, Some(120), vec![], Some(0)),
    )]));
    assert!(!simulated.is_first_turn);
    assert!(
        !simulated
            ._get_legal_actions_internal(0)
            .iter()
            .any(|a| a.action_type == ActionType::KyushuKyuhai)
    );
    s.apply_mjai_event(MjaiEvent::Kita { actor: 0 });
    assert!(!s.is_first_turn);
    s.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 0,
        pai: "P".to_string(),
    });
    assert!(
        !s._get_legal_actions_internal(0)
            .iter()
            .any(|a| a.action_type == ActionType::KyushuKyuhai)
    );
}

#[test]
fn kita_does_not_count_toward_four_kan_limit() {
    use riichienv_core::state_3p::{GameState3P, legal_actions::GameState3PLegalActions};
    let mut s = GameState3P::new(5, false, Some(42), 0, GameRule::default_tenhou());
    s._initialize_round(0, 0, 0, 0, None, None);
    s.players[0].melds = vec![kan(0), kan(8), kan(9)];
    s.players[0].hand = vec![44, 45, 46, 47, 108];
    s.players[0].kita_tiles = vec![120, 121, 122, 123];
    s.wall.rinshan_draw_count = 7;
    s.drawn_tile = Some(47);
    let ankan = s
        ._get_legal_actions_internal(0)
        .into_iter()
        .find(|a| a.action_type == ActionType::Ankan)
        .unwrap();
    s.step(&HashMap::from([(0, ankan)]));
    assert!(s.last_error.is_none());
    assert_eq!(s.wall.rinshan_draw_count, 8);
    assert_eq!(s.players[0].melds.len(), 4);
    assert!(!s.is_done);
}

#[test]
fn three_player_three_winds_and_three_riichi_continue() {
    use riichienv_core::state_3p::GameState3P;
    let mut s = GameState3P::new(5, false, Some(229), 0, GameRule::default_tenhou());
    s._initialize_round(0, 0, 0, 0, None, None);
    for pid in 0..3 {
        s.players[pid].hand = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            .iter()
            .map(|t| t * 4 + pid as u8)
            .collect();
    }
    s.mjai_log.clear();
    for pid in 0..3 {
        s.players[pid].hand.truncate(13);
        let tile = 108 + pid as u8;
        s.players[pid].hand.push(tile);
        s.drawn_tile = Some(tile);
        s.step(&HashMap::from([(
            pid as u8,
            Action::new(ActionType::Riichi, None, vec![], Some(pid as u8)),
        )]));
        s.step(&HashMap::from([(
            pid as u8,
            Action::new(ActionType::Discard, Some(tile), vec![], Some(pid as u8)),
        )]));
        assert!(s.last_error.is_none());
    }
    assert_eq!(s.honba, 0);
    assert_eq!(s.riichi_sticks, 3);
    assert!(s.players.iter().all(|p| p.riichi_declared));
    assert!(!events(&s.mjai_log).iter().any(|e| e["type"] == "ryukyoku"));
}
