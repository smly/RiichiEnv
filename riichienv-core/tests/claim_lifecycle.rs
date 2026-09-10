use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};
use std::collections::HashMap;

macro_rules! claim_tests {
    ($name:ident, $state:path, $legal:path, $mode:expr, $np:expr, $rule:expr $(, $kita:ident)?) => {
        mod $name {
            use super::*;
            use $legal;
            type State = $state;

            fn setup(followup: ActionType) -> State {
                let mut s = State::new($mode, false, Some(42), 0, $rule);
                for p in &mut s.players {
                    p.reset_round();
                }
                let draw = match followup {
                    ActionType::Kakan => {
                        // 234678p23456s EE: can chi 5p in 4P, but only ron souzu.
                        s.players[1].hand = vec![40, 44, 48, 56, 60, 64, 76, 80, 84, 89, 92, 108, 109];
                        s.players[2].melds = vec![Meld::new(
                            MeldType::Pon, vec![96, 97, 98], true, 0, Some(96),
                        )];
                        s.players[0].discards.push(96);
                        99
                    }
                    ActionType::Ankan => {
                        // Kokushi missing White; can rob an ankan under MjSoul rules.
                        s.players[1].hand = vec![0, 1, 32, 36, 68, 72, 104, 108, 112, 116, 120, 128, 132];
                        s.players[2].hand = vec![124, 125, 126];
                        127
                    }
                    ActionType::Kita => {
                        // Six pairs and a singleton North.
                        s.players[1].hand = vec![40, 41, 44, 45, 60, 61, 72, 73, 80, 81, 108, 109, 120];
                        123
                    }
                    _ => unreachable!(),
                };
                s.players[2].hand.extend([53, 54, 55]);
                let reserved: Vec<_> = s.players.iter()
                    .flat_map(|p| {
                        p.hand.iter().copied()
                            .chain(p.melds.iter().flat_map(|m| m.tiles.iter().copied()))
                    })
                    .chain([52, draw, 119])
                    .collect();
                let mut pool: Vec<u8> = (0..136)
                    .filter(|t| !reserved.contains(t) && ($np == 4 || *t < 4 || *t >= 32))
                    .collect();
                for p in &mut s.players {
                    let missing = 13 - 3 * p.melds.len() - p.hand.len();
                    p.hand.extend(pool.drain(..missing));
                }
                s.players[0].hand.push(52);
                pool.splice(..0, [draw, 119]);
                // load_wall also initializes the 3P dora/ura indicator arrays.
                s.wall.load_wall(pool.into_iter().rev().collect());
                s.wall.drawable_count = (s.wall.tiles.len() - 14) as u8;
                s.current_player = 0;
                s.active_players = vec![0];
                s.drawn_tile = Some(52);
                s.is_first_turn = false;
                s.phase = Phase::WaitAct;
                s.current_claims.clear();
                assert_inventory(&s);
                s
            }

            fn assert_inventory(s: &State) {
                let meld_tiles: Vec<_> = s.players.iter()
                    .flat_map(|p| p.melds.iter().flat_map(|m| m.tiles.iter().copied()))
                    .collect();
                let mut tiles: Vec<_> = s.wall.tiles.iter().copied()
                    .chain(meld_tiles.iter().copied())
                    .chain(s.players.iter().flat_map(|p| {
                        p.hand.iter().chain(&p.kita_tiles)
                            .chain(p.discards.iter().filter(|t| !meld_tiles.contains(t)))
                            .copied()
                    }))
                    .collect();
                tiles.sort();
                let expected: Vec<u8> = (0..136)
                    .filter(|t| $np == 4 || *t < 4 || *t >= 32)
                    .collect();
                assert_eq!(tiles, expected);
            }

            fn legal(s: &State, pid: u8, kind: ActionType) -> Action {
                s._get_legal_actions_internal(pid).into_iter()
                    .find(|a| {
                        // Select the White kan completed by the replacement draw.
                        a.action_type == kind
                            && (kind != ActionType::Ankan || a.consume_tiles.contains(&127))
                    })
                    .unwrap()
            }

            fn act(s: &mut State, pid: u8, kind: ActionType) {
                let mut actions = HashMap::from([(pid, legal(s, pid, kind))]);
                if s.phase == Phase::WaitResponse {
                    for &other in &s.active_players {
                        actions.entry(other).or_insert_with(|| {
                            Action::new(ActionType::Pass, None, vec![], Some(other))
                        });
                    }
                }
                s.step(&actions);
                assert_eq!(s.last_error, None);
                assert_inventory(s);
            }

            fn discard(s: &mut State) {
                s.step(&HashMap::from([(
                    0, Action::new(ActionType::Discard, Some(52), vec![], Some(0)),
                )]));
                assert_eq!(s.phase, Phase::WaitResponse);
                assert_eq!(s.last_error, None);
                assert_inventory(s);
            }

            fn after_daiminkan(followup: ActionType) -> State {
                let mut s = setup(followup);
                discard(&mut s);
                act(&mut s, 2, ActionType::Daiminkan);
                s
            }

            fn assert_ron_only(s: &mut State, pid: u8) {
                assert_eq!(s.phase, Phase::WaitResponse);
                assert_eq!(s.active_players, vec![pid]);
                let tile = s.last_discard.unwrap().1;
                for other in 0..$np {
                    let actions: Vec<Action> = s.get_observation(other)
                        .legal_actions_method().into_iter().map(Action::from).collect();
                    if other == pid {
                        assert_eq!(actions.len(), 2);
                        assert!(actions.iter().any(|a| {
                            a.action_type == ActionType::Ron && a.tile == Some(tile)
                        }));
                        assert!(actions.iter().any(|a| a.action_type == ActionType::Pass));
                    } else {
                        assert!(actions.is_empty());
                        assert!(s._get_legal_actions_internal(other).iter()
                            .all(|a| a.action_type == ActionType::Pass));
                    }
                }
            }

            #[test]
            fn accepting_pon_or_daiminkan_retires_discard_offers() {
                for kind in [ActionType::Pon, ActionType::Daiminkan] {
                    let mut s = setup(ActionType::Kakan);
                    discard(&mut s);
                    act(&mut s, 2, kind);
                    assert_eq!(s.phase, Phase::WaitAct);
                    assert_eq!(s.active_players, vec![2]);
                    assert!(s.current_claims.is_empty());
                }
            }

            #[test]
            fn kakan_offers_only_current_ron_and_resolves_ron_or_pass() {
                for response in [ActionType::Ron, ActionType::Pass] {
                    let mut s = after_daiminkan(ActionType::Kakan);
                    act(&mut s, 2, ActionType::Kakan);
                    assert_ron_only(&mut s, 1);
                    assert_eq!(s.wall.rinshan_draw_count, 1);
                    act(&mut s, 1, response);
                    assert!(s.current_claims.is_empty());
                    if response == ActionType::Ron {
                        assert!(s.is_done);
                        assert!(s.win_results.contains_key(&1));
                        assert_eq!(s.wall.rinshan_draw_count, 1);
                    } else {
                        assert!(!s.is_done);
                        assert_eq!(s.phase, Phase::WaitAct);
                        assert_eq!(s.current_player, 2);
                        assert_eq!(s.drawn_tile, Some(119));
                        assert_eq!(s.wall.rinshan_draw_count, 2);
                        assert!(s.players[1].missed_agari_doujun);
                    }
                }
            }

            #[test]
            fn submitting_a_stale_call_cannot_duplicate_tiles() {
                let mut s = setup(ActionType::Kakan);
                discard(&mut s);
                let pid = if $np == 4 { 1 } else { 2 };
                let kind = if $np == 4 {
                    ActionType::Chi
                } else {
                    ActionType::Daiminkan
                };
                let stale = legal(&s, pid, kind);
                act(&mut s, 2, ActionType::Daiminkan);
                act(&mut s, 2, ActionType::Kakan);
                let mut actions = HashMap::from([(pid, stale)]);
                actions.entry(1).or_insert_with(|| {
                    Action::new(ActionType::Pass, None, vec![], Some(1))
                });
                s.step(&actions);
                assert_eq!(
                    s.last_error,
                    Some(format!("Error: Illegal Action by Player {pid}")),
                );
                assert_inventory(&s);
            }

            #[test]
            fn ankan_does_not_restore_old_discard_offers() {
                let mut s = after_daiminkan(ActionType::Ankan);
                act(&mut s, 2, ActionType::Ankan);
                if s.rule.allows_ron_on_ankan_for_kokushi_musou {
                    assert_ron_only(&mut s, 1);
                    act(&mut s, 1, ActionType::Pass);
                    assert!(s.players[1].missed_agari_doujun);
                }
                assert_eq!(s.phase, Phase::WaitAct);
                assert!(s.current_claims.is_empty());
                assert_eq!(s.wall.rinshan_draw_count, 2);
                assert_inventory(&s);
            }

            $(claim_tests!(@kita_test $kita);)?
        }
    };
    (@kita_test $kita:ident) => {
        #[test]
        fn kita_offers_only_current_ron_and_resolves_ron_or_pass() {
            for response in [ActionType::Ron, ActionType::Pass] {
                let mut s = after_daiminkan(ActionType::Kita);
                act(&mut s, 2, ActionType::Kita);
                assert_ron_only(&mut s, 1);
                act(&mut s, 1, response);
                assert!(s.current_claims.is_empty());
                if response == ActionType::Ron {
                    assert!(s.is_done);
                    assert!(s.win_results.contains_key(&1));
                } else {
                    assert_eq!(s.phase, Phase::WaitAct);
                    assert_eq!(s.drawn_tile, Some(119));
                    assert_eq!(s.wall.rinshan_draw_count, 2);
                    assert!(s.players[1].missed_agari_doujun);
                }
            }
        }
    };
}

claim_tests!(
    tenhou_4p,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    0,
    4,
    GameRule::default_tenhou()
);
claim_tests!(
    mjsoul_4p,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    0,
    4,
    GameRule::default_mjsoul()
);
claim_tests!(
    tenhou_3p,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    3,
    GameRule::default_tenhou(),
    kita
);
claim_tests!(
    mjsoul_3p,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    3,
    GameRule::default_mjsoul(),
    kita
);
