use riichienv_core::replay::{Action as LogAction, MjaiEvent};
use riichienv_core::rule::GameRule;

macro_rules! discard_history_tests {
    ($name:ident, $state:path, $mode:expr, $np:expr) => {
        mod $name {
            use super::*;
            type State = $state;

            fn setup() -> State {
                let mut state = State::new($mode, false, Some(42), 0, GameRule::default_mjsoul());
                for player in &mut state.players {
                    player.reset_round();
                }
                state.players[1].hand = vec![48, 53, 56];
                state
            }

            fn check(state: &mut State, last: u8, tedashi: u8, riichi: Option<u8>) {
                for observer in 0..$np {
                    let obs = state.get_observation(observer);
                    assert_eq!(obs.last_discard, Some(last as u32));
                    assert_eq!(obs.last_tedashis[1], Some(tedashi));
                    assert_eq!(obs.riichi_sutehais[1], riichi);
                }
            }

            #[test]
            fn mjai_discard_history_tracks_riichi_and_tedashi() {
                for tsumogiri in [false, true] {
                    let mut state = setup();
                    state.apply_mjai_event(MjaiEvent::Dahai {
                        actor: 1,
                        pai: "4p".into(),
                        tsumogiri: false,
                    });
                    check(&mut state, 48, 48, None);
                    state.apply_mjai_event(MjaiEvent::Reach { actor: 1 });
                    state.apply_mjai_event(MjaiEvent::Dahai {
                        actor: 1,
                        pai: "5p".into(),
                        tsumogiri,
                    });
                    let tedashi = if tsumogiri { 48 } else { 53 };
                    check(&mut state, 53, tedashi, Some(53));
                    state.apply_mjai_event(MjaiEvent::Dahai {
                        actor: 1,
                        pai: "6p".into(),
                        tsumogiri: true,
                    });
                    check(&mut state, 56, tedashi, Some(53));
                }
            }

            #[test]
            fn log_discard_history_tracks_riichi_and_tedashi() {
                for (is_liqi, is_wliqi) in [(false, false), (true, false), (false, true)] {
                    for tsumogiri in [false, true] {
                        let mut state = setup();
                        state.drawn_tile = Some(56);
                        state.apply_log_action(&LogAction::DiscardTile {
                            seat: 1,
                            tile: 48,
                            is_liqi: false,
                            is_wliqi: false,
                            doras: None,
                        });
                        check(&mut state, 48, 48, None);
                        state.drawn_tile = Some(if tsumogiri { 53 } else { 56 });
                        state.apply_log_action(&LogAction::DiscardTile {
                            seat: 1,
                            tile: 53,
                            is_liqi,
                            is_wliqi,
                            doras: None,
                        });
                        let tedashi = if tsumogiri { 48 } else { 53 };
                        let riichi = if is_liqi || is_wliqi { Some(53) } else { None };
                        check(&mut state, 53, tedashi, riichi);
                        state.drawn_tile = Some(56);
                        state.apply_log_action(&LogAction::DiscardTile {
                            seat: 1,
                            tile: 56,
                            is_liqi: false,
                            is_wliqi: false,
                            doras: None,
                        });
                        check(&mut state, 56, tedashi, riichi);
                    }
                }
            }

            #[test]
            fn log_dealer_initial_discard_obeys_configured_convention() {
                for preset in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
                    for dealer in 0..$np {
                        for forced in [false, true] {
                            for interrupted in [false, true] {
                                let mut rule = preset;
                                rule.dealer_first_discard_is_tedashi = forced;
                                let mut state = State::new($mode, false, Some(42), 0, rule);
                                state._initialize_round(dealer, 0, 0, 0, None, None);
                                // Calls interrupt first-turn status even without a discard.
                                state.is_first_turn = !interrupted;
                                let tile = state.drawn_tile.unwrap();
                                let tedashi = forced && !interrupted;
                                assert_eq!(state.get_observation(dealer).forced_tedashi, tedashi);
                                for observer in 0..$np {
                                    if observer != dealer {
                                        assert!(!state.get_observation(observer).forced_tedashi);
                                    }
                                }
                                state.apply_log_action(&LogAction::DiscardTile {
                                    seat: dealer as usize,
                                    tile,
                                    is_liqi: false,
                                    is_wliqi: false,
                                    doras: None,
                                });
                                assert_eq!(
                                    state.players[dealer as usize].discard_from_hand,
                                    vec![tedashi]
                                );
                                assert!(state.drawn_tile.is_none());
                                for observer in 0..$np {
                                    let obs = state.get_observation(observer);
                                    assert_eq!(
                                        obs.last_tedashis[dealer as usize],
                                        tedashi.then_some(tile)
                                    );
                                    assert!(!obs.forced_tedashi);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn explicit_mjai_initial_discard_metadata_is_preserved() {
                for tsumogiri in [false, true] {
                    let mut state = setup();
                    state.apply_mjai_event(MjaiEvent::Tsumo {
                        actor: 0,
                        pai: "S".into(),
                    });
                    state.apply_mjai_event(MjaiEvent::Dahai {
                        actor: 0,
                        pai: "S".into(),
                        tsumogiri,
                    });
                    assert_eq!(state.last_tedashis[0], (!tsumogiri).then_some(112));
                }
            }
        }
    };
}

discard_history_tests!(four_player, riichienv_core::state::GameState, 0, 4);
discard_history_tests!(three_player, riichienv_core::state_3p::GameState3P, 3, 3);

#[test]
fn old_serialized_rules_keep_the_previous_discard_convention() {
    for rule in [GameRule::default_tenhou(), GameRule::default_mjsoul()] {
        let mut data = serde_json::to_value(rule).unwrap();
        assert_eq!(
            serde_json::from_value::<GameRule>(data.clone()).unwrap(),
            rule
        );
        data.as_object_mut()
            .unwrap()
            .remove("dealer_first_discard_is_tedashi");
        let restored: GameRule = serde_json::from_value(data).unwrap();
        assert!(!restored.dealer_first_discard_is_tedashi);
    }
}
