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
        }
    };
}

discard_history_tests!(four_player, riichienv_core::state::GameState, 0, 4);
discard_history_tests!(three_player, riichienv_core::state_3p::GameState3P, 3, 3);
