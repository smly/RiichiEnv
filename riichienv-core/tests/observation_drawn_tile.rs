//! `Observation::drawn_tile` belongs to the seat that drew it. Other seats,
//! including those asked whether to rob a kan, must not see it.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::replay::{Action as LogAction, MjaiEvent};
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};

macro_rules! drawn_tile_tests {
    ($name:ident, $state:path, $legal:path, $mode:expr, $np:expr) => {
        mod $name {
            use super::*;
            use $legal;
            type State = $state;

            /// Noten hands for the seats that only watch, disjoint from the
            /// tiles the fixtures below hand to seats 0 and 1.
            const BYSTANDERS: [[u8; 13]; 2] = [
                [2, 33, 37, 42, 46, 56, 62, 66, 69, 74, 78, 82, 86],
                [3, 34, 38, 43, 47, 58, 63, 67, 70, 75, 79, 83, 87],
            ];

            fn setup(rule: GameRule, drawer: Vec<u8>, melds: Vec<Meld>, other: Vec<u8>) -> State {
                let mut state = State::new($mode, false, Some(7), 0, rule);
                for player in &mut state.players {
                    player.reset_round();
                }
                state.players[0].hand = drawer;
                state.players[0].melds = melds;
                state.players[1].hand = other;
                for seat in 2..$np {
                    state.players[seat].hand = BYSTANDERS[seat - 2].to_vec();
                }
                state.current_player = 0;
                state.active_players = vec![0];
                state.phase = Phase::WaitAct;
                state.needs_tsumo = false;
                state.is_first_turn = false;
                state
            }

            fn action(state: &State, pid: u8, kind: ActionType) -> Action {
                state
                    ._get_legal_actions_internal(pid)
                    .into_iter()
                    .find(|action| action.action_type == kind)
                    .expect("fixture must offer the requested action")
            }

            /// Seat 0 holds the fourth 4p next to its pon of 4p and has just
            /// drawn haku; seat 1 waits on 1p/4p with 23p55p123456789s.
            fn added_kan() -> State {
                let mut state = setup(
                    GameRule::default_tenhou(),
                    vec![0, 32, 51, 60, 64, 68, 73, 89, 108, 112, 124],
                    vec![Meld::new(
                        MeldType::Pon,
                        vec![48, 49, 50],
                        true,
                        2,
                        Some(50),
                    )],
                    vec![40, 44, 53, 54, 72, 76, 80, 84, 88, 92, 96, 100, 104],
                );
                state.drawn_tile = Some(124);
                state
            }

            fn drawn_tiles(state: &mut State) -> Vec<Option<u8>> {
                (0..$np)
                    .map(|seat| state.get_observation(seat).drawn_tile)
                    .collect()
            }

            fn only_seat(seat: u8, tile: Option<u8>) -> Vec<Option<u8>> {
                (0..$np)
                    .map(|s| if s == seat { tile } else { None })
                    .collect()
            }

            #[test]
            fn only_the_seat_to_act_sees_its_draw() {
                let mut state = added_kan();
                assert_eq!(drawn_tiles(&mut state), only_seat(0, Some(124)));
            }

            #[test]
            fn seat_asked_to_rob_an_added_kan_does_not_see_the_draw() {
                let mut state = added_kan();
                state.step(&HashMap::from([(0, action(&state, 0, ActionType::Kakan))]));
                assert_eq!(state.phase, Phase::WaitResponse);
                assert_eq!(state.active_players, vec![1]);
                assert_eq!(state.current_player, 0);

                let robber = state.get_observation(1);
                assert!(
                    robber
                        .legal_actions_method()
                        .into_iter()
                        .map(Action::from)
                        .any(|a| a.action_type == ActionType::Ron && a.tile == Some(51))
                );
                assert_eq!(robber.drawn_tile, None);
                let payload = serde_json::to_value(&robber).unwrap();
                assert!(payload["drawn_tile"].is_null());
                // The kan maker still holds haku and still sees it.
                assert_eq!(drawn_tiles(&mut state), only_seat(0, Some(124)));

                state.step(&HashMap::from([(1, action(&state, 1, ActionType::Pass))]));
                assert_eq!(state.phase, Phase::WaitAct);
                let rinshan = state.drawn_tile;
                assert!(rinshan.is_some());
                assert_eq!(drawn_tiles(&mut state), only_seat(0, rinshan));
            }

            #[test]
            fn seat_asked_to_rob_a_closed_kan_does_not_see_the_draw() {
                // Seat 0 holds four East and draws 4s; seat 1 waits on East
                // with thirteen orphans.
                let mut state = setup(
                    GameRule::default_mjsoul(),
                    vec![41, 45, 49, 53, 57, 61, 65, 77, 81, 85, 108, 109, 110, 111],
                    vec![],
                    vec![0, 1, 32, 36, 68, 72, 104, 112, 116, 120, 124, 128, 132],
                );
                state.drawn_tile = Some(85);
                state.step(&HashMap::from([(0, action(&state, 0, ActionType::Ankan))]));
                assert_eq!(state.phase, Phase::WaitResponse);
                assert_eq!(state.active_players, vec![1]);
                assert_eq!(drawn_tiles(&mut state), only_seat(0, Some(85)));
            }

            #[test]
            fn replayed_robbing_of_an_added_kan_does_not_see_the_draw() {
                let mut state = added_kan();
                state.drawn_tile = None;
                state.players[0].hand.retain(|&t| t != 124);
                state.apply_log_action(&LogAction::DealTile {
                    seat: 0,
                    tile: 124,
                    doras: None,
                    left_tile_count: None,
                });
                state.apply_log_action(&LogAction::AnGangAddGang {
                    seat: 0,
                    meld_type: MeldType::Kakan,
                    tiles: vec![51],
                    tile_raw_id: 12,
                    doras: None,
                });
                let ron = Action::new(ActionType::Ron, Some(51), vec![], None);
                let robber = state
                    .get_observation_for_replay(1, &ron, "chankan")
                    .unwrap();
                assert_eq!(robber.drawn_tile, None);
                assert_eq!(state.get_observation(0).drawn_tile, Some(124));
            }

            /// After `start_game` and before `start_kyoku` no seat is to act,
            /// so no seat sees the tile the constructor dealt, which the
            /// state still holds.
            #[test]
            fn no_seat_sees_a_draw_while_no_seat_is_to_act() {
                let mut state = State::new($mode, false, Some(7), 0, GameRule::default_tenhou());
                state.apply_mjai_event(MjaiEvent::StartGame {
                    names: None,
                    id: None,
                });
                assert_eq!(state.current_player, u8::MAX);
                assert!(state.drawn_tile.is_some());
                assert_eq!(drawn_tiles(&mut state), vec![None; $np]);
            }
        }
    };
}

drawn_tile_tests!(
    four_player,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    0,
    4
);
drawn_tile_tests!(
    three_player,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    3
);

#[test]
fn seat_asked_to_ron_a_north_does_not_see_the_draw() {
    use riichienv_core::state_3p::GameState3P;
    use riichienv_core::state_3p::legal_actions::GameState3PLegalActions;

    let mut state = GameState3P::new(3, false, Some(7), 0, GameRule::default_mjsoul());
    for player in &mut state.players {
        player.reset_round();
    }
    // Seat 0 holds a North and draws 8s; seat 1 waits on North with seven pairs.
    state.players[0].hand = vec![0, 32, 48, 52, 56, 60, 64, 84, 88, 92, 96, 100, 108, 120];
    state.players[1].hand = vec![36, 37, 40, 41, 44, 45, 72, 73, 76, 77, 80, 81, 122];
    state.players[2].hand = vec![1, 33, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78];
    state.current_player = 0;
    state.active_players = vec![0];
    state.phase = Phase::WaitAct;
    state.needs_tsumo = false;
    state.is_first_turn = false;
    state.drawn_tile = Some(100);

    let kita = state
        ._get_legal_actions_internal(0)
        .into_iter()
        .find(|a| a.action_type == ActionType::Kita)
        .expect("seat 0 may extract its North");
    state.step(&HashMap::from([(0, kita)]));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, vec![1]);
    let seen: Vec<_> = (0..3)
        .map(|s| state.get_observation(s).drawn_tile)
        .collect();
    assert_eq!(seen, vec![Some(100), None, None]);
}

#[test]
fn no_seat_sees_the_draw_between_an_unclaimed_north_and_its_rinshan() {
    use riichienv_core::state_3p::GameState3P;

    let mut state = GameState3P::new(3, false, Some(7), 0, GameRule::default_mjsoul());
    for player in &mut state.players {
        player.reset_round();
    }
    // Seat 0 holds a North and has drawn 8s; no seat waits on North.
    state.players[0].hand = vec![0, 32, 48, 52, 56, 60, 64, 84, 88, 92, 96, 100, 108, 120];
    state.players[1].hand = vec![2, 36, 37, 40, 41, 44, 45, 72, 73, 76, 77, 80, 81];
    state.players[2].hand = vec![1, 33, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78];
    state.current_player = 0;
    state.active_players = vec![0];
    state.phase = Phase::WaitAct;
    state.is_first_turn = false;
    state.drawn_tile = Some(100);

    // An MJAI kita that opens no ron window leaves no seat to act until the
    // rinshan tsumo arrives.
    state.apply_mjai_event(MjaiEvent::Kita { actor: 0 });
    assert_eq!(state.current_player, u8::MAX);
    assert_eq!(state.drawn_tile, Some(100));
    let seen: Vec<_> = (0..3)
        .map(|s| state.get_observation(s).drawn_tile)
        .collect();
    assert_eq!(seen, vec![None, None, None]);

    state.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 0,
        pai: "7p".to_string(),
    });
    let rinshan = state.drawn_tile;
    assert!(rinshan.is_some());
    let seen: Vec<_> = (0..3)
        .map(|s| state.get_observation(s).drawn_tile)
        .collect();
    assert_eq!(seen, vec![rinshan, None, None]);
}
