//! Audit of the reports and follow-up comments in RiichiEnv issue #107.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};
use serde_json::Value;

macro_rules! regression_tests {
    ($module:ident, $state:ty, $legal:path, $mode:expr, $np:expr) => {
        mod $module {
            use super::*;
            use $legal;

            fn fresh() -> $state {
                let mut state =
                    <$state>::new($mode, false, Some(107), 0, GameRule::default_tenhou());
                for (pid, player) in state.players.iter_mut().enumerate() {
                    // Deliberately noten, with no 1p/2p/3p/4p.
                    player.hand = [12, 13, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30]
                        .iter()
                        .map(|t| t * 4 + pid as u8)
                        .collect();
                    player.melds.clear();
                    player.discards.clear();
                    player.nagashi_eligible = true;
                }
                state.current_player = 0;
                state.active_players = vec![0];
                state.phase = Phase::WaitAct;
                state.needs_tsumo = false;
                state.is_first_turn = false;
                state.mjai_log.clear();
                state
            }

            fn action(state: &$state, pid: u8, kind: ActionType) -> Action {
                state
                    ._get_legal_actions_internal(pid)
                    .into_iter()
                    .find(|action| action.action_type == kind)
                    .expect("fixture must offer the requested action")
            }

            fn discard_one_pin(state: &mut $state) {
                state.players[0].hand.push(39);
                state.drawn_tile = Some(39);
                state.step(&HashMap::from([(
                    0,
                    Action::new(ActionType::Discard, Some(39), vec![], Some(0)),
                )]));
                assert!(state.last_error.is_none());
                assert!(state.players[0].nagashi_eligible);
            }

            #[test]
            fn called_terminal_discard_revokes_nagashi() {
                let mut kinds = vec![ActionType::Pon, ActionType::Daiminkan];
                if $np == 4 {
                    kinds.push(ActionType::Chi);
                }
                for kind in kinds {
                    let mut state = fresh();
                    let consumed = match kind {
                        ActionType::Pon => vec![36, 37],
                        ActionType::Daiminkan => vec![36, 37, 38],
                        ActionType::Chi => vec![40, 44],
                        _ => unreachable!(),
                    };
                    state.players[1].hand.truncate(13 - consumed.len());
                    state.players[1].hand.extend(consumed);
                    discard_one_pin(&mut state);
                    state.step(&HashMap::from([(1, action(&state, 1, kind))]));
                    assert!(state.last_error.is_none());
                    assert!(!state.players[0].nagashi_eligible, "{kind:?}");
                    assert!(state.players[1].nagashi_eligible);
                    assert_eq!(state.players[1].melds.len(), 1);
                }
            }

            #[test]
            fn haitei_blocks_ankan_including_after_riichi() {
                for riichi in [false, true] {
                    let mut state = fresh();
                    // 1111p123456789sE: the kan preserves the single East wait.
                    state.players[0].hand =
                        vec![36, 37, 38, 39, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108];
                    state.players[0].riichi_declared = riichi;
                    state.drawn_tile = Some(39);
                    state.wall.drawable_count = 1;
                    assert_eq!(action(&state, 0, ActionType::Ankan).tile, Some(36));

                    state.wall.drawable_count = 0;
                    assert!(
                        state._get_legal_actions_internal(0).iter().all(|action| {
                            !matches!(action.action_type, ActionType::Ankan | ActionType::Kakan)
                        }),
                        "no kan is legal after the last live draw, riichi={riichi}"
                    );
                }
            }

            #[test]
            fn chankan_settlement_uses_four_pin_instead_of_stale_nine_pin() {
                let mut state = fresh();
                state.players[0].hand = vec![0, 32, 60, 64, 68, 73, 89, 108, 112, 116, 51];
                state.players[0].melds = vec![Meld::new(
                    MeldType::Pon,
                    vec![48, 49, 50],
                    true,
                    2,
                    Some(50),
                )];
                // 23p55p123456789s waits on 1p/4p, not 9p.
                state.players[1].hand = vec![40, 44, 53, 54, 72, 76, 80, 84, 88, 92, 96, 100, 104];
                state.drawn_tile = Some(51);
                state.last_discard = Some((2, 70));
                state.wall.dora_indicators = vec![108];
                let before = state.players[1].score;

                state.step(&HashMap::from([(0, action(&state, 0, ActionType::Kakan))]));
                assert_eq!(state.phase, Phase::WaitResponse);
                assert_eq!(state.last_discard, Some((0, 51)));
                let ron = action(&state, 1, ActionType::Ron);
                assert_eq!(ron.tile, Some(51));
                state.step(&HashMap::from([(1, ron)]));

                assert!(state.last_error.is_none());
                let result = &state.win_results[&1];
                assert!(result.is_win);
                assert!(result.han > 0);
                assert!(result.yaku.contains(&3), "chankan must be scored");
                assert!(state.players[1].score > before);
                let hora: Value = state
                    .mjai_log
                    .iter()
                    .map(|line| serde_json::from_str::<Value>(line).unwrap())
                    .find(|event| event["type"] == "hora")
                    .unwrap();
                assert_eq!(hora["actor"], 1);
                assert_eq!(hora["target"], 0);
            }

            #[test]
            fn daiminkan_does_not_reopen_chankan_or_reuse_other_claims() {
                let mut state = fresh();
                let caller = if $np == 4 { 2 } else { 1 };
                let ron_player = $np - 1;
                state.players[caller].hand.truncate(10);
                state.players[caller].hand.extend([36, 37, 38]);
                state.players[ron_player].hand =
                    vec![40, 44, 53, 54, 72, 76, 80, 84, 88, 92, 96, 100, 104];
                if $np == 4 {
                    // The lower-priority chi must not execute after the kan.
                    state.players[1].hand.truncate(11);
                    state.players[1].hand.extend([41, 45]);
                }
                // Give the ron player a yaku on the original discard.
                state.players[ron_player].riichi_declared = true;
                discard_one_pin(&mut state);
                assert_eq!(
                    action(&state, ron_player as u8, ActionType::Ron).tile,
                    Some(39)
                );
                let mut responses = HashMap::from([
                    (
                        caller as u8,
                        action(&state, caller as u8, ActionType::Daiminkan),
                    ),
                    (
                        ron_player as u8,
                        action(&state, ron_player as u8, ActionType::Pass),
                    ),
                ]);
                if $np == 4 {
                    responses.insert(1, action(&state, 1, ActionType::Chi));
                }
                let remaining = state.wall.drawable_count;
                state.step(&responses);

                assert!(state.last_error.is_none());
                assert_eq!(state.phase, Phase::WaitAct);
                assert_eq!(state.current_player, caller as u8);
                assert_eq!(state.active_players, vec![caller as u8]);
                assert!(state.pending_kan.is_none());
                assert!(state.is_rinshan_flag);
                assert!(state.drawn_tile.is_some());
                assert_eq!(state.wall.drawable_count, remaining - 1);
                assert_eq!(state.players[caller].hand.len(), 11);
                assert_eq!(state.players[caller].melds.len(), 1);
                assert_eq!(
                    state.players[caller].melds[0].meld_type,
                    MeldType::Daiminkan
                );
                assert_eq!(state.players[caller].melds[0].tiles, vec![36, 37, 38, 39]);
                if $np == 4 {
                    assert!(state.players[1].melds.is_empty());
                }
            }
        }
    };
}

regression_tests!(
    four_player,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    0,
    4
);
regression_tests!(
    three_player,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    3
);
