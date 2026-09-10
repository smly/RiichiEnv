use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use std::collections::HashMap;

// 222345p67s WhiteWhite GreenGreenGreen: a 5s/8s wait with a yaku.
const WAITING_HAND: [u8; 13] = [40, 41, 42, 44, 48, 53, 92, 96, 124, 125, 128, 129, 130];

macro_rules! furiten_tests {
    ($name:ident, $state:path, $legal_actions:path, $mode:expr, $np:expr, $rule:expr) => {
        mod $name {
            use super::*;
            use $legal_actions;
            type State = $state;

            fn setup(hand: &[u8], discarder: u8, discard: u8, draws: &[u8]) -> State {
                let mut s = State::new($mode, false, Some(42), 0, $rule);
                for player in &mut s.players {
                    player.reset_round();
                }
                s.players[0].hand = hand.to_vec();
                let reserved: Vec<_> = hand.iter().chain(draws).copied().chain([discard]).collect();
                let mut pool: Vec<u8> = (0..136)
                    .filter(|t| !reserved.contains(t) && ($np == 4 || *t < 4 || *t >= 32))
                    .collect();
                for player in s.players.iter_mut().skip(1) {
                    player.hand = pool.drain(0..13).collect();
                }
                s.players[discarder as usize].hand.push(discard);
                pool.extend(draws.iter().rev());
                s.wall.tiles = pool;
                s.wall.drawable_count = (s.wall.tiles.len() - 14) as u8;
                s.wall.dora_indicators = vec![s.wall.tiles[4]];
                s.wall.rinshan_draw_count = 0;
                s.wall.pending_kan_dora_count = 0;
                s.current_player = discarder;
                s.active_players = vec![discarder];
                s.drawn_tile = Some(discard);
                s.oya = discarder;
                s.is_first_turn = false;
                s.phase = Phase::WaitAct;
                s.current_claims.clear();
                s.mjai_log.clear();
                assert_inventory(&s);
                s
            }

            fn assert_inventory(s: &State) {
                let meld_tiles: Vec<_> = s
                    .players
                    .iter()
                    .flat_map(|p| p.melds.iter().flat_map(|m| m.tiles.iter().copied()))
                    .collect();
                // Called discards and dora indicators reference existing tiles.
                let mut actual: Vec<_> = s
                    .wall
                    .tiles
                    .iter()
                    .copied()
                    .chain(meld_tiles.iter().copied())
                    .chain(s.players.iter().flat_map(|p| {
                        p.hand
                            .iter()
                            .chain(p.discards.iter().filter(|t| !meld_tiles.contains(t)))
                            .copied()
                    }))
                    .collect();
                actual.sort();
                let expected: Vec<u8> = (0..136)
                    .filter(|t| $np == 4 || *t < 4 || *t >= 32)
                    .collect();
                assert_eq!(actual, expected, "physical tiles must be conserved");
            }

            fn offered(s: &State, pid: u8, kind: ActionType) -> bool {
                s._get_legal_actions_internal(pid)
                    .iter()
                    .any(|a| a.action_type == kind)
            }

            fn act(s: &mut State, pid: u8, kind: ActionType, tile: Option<u8>) {
                let action = s
                    ._get_legal_actions_internal(pid)
                    .into_iter()
                    .find(|a| a.action_type == kind && (tile.is_none() || a.tile == tile))
                    .unwrap_or_else(|| panic!("expected legal {kind:?} for P{pid}, tile {tile:?}"));
                let mut actions = HashMap::from([(pid, action)]);
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

            fn pass_all(s: &mut State) {
                if s.phase == Phase::WaitResponse {
                    let pid = s.active_players[0];
                    act(s, pid, ActionType::Pass, None);
                }
            }

            fn miss_ron(s: &mut State, discarder: u8, tile: u8) {
                act(s, discarder, ActionType::Discard, Some(tile));
                assert!(offered(s, 0, ActionType::Ron));
                pass_all(s);
                assert!(s.players[0].missed_agari_doujun);
            }

            #[test]
            fn own_pon_clears_furiten_on_discard() {
                let mut s = setup(&WAITING_HAND, 1, 89, &[126, 100]);
                miss_ron(&mut s, 1, 89);
                act(&mut s, 2, ActionType::Discard, Some(126));
                act(&mut s, 0, ActionType::Pon, Some(126));
                assert_eq!(s.drawn_tile, None);
                assert!(s.players[0].missed_agari_doujun);
                // White replaces the pair; 222p becomes the pair after discarding 2p.
                act(&mut s, 0, ActionType::Discard, Some(40));
                assert!(!s.players[0].missed_agari_doujun);
                pass_all(&mut s);
                act(&mut s, 1, ActionType::Discard, Some(100));
                assert!(offered(&s, 0, ActionType::Ron));

                // Declining this ron sets furiten again, until the next discard.
                pass_all(&mut s);
                assert!(s.players[0].missed_agari_doujun);
                for pid in 2..$np {
                    assert_eq!(s.current_player, pid);
                    let drawn = s.drawn_tile.unwrap();
                    act(&mut s, pid, ActionType::Discard, Some(drawn));
                    pass_all(&mut s);
                }
                assert_eq!(s.current_player, 0);
                let drawn = s.drawn_tile.unwrap();
                act(&mut s, 0, ActionType::Discard, Some(drawn));
                assert!(!s.players[0].missed_agari_doujun);
                assert!(
                    s._get_claim_actions_for_player(0, 1, 101)
                        .0
                        .iter()
                        .any(|a| a.action_type == ActionType::Ron)
                );
            }

            #[test]
            fn calls_from_kamicha_clear_furiten_on_discard() {
                for (kind, tile) in [(ActionType::Pon, 126), (ActionType::Chi, 49)] {
                    if $np == 3 && kind == ActionType::Chi {
                        continue; // Sanma has no chi.
                    }
                    let draws = if $np == 4 { vec![0, tile] } else { vec![tile] };
                    let mut s = setup(&WAITING_HAND, 1, 89, &draws);
                    miss_ron(&mut s, 1, 89);
                    if $np == 4 {
                        act(&mut s, 2, ActionType::Discard, Some(0));
                        pass_all(&mut s);
                    }
                    act(&mut s, $np - 1, ActionType::Discard, Some(tile));
                    act(&mut s, 0, kind, Some(tile));
                    assert_eq!(s.drawn_tile, None);
                    assert!(s.players[0].missed_agari_doujun);
                    act(&mut s, 0, ActionType::Discard, Some(42));
                    assert!(!s.players[0].missed_agari_doujun);
                }
            }

            #[test]
            fn choosing_pon_instead_of_ron_clears_furiten_on_discard() {
                // 2226667889p777s: win on 8p, or pon it and discard 7p for a 9p wait.
                let hand = [40, 41, 42, 56, 57, 58, 60, 64, 65, 68, 92, 93, 94];
                let mut s = setup(&hand, 1, 66, &[69]);
                act(&mut s, 1, ActionType::Discard, Some(66));
                assert!(offered(&s, 0, ActionType::Ron));
                act(&mut s, 0, ActionType::Pon, Some(66));
                assert!(s.players[0].missed_agari_doujun);
                act(&mut s, 0, ActionType::Discard, Some(60));
                assert!(!s.players[0].missed_agari_doujun);
                pass_all(&mut s);
                act(&mut s, 1, ActionType::Discard, Some(69));
                assert!(offered(&s, 0, ActionType::Ron));
            }

            #[test]
            fn another_players_reverse_pon_preserves_furiten() {
                let mut s = setup(&WAITING_HAND, 1, 89, &[134]);
                // Give P1 two Red dragons to pon P2's discard, and an 8s.
                for (index, tile) in [132, 133, 100].into_iter().enumerate() {
                    let wall_index = s.wall.tiles.iter().position(|&t| t == tile).unwrap();
                    std::mem::swap(&mut s.players[1].hand[index], &mut s.wall.tiles[wall_index]);
                }
                assert_inventory(&s);
                miss_ron(&mut s, 1, 89);
                act(&mut s, 2, ActionType::Discard, Some(134));
                act(&mut s, 1, ActionType::Pon, Some(134));
                assert!(s.players[0].missed_agari_doujun);
                act(&mut s, 1, ActionType::Discard, Some(100));
                assert!(s.players[0].missed_agari_doujun);
                assert!(!offered(&s, 0, ActionType::Ron));
            }

            #[test]
            fn draw_then_tsumogiri_or_tedashi_clears_temporary_furiten() {
                for discard in [43, 40] {
                    let mut s = setup(&WAITING_HAND, $np - 1, 89, &[43]);
                    miss_ron(&mut s, $np - 1, 89);
                    assert_eq!(s.current_player, 0);
                    assert_eq!(s.drawn_tile, Some(43));
                    act(&mut s, 0, ActionType::Discard, Some(discard));
                    assert!(!s.players[0].missed_agari_doujun);
                    assert!(
                        s._get_claim_actions_for_player(0, 1, 100)
                            .0
                            .iter()
                            .any(|a| a.action_type == ActionType::Ron)
                    );
                }
            }

            #[test]
            fn daiminkan_clears_furiten_only_after_rinshan_discard() {
                let mut s = setup(&WAITING_HAND, 1, 89, &[131]);
                miss_ron(&mut s, 1, 89);
                act(&mut s, 2, ActionType::Discard, Some(131));
                act(&mut s, 0, ActionType::Daiminkan, Some(131));
                assert_eq!(s.wall.rinshan_draw_count, 1);
                assert!(s.players[0].missed_agari_doujun);
                let drawn = s.drawn_tile.unwrap();
                act(&mut s, 0, ActionType::Discard, Some(drawn));
                assert!(!s.players[0].missed_agari_doujun);
                assert!(
                    s._get_claim_actions_for_player(0, 1, 100)
                        .0
                        .iter()
                        .any(|a| a.action_type == ActionType::Ron)
                );
            }

            #[test]
            fn temporary_furiten_does_not_prevent_tsumo() {
                let mut s = setup(&WAITING_HAND, $np - 1, 89, &[100]);
                miss_ron(&mut s, $np - 1, 89);
                assert_eq!(s.drawn_tile, Some(100));
                assert!(offered(&s, 0, ActionType::Tsumo));
            }

            #[test]
            fn riichi_furiten_survives_a_draw_and_discard() {
                let mut s = setup(&WAITING_HAND, $np - 1, 89, &[43]);
                s.players[0].riichi_declared = true;
                miss_ron(&mut s, $np - 1, 89);
                assert!(s.players[0].missed_agari_riichi);
                act(&mut s, 0, ActionType::Discard, Some(43));
                assert!(!s.players[0].missed_agari_doujun);
                assert!(s.players[0].missed_agari_riichi);
                assert!(
                    !s._get_claim_actions_for_player(0, 1, 100)
                        .0
                        .iter()
                        .any(|a| a.action_type == ActionType::Ron)
                );
            }
        }
    };
}

furiten_tests!(
    tenhou_four_player,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    0,
    4,
    GameRule::default_tenhou()
);
furiten_tests!(
    tenhou_three_player,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    3,
    GameRule::default_tenhou()
);
furiten_tests!(
    mjsoul_four_player,
    riichienv_core::state::GameState,
    riichienv_core::state::legal_actions::GameStateLegalActions,
    0,
    4,
    GameRule::default_mjsoul()
);
furiten_tests!(
    mjsoul_three_player,
    riichienv_core::state_3p::GameState3P,
    riichienv_core::state_3p::legal_actions::GameState3PLegalActions,
    3,
    3,
    GameRule::default_mjsoul()
);
