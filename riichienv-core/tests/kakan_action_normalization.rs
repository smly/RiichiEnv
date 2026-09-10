use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};
use std::collections::HashMap;

macro_rules! kakan_tests {
    ($name:ident, $state:path, $mode:expr, $np:expr, $rule:expr) => {
        mod $name {
            use super::*;
            type State = $state;

            fn setup(melds: Vec<Meld>, hand: Vec<u8>, ron_hand: Option<Vec<u8>>) -> State {
                let mut s = State::new($mode, false, Some(42), 0, $rule);
                for player in &mut s.players {
                    player.reset_round();
                }
                s.players[0].hand = hand;
                s.players[0].melds = melds;
                if let Some(hand) = ron_hand {
                    s.players[1].hand = hand;
                }
                let used: Vec<_> = s
                    .players
                    .iter()
                    .flat_map(|p| {
                        p.hand
                            .iter()
                            .copied()
                            .chain(p.melds.iter().flat_map(|m| m.tiles.iter().copied()))
                    })
                    .collect();
                let mut pool: Vec<u8> = (0..136)
                    .filter(|t| !used.contains(t) && ($np == 4 || *t < 4 || *t >= 32))
                    .collect();
                for player in s.players.iter_mut().skip(1) {
                    if player.hand.is_empty() {
                        player.hand = pool.drain(0..13).collect();
                    }
                }
                s.wall.tiles = pool;
                s.wall.drawable_count = (s.wall.tiles.len() - 14) as u8;
                s.wall.dora_indicators = vec![s.wall.tiles[4]];
                s.wall.rinshan_draw_count = 0;
                s.wall.pending_kan_dora_count = 0;
                s.current_player = 0;
                s.active_players = vec![0];
                s.drawn_tile = s.players[0].hand.first().copied();
                s.is_first_turn = false;
                s.phase = Phase::WaitAct;
                s.current_claims.clear();
                s.mjai_log.clear();
                s.mjai_log_per_player = Default::default();
                s.player_event_counts = [0; $np];
                assert_inventory(&s);
                s
            }

            fn pon(tiles: Vec<u8>) -> Meld {
                let called = tiles[0];
                Meld::new(MeldType::Pon, tiles, true, 1, Some(called))
            }

            fn single_pon(base: u8, added: u8) -> State {
                let tiles = (base..base + 4).filter(|&t| t != added).collect();
                setup(
                    vec![pon(tiles)],
                    vec![added, 0, 32, 60, 64, 68, 72, 76, 112, 120, 124],
                    None,
                )
            }

            fn assert_inventory(s: &State) {
                // Discard history and dora indicators are references to tiles,
                // not additional copies. This fixture has no uncalled discards.
                let mut actual: Vec<_> = s
                    .wall
                    .tiles
                    .iter()
                    .copied()
                    .chain(s.players.iter().flat_map(|p| {
                        p.hand
                            .iter()
                            .copied()
                            .chain(p.melds.iter().flat_map(|m| m.tiles.iter().copied()))
                    }))
                    .collect();
                actual.sort();
                let expected: Vec<u8> = (0..136)
                    .filter(|t| $np == 4 || *t < 4 || *t >= 32)
                    .collect();
                assert_eq!(actual, expected, "physical tiles must be conserved");
            }

            fn kakan(s: &mut State, tile: u8) -> Action {
                s.get_observation(0)
                    .legal_actions_method()
                    .into_iter()
                    .map(Action::from)
                    .find(|a| a.action_type == ActionType::Kakan && a.tile == Some(tile))
                    .unwrap()
            }

            fn assert_same_outcome(actual: &mut State, expected: &mut State) {
                assert_eq!(actual.last_error, None);
                assert_eq!(actual.phase, expected.phase);
                assert_eq!(actual.is_done, expected.is_done);
                assert_eq!(actual.active_players, expected.active_players);
                assert_eq!(actual.current_player, expected.current_player);
                assert_eq!(actual.pending_kan, expected.pending_kan);
                assert_eq!(actual.drawn_tile, expected.drawn_tile);
                assert_eq!(actual.wall.tiles, expected.wall.tiles);
                assert_eq!(actual.wall.drawable_count, expected.wall.drawable_count);
                assert_eq!(
                    actual.wall.rinshan_draw_count,
                    expected.wall.rinshan_draw_count
                );
                assert_eq!(actual.wall.dora_indicators, expected.wall.dora_indicators);
                assert_eq!(
                    actual.wall.pending_kan_dora_count,
                    expected.wall.pending_kan_dora_count
                );
                assert_eq!(actual.mjai_log, expected.mjai_log);
                assert_inventory(actual);
                for pid in 0..$np {
                    assert_eq!(
                        actual.get_observation(pid).serialize_to_base64().unwrap(),
                        expected.get_observation(pid).serialize_to_base64().unwrap(),
                    );
                }
            }

            fn abbreviate(action: &Action, omit_tile: bool) -> Action {
                let mut short = action.clone();
                short.actor = None;
                if omit_tile {
                    short.tile = None;
                } else {
                    short.consume_tiles.clear();
                }
                short
            }

            #[test]
            fn omitted_tile_preserves_physical_tiles_and_red_fives() {
                for (base, added) in [(36, 39), (52, 52), (52, 55)] {
                    let mut expected = single_pon(base, added);
                    let mut actual = expected.clone();
                    let normal = kakan(&mut expected, added);
                    let short = abbreviate(&normal, true);
                    expected.step(&HashMap::from([(0, normal)]));
                    actual.step(&HashMap::from([(0, short)]));
                    assert_same_outcome(&mut actual, &mut expected);
                    assert_eq!(
                        actual.players[0].melds[0].tiles,
                        (base..base + 4).collect::<Vec<_>>()
                    );
                    assert!(!actual.players[0].hand.contains(&added));
                }
            }

            #[test]
            fn omitted_consumed_preserves_mjai_and_observations() {
                for (base, added) in [(36, 39), (52, 52), (52, 55)] {
                    let mut expected = single_pon(base, added);
                    let mut actual = expected.clone();
                    let normal = kakan(&mut expected, added);
                    let short = abbreviate(&normal, false);
                    expected.step(&HashMap::from([(0, normal)]));
                    actual.step(&HashMap::from([(0, short)]));
                    assert_same_outcome(&mut actual, &mut expected);
                    let event: serde_json::Value =
                        serde_json::from_str(&actual.mjai_log[0]).unwrap();
                    assert_eq!(event["type"], "kakan");
                    assert_eq!(event["consumed"].as_array().unwrap().len(), 3);
                }
            }

            #[test]
            fn shorthand_selects_the_specified_pon_among_multiple_kans() {
                for added in [39, 52] {
                    for omit_tile in [true, false] {
                        let mut expected = setup(
                            vec![pon(vec![36, 37, 38]), pon(vec![53, 54, 55])],
                            vec![39, 52, 72, 76, 112, 120, 124, 132],
                            None,
                        );
                        // The kan tile need not be the current draw.
                        expected.drawn_tile = Some(132);
                        let mut actual = expected.clone();
                        let normal = kakan(&mut expected, added);
                        let short = abbreviate(&normal, omit_tile);
                        expected.step(&HashMap::from([(0, normal)]));
                        actual.step(&HashMap::from([(0, short)]));
                        assert_same_outcome(&mut actual, &mut expected);
                        assert_eq!(
                            actual.players[0]
                                .melds
                                .iter()
                                .filter(|m| m.meld_type == MeldType::Kakan)
                                .count(),
                            1
                        );
                    }
                }
            }

            #[test]
            fn contradictory_or_unspecified_kakan_is_rejected() {
                for (tile, consumed) in [
                    (Some(76), vec![36, 37, 38]), // Different tile kind, present in hand.
                    (Some(36), vec![36, 37, 38]), // A physical copy already in the pon.
                    (Some(39), vec![36, 37, 40]),
                    (None, vec![36, 36, 38]),
                    (None, vec![36, 37, 38, 39]),
                    (None, vec![]),
                ] {
                    let mut s = single_pon(36, 39);
                    let hand = s.players[0].hand.clone();
                    let melds = s.players[0].melds.clone();
                    s.step(&HashMap::from([(
                        0,
                        Action::new(ActionType::Kakan, tile, consumed, None),
                    )]));
                    assert_eq!(
                        s.last_error.as_deref(),
                        Some("Error: Illegal Action by Player 0")
                    );
                    assert_eq!(s.players[0].hand, hand);
                    assert_eq!(
                        serde_json::to_value(&s.players[0].melds).unwrap(),
                        serde_json::to_value(&melds).unwrap(),
                    );
                    assert_eq!(s.wall.rinshan_draw_count, 0);
                    assert_inventory(&s);
                }
            }

            #[test]
            fn shorthand_preserves_chankan_offer_and_resolution() {
                for omit_tile in [true, false] {
                    for claim_ron in [true, false] {
                        let mut expected = setup(
                            vec![pon(vec![53, 54, 55])],
                            vec![52, 0, 32, 36, 60, 64, 68, 112, 120, 124, 132],
                            // 34p234678s EEE WW waits on 2p/5p.
                            Some(vec![
                                44, 48, 76, 80, 84, 92, 96, 100, 108, 109, 110, 116, 117,
                            ]),
                        );
                        let mut actual = expected.clone();
                        let normal = kakan(&mut expected, 52);
                        let short = abbreviate(&normal, omit_tile);
                        expected.step(&HashMap::from([(0, normal)]));
                        actual.step(&HashMap::from([(0, short)]));
                        assert_eq!(actual.phase, Phase::WaitResponse);
                        assert!(
                            actual
                                .current_claims
                                .get(&1)
                                .unwrap()
                                .iter()
                                .any(|a| a.action_type == ActionType::Ron && a.tile == Some(52))
                        );
                        assert_same_outcome(&mut actual, &mut expected);
                        let response = if claim_ron {
                            ActionType::Ron
                        } else {
                            ActionType::Pass
                        };
                        let actions =
                            HashMap::from([(1, Action::new(response, None, vec![], Some(1)))]);
                        actual.step(&actions);
                        expected.step(&actions);
                        assert_same_outcome(&mut actual, &mut expected);
                        assert_eq!(actual.wall.rinshan_draw_count, u8::from(!claim_ron));
                    }
                }
            }
        }
    };
}

kakan_tests!(
    tenhou_4p,
    riichienv_core::state::GameState,
    0,
    4,
    GameRule::default_tenhou()
);
kakan_tests!(
    mjsoul_4p,
    riichienv_core::state::GameState,
    0,
    4,
    GameRule::default_mjsoul()
);
kakan_tests!(
    tenhou_3p,
    riichienv_core::state_3p::GameState3P,
    3,
    3,
    GameRule::default_tenhou()
);
kakan_tests!(
    mjsoul_3p,
    riichienv_core::state_3p::GameState3P,
    3,
    3,
    GameRule::default_mjsoul()
);
