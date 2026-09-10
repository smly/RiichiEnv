use riichienv_core::types::{Conditions, WinResult, Wind};
use riichienv_core::yaku::*;

fn assert_yaku(result: &WinResult, han: u32, expected: &[u32]) {
    assert!(result.is_win);
    assert_eq!(result.han, han);
    let mut actual = result.yaku.clone();
    let mut expected = expected.to_vec();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}

macro_rules! special_hand_tests {
    ($name:ident, $evaluator:path, $players:expr) => {
        mod $name {
            use super::*;
            type Evaluator = $evaluator;

            fn calc(text: &str, tile: u8, mut conditions: Conditions, bonus: bool) -> WinResult {
                conditions.is_sanma = $players == 3;
                conditions.num_players = $players;
                // For 224466p2205668s: two 4p dora, two 6p ura, and a red 5s.
                let (dora, ura) = if bonus {
                    (vec![44], vec![52])
                } else {
                    (vec![], vec![])
                };
                Evaluator::hand_from_text(text)
                    .unwrap()
                    .calc(tile, dora, ura, Some(conditions))
            }

            #[test]
            fn kokushi_stacks_with_tenhou_and_chiihou() {
                for (text, tile, units, id) in [
                    ("119m19p19s123467z", 124, 1, ID_KOKUSHI),
                    ("19m19p19s1234567z", 1, 2, ID_KOKUSHI_13),
                ] {
                    for wind in [Wind::East, Wind::South] {
                        let result = calc(
                            text,
                            tile,
                            Conditions {
                                tsumo: true,
                                tsumo_first_turn: true,
                                player_wind: wind,
                                honba: 2,
                                ..Default::default()
                            },
                            false,
                        );
                        let heavenly = if wind == Wind::East {
                            ID_TENHO
                        } else {
                            ID_CHIHO
                        };
                        // Tenhou uses all 14 initial tiles, so Kokushi is always
                        // interpreted as thirteen-sided regardless of the last dealt tile.
                        let (units, id) = if wind == Wind::East {
                            (2, ID_KOKUSHI_13)
                        } else {
                            (units, id)
                        };
                        assert_yaku(&result, 13 * (units + 1), &[id, heavenly]);
                        assert!(result.yakuman);
                        assert_eq!(result.fu, 0);
                        let base = 8000 * (units + 1);
                        assert_eq!(
                            result.tsumo_agari_ko,
                            base * if wind == Wind::East { 2 } else { 1 } + 200
                        );
                        if wind == Wind::South {
                            assert_eq!(result.tsumo_agari_oya, base * 2 + 200);
                        }
                    }
                }
            }

            #[test]
            fn tenhou_kokushi_is_independent_of_the_designated_winning_tile() {
                let evaluator = Evaluator::hand_from_text("119m19p19s1234567z").unwrap();
                for tile in [0, 1, 32, 36, 68, 72, 104, 108, 112, 116, 120, 124, 128, 132] {
                    let result = evaluator.calc(
                        tile,
                        vec![],
                        vec![],
                        Some(Conditions {
                            tsumo: true,
                            tsumo_first_turn: true,
                            player_wind: Wind::East,
                            is_sanma: $players == 3,
                            num_players: $players,
                            ..Default::default()
                        }),
                    );
                    assert_yaku(&result, 39, &[ID_TENHO, ID_KOKUSHI_13]);
                    assert_eq!(result.tsumo_agari_ko, 48000);
                }
            }

            #[test]
            fn kokushi_later_tsumo_and_ron_do_not_get_heavenly_yaku() {
                for (text, tile, units, id) in [
                    ("119m19p19s123467z", 124, 1, ID_KOKUSHI),
                    ("19m19p19s1234567z", 1, 2, ID_KOKUSHI_13),
                ] {
                    for tsumo in [false, true] {
                        let result = calc(
                            text,
                            tile,
                            Conditions {
                                tsumo,
                                tsumo_first_turn: !tsumo,
                                player_wind: Wind::South,
                                ..Default::default()
                            },
                            false,
                        );
                        assert_yaku(&result, 13 * units, &[id]);
                        assert!(result.yakuman);
                        if tsumo {
                            assert_eq!(result.tsumo_agari_oya, 16000 * units);
                            assert_eq!(result.tsumo_agari_ko, 8000 * units);
                        } else {
                            assert_eq!(result.ron_agari, 32000 * units);
                        }
                    }
                }
            }

            #[test]
            fn heavenly_seven_pairs_excludes_ordinary_yaku_and_dora() {
                for wind in [Wind::East, Wind::South] {
                    let result = calc(
                        "224466p2205668s",
                        101,
                        Conditions {
                            tsumo: true,
                            tsumo_first_turn: true,
                            player_wind: wind,
                            ..Default::default()
                        },
                        true,
                    );
                    assert_yaku(
                        &result,
                        13,
                        &[if wind == Wind::East {
                            ID_TENHO
                        } else {
                            ID_CHIHO
                        }],
                    );
                    assert!(result.yakuman);
                    assert_eq!(result.fu, 0);
                }
            }

            #[test]
            fn all_honors_seven_pairs_excludes_riichi_and_all_bonus_tiles() {
                for tsumo in [false, true] {
                    let result = Evaluator::hand_from_text("1122334455667z").unwrap().calc(
                        133,
                        vec![118],
                        vec![119],
                        Some(Conditions {
                            tsumo,
                            riichi: true,
                            player_wind: Wind::South,
                            // The two Norths outside the hand can be extracted in 3P.
                            kita_count: if $players == 3 { 2 } else { 0 },
                            is_sanma: $players == 3,
                            num_players: $players,
                            ..Default::default()
                        }),
                    );
                    assert_yaku(&result, 13, &[ID_TSUISO]);
                    assert!(result.yakuman);
                    assert_eq!(result.fu, 0);
                    if tsumo {
                        assert_eq!(result.tsumo_agari_oya, 16000);
                        assert_eq!(result.tsumo_agari_ko, 8000);
                    } else {
                        assert_eq!(result.ron_agari, 32000);
                    }
                }
            }

            #[test]
            fn many_dora_cannot_turn_all_honors_into_multiple_yakuman() {
                let result = Evaluator::hand_from_text("1122334455667z").unwrap().calc(
                    133,
                    vec![110, 114, 118, 122, 126],
                    vec![111, 115, 119, 123, 127],
                    Some(Conditions {
                        riichi: true,
                        player_wind: Wind::South,
                        is_sanma: $players == 3,
                        num_players: $players,
                        ..Default::default()
                    }),
                );
                // Ten dora and ten ura must not inflate a single all-honors yakuman.
                assert_eq!(result.ron_agari, 32000);
                assert_yaku(&result, 13, &[ID_TSUISO]);
                assert!(result.yakuman);
            }

            #[test]
            fn all_honors_seven_pairs_stacks_with_heavenly_yaku() {
                for wind in [Wind::East, Wind::South] {
                    let result = calc(
                        "1122334455667z",
                        133,
                        Conditions {
                            tsumo: true,
                            tsumo_first_turn: true,
                            player_wind: wind,
                            ..Default::default()
                        },
                        false,
                    );
                    assert_yaku(
                        &result,
                        26,
                        &[
                            ID_TSUISO,
                            if wind == Wind::East {
                                ID_TENHO
                            } else {
                                ID_CHIHO
                            },
                        ],
                    );
                    assert!(result.yakuman);
                    assert_eq!(result.fu, 0);
                }
            }

            #[test]
            fn ordinary_seven_pairs_keeps_yaku_dora_and_25_fu() {
                let result = calc(
                    "224466p2205668s",
                    101,
                    Conditions {
                        tsumo: true,
                        riichi: true,
                        player_wind: Wind::South,
                        kita_count: if $players == 3 { 2 } else { 0 },
                        ..Default::default()
                    },
                    true,
                );
                let mut expected = vec![
                    ID_CHITOITSU,
                    ID_TANYAO,
                    ID_TSUMO,
                    ID_RIICHI,
                    ID_DORA,
                    ID_URADORA,
                    ID_AKADORA,
                ];
                if $players == 3 {
                    expected.push(ID_NUKIDORA);
                }
                assert_yaku(&result, if $players == 3 { 12 } else { 10 }, &expected);
                assert!(!result.yakuman);
                assert_eq!(result.fu, 25);
                assert_eq!(
                    result.tsumo_agari_oya,
                    if $players == 3 { 12000 } else { 8000 }
                );
            }

            #[test]
            fn kazoe_seven_pairs_remains_a_single_counted_yakuman() {
                let result = Evaluator::hand_from_text("224466p2205668s").unwrap().calc(
                    101,
                    vec![44, 45, 46, 47],
                    vec![52],
                    Some(Conditions {
                        tsumo: true,
                        riichi: true,
                        player_wind: Wind::South,
                        kita_count: if $players == 3 { 2 } else { 0 },
                        is_sanma: $players == 3,
                        num_players: $players,
                        ..Default::default()
                    }),
                );
                assert!(result.is_win);
                assert!(!result.yakuman);
                assert_eq!(result.han, if $players == 3 { 18 } else { 16 });
                assert_eq!(result.fu, 25);
                assert!(result.yaku.contains(&ID_CHITOITSU));
                assert!(result.yaku.iter().all(|&id| id < ID_TENHO));
                assert_eq!(result.tsumo_agari_oya, 16000);
                assert_eq!(result.tsumo_agari_ko, 8000);
            }
        }
    };
}

special_hand_tests!(
    four_player,
    riichienv_core::hand_evaluator::HandEvaluator,
    4
);
special_hand_tests!(
    three_player,
    riichienv_core::hand_evaluator_3p::HandEvaluator3P,
    3
);
