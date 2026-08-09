//! Differential coverage for the SP-specific lean yaku/fu path.
//!
//! The optimized calculator is intentionally compared with the regular
//! `HandEvaluator`, not with hand-written expected values. This keeps the
//! test broad while preserving an independent implementation as the oracle.

use std::fs;

use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::sp::SpInput;
use riichienv_core::sp_yaku::compute_for_sp_tsumo;
use riichienv_core::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    cases: Vec<AgariCase>,
}

#[derive(Deserialize)]
struct AgariCase {
    tiles_136: Vec<u8>,
    melds: Vec<CorpusMeld>,
    win_tile_136: u8,
    dora_indicators: Vec<u8>,
    ura_indicators: Vec<u8>,
    conditions: CorpusConditions,
    expected: CorpusExpected,
}

#[derive(Deserialize)]
struct CorpusMeld {
    meld_type: String,
    tiles: Vec<u8>,
    opened: bool,
    from_who: i8,
}

#[derive(Deserialize)]
struct CorpusConditions {
    tsumo: bool,
    riichi: bool,
    double_riichi: bool,
    ippatsu: bool,
    haitei: bool,
    houtei: bool,
    rinshan: bool,
    chankan: bool,
    tsumo_first_turn: bool,
    player_wind: u8,
    round_wind: u8,
    kita_count: u8,
    num_players: u8,
}

#[derive(Deserialize)]
struct CorpusExpected {
    han: u32,
}

fn meld_type(name: &str) -> MeldType {
    match name {
        "chi" => MeldType::Chi,
        "pon" => MeldType::Pon,
        "daiminkan" => MeldType::Daiminkan,
        "ankan" => MeldType::Ankan,
        "kakan" => MeldType::Kakan,
        _ => panic!("unknown meld type {name}"),
    }
}

fn melds(case: &AgariCase) -> Vec<Meld> {
    case.melds
        .iter()
        .map(|meld| {
            Meld::new(
                meld_type(&meld.meld_type),
                meld.tiles.clone(),
                meld.opened,
                meld.from_who,
                None,
            )
        })
        .collect()
}

fn pre_win_tiles(case: &AgariCase) -> Vec<u8> {
    let mut tiles = case.tiles_136.clone();
    let effective_tile_count = tiles.len() + case.melds.len() * 3;
    match effective_tile_count {
        13 => tiles,
        14 => {
            let position = tiles
                .iter()
                .position(|&tile| tile == case.win_tile_136)
                .or_else(|| {
                    tiles
                        .iter()
                        .position(|&tile| tile / 4 == case.win_tile_136 / 4)
                })
                .expect("14-tile corpus case must contain its winning tile");
            tiles.remove(position);
            tiles
        }
        count => panic!("unexpected effective tile count {count}"),
    }
}

fn counts(tiles: &[u8]) -> [u8; TILE_MAX] {
    let mut counts = [0u8; TILE_MAX];
    for &tile in tiles {
        counts[(tile / 4) as usize] += 1;
    }
    counts
}

fn aka_flags(tiles: &[u8], win_tile: u8) -> [bool; 3] {
    [16u8, 52, 88].map(|red| win_tile == red || tiles.contains(&red))
}

fn is_lean_compatible(case: &AgariCase) -> bool {
    case.conditions.tsumo
        && !case.conditions.double_riichi
        && !case.conditions.ippatsu
        && !case.conditions.haitei
        && !case.conditions.houtei
        && !case.conditions.rinshan
        && !case.conditions.chankan
        && !case.conditions.tsumo_first_turn
        && case.conditions.kita_count == 0
        && case.conditions.num_players == 4
        && case.ura_indicators.is_empty()
        && case.expected.han < 13
        && case
            .melds
            .iter()
            .all(|meld| !matches!(meld.meld_type.as_str(), "daiminkan" | "ankan" | "kakan"))
}

#[test]
fn lean_sp_yaku_matches_hand_evaluator_on_agari_corpus() {
    let raw = fs::read_to_string("benches/data/agari_4p.json")
        .expect("agari correctness corpus must be available");
    let corpus: Corpus = serde_json::from_str(&raw).expect("agari corpus must be valid JSON");

    let mut compared = 0usize;
    let mut compared_with_dora = 0usize;
    let mut compared_with_aka = 0usize;

    for (case_index, case) in corpus.cases.iter().enumerate() {
        if !is_lean_compatible(case) {
            continue;
        }

        let pre_win = pre_win_tiles(case);
        let concealed_counts = counts(&pre_win);
        let melds = melds(case);
        let is_menzen = melds.iter().all(|meld| !meld.opened);
        let akas_in_hand = aka_flags(&pre_win, case.win_tile_136);
        let input = SpInput {
            tehai: concealed_counts,
            akas_in_hand,
            tiles_seen: concealed_counts,
            akas_seen: akas_in_hand,
            dora_indicators: case.dora_indicators.clone(),
            melds: melds.clone(),
            bakaze: 27 + case.conditions.round_wind,
            jikaze: 27 + case.conditions.player_wind,
            is_menzen,
            can_riichi: case.conditions.riichi,
            can_double_riichi: false,
            tsumos_left: 1,
            discard_candidates: vec![],
            kita_count: 0,
        };

        let Some(lean) = compute_for_sp_tsumo(
            &input,
            &concealed_counts,
            case.win_tile_136 / 4,
            akas_in_hand,
        ) else {
            continue;
        };

        let conditions = Conditions {
            tsumo: true,
            riichi: case.conditions.riichi,
            player_wind: Wind::from(case.conditions.player_wind),
            round_wind: Wind::from(case.conditions.round_wind),
            num_players: 4,
            ..Conditions::default()
        };
        let regular = HandEvaluator::new(pre_win.clone(), melds).calc(
            case.win_tile_136,
            case.dora_indicators.clone(),
            vec![],
            Some(conditions),
        );

        assert!(regular.is_win, "corpus case {case_index} must be a win");
        assert_eq!(
            (lean.han, lean.fu),
            (regular.han, regular.fu),
            "SP lean score differs for corpus case {case_index}: hand={pre_win:?}, win={}",
            case.win_tile_136,
        );

        compared += 1;
        compared_with_dora += usize::from(!case.dora_indicators.is_empty());
        compared_with_aka += usize::from(akas_in_hand.iter().any(|&is_aka| is_aka));
    }

    assert!(
        compared >= 40,
        "only {compared} corpus cases exercised the lean path"
    );
    assert!(
        compared_with_dora >= 20,
        "dora coverage regressed to {compared_with_dora} cases"
    );
    assert!(
        compared_with_aka >= 5,
        "red-five coverage regressed to {compared_with_aka} cases"
    );
}

#[test]
fn sp_scoring_falls_back_and_matches_open_tanyao_score() {
    // Open 456m + concealed 234p 234s 66p 67s, tsumo 5s.
    let mut concealed = [0u8; TILE_MAX];
    for tile in [10usize, 11, 12, 19, 20, 21, 14, 14, 23, 24] {
        concealed[tile] += 1;
    }
    let melds = vec![Meld::new(
        MeldType::Chi,
        vec![12, 16, 20],
        true,
        3,
        Some(12),
    )];
    let input = SpInput {
        tehai: concealed,
        akas_in_hand: [false; 3],
        tiles_seen: concealed,
        akas_seen: [false; 3],
        dora_indicators: vec![],
        melds: melds.clone(),
        bakaze: 27,
        jikaze: 28,
        is_menzen: false,
        can_riichi: false,
        can_double_riichi: false,
        tsumos_left: 1,
        discard_candidates: vec![],
        kita_count: 0,
    };

    assert_eq!(
        compute_for_sp_tsumo(&input, &concealed, 22, [false; 3]),
        None,
        "open hands must use the full evaluator fallback"
    );
    let fallback = riichienv_core::sp::__debug_score_for_win(&input, &concealed, 22)
        .expect("the full evaluator fallback must score open tanyao");
    let concealed_tiles = [40u8, 44, 48, 76, 80, 84, 56, 57, 92, 96];
    let regular = HandEvaluator::new(concealed_tiles.to_vec(), melds).calc(
        89,
        vec![],
        vec![],
        Some(Conditions {
            tsumo: true,
            player_wind: Wind::South,
            round_wind: Wind::East,
            ..Conditions::default()
        }),
    );

    assert!(regular.is_win);
    assert_eq!((fallback.0, fallback.1), (regular.han, regular.fu));
}

#[test]
fn lean_sp_yaku_falls_back_for_all_kan_types() {
    let mut concealed = [0u8; TILE_MAX];
    for tile in [9usize, 10, 11, 18, 19, 20, 22] {
        concealed[tile] += 1;
    }
    concealed[23] = 2;

    for meld_type in [MeldType::Daiminkan, MeldType::Ankan, MeldType::Kakan] {
        let input = SpInput {
            tehai: concealed,
            akas_in_hand: [false; 3],
            tiles_seen: concealed,
            akas_seen: [false; 3],
            dora_indicators: vec![],
            melds: vec![Meld::new(meld_type, vec![0, 1, 2, 3], true, 1, Some(0))],
            bakaze: 27,
            jikaze: 28,
            is_menzen: meld_type == MeldType::Ankan,
            can_riichi: false,
            can_double_riichi: false,
            tsumos_left: 1,
            discard_candidates: vec![],
            kita_count: 0,
        };

        assert_eq!(
            compute_for_sp_tsumo(&input, &concealed, 21, [false; 3]),
            None,
            "{meld_type:?} must use the full evaluator fallback"
        );
    }
}
