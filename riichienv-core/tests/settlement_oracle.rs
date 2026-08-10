//! Table-driven score settlement and round-transition oracles.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::engine::{Decision, EngineConfig, GameEngine, GameMode, StepOutcome};
use riichienv_core::game_variant::GameStateVariant;
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};
use serde_json::Value;

const BASE_SCORE: i32 = 100_000;
const DEALER: usize = 0;
const WINNER: usize = 1;
const PAO_PAYER: usize = 2;

#[test]
fn legacy_game_rule_payload_defaults_nagashi_to_draw_settlement() {
    let legacy = serde_json::json!({
        "allows_ron_on_ankan_for_kokushi_musou": false,
        "is_kokushi_musou_13machi_double": false,
        "is_suuankou_tanki_double": false,
        "is_junsei_chuurenpoutou_double": false,
        "is_daisuushii_double": false,
        "yakuman_pao_is_liability_only": false,
        "sanchaho_is_draw": true,
        "kuikae_forbidden": true
    });
    let rule: GameRule = serde_json::from_value(legacy).expect("legacy rule payload");
    assert!(!rule.nagashi_mangan_is_win);

    let mut sega_style = rule;
    sega_style.nagashi_mangan_is_win = true;
    let round_trip: GameRule = serde_json::from_value(
        serde_json::to_value(sega_style).expect("serialize extended rule payload"),
    )
    .expect("deserialize extended rule payload");
    assert!(round_trip.nagashi_mangan_is_win);
}

fn engine(mode: GameMode, rule: GameRule) -> GameEngine {
    GameEngine::new(EngineConfig::new(mode).with_seed(20260810).with_rule(rule))
        .expect("settlement fixture engine must initialize")
}

fn tenpai_hand() -> Vec<u8> {
    // 111p 234p 123s EEE S: tanki on South.
    vec![36, 37, 38, 40, 44, 48, 72, 76, 80, 108, 109, 110, 112]
}

fn noten_hand() -> Vec<u8> {
    vec![36, 40, 48, 56, 64, 72, 80, 88, 96, 104, 108, 116, 124]
}

fn configure_draw(
    game: &mut GameEngine,
    tenpai_seats: &[usize],
    nagashi_winners: &[usize],
    honba: u8,
    riichi_sticks: u32,
) {
    match game.state_mut() {
        GameStateVariant::FourPlayer(state) => {
            state.oya = DEALER as u8;
            state.kyoku_idx = DEALER as u8;
            state.round_wind = 0;
            state.honba = honba;
            state.riichi_sticks = riichi_sticks;
            state.is_done = false;
            state.mjai_log.clear();
            for (seat, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.score = BASE_SCORE;
                player.hand = if tenpai_seats.contains(&seat) {
                    tenpai_hand()
                } else {
                    noten_hand()
                };
                player.nagashi_eligible = nagashi_winners.contains(&seat);
            }
        }
        GameStateVariant::ThreePlayer(state) => {
            state.oya = DEALER as u8;
            state.kyoku_idx = DEALER as u8;
            state.round_wind = 0;
            state.honba = honba;
            state.riichi_sticks = riichi_sticks;
            state.is_done = false;
            state.mjai_log.clear();
            for (seat, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.score = BASE_SCORE;
                player.hand = if tenpai_seats.contains(&seat) {
                    tenpai_hand()
                } else {
                    noten_hand()
                };
                player.nagashi_eligible = nagashi_winners.contains(&seat);
            }
        }
    }
}

fn trigger_exhaustive_draw(game: &mut GameEngine) {
    match game.state_mut() {
        GameStateVariant::FourPlayer(state) => state._trigger_ryukyoku("exhaustive_draw"),
        GameStateVariant::ThreePlayer(state) => state._trigger_ryukyoku("exhaustive_draw"),
    }
}

fn parsed_events(game: &GameEngine) -> Vec<Value> {
    game.mjai_log()
        .iter()
        .map(|raw| serde_json::from_str(raw).expect("engine event must be valid JSON"))
        .collect()
}

fn settlement_event(game: &GameEngine, event_type: &str) -> Value {
    parsed_events(game)
        .into_iter()
        .find(|event| event["type"] == event_type)
        .unwrap_or_else(|| panic!("missing {event_type} event"))
}

struct DrawCase {
    name: &'static str,
    tenpai_seats: &'static [usize],
    expected_deltas: &'static [i32],
    next_dealer: u8,
}

#[test]
fn exhaustive_draw_payment_and_dealer_continuation_matrix() {
    let four_player_cases = [
        DrawCase {
            name: "none tenpai",
            tenpai_seats: &[],
            expected_deltas: &[0, 0, 0, 0],
            next_dealer: 1,
        },
        DrawCase {
            name: "dealer only",
            tenpai_seats: &[0],
            expected_deltas: &[3000, -1000, -1000, -1000],
            next_dealer: 0,
        },
        DrawCase {
            name: "one child",
            tenpai_seats: &[1],
            expected_deltas: &[-1000, 3000, -1000, -1000],
            next_dealer: 1,
        },
        DrawCase {
            name: "dealer and one child",
            tenpai_seats: &[0, 1],
            expected_deltas: &[1500, 1500, -1500, -1500],
            next_dealer: 0,
        },
        DrawCase {
            name: "two children",
            tenpai_seats: &[1, 2],
            expected_deltas: &[-1500, 1500, 1500, -1500],
            next_dealer: 1,
        },
        DrawCase {
            name: "dealer and two children",
            tenpai_seats: &[0, 1, 2],
            expected_deltas: &[1000, 1000, 1000, -3000],
            next_dealer: 0,
        },
        DrawCase {
            name: "three children",
            tenpai_seats: &[1, 2, 3],
            expected_deltas: &[-3000, 1000, 1000, 1000],
            next_dealer: 1,
        },
        DrawCase {
            name: "all tenpai",
            tenpai_seats: &[0, 1, 2, 3],
            expected_deltas: &[0, 0, 0, 0],
            next_dealer: 0,
        },
    ];
    let three_player_cases = [
        DrawCase {
            name: "none tenpai",
            tenpai_seats: &[],
            expected_deltas: &[0, 0, 0],
            next_dealer: 1,
        },
        DrawCase {
            name: "dealer only",
            tenpai_seats: &[0],
            expected_deltas: &[2000, -1000, -1000],
            next_dealer: 0,
        },
        DrawCase {
            name: "one child",
            tenpai_seats: &[1],
            expected_deltas: &[-1000, 2000, -1000],
            next_dealer: 1,
        },
        DrawCase {
            name: "dealer and child",
            tenpai_seats: &[0, 1],
            expected_deltas: &[1000, 1000, -2000],
            next_dealer: 0,
        },
        DrawCase {
            name: "two children",
            tenpai_seats: &[1, 2],
            expected_deltas: &[-2000, 1000, 1000],
            next_dealer: 1,
        },
        DrawCase {
            name: "all tenpai",
            tenpai_seats: &[0, 1, 2],
            expected_deltas: &[0, 0, 0],
            next_dealer: 0,
        },
    ];

    for (mode, cases) in [
        (GameMode::FourPlayerEast, four_player_cases.as_slice()),
        (GameMode::ThreePlayerEast, three_player_cases.as_slice()),
    ] {
        for case in cases {
            let mut game = engine(mode, GameRule::default_tenhou());
            configure_draw(&mut game, case.tenpai_seats, &[], 2, 1);
            trigger_exhaustive_draw(&mut game);

            let draw = settlement_event(&game, "ryukyoku");
            assert_eq!(
                draw["reason"], "exhaustive_draw",
                "{} in {mode:?}",
                case.name
            );
            assert_eq!(
                draw["deltas"],
                serde_json::json!(case.expected_deltas),
                "{} in {mode:?}",
                case.name
            );

            let snapshot = game.snapshot();
            let expected_scores = case
                .expected_deltas
                .iter()
                .map(|delta| BASE_SCORE + delta)
                .collect::<Vec<_>>();
            assert_eq!(
                snapshot.scores, expected_scores,
                "{} in {mode:?}",
                case.name
            );
            assert_eq!(
                snapshot.current_player, case.next_dealer,
                "{} in {mode:?}",
                case.name
            );
            assert_eq!(
                snapshot.kyoku, case.next_dealer,
                "{} in {mode:?}",
                case.name
            );
            assert_eq!(snapshot.honba, 3, "{} in {mode:?}", case.name);
            assert_eq!(snapshot.riichi_sticks, 1, "{} in {mode:?}", case.name);
            assert_eq!(case.expected_deltas.iter().sum::<i32>(), 0);
        }
    }
}

fn nagashi_deltas(mode: GameMode, winner: usize) -> Vec<i32> {
    match (mode.num_players(), winner) {
        (4, 0) => vec![12_000, -4_000, -4_000, -4_000],
        (4, 1) => vec![-4_000, 8_000, -2_000, -2_000],
        (3, 0) => vec![8_000, -4_000, -4_000],
        (3, 1) => vec![-4_000, 6_000, -2_000],
        _ => unreachable!("fixture only covers dealer and first child"),
    }
}

#[test]
fn nagashi_mangan_win_vs_draw_transition_matrix() {
    assert!(!GameRule::default_tenhou().nagashi_mangan_is_win);
    assert!(!GameRule::default_mjsoul().nagashi_mangan_is_win);

    // Draw-style cases deliberately make nagashi eligibility disagree with
    // dealer tenpai, proving that continuation follows the configured rule.
    let cases = [
        (false, 0usize, &[][..], 1u8, 3u8),
        (false, 1usize, &[0][..], 0u8, 3u8),
        (true, 0usize, &[][..], 0u8, 3u8),
        (true, 1usize, &[0][..], 1u8, 0u8),
    ];

    for mode in [GameMode::FourPlayerEast, GameMode::ThreePlayerEast] {
        for (is_win, winner, tenpai_seats, next_dealer, next_honba) in cases {
            let mut rule = GameRule::default_tenhou();
            rule.nagashi_mangan_is_win = is_win;
            let mut game = engine(mode, rule);
            configure_draw(&mut game, tenpai_seats, &[winner], 2, 1);
            trigger_exhaustive_draw(&mut game);

            let expected_deltas = nagashi_deltas(mode, winner);
            let draw = settlement_event(&game, "ryukyoku");
            assert_eq!(draw["reason"], "nagashimangan");
            assert_eq!(draw["deltas"], serde_json::json!(expected_deltas));

            let snapshot = game.snapshot();
            assert_eq!(
                snapshot.scores,
                expected_deltas
                    .iter()
                    .map(|delta| BASE_SCORE + delta)
                    .collect::<Vec<_>>()
            );
            assert_eq!(snapshot.current_player, next_dealer);
            assert_eq!(snapshot.kyoku, next_dealer);
            assert_eq!(snapshot.honba, next_honba);
            assert_eq!(snapshot.riichi_sticks, 1);
        }
    }
}

fn open_pon(tile_type: u8) -> Meld {
    let first = tile_type * 4;
    Meld::new(
        MeldType::Pon,
        vec![first, first + 1, first + 2],
        true,
        PAO_PAYER as i8,
        Some(first),
    )
}

fn configure_pao_hand(game: &mut GameEngine, is_tsumo: bool) {
    let num_players = game.num_players() as usize;
    let mut hands = vec![Vec::new(); num_players];
    let mut melds = vec![Vec::new(); num_players];
    melds[WINNER] = vec![open_pon(31), open_pon(32), open_pon(33)];

    let (current_player, drawn_tile) = if is_tsumo {
        // Daisangen + Tsuuiisou: EEE SS completes in the winner's hand.
        hands[WINNER] = vec![108, 109, 110, 112, 113];
        (WINNER as u8, 110)
    } else {
        // The discarder drew 1p but deals East from the hand. This also
        // guards the historical last_discard/ron-tile mismatch.
        hands[DEALER] = vec![36, 40, 44, 48, 52, 56, 60, 72, 76, 80, 84, 88, 110, 116];
        hands[WINNER] = vec![108, 109, 112, 113];
        (DEALER as u8, 36)
    };

    match game.state_mut() {
        GameStateVariant::FourPlayer(state) => {
            state.mjai_log.clear();
            state.oya = DEALER as u8;
            state.kyoku_idx = DEALER as u8;
            state.round_wind = 0;
            state.honba = 2;
            state.riichi_sticks = 2;
            for (seat, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.score = BASE_SCORE;
                player.hand = hands[seat].clone();
                player.melds = melds[seat].clone();
            }
            state.players[WINNER].pao.insert(37, PAO_PAYER as u8);
            state.current_player = current_player;
            state.active_players = vec![current_player];
            state.phase = Phase::WaitAct;
            state.needs_tsumo = false;
            state.drawn_tile = Some(drawn_tile);
            state.is_done = false;
            state.is_first_turn = false;
            state.is_rinshan_flag = false;
            state.is_after_kan = false;
            state.last_discard = None;
            state.current_claims.clear();
            state.pending_kan = None;
            state.last_error = None;
            state.wall.drawable_count = 40;
        }
        GameStateVariant::ThreePlayer(state) => {
            state.mjai_log.clear();
            state.oya = DEALER as u8;
            state.kyoku_idx = DEALER as u8;
            state.round_wind = 0;
            state.honba = 2;
            state.riichi_sticks = 2;
            for (seat, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.score = BASE_SCORE;
                player.hand = hands[seat].clone();
                player.melds = melds[seat].clone();
            }
            state.players[WINNER].pao.insert(37, PAO_PAYER as u8);
            state.current_player = current_player;
            state.active_players = vec![current_player];
            state.phase = Phase::WaitAct;
            state.needs_tsumo = false;
            state.drawn_tile = Some(drawn_tile);
            state.is_done = false;
            state.is_first_turn = false;
            state.is_rinshan_flag = false;
            state.is_after_kan = false;
            state.last_discard = None;
            state.current_claims.clear();
            state.pending_kan = None;
            state.last_error = None;
            state.wall.drawable_count = 40;
        }
    }
}

fn require_action(
    decisions: &[Decision],
    player_id: u8,
    action_type: ActionType,
    tile: Option<u8>,
) -> Action {
    decisions
        .iter()
        .find(|decision| decision.player_id == player_id)
        .unwrap_or_else(|| panic!("missing decision for player {player_id}"))
        .observation
        .legal_actions()
        .into_iter()
        .find(|action| {
            action.action_type == action_type && tile.is_none_or(|t| action.tile == Some(t))
        })
        .unwrap_or_else(|| panic!("missing {action_type:?} for player {player_id}"))
}

fn step(game: &mut GameEngine, actions: HashMap<u8, Action>) -> StepOutcome {
    let outcome = game.step(&actions);
    assert_eq!(
        outcome.error, None,
        "settlement step failed: {:?}",
        outcome.error
    );
    outcome
}

fn settle_pao(game: &mut GameEngine, is_tsumo: bool) -> StepOutcome {
    let decisions = game.decisions();
    if is_tsumo {
        let tsumo = require_action(&decisions, WINNER as u8, ActionType::Tsumo, None);
        return step(game, HashMap::from([(WINNER as u8, tsumo)]));
    }

    let discard = require_action(&decisions, DEALER as u8, ActionType::Discard, Some(110));
    let response = step(game, HashMap::from([(DEALER as u8, discard)]));
    let ron = require_action(
        &response.decisions,
        WINNER as u8,
        ActionType::Ron,
        Some(110),
    );
    let actions = response
        .decisions
        .iter()
        .map(|decision| {
            if decision.player_id == WINNER as u8 {
                (decision.player_id, ron.clone())
            } else {
                (
                    decision.player_id,
                    require_action(
                        &response.decisions,
                        decision.player_id,
                        ActionType::Pass,
                        None,
                    ),
                )
            }
        })
        .collect();
    step(game, actions)
}

fn expected_pao_deltas(mode: GameMode, liability_only: bool, is_tsumo: bool) -> Vec<i32> {
    match (mode.num_players(), liability_only, is_tsumo) {
        (4, false, true) => vec![0, 66_600, -64_600, 0],
        (4, true, true) => vec![-16_000, 66_600, -40_600, -8_000],
        (4, false, false) => vec![-32_000, 66_600, -32_600, 0],
        (4, true, false) => vec![-48_000, 66_600, -16_600, 0],
        (3, false, true) => vec![0, 50_400, -48_400],
        (3, true, true) => vec![-16_000, 50_400, -32_400],
        (3, false, false) => vec![-32_000, 66_400, -32_400],
        (3, true, false) => vec![-48_000, 66_400, -16_400],
        _ => unreachable!(),
    }
}

#[test]
fn composite_yakuman_pao_honba_and_kyotaku_matrix() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for liability_only in [false, true] {
            for is_tsumo in [false, true] {
                let mut rule = if liability_only {
                    GameRule::default_mjsoul()
                } else {
                    GameRule::default_tenhou()
                };
                rule.yakuman_pao_is_liability_only = liability_only;
                let mut game = engine(mode, rule);
                configure_pao_hand(&mut game, is_tsumo);
                let outcome = settle_pao(&mut game, is_tsumo);
                let expected = expected_pao_deltas(mode, liability_only, is_tsumo);

                let hora = outcome
                    .events
                    .iter()
                    .map(|raw| serde_json::from_str::<Value>(raw).expect("valid hora JSON"))
                    .find(|event| event["type"] == "hora")
                    .expect("pao hand must win");
                assert_eq!(hora["actor"], WINNER as u8);
                assert_eq!(
                    hora["target"],
                    if is_tsumo { WINNER as u8 } else { DEALER as u8 }
                );
                assert_eq!(hora["deltas"], serde_json::json!(expected));
                assert_eq!(
                    outcome.snapshot.scores,
                    expected
                        .iter()
                        .map(|delta| BASE_SCORE + delta)
                        .collect::<Vec<_>>()
                );
                assert_eq!(outcome.snapshot.riichi_sticks, 0);
                assert!(outcome.snapshot.done);
                assert_eq!(expected.iter().sum::<i32>(), 2_000);
            }
        }
    }
}

fn configure_last_tile_kan(game: &mut GameEngine, kakan: bool) {
    let num_players = game.num_players() as usize;
    let mut hands = vec![Vec::new(); num_players];
    let mut melds = vec![Vec::new(); num_players];
    if kakan {
        hands[0] = vec![39, 40, 44, 48, 52, 56, 60, 72, 76, 80, 84];
        melds[0] = vec![open_pon(9)];
    } else {
        hands[0] = vec![36, 37, 38, 39, 40, 44, 48, 52, 56, 72, 76, 80, 84, 88];
    }

    match game.state_mut() {
        GameStateVariant::FourPlayer(state) => {
            for (seat, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.hand = hands[seat].clone();
                player.melds = melds[seat].clone();
            }
            state.current_player = 0;
            state.active_players = vec![0];
            state.phase = Phase::WaitAct;
            state.drawn_tile = Some(39);
            state.needs_tsumo = false;
            state.is_done = false;
            state.is_first_turn = false;
            state.wall.drawable_count = 0;
        }
        GameStateVariant::ThreePlayer(state) => {
            for (seat, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.hand = hands[seat].clone();
                player.melds = melds[seat].clone();
            }
            state.current_player = 0;
            state.active_players = vec![0];
            state.phase = Phase::WaitAct;
            state.drawn_tile = Some(39);
            state.needs_tsumo = false;
            state.is_done = false;
            state.is_first_turn = false;
            state.wall.drawable_count = 0;
        }
    }
}

#[test]
fn last_live_draw_forbids_ankan_and_kakan() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for kakan in [false, true] {
            let mut game = engine(mode, GameRule::default_tenhou());
            configure_last_tile_kan(&mut game, kakan);
            let legal = game.decisions()[0].observation.legal_actions();
            assert!(
                legal
                    .iter()
                    .any(|action| action.action_type == ActionType::Discard)
            );
            assert!(
                legal.iter().all(|action| {
                    !matches!(
                        action.action_type,
                        ActionType::Ankan | ActionType::Kakan | ActionType::Daiminkan
                    )
                }),
                "kan must be unavailable after the last live draw in {mode:?}"
            );
        }
    }
}
