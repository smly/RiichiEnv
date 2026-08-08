use std::collections::{HashMap, HashSet};

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::engine::{Decision, EngineConfig, GameEngine, GameMode, StepOutcome};
use riichienv_core::game_variant::GameStateVariant;
use riichienv_core::replay::MjaiEvent;
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};

fn engine(mode: GameMode, kuikae_forbidden: bool) -> GameEngine {
    let mut rule = GameRule::default_tenhou();
    rule.kuikae_forbidden = kuikae_forbidden;
    GameEngine::new(EngineConfig::new(mode).with_seed(42).with_rule(rule))
        .expect("test engine must initialize")
}

fn prepare_wait_act(
    engine: &mut GameEngine,
    current_player: u8,
    drawn_tile: u8,
    hands: &[Vec<u8>],
    melds: &[Vec<Meld>],
) {
    match engine.state_mut() {
        GameStateVariant::FourPlayer(state) => {
            for (player_id, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.hand = hands.get(player_id).cloned().unwrap_or_default();
                player.melds = melds.get(player_id).cloned().unwrap_or_default();
            }
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
            state.wall.pending_kan_dora_count = 0;
        }
        GameStateVariant::ThreePlayer(state) => {
            for (player_id, player) in state.players.iter_mut().enumerate() {
                player.reset_round();
                player.hand = hands.get(player_id).cloned().unwrap_or_default();
                player.melds = melds.get(player_id).cloned().unwrap_or_default();
            }
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
            state.wall.pending_kan_dora_count = 0;
        }
    }
}

fn find_action<F>(
    decisions: &[Decision],
    player_id: u8,
    action_type: ActionType,
    predicate: F,
) -> Action
where
    F: Fn(&Action) -> bool,
{
    decisions
        .iter()
        .find(|decision| decision.player_id == player_id)
        .unwrap_or_else(|| panic!("missing decision for player {player_id}"))
        .observation
        .legal_actions()
        .into_iter()
        .find(|action| action.action_type == action_type && predicate(action))
        .unwrap_or_else(|| panic!("missing {action_type:?} action for player {player_id}"))
}

fn step_one(engine: &mut GameEngine, player_id: u8, action: Action) -> StepOutcome {
    let outcome = engine.step(&HashMap::from([(player_id, action)]));
    assert_eq!(outcome.error, None, "step failed: {:?}", outcome.error);
    outcome
}

fn discard_types(decisions: &[Decision], player_id: u8) -> HashSet<u8> {
    decisions
        .iter()
        .find(|decision| decision.player_id == player_id)
        .expect("discard decision must exist")
        .observation
        .legal_actions()
        .into_iter()
        .filter(|action| action.action_type == ActionType::Discard)
        .filter_map(|action| action.tile.map(|tile| tile / 4))
        .collect()
}

fn event_types(outcome: &StepOutcome) -> Vec<String> {
    outcome
        .events
        .iter()
        .map(|raw| {
            serde_json::from_str::<serde_json::Value>(raw)
                .expect("engine event must be JSON")["type"]
                .as_str()
                .expect("engine event must have a type")
                .to_owned()
        })
        .collect()
}

fn dora_count(engine: &GameEngine) -> usize {
    match engine.state() {
        GameStateVariant::FourPlayer(state) => state.wall.dora_indicators.len(),
        GameStateVariant::ThreePlayer(state) => state.wall.dora_indicators.len(),
    }
}

fn pending_kan_dora_count(engine: &GameEngine) -> u8 {
    match engine.state() {
        GameStateVariant::FourPlayer(state) => state.wall.pending_kan_dora_count,
        GameStateVariant::ThreePlayer(state) => state.wall.pending_kan_dora_count,
    }
}

fn rinshan_draw_count(engine: &GameEngine) -> u8 {
    match engine.state() {
        GameStateVariant::FourPlayer(state) => state.wall.rinshan_draw_count,
        GameStateVariant::ThreePlayer(state) => state.wall.rinshan_draw_count,
    }
}

#[test]
fn chi_kuikae_rule_matches_live_and_replay_paths() {
    for kuikae_forbidden in [true, false] {
        let mut live = engine(GameMode::FourPlayerSingle, kuikae_forbidden);
        let hands = vec![
            vec![0, 4, 8, 24, 28, 32, 36, 40, 44, 48, 72, 76, 80, 108],
            vec![1, 5, 9, 12, 16, 20, 25, 29, 33, 41, 45, 49, 73],
            vec![],
            vec![],
        ];
        prepare_wait_act(&mut live, 0, 8, &hands, &[vec![], vec![], vec![], vec![]]);

        let initial = live.decisions();
        let discard = find_action(&initial, 0, ActionType::Discard, |action| {
            action.tile == Some(8)
        });
        let response = step_one(&mut live, 0, discard);
        let chi = find_action(&response.decisions, 1, ActionType::Chi, |action| {
            action.consume_tiles == [12, 16]
        });
        let after_chi = step_one(&mut live, 1, chi);
        let live_discards = discard_types(&after_chi.decisions, 1);

        assert_eq!(
            live_discards.contains(&2),
            !kuikae_forbidden,
            "called 3m must follow the configured kuikae rule"
        );
        assert_eq!(
            live_discards.contains(&5),
            !kuikae_forbidden,
            "opposite-side 6m must follow the configured suji-kuikae rule"
        );

        let mut replay_rule = GameRule::default_tenhou();
        replay_rule.kuikae_forbidden = kuikae_forbidden;
        let mut replay = riichienv_core::state::GameState::new(0, true, Some(42), 0, replay_rule);
        replay.players[1].hand = hands[1].clone();
        replay.apply_mjai_event(MjaiEvent::Chi {
            actor: 1,
            target: 0,
            pai: "3m".to_owned(),
            consumed: vec!["4m".to_owned(), "5m".to_owned()],
        });
        let replay_forbidden = replay.players[1]
            .forbidden_discards
            .iter()
            .map(|tile| tile / 4)
            .collect::<HashSet<_>>();

        assert_eq!(replay_forbidden.contains(&2), kuikae_forbidden);
        assert_eq!(replay_forbidden.contains(&5), kuikae_forbidden);
        assert_eq!(
            live_discards.contains(&2),
            !replay_forbidden.contains(&2),
            "live and replay must enforce the same called-tile restriction"
        );
        assert_eq!(
            live_discards.contains(&5),
            !replay_forbidden.contains(&5),
            "live and replay must enforce the same suji restriction"
        );
    }
}

#[test]
fn pon_kuikae_rule_matches_live_and_replay_in_both_variants() {
    for (mode, num_players) in [
        (GameMode::FourPlayerSingle, 4usize),
        (GameMode::ThreePlayerSingle, 3usize),
    ] {
        for kuikae_forbidden in [true, false] {
            let mut live = engine(mode, kuikae_forbidden);
            let mut hands = vec![vec![]; num_players];
            hands[0] = vec![0, 4, 8, 32, 36, 40, 44, 48, 72, 76, 80, 108, 112, 116];
            hands[1] = vec![37, 38, 39, 1, 33, 41, 45, 49, 73, 77, 81, 109, 113];
            let melds = vec![vec![]; num_players];
            prepare_wait_act(&mut live, 0, 36, &hands, &melds);

            let initial = live.decisions();
            let discard = find_action(&initial, 0, ActionType::Discard, |action| {
                action.tile == Some(36)
            });
            let response = step_one(&mut live, 0, discard);
            let pon = find_action(&response.decisions, 1, ActionType::Pon, |action| {
                action.consume_tiles == [37, 38]
            });
            let after_pon = step_one(&mut live, 1, pon);
            let discards = discard_types(&after_pon.decisions, 1);

            assert_eq!(
                discards.contains(&9),
                !kuikae_forbidden,
                "remaining 1p after pon must follow the configured kuikae rule in {mode:?}"
            );

            let mut replay_rule = GameRule::default_tenhou();
            replay_rule.kuikae_forbidden = kuikae_forbidden;
            let pon_event = MjaiEvent::Pon {
                actor: 1,
                target: 0,
                pai: "1p".to_owned(),
                consumed: vec!["1p".to_owned(), "1p".to_owned()],
            };
            let replay_forbidden = match mode {
                GameMode::FourPlayerSingle => {
                    let mut replay =
                        riichienv_core::state::GameState::new(0, true, Some(42), 0, replay_rule);
                    replay.players[1].hand = hands[1].clone();
                    replay.apply_mjai_event(pon_event);
                    replay.players[1].forbidden_discards.clone()
                }
                GameMode::ThreePlayerSingle => {
                    let mut replay = riichienv_core::state_3p::GameState3P::new(
                        0,
                        true,
                        Some(42),
                        0,
                        replay_rule,
                    );
                    replay.players[1].hand = hands[1].clone();
                    replay.apply_mjai_event(pon_event);
                    replay.players[1].forbidden_discards.clone()
                }
                _ => unreachable!(),
            };
            assert_eq!(
                replay_forbidden.iter().any(|tile| tile / 4 == 9),
                kuikae_forbidden
            );
            assert_eq!(
                discards.contains(&9),
                !replay_forbidden.iter().any(|tile| tile / 4 == 9),
                "live and replay pon restrictions must match in {mode:?}"
            );
        }
    }
}

#[test]
fn ankan_lifecycle_is_consistent_in_four_and_three_player_engines() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        let mut game = engine(mode, true);
        let num_players = mode.num_players() as usize;
        let mut hands = vec![vec![]; num_players];
        hands[0] = vec![0, 32, 36, 37, 38, 39, 40, 44, 48, 72, 76, 80, 108, 112];
        prepare_wait_act(&mut game, 0, 39, &hands, &vec![vec![]; num_players]);
        match game.state_mut() {
            GameStateVariant::FourPlayer(state) => state
                .players
                .iter_mut()
                .for_each(|player| player.ippatsu_cycle = true),
            GameStateVariant::ThreePlayer(state) => state
                .players
                .iter_mut()
                .for_each(|player| player.ippatsu_cycle = true),
        }

        let before_dora = dora_count(&game);
        let before_wall = game.snapshot().wall_tiles_remaining;
        let decisions = game.decisions();
        let ankan = find_action(&decisions, 0, ActionType::Ankan, |action| {
            action.tile.map(|tile| tile / 4) == Some(9)
        });
        let outcome = step_one(&mut game, 0, ankan);

        assert_eq!(event_types(&outcome), ["ankan", "dora", "tsumo"]);
        assert_eq!(outcome.snapshot.phase, Phase::WaitAct);
        assert_eq!(outcome.snapshot.active_players, [0]);
        assert_eq!(outcome.snapshot.wall_tiles_remaining, before_wall - 1);
        assert_eq!(dora_count(&game), before_dora + 1);
        assert_eq!(pending_kan_dora_count(&game), 0);
        assert_eq!(rinshan_draw_count(&game), 1);
        match game.state() {
            GameStateVariant::FourPlayer(state) => {
                assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
                assert_eq!(state.players[0].melds[0].meld_type, MeldType::Ankan);
            }
            GameStateVariant::ThreePlayer(state) => {
                assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
                assert_eq!(state.players[0].melds[0].meld_type, MeldType::Ankan);
            }
        }
    }
}

#[test]
fn open_and_added_kan_defer_dora_until_the_rinshan_discard() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        let num_players = mode.num_players() as usize;

        for action_type in [ActionType::Daiminkan, ActionType::Kakan] {
            let mut game = engine(mode, true);
            let mut hands = vec![vec![]; num_players];
            let mut melds = vec![vec![]; num_players];
            if action_type == ActionType::Daiminkan {
                hands[0] = vec![0, 4, 8, 32, 36, 40, 44, 48, 72, 76, 80, 108, 112, 116];
                hands[1] = vec![37, 38, 39, 1, 33, 41, 45, 49, 73, 77, 81, 109, 113];
                prepare_wait_act(&mut game, 0, 36, &hands, &melds);
            } else {
                hands[0] = vec![0, 32, 39, 40, 44, 48, 72, 76, 80, 108, 112];
                melds[0] = vec![Meld::new(
                    MeldType::Pon,
                    vec![36, 37, 38],
                    true,
                    1,
                    Some(36),
                )];
                prepare_wait_act(&mut game, 0, 39, &hands, &melds);
            }

            let before_dora = dora_count(&game);
            let kan_outcome = if action_type == ActionType::Daiminkan {
                let initial = game.decisions();
                let discard = find_action(&initial, 0, ActionType::Discard, |action| {
                    action.tile == Some(36)
                });
                let response = step_one(&mut game, 0, discard);
                let kan = find_action(&response.decisions, 1, ActionType::Daiminkan, |_| true);
                step_one(&mut game, 1, kan)
            } else {
                let initial = game.decisions();
                let kan = find_action(&initial, 0, ActionType::Kakan, |action| {
                    action.tile == Some(39)
                });
                step_one(&mut game, 0, kan)
            };

            let actor = if action_type == ActionType::Daiminkan {
                1
            } else {
                0
            };
            let kan_events = event_types(&kan_outcome);
            assert_eq!(
                kan_events.first().map(String::as_str),
                Some(match action_type {
                    ActionType::Daiminkan => "daiminkan",
                    ActionType::Kakan => "kakan",
                    _ => unreachable!(),
                })
            );
            assert_eq!(kan_events.last().map(String::as_str), Some("tsumo"));
            assert!(!kan_events.iter().any(|event| event == "dora"));
            assert_eq!(dora_count(&game), before_dora);
            assert_eq!(pending_kan_dora_count(&game), 1);

            let rinshan = match game.state() {
                GameStateVariant::FourPlayer(state) => state.drawn_tile,
                GameStateVariant::ThreePlayer(state) => state.drawn_tile,
            }
            .expect("kan must draw a rinshan tile");
            let discard = find_action(
                &kan_outcome.decisions,
                actor,
                ActionType::Discard,
                |action| action.tile == Some(rinshan),
            );
            let discard_outcome = step_one(&mut game, actor, discard);
            let discard_events = event_types(&discard_outcome);
            let dora_index = discard_events
                .iter()
                .position(|event| event == "dora")
                .expect("deferred kan dora event must be emitted");
            let dahai_index = discard_events
                .iter()
                .position(|event| event == "dahai")
                .expect("rinshan discard event must be emitted");
            assert!(dora_index < dahai_index);
            assert_eq!(dora_count(&game), before_dora + 1);
            assert_eq!(pending_kan_dora_count(&game), 0);
        }
    }
}

#[test]
fn kakan_waits_for_chankan_instead_of_drawing_rinshan() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        let num_players = mode.num_players() as usize;
        let mut game = engine(mode, true);
        let mut hands = vec![vec![]; num_players];
        let mut melds = vec![vec![]; num_players];

        // Player 0 adds the fourth 1p to an existing pon. Player 1 has a
        // yaku-less hand waiting on 1p, so the chankan condition is what makes
        // the ron legal: 23p 456p 456s 789s EE.
        hands[0] = vec![39, 60, 64, 68, 72, 76, 80, 110, 112, 116, 120];
        melds[0] = vec![Meld::new(
            MeldType::Pon,
            vec![36, 37, 38],
            true,
            1,
            Some(36),
        )];
        hands[1] = vec![40, 44, 48, 52, 56, 84, 88, 92, 96, 100, 104, 108, 109];
        prepare_wait_act(&mut game, 0, 39, &hands, &melds);

        let before_dora = dora_count(&game);
        let before_snapshot = game.snapshot();
        let before_wall = before_snapshot.wall_tiles_remaining;
        let kakan = find_action(&game.decisions(), 0, ActionType::Kakan, |action| {
            action.tile == Some(39)
        });
        let response = step_one(&mut game, 0, kakan);

        assert_eq!(event_types(&response), ["kakan"]);
        assert_eq!(response.snapshot.phase, Phase::WaitResponse);
        assert_eq!(response.snapshot.active_players, [1]);
        assert_eq!(response.snapshot.wall_tiles_remaining, before_wall);
        assert_eq!(dora_count(&game), before_dora);
        assert_eq!(rinshan_draw_count(&game), 0);
        match game.state() {
            GameStateVariant::FourPlayer(state) => {
                assert!(state.pending_kan.is_some());
                assert_eq!(state.last_discard, Some((0, 39)));
            }
            GameStateVariant::ThreePlayer(state) => {
                assert!(state.pending_kan.is_some());
                assert_eq!(state.last_discard, Some((0, 39)));
            }
        }

        let ron = find_action(&response.decisions, 1, ActionType::Ron, |action| {
            action.tile == Some(39)
        });
        let outcome = step_one(&mut game, 1, ron);
        let hora = outcome
            .events
            .iter()
            .map(|raw| serde_json::from_str::<serde_json::Value>(raw).expect("valid event"))
            .find(|event| event["type"] == "hora")
            .expect("chankan must produce hora");

        assert_eq!(hora["actor"], 1);
        assert_eq!(hora["target"], 0);
        assert!(!event_types(&outcome).iter().any(|event| event == "tsumo"));
        assert!(outcome.snapshot.scores[1] > before_snapshot.scores[1]);
        assert!(outcome.snapshot.scores[0] < before_snapshot.scores[0]);
    }
}
