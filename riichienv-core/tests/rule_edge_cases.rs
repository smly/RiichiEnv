//! High-risk rule boundaries exercised through the public engine facade.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::engine::{Decision, EngineConfig, GameEngine, GameMode, StepOutcome};
use riichienv_core::game_variant::GameStateVariant;
use riichienv_core::rule::GameRule;
use riichienv_core::types::{Meld, MeldType};

fn engine(mode: GameMode, rule: GameRule) -> GameEngine {
    GameEngine::new(EngineConfig::new(mode).with_seed(73).with_rule(rule))
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
) -> Option<Action>
where
    F: Fn(&Action) -> bool,
{
    decisions
        .iter()
        .find(|decision| decision.player_id == player_id)?
        .observation
        .legal_actions()
        .into_iter()
        .find(|action| action.action_type == action_type && predicate(action))
}

fn require_action<F>(
    decisions: &[Decision],
    player_id: u8,
    action_type: ActionType,
    predicate: F,
) -> Action
where
    F: Fn(&Action) -> bool,
{
    find_action(decisions, player_id, action_type, predicate)
        .unwrap_or_else(|| panic!("missing {action_type:?} action for player {player_id}"))
}

fn step_one(engine: &mut GameEngine, player_id: u8, action: Action) -> StepOutcome {
    let outcome = engine.step(&HashMap::from([(player_id, action)]));
    assert_eq!(outcome.error, None, "step failed: {:?}", outcome.error);
    outcome
}

fn parsed_events(outcome: &StepOutcome) -> Vec<serde_json::Value> {
    outcome
        .events
        .iter()
        .map(|raw| serde_json::from_str(raw).expect("engine event must be valid JSON"))
        .collect()
}

fn event_types(outcome: &StepOutcome) -> Vec<String> {
    outcome
        .events
        .iter()
        .map(|raw| {
            serde_json::from_str::<serde_json::Value>(raw).expect("engine event must be valid JSON")
                ["type"]
                .as_str()
                .expect("engine event must have a type")
                .to_owned()
        })
        .collect()
}

fn set_riichi_declared(engine: &mut GameEngine, player_id: usize) {
    match engine.state_mut() {
        GameStateVariant::FourPlayer(state) => {
            state.players[player_id].riichi_declared = true;
        }
        GameStateVariant::ThreePlayer(state) => {
            state.players[player_id].riichi_declared = true;
        }
    }
}

fn ankan_meld(tile_type: u8) -> Meld {
    let first = tile_type * 4;
    Meld::new(
        MeldType::Ankan,
        vec![first, first + 1, first + 2, first + 3],
        false,
        -1,
        None,
    )
}

fn drawn_tile(engine: &GameEngine) -> u8 {
    match engine.state() {
        GameStateVariant::FourPlayer(state) => state.drawn_tile,
        GameStateVariant::ThreePlayer(state) => state.drawn_tile,
    }
    .expect("engine must have a drawn tile")
}

fn has_discard(decisions: &[Decision], player_id: u8, tile: u8) -> bool {
    find_action(decisions, player_id, ActionType::Discard, |action| {
        action.tile == Some(tile)
    })
    .is_some()
}

#[test]
fn kokushi_can_rob_ankan_only_when_the_rule_allows_it() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for allow_chankan in [false, true] {
            let mut rule = GameRule::default_tenhou();
            rule.allows_ron_on_ankan_for_kokushi_musou = allow_chankan;
            let mut game = engine(mode, rule);
            let num_players = mode.num_players() as usize;
            let mut hands = vec![vec![]; num_players];

            // Player 0 declares ankan on 9p. Player 1 holds kokushi missing 9p.
            hands[0] = vec![40, 44, 48, 52, 56, 68, 69, 70, 71, 76, 80, 84, 88, 92];
            hands[1] = vec![0, 1, 32, 36, 72, 104, 108, 112, 116, 120, 124, 128, 132];
            prepare_wait_act(&mut game, 0, 71, &hands, &vec![vec![]; num_players]);

            let before = game.snapshot();
            let ankan = require_action(&game.decisions(), 0, ActionType::Ankan, |action| {
                action.tile.map(|tile| tile / 4) == Some(17)
            });
            let response = step_one(&mut game, 0, ankan);

            if allow_chankan {
                assert!(
                    response.events.is_empty(),
                    "ankan is not final before chankan"
                );
                assert_eq!(response.snapshot.phase, Phase::WaitResponse);
                assert_eq!(response.snapshot.active_players, [1]);
                assert_eq!(
                    response.snapshot.wall_tiles_remaining,
                    before.wall_tiles_remaining
                );

                let ron = require_action(&response.decisions, 1, ActionType::Ron, |action| {
                    action.tile.map(|tile| tile / 4) == Some(17)
                });
                let outcome = step_one(&mut game, 1, ron);
                let hora = parsed_events(&outcome)
                    .into_iter()
                    .find(|event| event["type"] == "hora")
                    .expect("kokushi chankan must produce hora");
                assert_eq!(hora["actor"], 1);
                assert_eq!(hora["target"], 0);
                assert!(
                    !outcome
                        .events
                        .iter()
                        .any(|event| event.contains("\"type\":\"ankan\""))
                );
            } else {
                assert_eq!(event_types(&response), ["ankan", "dora", "tsumo"]);
                assert_eq!(response.snapshot.phase, Phase::WaitAct);
                assert_eq!(response.snapshot.active_players, [0]);
                assert_eq!(
                    response.snapshot.wall_tiles_remaining,
                    before.wall_tiles_remaining - 1
                );
                assert!(find_action(&response.decisions, 1, ActionType::Ron, |_| true).is_none());
            }
        }
    }
}

#[test]
fn riichi_ankan_requires_the_wait_set_to_stay_unchanged() {
    let fixtures = [
        (
            true,
            vec![36, 37, 38, 39, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108],
        ),
        (
            false,
            vec![36, 37, 38, 39, 40, 44, 52, 56, 60, 80, 88, 96, 100, 104],
        ),
    ];

    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for (waits_unchanged, hand) in &fixtures {
            let mut game = engine(mode, GameRule::default_tenhou());
            let num_players = mode.num_players() as usize;
            let mut hands = vec![vec![]; num_players];
            hands[0] = hand.clone();
            prepare_wait_act(&mut game, 0, 39, &hands, &vec![vec![]; num_players]);
            set_riichi_declared(&mut game, 0);

            let decisions = game.decisions();
            let ankan = find_action(&decisions, 0, ActionType::Ankan, |action| {
                action.tile.map(|tile| tile / 4) == Some(9)
            });
            assert_eq!(
                ankan.is_some(),
                *waits_unchanged,
                "riichi ankan legality differs in {mode:?}"
            );

            if let Some(ankan) = ankan {
                let outcome = step_one(&mut game, 0, ankan);
                assert_eq!(event_types(&outcome), ["ankan", "dora", "tsumo"]);
            } else {
                let legal = decisions[0].observation.legal_actions();
                assert!(
                    legal
                        .iter()
                        .all(|action| action.action_type != ActionType::Kakan)
                );
                assert!(legal.iter().any(|action| {
                    action.action_type == ActionType::Discard && action.tile == Some(39)
                }));
            }
        }
    }
}

#[test]
fn fourth_kan_aborts_only_when_multiple_players_own_the_kans() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for split_owners in [false, true] {
            let mut game = engine(mode, GameRule::default_tenhou());
            let num_players = mode.num_players() as usize;
            let mut hands = vec![vec![]; num_players];
            let mut melds = vec![vec![]; num_players];
            hands[0] = vec![36, 37, 38, 39, 40, 44, 48, 52, 56, 72, 76, 80, 84, 88];

            if split_owners {
                melds[0].push(ankan_meld(27));
                melds[1].push(ankan_meld(28));
                melds[2].push(ankan_meld(29));
            } else {
                melds[0] = vec![ankan_meld(27), ankan_meld(28), ankan_meld(29)];
            }
            prepare_wait_act(&mut game, 0, 39, &hands, &melds);

            let fourth = require_action(&game.decisions(), 0, ActionType::Ankan, |action| {
                action.tile.map(|tile| tile / 4) == Some(9)
            });
            let after_kan = step_one(&mut game, 0, fourth);
            assert_eq!(event_types(&after_kan), ["ankan", "dora", "tsumo"]);

            let rinshan = drawn_tile(&game);
            let discard = require_action(&after_kan.decisions, 0, ActionType::Discard, |action| {
                action.tile == Some(rinshan)
            });
            let outcome = step_one(&mut game, 0, discard);
            let events = parsed_events(&outcome);
            let abort = events.iter().find(|event| event["type"] == "ryukyoku");

            if split_owners {
                let abort = abort.expect("four kans owned by multiple players must abort");
                assert_eq!(abort["reason"], "suukansansen");
                assert!(!events.iter().any(|event| event["type"] == "tsumo"));
            } else {
                assert!(abort.is_none(), "one player's four kans must continue");
                assert!(events.iter().any(|event| event["type"] == "tsumo"));
                assert_eq!(outcome.snapshot.phase, Phase::WaitAct);
                assert_eq!(outcome.snapshot.active_players, [1]);
            }
        }
    }
}

#[test]
fn red_five_chi_kuikae_applies_to_all_copies_of_the_called_tile() {
    for kuikae_forbidden in [false, true] {
        let mut rule = GameRule::default_tenhou();
        rule.kuikae_forbidden = kuikae_forbidden;
        let mut game = engine(GameMode::FourPlayerSingle, rule);
        let hands = vec![
            vec![0, 4, 16, 24, 28, 32, 36, 40, 44, 48, 72, 76, 80, 108],
            vec![8, 12, 17, 20, 25, 29, 33, 41, 45, 49, 73, 77, 81],
            vec![],
            vec![],
        ];
        prepare_wait_act(&mut game, 0, 16, &hands, &vec![vec![]; 4]);

        let discard = require_action(&game.decisions(), 0, ActionType::Discard, |action| {
            action.tile == Some(16)
        });
        let response = step_one(&mut game, 0, discard);
        let chi = require_action(&response.decisions, 1, ActionType::Chi, |action| {
            action.consume_tiles == [8, 12]
        });
        let after_chi = step_one(&mut game, 1, chi);

        assert_eq!(
            has_discard(&after_chi.decisions, 1, 17),
            !kuikae_forbidden,
            "normal 5m remaining after calling red 5m must use type-level kuikae"
        );
    }
}

#[test]
fn red_five_pon_kuikae_applies_in_four_and_three_player_games() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for kuikae_forbidden in [false, true] {
            let mut rule = GameRule::default_tenhou();
            rule.kuikae_forbidden = kuikae_forbidden;
            let mut game = engine(mode, rule);
            let num_players = mode.num_players() as usize;
            let mut hands = vec![vec![]; num_players];
            hands[0] = vec![0, 32, 52, 72, 76, 80, 96, 100, 104, 108, 112, 116, 120, 124];
            hands[1] = vec![53, 54, 55, 40, 44, 48, 56, 60, 64, 84, 88, 92, 128];
            prepare_wait_act(&mut game, 0, 52, &hands, &vec![vec![]; num_players]);

            let discard = require_action(&game.decisions(), 0, ActionType::Discard, |action| {
                action.tile == Some(52)
            });
            let response = step_one(&mut game, 0, discard);
            let pon = require_action(&response.decisions, 1, ActionType::Pon, |action| {
                action.consume_tiles == [53, 54]
            });
            let after_pon = step_one(&mut game, 1, pon);

            assert_eq!(
                has_discard(&after_pon.decisions, 1, 55),
                !kuikae_forbidden,
                "normal 5p remaining after calling red 5p must use type-level kuikae in {mode:?}"
            );
        }
    }
}

#[test]
fn all_opponents_ron_distinguishes_four_player_sanchaho_from_sanma_double_ron() {
    for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
        for sanchaho_is_draw in [false, true] {
            let mut rule = GameRule::default_tenhou();
            rule.sanchaho_is_draw = sanchaho_is_draw;
            let mut game = engine(mode, rule);
            let num_players = mode.num_players() as usize;
            let mut hands = vec![vec![]; num_players];
            hands[0] = vec![36, 43, 47, 51, 55, 59, 87, 91, 95, 99, 103, 107, 120, 121];

            // 23p 456p 456s 789s plus a wind pair waits on 1p. Riichi is set so each
            // responder has a legal ron independent of other shape yaku. Use
            // distinct physical copies and a different wind pair per player.
            for (offset, hand) in hands[1..].iter_mut().enumerate() {
                let copy = offset as u8;
                *hand = vec![
                    40 + copy,
                    44 + copy,
                    48 + copy,
                    52 + copy,
                    56 + copy,
                    84 + copy,
                    88 + copy,
                    92 + copy,
                    96 + copy,
                    100 + copy,
                    104 + copy,
                    108 + offset as u8 * 4,
                    109 + offset as u8 * 4,
                ];
            }
            prepare_wait_act(&mut game, 0, 36, &hands, &vec![vec![]; num_players]);
            for player_id in 1..num_players {
                set_riichi_declared(&mut game, player_id);
            }

            let discard = require_action(&game.decisions(), 0, ActionType::Discard, |action| {
                action.tile == Some(36)
            });
            let response = step_one(&mut game, 0, discard);
            assert_eq!(response.snapshot.phase, Phase::WaitResponse);
            assert_eq!(
                response.snapshot.active_players,
                (1..num_players as u8).collect::<Vec<_>>()
            );

            let actions = (1..num_players as u8)
                .map(|player_id| {
                    let ron =
                        require_action(&response.decisions, player_id, ActionType::Ron, |action| {
                            action.tile == Some(36)
                        });
                    (player_id, ron)
                })
                .collect::<HashMap<_, _>>();
            let outcome = game.step(&actions);
            assert_eq!(outcome.error, None, "multi-ron batch must be accepted");
            let events = parsed_events(&outcome);

            let should_abort = num_players == 4 && sanchaho_is_draw;
            if should_abort {
                let draw = events
                    .iter()
                    .find(|event| event["type"] == "ryukyoku")
                    .expect("all-opponent ron must become sanchaho");
                assert_eq!(draw["reason"], "sanchaho");
                assert!(!events.iter().any(|event| event["type"] == "hora"));
            } else {
                let horas = events
                    .iter()
                    .filter(|event| event["type"] == "hora")
                    .collect::<Vec<_>>();
                assert_eq!(horas.len(), num_players - 1);
                assert!(horas.iter().all(|event| event["target"] == 0));
                assert_eq!(
                    horas
                        .iter()
                        .map(|event| event["actor"].as_u64().expect("numeric actor") as u8)
                        .collect::<Vec<_>>(),
                    (1..num_players as u8).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn response_priority_is_ron_then_pon_then_chi() {
    for take_ron in [false, true] {
        let mut game = engine(GameMode::FourPlayerSingle, GameRule::default_tenhou());
        let hands = vec![
            vec![36, 61, 65, 70, 73, 77, 81, 85, 89, 93, 108, 109, 110, 112],
            vec![40, 44, 48, 52, 56, 60, 64, 84, 88, 92, 124, 128, 132],
            vec![37, 38, 53, 54, 55, 68, 69, 96, 97, 100, 101, 104, 105],
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 39, 72, 76, 80],
        ];
        prepare_wait_act(&mut game, 0, 36, &hands, &vec![vec![]; 4]);
        set_riichi_declared(&mut game, 3);

        let discard = require_action(&game.decisions(), 0, ActionType::Discard, |action| {
            action.tile == Some(36)
        });
        let response = step_one(&mut game, 0, discard);
        assert_eq!(response.snapshot.active_players, [1, 2, 3]);

        let chi = require_action(&response.decisions, 1, ActionType::Chi, |action| {
            action.consume_tiles == [40, 44]
        });
        let pon = require_action(&response.decisions, 2, ActionType::Pon, |action| {
            action.consume_tiles == [37, 38]
        });
        let player_three = require_action(
            &response.decisions,
            3,
            if take_ron {
                ActionType::Ron
            } else {
                ActionType::Pass
            },
            |_| true,
        );
        let outcome = game.step(&HashMap::from([(1, chi), (2, pon), (3, player_three)]));
        assert_eq!(outcome.error, None);
        let events = parsed_events(&outcome);

        if take_ron {
            let hora = events
                .iter()
                .find(|event| event["type"] == "hora")
                .expect("ron must override every call");
            assert_eq!(hora["actor"], 3);
            assert_eq!(hora["target"], 0);
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event["type"].as_str(), Some("chi" | "pon")))
            );
        } else {
            let pon = events
                .iter()
                .find(|event| event["type"] == "pon")
                .expect("pon must override chi");
            assert_eq!(pon["actor"], 2);
            assert_eq!(pon["target"], 0);
            assert!(!events.iter().any(|event| event["type"] == "chi"));
            assert_eq!(outcome.snapshot.phase, Phase::WaitAct);
            assert_eq!(outcome.snapshot.active_players, [2]);
        }
    }
}
