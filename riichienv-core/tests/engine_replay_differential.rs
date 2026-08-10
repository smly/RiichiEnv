//! End-to-end deterministic engine/replay differential traces.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::engine::{Decision, EngineConfig, GameEngine, GameMode, GameSnapshot};
use riichienv_core::parser::tid_to_mjai;
use riichienv_core::replay::{Action as ReplayAction, EventCursor, EventJournal, ReplayLog};
use riichienv_core::rule::GameRule;
use riichienv_core::types::MeldType;
use serde_json::Value;

const MAX_STEPS: usize = 20_000;

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecisionTrace {
    player_id: u8,
    legal_action_ids: Vec<usize>,
    legal_action_ids_v1: Vec<usize>,
    selected: Action,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StepTrace {
    before: GameSnapshot,
    decisions: Vec<DecisionTrace>,
    events: Vec<String>,
    after: GameSnapshot,
}

#[derive(Debug, Clone)]
struct Simulation {
    trace: Vec<StepTrace>,
    events: Vec<String>,
    final_snapshot: GameSnapshot,
    journal: EventJournal,
}

fn sorted_ones(mask: &[u8]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(index, &value)| (value == 1).then_some(index))
        .collect()
}

fn choose_action(decision: &Decision, phase: Phase) -> Action {
    let mut legal = decision.observation.legal_actions();
    legal.sort_by_key(|action| {
        (
            action.tile.unwrap_or(u8::MAX),
            action.consume_tiles.clone(),
            action.action_type as i32,
        )
    });

    let preferred = match phase {
        Phase::WaitAct => [
            ActionType::Tsumo,
            ActionType::Discard,
            ActionType::KyushuKyuhai,
            ActionType::Kita,
            ActionType::Ankan,
            ActionType::Kakan,
            ActionType::Riichi,
            ActionType::Pass,
            ActionType::Ron,
            ActionType::Daiminkan,
            ActionType::Pon,
            ActionType::Chi,
        ],
        Phase::WaitResponse => [
            ActionType::Ron,
            ActionType::Pass,
            ActionType::Daiminkan,
            ActionType::Pon,
            ActionType::Chi,
            ActionType::Tsumo,
            ActionType::Discard,
            ActionType::KyushuKyuhai,
            ActionType::Kita,
            ActionType::Ankan,
            ActionType::Kakan,
            ActionType::Riichi,
        ],
    };

    preferred
        .into_iter()
        .find_map(|kind| {
            legal
                .iter()
                .find(|action| action.action_type == kind)
                .cloned()
        })
        .unwrap_or_else(|| panic!("player {} has no selectable action", decision.player_id))
}

fn trace_decision(decision: &Decision, phase: Phase) -> DecisionTrace {
    assert_eq!(decision.player_id, decision.observation.player_id());

    let mut legal_action_ids = decision
        .observation
        .legal_action_ids()
        .expect("legacy action IDs must encode");
    legal_action_ids.sort_unstable();
    let mut legal_action_ids_v1 = decision
        .observation
        .legal_action_ids_v1()
        .expect("v1 action IDs must encode");
    legal_action_ids_v1.sort_unstable();

    assert_eq!(
        sorted_ones(
            &decision
                .observation
                .action_mask()
                .expect("legacy action mask")
        ),
        legal_action_ids
    );
    assert_eq!(
        sorted_ones(
            &decision
                .observation
                .action_mask_v1()
                .expect("v1 action mask")
        ),
        legal_action_ids_v1
    );
    for &action_id in &legal_action_ids {
        decision
            .observation
            .select_action(action_id)
            .expect("every set legacy mask bit must select a legal action");
    }
    for &action_id in &legal_action_ids_v1 {
        decision
            .observation
            .select_action_v1(action_id)
            .expect("every set v1 mask bit must select a legal action");
    }

    DecisionTrace {
        player_id: decision.player_id,
        legal_action_ids,
        legal_action_ids_v1,
        selected: choose_action(decision, phase),
    }
}

fn simulate(mode: GameMode, seed: u64) -> Simulation {
    let mut engine = GameEngine::new(
        EngineConfig::new(mode)
            .with_seed(seed)
            .with_rule(GameRule::default_tenhou()),
    )
    .expect("full-game engine must initialize");
    let mut decisions = engine.decisions();
    let mut trace = Vec::new();

    while !engine.is_done() {
        assert!(
            trace.len() < MAX_STEPS,
            "{mode:?} seed {seed} exceeded step cap"
        );
        let before = engine.snapshot();
        assert_eq!(
            decisions
                .iter()
                .map(|item| item.player_id)
                .collect::<Vec<_>>(),
            before.active_players,
            "pending decisions must mirror the snapshot"
        );

        let decision_trace = decisions
            .iter()
            .map(|decision| trace_decision(decision, before.phase))
            .collect::<Vec<_>>();
        let actions = decision_trace
            .iter()
            .map(|decision| (decision.player_id, decision.selected.clone()))
            .collect::<HashMap<_, _>>();
        let previous_log_len = engine.mjai_log().len();
        let outcome = engine.step(&actions);
        assert_eq!(outcome.error, None, "engine step must remain legal");
        assert_eq!(outcome.snapshot, engine.snapshot());
        assert_eq!(
            outcome.events,
            engine.mjai_log()[previous_log_len..],
            "StepOutcome must expose exactly the appended MJAI suffix"
        );

        trace.push(StepTrace {
            before,
            decisions: decision_trace,
            events: outcome.events.clone(),
            after: outcome.snapshot.clone(),
        });
        decisions = outcome.decisions;
    }

    assert!(decisions.is_empty());
    let events = engine.mjai_log().to_vec();
    assert_eq!(
        serde_json::from_str::<Value>(events.last().expect("completed game has events"))
            .expect("valid final event")["type"],
        "end_game"
    );
    let journal = engine.event_journal().expect("engine log must index");
    assert_eq!(journal.events(), events);

    Simulation {
        trace,
        events,
        final_snapshot: engine.snapshot(),
        journal,
    }
}

fn json(raw: &str) -> Value {
    serde_json::from_str(raw).expect("engine MJAI event must be valid JSON")
}

fn event_type(raw: &str) -> String {
    let value = json(raw);
    value["type"].as_str().expect("MJAI event type").to_owned()
}

fn raw_action_trace(events: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < events.len() {
        let event = json(&events[index]);
        match event["type"].as_str().expect("event type") {
            "tsumo" => result.push(format!(
                "tsumo:{}:{}",
                event["actor"].as_u64().expect("tsumo actor"),
                event["pai"].as_str().expect("tsumo tile")
            )),
            "dahai" => result.push(format!(
                "dahai:{}:{}",
                event["actor"].as_u64().expect("dahai actor"),
                event["pai"].as_str().expect("dahai tile")
            )),
            "chi" | "pon" | "daiminkan" | "kan" => result.push(format!(
                "call:{}:{}",
                event["type"].as_str().expect("call type"),
                event["actor"].as_u64().expect("call actor")
            )),
            "ankan" | "kakan" => result.push(format!(
                "closed_kan:{}:{}",
                event["type"].as_str().expect("kan type"),
                event["actor"].as_u64().expect("kan actor")
            )),
            "dora" => result.push(format!(
                "dora:{}",
                event["dora_marker"].as_str().expect("dora marker")
            )),
            "kita" => result.push(format!(
                "kita:{}",
                event["actor"].as_u64().expect("kita actor")
            )),
            "hora" => {
                let mut actors = Vec::new();
                while index < events.len() {
                    let hora = json(&events[index]);
                    if hora["type"] != "hora" {
                        break;
                    }
                    actors.push(hora["actor"].as_u64().expect("hora actor"));
                    index += 1;
                }
                result.push(format!("hora:{actors:?}"));
                continue;
            }
            "ryukyoku" => result.push("ryukyoku".to_string()),
            _ => {}
        }
        index += 1;
    }
    result
}

fn replay_action_trace(actions: &[ReplayAction]) -> Vec<String> {
    actions
        .iter()
        .map(|action| match action {
            ReplayAction::DealTile { seat, tile, .. } => {
                format!("tsumo:{seat}:{}", tid_to_mjai(*tile))
            }
            ReplayAction::DiscardTile { seat, tile, .. } => {
                format!("dahai:{seat}:{}", tid_to_mjai(*tile))
            }
            ReplayAction::ChiPengGang {
                seat, meld_type, ..
            } => {
                let kind = match meld_type {
                    MeldType::Chi => "chi",
                    MeldType::Pon => "pon",
                    MeldType::Daiminkan => "daiminkan",
                    other => panic!("unexpected open-call type {other:?}"),
                };
                format!("call:{kind}:{seat}")
            }
            ReplayAction::AnGangAddGang {
                seat, meld_type, ..
            } => {
                let kind = match meld_type {
                    MeldType::Ankan => "ankan",
                    MeldType::Kakan => "kakan",
                    other => panic!("unexpected closed-kan type {other:?}"),
                };
                format!("closed_kan:{kind}:{seat}")
            }
            ReplayAction::Dora { dora_marker } => {
                format!("dora:{}", tid_to_mjai(*dora_marker))
            }
            ReplayAction::Hule { hules } => format!(
                "hora:{:?}",
                hules
                    .iter()
                    .map(|hule| hule.seat as u64)
                    .collect::<Vec<_>>()
            ),
            ReplayAction::NoTile => "ryukyoku".to_string(),
            ReplayAction::BaBei { seat, .. } => format!("kita:{seat}"),
            ReplayAction::LiuJu { .. } => "ryukyoku".to_string(),
            ReplayAction::Other(value) => format!("other:{value}"),
        })
        .collect()
}

fn scores_after_events(events: &[String]) -> Vec<i32> {
    let start = json(events.first().expect("kyoku starts with start_kyoku"));
    assert_eq!(start["type"], "start_kyoku");
    let mut scores = start["scores"]
        .as_array()
        .expect("start scores")
        .iter()
        .map(|score| score.as_i64().expect("integer score") as i32)
        .collect::<Vec<_>>();

    for raw in &events[1..] {
        let event = json(raw);
        match event["type"].as_str().expect("event type") {
            "reach_accepted" => {
                let actor = event["actor"].as_u64().expect("reach actor") as usize;
                scores[actor] -= 1_000;
            }
            "hora" | "ryukyoku" => {
                if let Some(deltas) = event.get("deltas").and_then(Value::as_array) {
                    for (score, delta) in scores.iter_mut().zip(deltas) {
                        *score += delta.as_i64().expect("integer delta") as i32;
                    }
                }
            }
            _ => {}
        }
    }
    scores
}

fn validate_replay(simulation: &Simulation, mode: GameMode) {
    let jsonl = format!("{}\n", simulation.events.join("\n"));
    let replay = ReplayLog::from_jsonl(&jsonl, GameRule::default_tenhou())
        .expect("engine-generated full game must parse as typed replay");
    let journal = &simulation.journal;

    assert!(journal.is_complete());
    assert!(!journal.has_in_progress_kyoku());
    assert_eq!(journal.revision(), EventCursor(simulation.events.len()));
    assert_eq!(journal.completed_kyokus().len(), replay.len());
    assert_eq!(journal.spectator_prefix(1), simulation.events);
    assert!(replay.len() >= mode.num_players() as usize);

    for (index, (span, round)) in journal
        .completed_kyokus()
        .iter()
        .zip(replay.rounds())
        .enumerate()
    {
        let events = journal.events_for_kyoku(index).expect("indexed kyoku");
        assert_eq!(
            events.last().map(|raw| event_type(raw)),
            Some("end_kyoku".to_string())
        );
        let start = json(events.first().expect("start_kyoku event"));
        let expected_wind = ["E", "S", "W", "N"][round.chang as usize];
        assert_eq!(start["bakaze"], expected_wind);
        assert_eq!(start["kyoku"], round.ju + 1);
        assert_eq!(start["honba"], round.ben);
        assert_eq!(start["scores"], serde_json::json!(round.scores));
        assert_eq!(span.key.bakaze.as_deref(), Some(expected_wind));
        assert_eq!(span.key.kyoku, Some(round.ju + 1));
        assert_eq!(span.key.honba, Some(round.ben));

        assert_eq!(
            raw_action_trace(events),
            replay_action_trace(round.actions()),
            "typed action order differs in kyoku {index} for {mode:?}"
        );
        assert_eq!(
            round.end_scores,
            scores_after_events(events),
            "score settlement differs in kyoku {index} for {mode:?}"
        );
        if let Some(next) = replay.rounds().get(index + 1) {
            assert_eq!(
                round.end_scores, next.scores,
                "score continuity at kyoku {index}"
            );
        }
    }
    assert_eq!(
        replay
            .rounds()
            .last()
            .expect("at least one kyoku")
            .end_scores,
        simulation.final_snapshot.scores
    );

    let mut cursor = replay.cursor();
    for expected in replay.rounds() {
        assert!(std::ptr::eq(cursor.next().expect("cursor round"), expected));
    }
    assert_eq!(cursor.remaining(), 0);

    let mut live = EventJournal::new();
    let mut completed = 0;
    for raw in &simulation.events {
        let revision = live.push_json(raw).expect("incremental engine event");
        assert_eq!(revision, live.revision());
        if event_type(raw) == "end_kyoku" {
            completed += 1;
            let prefix_replay = ReplayLog::from_jsonl(&live.to_jsonl(), GameRule::default_tenhou())
                .expect("every completed engine prefix must replay");
            assert_eq!(prefix_replay.len(), completed);
            assert_eq!(live.completed_kyokus().len(), completed);
        }
    }
    assert_eq!(live.events(), simulation.events);
    assert!(live.is_complete());

    // Cut inside the second kyoku: typed replay exposes only the first
    // completed round, while the journal keeps the unfinished suffix framed.
    let second = journal.completed_kyokus().get(1).expect("multi-kyoku game");
    let cut = (second.start.0 + 2).min(second.end.0 - 1);
    let truncated = EventJournal::from_events(simulation.events[..cut].iter().cloned())
        .expect("truncated authoritative prefix must remain valid");
    let truncated_replay = ReplayLog::from_jsonl(&truncated.to_jsonl(), GameRule::default_tenhou())
        .expect("typed replay accepts unfinished EOF");
    assert!(truncated.has_in_progress_kyoku());
    assert_eq!(truncated_replay.len(), 1);
    assert_eq!(truncated.completed_kyokus().len(), 1);
    assert_eq!(
        truncated.spectator_end(1),
        journal.completed_kyokus()[0].end
    );

    // A corrupted actor must fail at the typed replay boundary rather than
    // producing a subtly different action trace.
    let mut corrupted = simulation.events.clone();
    let discard_index = corrupted
        .iter()
        .position(|raw| event_type(raw) == "dahai")
        .expect("full game contains a discard");
    let mut discard = json(&corrupted[discard_index]);
    discard["actor"] = serde_json::json!(mode.num_players());
    corrupted[discard_index] = discard.to_string();
    assert!(
        ReplayLog::from_jsonl(&corrupted.join("\n"), GameRule::default_tenhou()).is_err(),
        "out-of-range replay actor must be rejected"
    );
}

#[test]
fn deterministic_east_and_half_games_round_trip_through_replay() {
    let fixtures = [
        (GameMode::FourPlayerEast, 0x4e01),
        (GameMode::FourPlayerHalf, 0x4e02),
        (GameMode::ThreePlayerEast, 0x3e01),
        (GameMode::ThreePlayerHalf, 0x3e02),
    ];

    for (mode, seed) in fixtures {
        let first = simulate(mode, seed);
        let second = simulate(mode, seed);
        assert_eq!(
            first.trace, second.trace,
            "decision trace changed for {mode:?}"
        );
        assert_eq!(first.events, second.events, "MJAI log changed for {mode:?}");
        assert_eq!(first.final_snapshot, second.final_snapshot);
        validate_replay(&first, mode);
    }
}
