use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::parser::tid_to_mjai;
use riichienv_core::rule::GameRule;
use riichienv_core::state::{GameState, legal_actions::GameStateLegalActions};
use serde_json::{Value, json};

fn take_tile(unused: &mut Vec<u8>, pai: &Value) -> u8 {
    let index = unused
        .iter()
        .position(|&tile| tid_to_mjai(tile) == pai.as_str().unwrap())
        .expect("the log must not use more copies of a tile than exist");
    unused.remove(index)
}

#[test]
fn issue_214_south_1_repeats_after_non_dealer_nagashi() {
    // Unmodified South 1 excerpt from the reported game (no calls or kans):
    // https://logs.riichi.dev/mjai-logs/2026/05/11/8f6b51ad-3847-4a4f-83a2-9f86501bfec5.jsonl.gz
    let events: Vec<Value> = include_str!("fixtures/issue_214_south_1.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let start = &events[0];
    let mut unused: Vec<u8> = (0..136).collect();
    let hands: Vec<Vec<u8>> = start["tehais"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hand| {
            hand.as_array()
                .unwrap()
                .iter()
                .map(|pai| take_tile(&mut unused, pai))
                .collect()
        })
        .collect();

    // Reconstruct a wall consistent with every visible tile. Physical copies
    // of identical non-red tiles and unobserved dead-wall slots are arbitrary.
    let mut wall = Vec::new();
    for packet in 0..3 {
        for hand in &hands {
            wall.extend_from_slice(&hand[packet * 4..packet * 4 + 4]);
        }
    }
    wall.extend(hands.iter().map(|hand| hand[12]));
    for event in &events {
        if event["type"] == "tsumo" {
            wall.push(take_tile(&mut unused, &event["pai"]));
        }
    }
    assert_eq!(unused.len(), 14);
    let dora_index = unused
        .iter()
        .position(|&tile| tid_to_mjai(tile) == start["dora_marker"].as_str().unwrap())
        .unwrap();
    unused.swap(9, dora_index); // First dora marker is fifth from the wall's end.
    wall.extend(unused);
    assert_eq!(wall.len(), 136);

    let scores: Vec<i32> = serde_json::from_value(start["scores"].clone()).unwrap();
    let mut state = GameState::new(2, false, Some(214), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state._initialize_round(0, 1, 0, 0, Some(wall), Some(scores));

    for event in &events {
        let action_type = match event["type"].as_str().unwrap() {
            "reach" => ActionType::Riichi,
            "dahai" => ActionType::Discard,
            _ => continue,
        };
        let actor = event["actor"].as_u64().unwrap() as u8;
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.current_player, actor);
        let action = state
            ._get_legal_actions_internal(actor)
            .into_iter()
            .find(|action| {
                action.action_type == action_type
                    && (action_type == ActionType::Riichi
                        || action.tile.is_some_and(|tile| {
                            tid_to_mjai(tile) == event["pai"].as_str().unwrap()
                                && (Some(tile) == state.drawn_tile)
                                    == event["tsumogiri"].as_bool().unwrap()
                        }))
            })
            .expect("every recorded action must be legal");
        state.step(&HashMap::from([(actor, action)]));
        // The log contains no calls: all offered claims were passed.
        if state.phase == Phase::WaitResponse {
            let passes = state
                .active_players
                .iter()
                .map(|&pid| (pid, Action::new(ActionType::Pass, None, vec![], Some(pid))))
                .collect();
            state.step(&passes);
        }
        assert!(state.last_error.is_none(), "{:?}", state.last_error);
    }

    // All events through settlement, including the riichi deposit in deltas,
    // match the report. Only the dealer of the following round changes.
    let generated: Vec<Value> = state
        .mjai_log
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(&generated[..events.len()], events.as_slice());
    let next = &generated[events.len()];
    assert_eq!(next["type"], "start_kyoku");
    assert_eq!(next["bakaze"], "S");
    assert_eq!(
        next["kyoku"], 1,
        "the reported log incorrectly advances to South 2"
    );
    assert_eq!(next["oya"], 0, "the riichi dealer remains tenpai");
    assert_eq!(next["honba"], 1);
    assert_eq!(next["kyotaku"], 1);
    assert_eq!(next["scores"], json!([30600, 17000, 36600, 14800]));
}
