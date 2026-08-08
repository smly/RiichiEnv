//! Multi-kyoku replay boundaries shared by the typed replay and event journal.

use riichienv_core::replay::{Action, EventCursor, EventJournal, ReplayLog};
use riichienv_core::rule::GameRule;
use serde_json::{Value, json};

fn two_kyoku_events(num_players: usize) -> (Vec<String>, Vec<i32>, Vec<i32>) {
    let initial_score = if num_players == 4 { 25_000 } else { 35_000 };
    let initial_scores = vec![initial_score; num_players];
    let mut second_scores = initial_scores.clone();
    second_scores[0] += 1_000;
    second_scores[1] -= 1_000;

    let mut first_delta = vec![0; num_players];
    first_delta[0] = 1_000;
    first_delta[1] = -1_000;
    let zero_delta = vec![0; num_players];
    let first_hands = (0..num_players)
        .map(|index| vec![["1m", "1p", "1s", "E"][index]])
        .collect::<Vec<_>>();
    let second_hands = (0..num_players)
        .map(|index| vec![["9m", "9p", "9s", "S"][index]])
        .collect::<Vec<_>>();

    let events = vec![
        json!({"type": "start_game"}),
        json!({
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": initial_scores,
            "dora_marker": "1m",
            "tehais": first_hands,
        }),
        json!({"type": "tsumo", "actor": 0, "pai": "5p"}),
        json!({"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": true}),
        json!({"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": first_delta}),
        json!({"type": "end_kyoku"}),
        json!({
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 2,
            "honba": 1,
            "kyotaku": 0,
            "oya": 1,
            "scores": second_scores,
            "dora_marker": "2p",
            "tehais": second_hands,
        }),
        json!({"type": "tsumo", "actor": 1, "pai": "6s"}),
        json!({"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": true}),
        json!({"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": zero_delta}),
        json!({"type": "end_kyoku"}),
        json!({"type": "end_game"}),
    ]
    .into_iter()
    .map(|event: Value| event.to_string())
    .collect();

    (events, initial_scores, second_scores)
}

#[test]
fn typed_replay_and_journal_share_two_kyoku_boundaries_and_scores() {
    for num_players in [4, 3] {
        let (events, initial_scores, second_scores) = two_kyoku_events(num_players);
        let jsonl = format!("{}\n", events.join("\n"));

        let replay = ReplayLog::from_jsonl(&jsonl, GameRule::default_tenhou())
            .expect("two complete kyokus must parse");
        assert_eq!(replay.len(), 2);
        assert_eq!(replay.rounds()[0].scores, initial_scores);
        assert_eq!(replay.rounds()[0].end_scores, second_scores);
        assert_eq!(replay.rounds()[1].scores, second_scores);
        assert_eq!(replay.rounds()[1].end_scores, second_scores);
        assert_eq!(
            (
                replay.rounds()[0].chang,
                replay.rounds()[0].ju,
                replay.rounds()[0].ben
            ),
            (0, 0, 0)
        );
        assert_eq!(
            (
                replay.rounds()[1].chang,
                replay.rounds()[1].ju,
                replay.rounds()[1].ben
            ),
            (0, 1, 1)
        );
        assert!(matches!(
            replay.rounds()[0].actions()[0],
            Action::DealTile { seat: 0, .. }
        ));
        assert!(matches!(
            replay.rounds()[0].actions()[1],
            Action::DiscardTile { seat: 0, .. }
        ));
        assert!(matches!(replay.rounds()[0].actions()[2], Action::NoTile));
        assert!(matches!(
            replay.rounds()[1].actions()[0],
            Action::DealTile { seat: 1, .. }
        ));

        let mut cursor = replay.cursor();
        assert_eq!(cursor.remaining(), 2);
        cursor
            .seek(1)
            .expect("second kyoku is a valid cursor target");
        assert_eq!(cursor.next().expect("second kyoku").ju, 1);
        assert_eq!(cursor.remaining(), 0);

        let journal = EventJournal::from_jsonl(&jsonl).expect("journal must accept replay JSONL");
        assert!(journal.is_complete());
        assert_eq!(journal.completed_kyokus().len(), 2);
        assert_eq!(journal.completed_kyokus()[0].start, EventCursor(1));
        assert_eq!(journal.completed_kyokus()[0].end, EventCursor(6));
        assert_eq!(journal.completed_kyokus()[0].key.kyoku, Some(1));
        assert_eq!(journal.completed_kyokus()[0].key.honba, Some(0));
        assert_eq!(journal.completed_kyokus()[1].start, EventCursor(6));
        assert_eq!(journal.completed_kyokus()[1].end, EventCursor(11));
        assert_eq!(journal.completed_kyokus()[1].key.kyoku, Some(2));
        assert_eq!(journal.completed_kyokus()[1].key.honba, Some(1));
        assert_eq!(
            journal.events_for_kyoku(0).expect("first kyoku"),
            &events[1..6]
        );
        assert_eq!(
            journal
                .prefix_through_completed_kyoku(0)
                .expect("first complete prefix"),
            &events[..6]
        );
        assert_eq!(journal.spectator_prefix(1), events.as_slice());

        let mut live = EventJournal::new();
        for event in &events[..6] {
            live.push_json(event).expect("first kyoku event");
        }
        assert_eq!(live.spectator_prefix(1), &events[..1]);

        live.push_json(&events[6]).expect("second start_kyoku");
        assert_eq!(live.spectator_end(1), EventCursor(6));
        assert_eq!(live.spectator_prefix(1), &events[..6]);
        assert_eq!(
            live.spectator_events_since(EventCursor(1), 1)
                .expect("newly visible first kyoku"),
            &events[1..6]
        );

        for event in &events[7..] {
            live.push_json(event).expect("remaining replay event");
        }
        assert!(live.is_complete());
        assert_eq!(live.revision(), EventCursor(events.len()));
        assert_eq!(live.spectator_prefix(1), events.as_slice());
    }
}
