use riichienv_core::drev_validation::validate_drev_replay;
use riichienv_core::replay::ReplayLog;
use riichienv_core::rule::GameRule;

struct Fixture {
    name: &'static str,
    jsonl: &'static str,
    rule: GameRule,
    allow_trailing_start_kyoku: bool,
}

fn fixtures() -> [Fixture; 4] {
    [
        Fixture {
            name: "tenhou-4p-ranked",
            jsonl: include_str!("../../tests/data/external_replay/tenhou_4p_ranked_excerpt.jsonl"),
            rule: GameRule::default_tenhou(),
            allow_trailing_start_kyoku: true,
        },
        Fixture {
            name: "tenhou-4p-dealer-tsumo",
            jsonl: include_str!(
                "../../tests/data/external_replay/tenhou_4p_dealer_tsumo_excerpt.jsonl"
            ),
            rule: GameRule::default_tenhou(),
            allow_trailing_start_kyoku: true,
        },
        Fixture {
            name: "tenhou-4p-honba-kyotaku",
            jsonl: include_str!(
                "../../tests/data/external_replay/tenhou_4p_honba_kyotaku_excerpt.jsonl"
            ),
            rule: GameRule::default_tenhou(),
            allow_trailing_start_kyoku: false,
        },
        Fixture {
            name: "mjsoul-3p-double-ron",
            jsonl: include_str!(
                "../../tests/data/external_replay/mjsoul_3p_double_ron_excerpt.jsonl"
            ),
            rule: GameRule::default_mjsoul(),
            allow_trailing_start_kyoku: true,
        },
    ]
}

#[test]
fn committed_replays_satisfy_drev_hidden_state_contract() {
    let mut total_cells = 0;
    let mut total_hard_safe = 0;
    for fixture in fixtures() {
        let replay = if fixture.allow_trailing_start_kyoku {
            ReplayLog::from_jsonl_strict_allowing_trailing_start_kyoku(fixture.jsonl, fixture.rule)
        } else {
            ReplayLog::from_jsonl_strict(fixture.jsonl, fixture.rule)
        }
        .unwrap();
        let report = validate_drev_replay(&replay).unwrap();
        eprintln!(
            "{}: {}",
            fixture.name,
            serde_json::to_string(&report).unwrap()
        );
        report.ensure_valid().unwrap();
        assert_eq!(report.rounds, 1);
        assert!(report.decisions > 20);
        assert!(report.candidate_tile_types > report.decisions);
        total_cells += report.opponent_candidate_cells;
        total_hard_safe += report.hard_safe_cells;
    }
    assert!(total_cells > 1_000);
    assert!(total_hard_safe > 100);
}

#[test]
fn incomplete_replay_is_rejected() {
    let replay = ReplayLog::from_jsonl(
        r#"{"type":"start_game","names":["A","B","C","D"]}
{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[[],[],[],[]]}
"#,
        GameRule::default_tenhou(),
    )
    .unwrap();
    assert!(replay.is_empty());
    let error = validate_drev_replay(&replay).unwrap_err().to_string();
    assert!(error.contains("at least one complete round"));
}

#[test]
fn tracked_multi_round_replay_exercises_riichi_public_history() {
    let replay = ReplayLog::from_jsonl_strict(
        include_str!("../../tests/data/126_204_0_mjai.jsonl"),
        GameRule::default_tenhou(),
    )
    .unwrap();
    let report = validate_drev_replay(&replay).unwrap();

    report.ensure_valid().unwrap();
    assert_eq!(report.rounds, 12);
    assert_eq!(report.decisions, 659);
    assert_eq!(report.opponent_candidate_cells, 19_734);
    assert_eq!(report.legal_ron_cells, 210);
    assert_eq!(report.hard_safe_cells, 2_556);
    assert_eq!(
        report.semantic_sha256,
        "4c43da255d76bcaa9d2606a5cb317b37b646c8ae404822d8b8a0a80e188d633f"
    );
    assert_eq!(report.yaku_support.get("riichi"), Some(&48));
}
