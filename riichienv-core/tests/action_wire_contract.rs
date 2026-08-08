use riichienv_core::action::{ACTION_SPACE_3P, ACTION_SPACE_4P, Action, ActionEncoder, ActionType};
use serde_json::{Value, json};

fn wire(action: Action) -> Value {
    serde_json::from_str(&action.to_mjai()).unwrap()
}

#[test]
fn all_action_types_keep_the_v048_mjai_wire() {
    let cases = [
        (
            Action::new(ActionType::Discard, Some(16), vec![], Some(0)),
            json!({"type": "dahai", "actor": 0, "pai": "5mr"}),
        ),
        (
            Action::new(ActionType::Chi, Some(53), vec![57, 49], Some(1)),
            json!({"type": "chi", "actor": 1, "pai": "5p", "consumed": ["4p", "6p"]}),
        ),
        (
            Action::new(ActionType::Pon, Some(88), vec![89, 90], Some(2)),
            json!({"type": "pon", "actor": 2, "pai": "5sr", "consumed": ["5s", "5s"]}),
        ),
        (
            Action::new(
                ActionType::Daiminkan,
                Some(108),
                vec![109, 110, 111],
                Some(3),
            ),
            json!({"type": "daiminkan", "actor": 3, "pai": "E", "consumed": ["E", "E", "E"]}),
        ),
        (
            Action::new(
                ActionType::Ankan,
                Some(124),
                vec![124, 125, 126, 127],
                Some(0),
            ),
            json!({"type": "ankan", "actor": 0, "pai": "P", "consumed": ["P", "P", "P", "P"]}),
        ),
        (
            Action::new(
                ActionType::Kakan,
                Some(132),
                vec![132, 133, 134, 135],
                Some(1),
            ),
            json!({"type": "kakan", "actor": 1, "pai": "C", "consumed": ["C", "C", "C", "C"]}),
        ),
        (
            Action::new(ActionType::Riichi, Some(52), vec![], Some(2)),
            json!({"type": "reach", "actor": 2}),
        ),
        (
            Action::new(ActionType::Tsumo, Some(52), vec![], Some(3)),
            json!({"type": "hora", "actor": 3}),
        ),
        (
            Action::new(ActionType::Ron, Some(52), vec![], Some(0)),
            json!({"type": "hora", "actor": 0}),
        ),
        (
            Action::new(ActionType::KyushuKyuhai, None, vec![], Some(1)),
            json!({"type": "ryukyoku", "actor": 1}),
        ),
        (
            Action::new(ActionType::Pass, None, vec![], Some(2)),
            json!({"type": "none", "actor": 2}),
        ),
        (
            Action::new(ActionType::Kita, Some(120), vec![], Some(0)),
            json!({"type": "kita", "actor": 0, "pai": "N"}),
        ),
    ];

    for (action, expected) in cases {
        assert_eq!(wire(action), expected);
    }
}

#[test]
fn four_player_action_id_layout_is_stable() {
    let encoder = ActionEncoder::FourPlayer;
    assert_eq!(encoder.action_space_size(), ACTION_SPACE_4P);
    for tile_type in 0..34u8 {
        let tile = tile_type * 4;
        assert_eq!(
            encoder
                .encode(&Action::new(ActionType::Discard, Some(tile), vec![], None))
                .unwrap(),
            i32::from(tile_type)
        );
        assert_eq!(
            encoder
                .encode(&Action::new(
                    ActionType::Daiminkan,
                    Some(tile),
                    vec![],
                    None
                ))
                .unwrap(),
            42 + i32::from(tile_type)
        );
    }

    let fixed = [
        (ActionType::Riichi, 37),
        (ActionType::Pon, 41),
        (ActionType::Ron, 79),
        (ActionType::Tsumo, 79),
        (ActionType::KyushuKyuhai, 80),
        (ActionType::Pass, 81),
    ];
    for (action_type, expected) in fixed {
        assert_eq!(
            encoder
                .encode(&Action::new(action_type, None, vec![], None))
                .unwrap(),
            expected
        );
    }

    assert_eq!(
        encoder
            .encode(&Action::new(ActionType::Chi, Some(4), vec![8, 12], None))
            .unwrap(),
        38
    );
    assert_eq!(
        encoder
            .encode(&Action::new(ActionType::Chi, Some(8), vec![4, 12], None))
            .unwrap(),
        39
    );
    assert_eq!(
        encoder
            .encode(&Action::new(ActionType::Chi, Some(12), vec![4, 8], None))
            .unwrap(),
        40
    );
}

#[test]
fn three_player_action_id_layout_is_stable() {
    let encoder = ActionEncoder::ThreePlayer;
    assert_eq!(encoder.action_space_size(), ACTION_SPACE_3P);
    let valid_tile_types = [
        0u8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33,
    ];
    for (compact, tile_type) in valid_tile_types.into_iter().enumerate() {
        let tile = tile_type * 4;
        assert_eq!(
            encoder
                .encode(&Action::new(ActionType::Discard, Some(tile), vec![], None))
                .unwrap(),
            compact as i32
        );
        assert_eq!(
            encoder
                .encode(&Action::new(
                    ActionType::Daiminkan,
                    Some(tile),
                    vec![],
                    None
                ))
                .unwrap(),
            29 + compact as i32
        );
    }

    let fixed = [
        (ActionType::Riichi, 27),
        (ActionType::Pon, 28),
        (ActionType::Ron, 56),
        (ActionType::Tsumo, 56),
        (ActionType::KyushuKyuhai, 57),
        (ActionType::Pass, 58),
        (ActionType::Kita, 59),
    ];
    for (action_type, expected) in fixed {
        assert_eq!(
            encoder
                .encode(&Action::new(action_type, None, vec![], None))
                .unwrap(),
            expected
        );
    }

    assert!(
        encoder
            .encode(&Action::new(ActionType::Discard, Some(4), vec![], None))
            .is_err()
    );
    assert!(
        encoder
            .encode(&Action::new(ActionType::Chi, Some(36), vec![40, 44], None))
            .is_err()
    );
}
