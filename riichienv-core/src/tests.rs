#[cfg(test)]
mod unit_tests {
    use crate::action::Phase;
    use crate::agari::{is_agari, is_chiitoitsu, is_kokushi};
    use crate::score::calculate_score;
    use crate::types::Hand;

    #[test]
    fn test_agari_standard() {
        // Pinfu Tsumo: 123 456 789 m 234 p 55 s
        let tiles = [
            0, 1, 2, // 123m
            3, 4, 5, // 456m
            6, 7, 8, // 789m
            9, 10, 11, // 123p (mapped to 9,10,11)
            18, 18, // 1s pair (mapped to 18)
        ];
        let mut hand = Hand::new(Some(tiles.to_vec()));
        assert!(is_agari(&mut hand), "Should be agari");
    }

    #[test]
    fn test_basic_pinfu() {
        // 123m 456m 789m 123p 11s
        // m: 0-8, p: 9-17, s: 18-26_
        // 123p -> 9, 10, 11
        // 11s -> 18, 18
        let mut hand = Hand::new(None);
        // 123m
        hand.add(0);
        hand.add(1);
        hand.add(2);
        // 456m
        hand.add(3);
        hand.add(4);
        hand.add(5);
        // 789m
        hand.add(6);
        hand.add(7);
        hand.add(8);
        // 123p
        hand.add(9);
        hand.add(10);
        hand.add(11);
        // 11s (pair)
        hand.add(18);
        hand.add(18);

        assert!(is_agari(&mut hand));
    }

    #[test]
    fn test_chiitoitsu() {
        let mut hand = Hand::new(None);
        let pairs = [0, 2, 4, 6, 8, 10, 12];
        for &t in &pairs {
            hand.add(t);
            hand.add(t);
        }
        assert!(is_chiitoitsu(&hand));
        assert!(is_agari(&mut hand));
    }

    #[test]
    fn test_kokushi() {
        let mut hand = Hand::new(None);
        // 1m,9m, 1p,9p, 1s,9s, 1z-7z
        let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
        for &t in &terminals {
            hand.add(t);
        }
        hand.add(0); // Double 1m
        assert!(is_kokushi(&hand));
        assert!(is_agari(&mut hand));
    }

    #[test]
    fn test_score_calculation() {
        // Current implementation does NOT do Kiriage Mangan (rounding 1920->2000).
        // So base is 1920.
        // Oya pays: ceil(1920*2/100)*100 = 3900.
        // Ko pays: ceil(1920/100)*100 = 2000.
        // Total: 3900 + 2000*2 = 7900.

        let score = calculate_score(4, 30, false, true, 0, 4); // Ko Tsumo

        assert_eq!(score.pay_tsumo_oya, 3900);
        assert_eq!(score.pay_tsumo_ko, 2000);
        assert_eq!(score.total, 7900); // 3900 + 2000 + 2000
    }

    #[test]
    fn test_tsuu_iisou() {
        use crate::yaku::{YakuContext, calculate_yaku};
        let mut hand = Hand::new(None);
        // 111z, 222z, 333z, 444z, 55z
        for &t in &[27, 28, 29, 30] {
            hand.add(t);
            hand.add(t);
            hand.add(t);
        }
        hand.add(31);
        hand.add(31);

        let res = calculate_yaku(&hand, &[], &YakuContext::default(), 31);
        assert!(res.han >= 13);
        assert!(res.yaku_ids.contains(&39));
    }

    #[test]
    fn test_ryuu_iisou() {
        use crate::yaku::{YakuContext, calculate_yaku};
        let mut hand = Hand::new(None);
        // 234s, 666s, 888s, 6s6s6s (Wait, 6s6s6s is already there)
        // Correct 234s, 666s, 888s, Hatsuz, 6s6s (pair)
        let tiles = [
            19, 20, 21, // 234s
            23, 23, 23, // 666s
            25, 25, 25, // 888s
            32, 32, 32, // Hatsuz
            19, 19, // 2s pair
        ];
        for &t in &tiles {
            hand.add(t);
        }

        let res = calculate_yaku(&hand, &[], &YakuContext::default(), 19);
        assert!(res.han >= 13);
        assert!(res.yaku_ids.contains(&40));
    }

    #[test]
    fn test_daisushii() {
        use crate::yaku::{YakuContext, calculate_yaku};
        let mut hand = Hand::new(None);
        // EEEz, SSSz, WWWz, NNNz, 11m
        for &t in &[27, 28, 29, 30] {
            hand.add(t);
            hand.add(t);
            hand.add(t);
        }
        hand.add(0);
        hand.add(0);

        let res = calculate_yaku(&hand, &[], &YakuContext::default(), 0);
        assert!(res.han >= 26);
        assert!(res.yaku_ids.contains(&50));
    }

    fn create_test_state(game_type: u8) -> crate::state::GameState {
        crate::state::GameState::new(game_type, false, None, 0, crate::rule::GameRule::default())
    }

    #[test]
    fn test_seeded_shuffle_changes_between_rounds() {
        let mut state = create_test_state(2);
        state.seed = Some(42);

        state._initialize_next_round(true, false);
        let digest1 = state.wall.wall_digest.clone();

        state._initialize_next_round(true, false);
        let digest2 = state.wall.wall_digest.clone();

        assert_ne!(
            digest1, digest2,
            "Wall digest should differ between rounds when seed is fixed"
        );
    }

    #[test]
    fn test_sudden_death_hanchan_logic() {
        use serde_json::Value;

        let mut state = create_test_state(2);
        state.round_wind = 1;
        state.kyoku_idx = 3;
        state.oya = 3;
        for i in 0..4 {
            state.players[i].score = 25000;
            state.players[i].nagashi_eligible = false;
        }
        state.needs_initialize_next_round = false;

        state._trigger_ryukyoku("exhaustive_draw");

        if state.needs_initialize_next_round {
            state._initialize_next_round(state.pending_oya_won, state.pending_is_draw);
            state.needs_initialize_next_round = false;
        }

        assert!(
            !state.is_done,
            "Game should not be done (Sudden Death should trigger)"
        );
        assert_eq!(state.round_wind, 2, "Should enter West round");
        assert_eq!(state.kyoku_idx, 0, "Should be West 1 (Kyoku 0)");
        assert_eq!(state.oya, 0, "Oya should rotate to player 0");

        let new_scores = [31000, 25000, 24000, 20000];
        for (player, &score) in state.players.iter_mut().zip(new_scores.iter()) {
            player.score = score;
        }

        state._trigger_ryukyoku("exhaustive_draw");
        if state.needs_initialize_next_round {
            state._initialize_next_round(state.pending_oya_won, state.pending_is_draw);
            state.needs_initialize_next_round = false;
        }

        assert!(
            state.is_done,
            "Game should be done (Score >= 30000 in West)"
        );

        let logs = &state.mjai_log;
        let event_types: Vec<String> = logs
            .iter()
            .filter_map(|s| {
                let v: Value = serde_json::from_str(s).ok()?;
                v.get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t.to_string())
            })
            .collect();

        let last_event = event_types.last().expect("Should have events");
        assert_eq!(last_event, "end_game");

        assert!(event_types.contains(&"ryukyoku".to_string()));
    }

    #[test]
    fn test_ryukyoku_deltas_are_reset_each_round_4p() {
        use serde_json::Value;

        let mut state = create_test_state(2);

        state.players[0].score_delta = 2300;
        state.players[1].score_delta = -2300;
        state.players[2].score_delta = 0;
        state.players[3].score_delta = 0;

        state._initialize_round(0, 0, 0, 0, None, None);
        state._trigger_ryukyoku("sufuurenta");

        let v: Value = state
            .mjai_log
            .iter()
            .rev()
            .find_map(|s| {
                let ev: Value = serde_json::from_str(s).ok()?;
                (ev["type"] == "ryukyoku").then_some(ev)
            })
            .expect("ryukyoku event should be logged");
        assert_eq!(v["reason"], "sufuurenta");

        let deltas: Vec<i32> = serde_json::from_value(v["deltas"].clone()).expect("deltas");
        assert_eq!(deltas, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_is_tenpai() {
        use crate::hand_evaluator::HandEvaluator;
        // 111,222,333m, 444p, 11s (Tenpai on 1s)
        let hand = vec![0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 72];
        let calc = HandEvaluator::new(hand, Vec::new());
        assert!(calc.is_tenpai());
        let waits = calc.get_waits_u8();
        assert!(waits.contains(&18)); // 1s
    }

    #[test]
    fn test_kuikae_deadlock_repro() {
        use crate::action::{Action, ActionType};
        use std::collections::HashMap;

        let mut state = create_test_state(2);
        let pid = 0;

        // Hand: 4m, 5m, 6m, 6m. (12, 16, 20, 21)
        // 3m is 8.
        state.players[pid as usize].hand = vec![12, 16, 20, 21];

        // Setup P3 (Kamicha of P0)
        state.current_player = 3;
        state.phase = Phase::WaitAct;
        state.active_players = vec![3];
        state.players[3].hand.push(8); // Give 3m

        // Action: P3 discards 3m
        let mut actions = HashMap::new();
        actions.insert(3, Action::new(ActionType::Discard, Some(8), vec![], None));

        state.step(&actions);

        state.step(&actions);

        assert_eq!(
            state.phase,
            Phase::WaitAct,
            "Should proceed to WaitAct as deadlock Chi is filtered out"
        );
        assert_eq!(state.current_player, 0, "Should be P0's turn");

        // Verify current_claims is empty or does not contain 0
        if let Some(claims) = state.current_claims.get(&0) {
            assert!(claims.is_empty(), "P0 should have no legal claims");
        }
    }
    #[test]
    fn test_match_84_agari_check() {
        use crate::hand_evaluator::HandEvaluator;
        use crate::types::{Conditions, Wind};

        // Hand: 111m, 78p, 11123s, 789s
        // 1m: 0
        // 7p: 15. 8p: 16.
        // 1s: 18. 2s: 19. 3s: 20.
        // 7s: 24. 8s: 25. 9s: 26.

        let mut tiles = vec![
            0, 1, 2,  // 1m x3
            60, // 7p (15*4)
            64, // 8p (16*4)
            72, 73, 74,  // 1s x3
            76,  // 2s (19*4)
            80,  // 3s (20*4)
            96,  // 7s (24*4)
            100, // 8s (25*4)
            104, // 9s (26*4)
        ];
        tiles.sort();

        let calc = HandEvaluator::new(tiles, Vec::new());

        let cond = Conditions {
            tsumo: false,
            riichi: false,
            double_riichi: false,
            ippatsu: false,
            haitei: false,
            houtei: false,
            rinshan: false,
            chankan: false,
            tsumo_first_turn: false,
            player_wind: Wind::West,
            round_wind: Wind::East,
            riichi_sticks: 0,
            honba: 0,
            ..Default::default()
        };

        // 1. Check 6p (14 -> 56)
        let res6p = calc.calc(56, vec![], vec![], Some(cond.clone()));
        println!(
            "6p Result: is_win={}, Shape={}, Han={}, Yaku={:?}",
            res6p.is_win, res6p.has_win_shape, res6p.han, res6p.yaku
        );
        assert!(!res6p.is_win, "6p should NOT be a win (No Yaku)");
        assert!(res6p.has_win_shape, "6p should have win shape");
        assert_eq!(res6p.han, 0, "6p should have 0 Han");

        // 2. Check 9p (17 -> 68)
        let res9p = calc.calc(68, vec![], vec![], Some(cond));
        println!(
            "9p Result: is_win={}, Han={}, Yaku={:?}",
            res9p.is_win, res9p.han, res9p.yaku
        );
        assert!(res9p.is_win, "9p should be a win");
        assert!(res9p.han >= 3, "9p should be Junchan (>= 3 Han)"); // Junchan (3)
    }

    #[test]
    fn test_tobi_ends_game() {
        let mut state = create_test_state(2);

        // Set scores with one player having negative score
        state.players[0].score = 30000;
        state.players[1].score = 40000;
        state.players[2].score = 35000;
        state.players[3].score = -5000; // Negative score - should trigger tobi

        state.needs_initialize_next_round = false;

        // Try to initialize next round - should end game due to tobi
        state._initialize_next_round(false, false);

        assert!(
            state.is_done,
            "Game should be done due to tobi (player with negative score)"
        );

        // Verify MJAI log contains end_kyoku followed by end_game
        let log = &state.mjai_log;
        let ek_pos = log.iter().position(|s| s.contains("\"end_kyoku\""));
        let eg_pos = log.iter().position(|s| s.contains("\"end_game\""));
        assert!(
            ek_pos.is_some(),
            "end_kyoku must be emitted before end_game on tobi"
        );
        assert!(eg_pos.is_some(), "end_game must be emitted on tobi");
        assert!(
            ek_pos.unwrap() < eg_pos.unwrap(),
            "end_kyoku must appear before end_game"
        );
    }

    #[test]
    fn test_oyayame_requires_target_in_orasu_4p() {
        let mut state = create_test_state(2);
        state.round_wind = 1;
        state.oya = 3;
        state.players[0].score = 28900;
        state.players[1].score = 20000;
        state.players[2].score = 20100;
        state.players[3].score = 29000;

        state._initialize_next_round(true, false);

        assert!(
            !state.is_done,
            "Game should continue if the orasu dealer is top but below 30000"
        );
    }

    #[test]
    fn test_apply_mjai_event_honor_and_red_tiles() {
        use crate::replay::MjaiEvent;

        let mut state =
            crate::state::GameState::new(2, true, None, 0, crate::rule::GameRule::default());

        // start_kyoku with mjai-format tiles: honors (E, S, W, N, P, F, C) and red fives (5pr, 5sr)
        let start = MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![25000, 25000, 25000, 25000],
            dora_marker: "P".to_string(), // White dragon (tid 124)
            tehais: vec![
                // Player 0: E, S, W, N, P, F, C, 1m, 2m, 3m, 4m, 5m, 6m
                vec![
                    "E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                // Player 1: 1s, 2s, 3s, 4s, 5sr, 6s, 7s, 8s, 9s, 1p, 2p, 3p, 4p
                vec![
                    "1s", "2s", "3s", "4s", "5sr", "6s", "7s", "8s", "9s", "1p", "2p", "3p", "4p",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                // Player 2: 5pr, 1m, 2m, 3m, 4m, 6m, 7m, 8m, 9m, 1p, 2p, 3p, 4p
                vec![
                    "5pr", "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                // Player 3: all number tiles
                vec![
                    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "1m", "2m", "3m", "4m",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
            ],
        };
        state.apply_mjai_event(start);

        // Player 0: verify honor tiles are parsed correctly
        let hand0 = &state.players[0].hand;
        // E=108, S=112, W=116, N=120, P=124, F=128, C=132
        assert!(
            hand0.contains(&108),
            "E should be tid 108, hand: {:?}",
            hand0
        );
        assert!(
            hand0.contains(&112),
            "S should be tid 112, hand: {:?}",
            hand0
        );
        assert!(
            hand0.contains(&116),
            "W should be tid 116, hand: {:?}",
            hand0
        );
        assert!(
            hand0.contains(&120),
            "N should be tid 120, hand: {:?}",
            hand0
        );
        assert!(
            hand0.contains(&124),
            "P should be tid 124, hand: {:?}",
            hand0
        );
        assert!(
            hand0.contains(&128),
            "F should be tid 128, hand: {:?}",
            hand0
        );
        assert!(
            hand0.contains(&132),
            "C should be tid 132, hand: {:?}",
            hand0
        );

        // Player 1: verify red 5s (5sr = tid 88)
        let hand1 = &state.players[1].hand;
        assert!(
            hand1.contains(&88),
            "5sr should be tid 88, hand: {:?}",
            hand1
        );

        // Player 2: verify red 5p (5pr = tid 52)
        let hand2 = &state.players[2].hand;
        assert!(
            hand2.contains(&52),
            "5pr should be tid 52, hand: {:?}",
            hand2
        );

        // Dora marker "P" should be tid 124
        assert_eq!(
            state.wall.dora_indicators[0], 124,
            "dora_marker P should be tid 124, got: {}",
            state.wall.dora_indicators[0]
        );

        // Test tsumo with honor tile
        let tsumo = MjaiEvent::Tsumo {
            actor: 0,
            pai: "C".to_string(), // Red dragon (tid 132)
        };
        state.apply_mjai_event(tsumo);
        assert!(
            state.players[0].hand.contains(&132),
            "Tsumo C should add tid 132 to hand, hand: {:?}",
            state.players[0].hand
        );

        // Test dahai with honor tile
        let dahai = MjaiEvent::Dahai {
            actor: 0,
            pai: "E".to_string(), // East (tid 108)
            tsumogiri: false,
        };
        state.apply_mjai_event(dahai);
        assert!(
            state.players[0].discards.contains(&108),
            "Dahai E should discard tid 108, discards: {:?}",
            state.players[0].discards
        );

        // Test dora event with mjai honor
        let dora = MjaiEvent::Dora {
            dora_marker: "F".to_string(), // Green dragon (tid 128)
        };
        state.apply_mjai_event(dora);
        assert_eq!(
            state.wall.dora_indicators[1], 128,
            "dora F should be tid 128, got: {}",
            state.wall.dora_indicators[1]
        );
    }

    #[test]
    fn test_apply_mjai_event_start_kyoku_resets_pre_tsumo_wall_4p() {
        use crate::replay::MjaiEvent;

        let mut state =
            crate::state::GameState::new(2, true, None, 0, crate::rule::GameRule::default());

        assert_eq!(
            state.wall.tiles.len(),
            83,
            "fresh state should start post-tsumo"
        );

        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![25000, 25000, 25000, 25000],
            dora_marker: "1m".to_string(),
            tehais: vec![
                vec!["1m".to_string(); 13],
                vec!["2m".to_string(); 13],
                vec!["3m".to_string(); 13],
                vec!["4m".to_string(); 13],
            ],
        });

        assert_eq!(
            state.wall.tiles.len(),
            84,
            "start_kyoku should rewind to pre-tsumo"
        );
        assert!(state.needs_tsumo, "dealer draw should still be pending");
        assert!(state.drawn_tile.is_none(), "no tile should be drawn yet");

        state.apply_mjai_event(MjaiEvent::Tsumo {
            actor: 0,
            pai: "5m".to_string(),
        });

        assert_eq!(
            state.wall.tiles.len(),
            83,
            "first tsumo should consume exactly one tile"
        );
    }

    #[test]
    fn test_apply_mjai_event_start_kyoku_resets_pre_tsumo_wall_3p() {
        use crate::replay::MjaiEvent;

        let mut state =
            crate::state_3p::GameState3P::new(5, true, None, 0, crate::rule::GameRule::default());

        assert_eq!(
            state.wall.tiles.len(),
            68,
            "fresh sanma state should start post-tsumo"
        );

        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![35000, 35000, 35000],
            dora_marker: "1p".to_string(),
            tehais: vec![
                vec!["1p".to_string(); 13],
                vec!["2p".to_string(); 13],
                vec!["3p".to_string(); 13],
            ],
        });

        assert_eq!(
            state.wall.tiles.len(),
            69,
            "sanma start_kyoku should rewind to pre-tsumo"
        );
        assert!(state.needs_tsumo, "dealer draw should still be pending");
        assert!(state.drawn_tile.is_none(), "no tile should be drawn yet");

        state.apply_mjai_event(MjaiEvent::Tsumo {
            actor: 0,
            pai: "4p".to_string(),
        });

        assert_eq!(
            state.wall.tiles.len(),
            68,
            "first tsumo should consume exactly one tile"
        );
    }

    #[test]
    fn test_reach_to_mjai_includes_actor() {
        use crate::action::{Action, ActionType};

        // actor が Some のとき、to_mjai() の JSON に "actor" が含まれること
        let action = Action::new(ActionType::Riichi, None, vec![], Some(2));
        let json_str = action.to_mjai();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["type"], "reach");
        assert_eq!(v["actor"], 2, "reach event should contain actor=2");

        // actor が None のとき、"actor" キーが存在しないこと
        let action_no_actor = Action::new(ActionType::Riichi, None, vec![], None);
        let json_str2 = action_no_actor.to_mjai();
        let v2: serde_json::Value = serde_json::from_str(&json_str2).unwrap();
        assert_eq!(v2["type"], "reach");
        assert!(
            v2.get("actor").is_none(),
            "reach event without actor should not have actor key"
        );
    }

    #[test]
    fn test_reach_accepted_mjai_includes_actor() {
        use crate::action::{Action, ActionType};
        use std::collections::HashMap;

        // リーチ宣言→打牌→他家パスの流れで reach_accepted イベントが生成され、
        // actor が正しく含まれることを確認する
        let mut state = create_test_state(2);
        let pid: u8 = state.current_player;
        let pid_us = pid as usize;

        // テンパイ形の手牌を構築 (14枚 = 13 + ツモ牌):
        //   123m 456m 789m 12p 11s + ツモ 5sr(88)
        //   5sr を切ると 123456789m 12p 11s → 3p 待ちテンパイ
        state.players[pid_us].hand = vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 72, 73, 88];
        state.players[pid_us].hand.sort();
        state.players[pid_us].melds.clear();
        state.players[pid_us].score = 25000;
        state.players[pid_us].riichi_declared = false;
        state.players[pid_us].riichi_stage = false;
        state.players[pid_us].forbidden_discards.clear();
        state.drawn_tile = Some(88);
        state.phase = Phase::WaitAct;
        state.active_players = vec![pid];

        // Step 1: リーチ宣言 (tile=None)
        let mut actions = HashMap::new();
        actions.insert(pid, Action::new(ActionType::Riichi, None, vec![], None));
        state.step(&actions);

        // Step 2: 打牌 5sr(88) — リーチ宣言後の捨て牌
        let mut actions2 = HashMap::new();
        actions2.insert(
            pid,
            Action::new(ActionType::Discard, Some(88), vec![], None),
        );
        state.step(&actions2);

        // 他家にクレームがある場合は WaitResponse → 全員パス
        if state.phase == Phase::WaitResponse {
            let mut pass_actions = HashMap::new();
            for &ap in &state.active_players.clone() {
                pass_actions.insert(ap, Action::new(ActionType::Pass, None, vec![], None));
            }
            state.step(&pass_actions);
        }

        // mjai_log から reach_accepted イベントを探す
        let reach_accepted_event = state.mjai_log.iter().find_map(|s| {
            let v: serde_json::Value = serde_json::from_str(s).ok()?;
            if v["type"] == "reach_accepted" {
                Some(v)
            } else {
                None
            }
        });

        assert!(
            reach_accepted_event.is_some(),
            "mjai_log should contain a reach_accepted event. Log: {:?}",
            state.mjai_log
        );

        let v = reach_accepted_event.unwrap();
        assert_eq!(
            v["actor"],
            serde_json::Value::Number(pid.into()),
            "reach_accepted event should contain actor={}",
            pid
        );

        // riichi_pending_acceptance がクリアされていること
        assert!(state.riichi_pending_acceptance.is_none());
    }

    #[test]
    fn test_no_tobi_with_positive_scores() {
        let mut state = create_test_state(2);
        state.game_mode = 2; // 4p-red-half (Hanchan)
        state.round_wind = 0; // East round

        // Set scores with all players having positive scores
        state.players[0].score = 25000;
        state.players[1].score = 25000;
        state.players[2].score = 25000;
        state.players[3].score = 25000;

        state.needs_initialize_next_round = false;

        // Try to initialize next round - should NOT end game
        state._initialize_next_round(false, false);

        assert!(
            !state.is_done,
            "Game should NOT be done (all players have positive scores)"
        );
    }

    // ========== Sanma (3-player mahjong) Tests ==========

    fn create_sanma_test_state(game_type: u8) -> crate::state_3p::GameState3P {
        crate::state_3p::GameState3P::new(
            game_type,
            false,
            None,
            0,
            crate::rule::GameRule::default(),
        )
    }

    #[test]
    fn test_sanma_game_mode_config() {
        use crate::state_3p::game_mode;

        assert_eq!(game_mode::num_players(), 3);
        assert_eq!(game_mode::starting_score(), 35000);
        assert_eq!(game_mode::tenpai_pool(), 2000);
    }

    #[test]
    fn test_oyayame_requires_target_in_orasu_3p() {
        let mut state = create_sanma_test_state(5);
        state.round_wind = 1;
        state.oya = 2;
        state.players[0].score = 34900;
        state.players[1].score = 34000;
        state.players[2].score = 35100;

        state._initialize_next_round(true, false);

        assert!(
            !state.is_done,
            "Sanma should continue if the final dealer is top but below 40000"
        );
    }

    #[test]
    fn test_sanma_starting_scores() {
        let state = create_sanma_test_state(3);
        assert_eq!(state.players.len(), 3, "Should have exactly 3 players");
        for p in &state.players {
            assert_eq!(p.score, 35000, "Each player should start with 35000");
        }
    }

    #[test]
    fn test_sanma_wall_108_tiles() {
        let state = create_sanma_test_state(3);

        let total_tiles =
            state.wall.tiles.len() + state.players.iter().map(|p| p.hand.len()).sum::<usize>();
        assert_eq!(total_tiles, 108, "Total tiles should be 108 for sanma");

        // Verify no manzu 2-8 tiles (tile types 1-7, tile IDs 4-31)
        for p in &state.players {
            for &t in &p.hand {
                let tile_type = t / 4;
                assert!(
                    !(1..=7).contains(&tile_type),
                    "Hand should not contain manzu 2-8 (tile type {}), but found tile {}",
                    tile_type,
                    t
                );
            }
        }
        for &t in &state.wall.tiles {
            let tile_type = t / 4;
            assert!(
                !(1..=7).contains(&tile_type),
                "Wall should not contain manzu 2-8 (tile type {}), but found tile {}",
                tile_type,
                t
            );
        }
    }

    #[test]
    fn test_sanma_deal_3_players() {
        let state = create_sanma_test_state(3);

        assert_eq!(state.players.len(), 3);
        assert_eq!(
            state.players[0].hand.len(),
            14,
            "Oya (player 0) should have 14 tiles after deal"
        );
        for i in 1..3 {
            assert_eq!(
                state.players[i].hand.len(),
                13,
                "Player {} should have 13 tiles after deal",
                i
            );
        }
    }

    #[test]
    fn test_sanma_no_chi() {
        use crate::action::{Action, ActionType};
        use crate::state_3p::legal_actions::GameState3PLegalActions;
        use std::collections::HashMap;

        let mut state = create_sanma_test_state(5); // 3p-red-half
        state._initialize_round(0, 0, 0, 0, None, None);

        // Give player 1 a sequential hand that could chi
        state.players[1].hand = vec![36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84];
        state.current_player = 0;
        state.drawn_tile = Some(state.players[0].hand[0]);
        state.phase = Phase::WaitAct;
        state.active_players = vec![0];
        state.needs_tsumo = false;

        let discard_tile = state.players[0].hand[0];
        let mut actions = HashMap::new();
        actions.insert(
            0,
            Action::new(ActionType::Discard, Some(discard_tile), vec![], None),
        );
        state.step(&actions);

        let legal = state._get_legal_actions_internal(1);
        let has_chi = legal.iter().any(|a| a.action_type == ActionType::Chi);
        assert!(!has_chi, "Chi should not be available in sanma");
    }

    #[test]
    fn test_sanma_player_rotation() {
        let state = create_sanma_test_state(3);
        let np = state.np() as u8;

        // In sanma, players cycle 0 → 1 → 2 → 0
        assert_eq!(1u8 % np, 1, "Next player after 0 should be 1");
        assert_eq!((1u8 + 1) % np, 2, "Next player after 1 should be 2");
        assert_eq!((2u8 + 1) % np, 0, "Next player after 2 should be 0");
    }

    #[test]
    fn test_sanma_kita_action() {
        use crate::action::ActionType;
        use crate::state_3p::legal_actions::GameState3PLegalActions;

        let mut state = create_sanma_test_state(3);
        state._initialize_round(0, 0, 0, 0, None, None);

        state.players[0].hand = vec![0, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 120];
        state.drawn_tile = Some(120);
        state.current_player = 0;
        state.phase = Phase::WaitAct;
        state.active_players = vec![0];
        state.needs_tsumo = false;

        let legal = state._get_legal_actions_internal(0);
        let has_kita = legal.iter().any(|a| a.action_type == ActionType::Kita);
        assert!(
            has_kita,
            "Kita should be available when holding North tile in sanma"
        );
    }

    #[test]
    fn test_sanma_dora_wrapping() {
        use crate::state_3p::game_mode;

        // 1m (type 0) → 9m (type 8)
        assert_eq!(game_mode::get_next_dora_tile(0), 8);
        // 9m (type 8) → 1m (type 0)
        assert_eq!(game_mode::get_next_dora_tile(8), 0);
        // Pin/sou/honor wrapping should be standard
        assert_eq!(game_mode::get_next_dora_tile(9), 10); // 1p → 2p
        assert_eq!(game_mode::get_next_dora_tile(17), 9); // 9p → 1p
        assert_eq!(game_mode::get_next_dora_tile(27), 28); // East → South
        assert_eq!(game_mode::get_next_dora_tile(30), 27); // North → East
    }

    #[test]
    fn test_sanma_tsumo_scoring() {
        let score = calculate_score(4, 30, false, true, 0, 3); // Ko Tsumo, 3 players
        assert_eq!(score.pay_tsumo_oya, 3900);
        assert_eq!(score.pay_tsumo_ko, 2000);
        assert_eq!(
            score.total, 5900,
            "3P tsumo total should be 5900 (2 payers)"
        );
    }

    #[test]
    fn test_sanma_tenpai_payment() {
        use crate::state_3p::game_mode;
        assert_eq!(
            game_mode::tenpai_pool(),
            2000,
            "Sanma tenpai pool should be 2000"
        );
    }

    #[test]
    fn test_ryukyoku_deltas_are_reset_each_round_3p() {
        use serde_json::Value;

        let mut state = create_sanma_test_state(5);

        state.players[0].score_delta = 2300;
        state.players[1].score_delta = -2300;
        state.players[2].score_delta = 0;

        state._initialize_round(0, 0, 0, 0, None, None);
        state._trigger_ryukyoku("suukansansen");

        let v: Value = state
            .mjai_log
            .iter()
            .rev()
            .find_map(|s| {
                let ev: Value = serde_json::from_str(s).ok()?;
                (ev["type"] == "ryukyoku").then_some(ev)
            })
            .expect("ryukyoku event should be logged");
        assert_eq!(v["reason"], "suukansansen");

        let deltas: Vec<i32> = serde_json::from_value(v["deltas"].clone()).expect("deltas");
        assert_eq!(deltas, vec![0, 0, 0]);
    }

    // ========== Action Encode Tests (4P/3P) ==========

    #[test]
    fn test_action_encode_4p_discard() {
        use crate::action::{Action, ActionType};

        // tile 0 (1m, type 0) → ID 0
        let a = Action::new(ActionType::Discard, Some(0), vec![], None);
        assert_eq!(a.encode().unwrap(), 0);

        // tile 4 (2m, type 1) → ID 1
        let a = Action::new(ActionType::Discard, Some(4), vec![], None);
        assert_eq!(a.encode().unwrap(), 1);

        // tile 132 (C/7z, type 33) → ID 33
        let a = Action::new(ActionType::Discard, Some(132), vec![], None);
        assert_eq!(a.encode().unwrap(), 33);
    }

    #[test]
    fn test_action_encode_4p_special() {
        use crate::action::{Action, ActionType};

        assert_eq!(
            Action::new(ActionType::Riichi, None, vec![], None)
                .encode()
                .unwrap(),
            37
        );
        assert_eq!(
            Action::new(ActionType::Pon, None, vec![], None)
                .encode()
                .unwrap(),
            41
        );
        // Daiminkan tile 0 (1m) → 42 + 0 = 42
        assert_eq!(
            Action::new(ActionType::Daiminkan, Some(0), vec![], None)
                .encode()
                .unwrap(),
            42
        );
        assert_eq!(
            Action::new(ActionType::Ron, None, vec![], None)
                .encode()
                .unwrap(),
            79
        );
        assert_eq!(
            Action::new(ActionType::KyushuKyuhai, None, vec![], None)
                .encode()
                .unwrap(),
            80
        );
        assert_eq!(
            Action::new(ActionType::Pass, None, vec![], None)
                .encode()
                .unwrap(),
            81
        );
        // Kita is not valid in 4P mode
        assert!(
            Action::new(ActionType::Kita, None, vec![], None)
                .encode()
                .is_err()
        );
    }

    #[test]
    fn test_action_encode_3p_discard() {
        use crate::action::{Action, ActionEncoder, ActionType};
        let enc = ActionEncoder::ThreePlayer;

        // 1m (type 0) → compact 0
        let a = Action::new(ActionType::Discard, Some(0), vec![], None);
        assert_eq!(enc.encode(&a).unwrap(), 0);

        // 9m (type 8, tile 32) → compact 1
        let a = Action::new(ActionType::Discard, Some(32), vec![], None);
        assert_eq!(enc.encode(&a).unwrap(), 1);

        // 1p (type 9, tile 36) → compact 2
        let a = Action::new(ActionType::Discard, Some(36), vec![], None);
        assert_eq!(enc.encode(&a).unwrap(), 2);

        // 9p (type 17, tile 68) → compact 10
        let a = Action::new(ActionType::Discard, Some(68), vec![], None);
        assert_eq!(enc.encode(&a).unwrap(), 10);

        // 1s (type 18, tile 72) → compact 11
        let a = Action::new(ActionType::Discard, Some(72), vec![], None);
        assert_eq!(enc.encode(&a).unwrap(), 11);

        // C (type 33, tile 132) → compact 26
        let a = Action::new(ActionType::Discard, Some(132), vec![], None);
        assert_eq!(enc.encode(&a).unwrap(), 26);
    }

    #[test]
    fn test_action_encode_3p_special() {
        use crate::action::{Action, ActionEncoder, ActionType};
        let enc = ActionEncoder::ThreePlayer;

        assert_eq!(
            enc.encode(&Action::new(ActionType::Riichi, None, vec![], None))
                .unwrap(),
            27
        );
        assert_eq!(
            enc.encode(&Action::new(ActionType::Pon, None, vec![], None))
                .unwrap(),
            28
        );
        // Kan 1m (type 0) → 29 + 0 = 29
        assert_eq!(
            enc.encode(&Action::new(ActionType::Daiminkan, Some(0), vec![], None))
                .unwrap(),
            29
        );
        // Kan 9m (type 8, tile 32) → 29 + 1 = 30
        assert_eq!(
            enc.encode(&Action::new(ActionType::Daiminkan, Some(32), vec![], None))
                .unwrap(),
            30
        );
        // Kan C (type 33, tile 132) → 29 + 26 = 55
        assert_eq!(
            enc.encode(&Action::new(ActionType::Ankan, None, vec![132], None))
                .unwrap(),
            55
        );
        assert_eq!(
            enc.encode(&Action::new(ActionType::Ron, None, vec![], None))
                .unwrap(),
            56
        );
        assert_eq!(
            enc.encode(&Action::new(ActionType::KyushuKyuhai, None, vec![], None))
                .unwrap(),
            57
        );
        assert_eq!(
            enc.encode(&Action::new(ActionType::Pass, None, vec![], None))
                .unwrap(),
            58
        );
        assert_eq!(
            enc.encode(&Action::new(ActionType::Kita, None, vec![], None))
                .unwrap(),
            59
        );
    }

    #[test]
    fn test_action_encode_3p_invalid_manzu() {
        use crate::action::{Action, ActionEncoder, ActionType};
        let enc = ActionEncoder::ThreePlayer;

        // 2m (type 1, tile 4) should be invalid in 3P
        let a = Action::new(ActionType::Discard, Some(4), vec![], None);
        assert!(enc.encode(&a).is_err());

        // 5m (type 4, tile 16) should be invalid in 3P
        let a = Action::new(ActionType::Discard, Some(16), vec![], None);
        assert!(enc.encode(&a).is_err());

        // 8m (type 7, tile 28) should be invalid in 3P
        let a = Action::new(ActionType::Discard, Some(28), vec![], None);
        assert!(enc.encode(&a).is_err());
    }

    #[test]
    fn test_action_encode_3p_chi_not_allowed() {
        use crate::action::{Action, ActionEncoder, ActionType};
        let enc = ActionEncoder::ThreePlayer;

        let a = Action::new(ActionType::Chi, Some(36), vec![40, 44], None);
        assert!(enc.encode(&a).is_err());
    }

    #[test]
    fn test_action_space_size() {
        use crate::action::ActionEncoder;

        assert_eq!(ActionEncoder::FourPlayer.action_space_size(), 82);
        assert_eq!(ActionEncoder::ThreePlayer.action_space_size(), 60);
        assert_eq!(ActionEncoder::from_num_players(4).action_space_size(), 82);
        assert_eq!(ActionEncoder::from_num_players(3).action_space_size(), 60);
    }

    #[test]
    fn test_3p_encode_all_discard_ids_contiguous() {
        use crate::action::{Action, ActionEncoder, ActionType};
        let enc = ActionEncoder::ThreePlayer;

        // All 27 valid tile types in 3P should map to discard IDs 0-26 contiguously
        let valid_types: Vec<u8> = vec![
            0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33,
        ];
        assert_eq!(valid_types.len(), 27);

        let mut ids: Vec<i32> = Vec::new();
        for &tile_type in &valid_types {
            let tile_136 = tile_type * 4;
            let a = Action::new(ActionType::Discard, Some(tile_136), vec![], None);
            ids.push(enc.encode(&a).unwrap());
        }

        ids.sort();
        let expected: Vec<i32> = (0..27).collect();
        assert_eq!(ids, expected, "3P discard IDs should be contiguous 0-26");
    }

    #[test]
    fn test_sanma_observation_num_players() {
        let mut state = create_sanma_test_state(3);
        state._initialize_round(0, 0, 0, 0, None, None);

        state.drawn_tile = Some(state.players[0].hand[0]);
        state.current_player = 0;
        state.phase = Phase::WaitAct;
        state.active_players = vec![0];
        state.needs_tsumo = false;

        let obs = state.get_observation(0);
        // hands, scores, discards, melds are now [T; 3] fixed-size arrays,
        // so .len() checks are trivially true and have been removed.
        assert!(
            !obs.hands[0].is_empty(),
            "Player 0 hand should not be empty"
        );
    }

    // ========== Yakuman Scoring & PAO Tests ==========

    #[test]
    fn test_kazoe_yakuman_score_boundary() {
        // calculate_score uses `8000 * (han / 13)` for han >= 13.
        // han 13 → single yakuman (8000 base)
        let s13 = calculate_score(13, 0, false, false, 0, 4);
        assert_eq!(s13.pay_ron, 32000, "han=13 ko ron should be 32000");

        // han 14 → 8000*(14/13) = 8000*1 = still single yakuman
        let s14 = calculate_score(14, 0, false, false, 0, 4);
        assert_eq!(s14.pay_ron, 32000, "han=14 ko ron should still be 32000");

        // han 25 → 8000*(25/13) = 8000*1 = still single yakuman
        let s25 = calculate_score(25, 0, false, false, 0, 4);
        assert_eq!(s25.pay_ron, 32000, "han=25 ko ron should still be 32000");

        // han 26 → 8000*(26/13) = 8000*2 = double yakuman
        let s26 = calculate_score(26, 0, false, false, 0, 4);
        assert_eq!(s26.pay_ron, 64000, "han=26 ko ron should be 64000");

        // Oya equivalents
        let s13_oya = calculate_score(13, 0, true, false, 0, 4);
        assert_eq!(s13_oya.pay_ron, 48000, "han=13 oya ron should be 48000");

        let s26_oya = calculate_score(26, 0, true, false, 0, 4);
        assert_eq!(s26_oya.pay_ron, 96000, "han=26 oya ron should be 96000");
    }

    #[test]
    fn test_kazoe_yakuman_hand_evaluator_cap() {
        use crate::hand_evaluator::HandEvaluator;
        use crate::types::{Conditions, Wind};

        // Build a hand with 14+ han but no yakuman yaku:
        // Chinitsu (6) + Pinfu (1) + Iipeiko (1) + Riichi (1) + Tsumo (1) + Ippatsu (1) + Dora 3
        // = 14 han. Should be capped to single yakuman, NOT double.
        //
        // Hand: 112233 456 789 m + 55m pair (all manzu = chinitsu)
        // 1m=0, 2m=1, 3m=2, 4m=3, 5m=4, 6m=5, 7m=6, 8m=7, 9m=8
        // 136-tile: 1m=0,1; 2m=4,5; 3m=8,9; 4m=12; 5m=16(red); 6m=20; 7m=24; 8m=28; 9m=32; 5m=17(pair)
        let tiles_136 = vec![
            0, 1, // 1m x2
            4, 5, // 2m x2
            8, 9,  // 3m x2
            12, // 4m
            16, // 5m (red → 1 aka dora)
            20, // 6m
            24, // 7m
            28, // 8m
            32, // 9m
            17, // 5m (pair piece 1)
        ];

        let calc = HandEvaluator::new(tiles_136, Vec::new());

        let cond = Conditions {
            tsumo: true,
            riichi: true,
            ippatsu: true,
            player_wind: Wind::South,
            round_wind: Wind::East,
            ..Default::default()
        };

        // Win tile: 5m (tile 18 in 136-format → type 4)
        // This completes the 55m pair.
        // Dora indicators: set 3 indicators that produce dora on tiles we hold.
        // Indicator 0m (tile_type=0, but 0-1 → next=1m). Use 3m indicator → dora is 4m.
        // Indicator for 3m = tile 8..11 → next tile type is 4 (4m). We hold 4m.
        // Let's use 3 copies of 3m indicator so dora count = 3 (we hold one 4m).
        // Actually to get 3 dora, let's use indicator pointing to a tile we hold multiple of.
        // Indicator 0m (type 0) → dora is 1m (type 0+1=1... wait, 0→1 means 2m).
        // get_next_tile: if tile < 9, tile == 8 → 0, else tile+1
        // So indicator type 0 (1m) → dora type 1 (2m). We hold 2x 2m = 2 dora.
        // Indicator type 1 (2m) → dora type 2 (3m). We hold 2x 3m = 2 dora.
        // Indicator type 3 (4m) → dora type 4 (5m). We hold 2x 5m in hand = 2 dora.
        // With aka 5m (red) that's +1 aka_dora.
        // So: chinitsu(6) + pinfu(1) + iipeiko(1) + riichi(1) + tsumo(1) + ippatsu(1) + 2 dora + 1 aka = 14 han
        // indicator type 0 (1m) → dora type 1 (2m), we hold 2 copies = 2 dora.
        // Plus 1 aka dora from red 5m.
        // chinitsu(6) + pinfu(1) + iipeiko(1) + riichi(1) + tsumo(1) + ippatsu(1) + 2 dora + 1 aka = 14
        let dora_indicators = vec![0]; // type 0 → dora type 1 (2m), we hold 2
        let res = calc.calc(18, dora_indicators, vec![], Some(cond));

        assert!(res.is_win, "Should be a winning hand");
        assert!(
            !res.yakuman,
            "Should NOT be flagged as yakuman (it's kazoe)"
        );
        // han should be reported as the raw total (>=14), but scores capped at single yakuman
        assert!(res.han >= 14, "Raw han should be >= 14, got {}", res.han);

        // Ko tsumo: single yakuman = base 8000 → oya pays 16000, ko pays 8000
        assert_eq!(
            res.tsumo_agari_oya, 16000,
            "Kazoe yakuman tsumo: oya should pay 16000"
        );
        assert_eq!(
            res.tsumo_agari_ko, 8000,
            "Kazoe yakuman tsumo: ko should pay 8000"
        );
    }

    #[test]
    fn test_daiminkan_pao_daisangen() {
        use crate::action::{Action, ActionType};
        use crate::types::{Meld, MeldType};

        let mut state = create_test_state(2);
        state._initialize_next_round(true, false);

        let pid: u8 = 0;

        // Give player 0 two open dragon pon melds (haku=31, hatsu=32).
        // Tiles in melds use 34-tile notation.
        state.players[0].melds = vec![
            Meld {
                meld_type: MeldType::Pon,
                tiles: vec![124, 125, 126], // haku (31*4=124)
                opened: true,
                from_who: 1,
                called_tile: Some(124),
            },
            Meld {
                meld_type: MeldType::Pon,
                tiles: vec![128, 129, 130], // hatsu (32*4=128)
                opened: true,
                from_who: 2,
                called_tile: Some(128),
            },
        ];

        // Player 3 discards chun (33*4=132). Player 0 calls daiminkan.
        let discarder: u8 = 3;
        state.last_discard = Some((discarder, 132));

        // consume_tiles = the 3 chun tiles from hand (we use 133,134,135)
        let action = Action::new(
            ActionType::Daiminkan,
            Some(132),
            vec![133, 134, 135],
            Some(pid),
        );

        state._resolve_kan(pid, action);

        // PAO should be set: yaku 37 (daisangen) → discarder 3
        assert_eq!(
            state.players[0].pao.get(&37),
            Some(&discarder),
            "Daisangen PAO should point to discarder {}",
            discarder
        );
    }

    #[test]
    fn test_daiminkan_pao_daisuushii() {
        use crate::action::{Action, ActionType};
        use crate::types::{Meld, MeldType};

        let mut state = create_test_state(2);
        state._initialize_next_round(true, false);

        let pid: u8 = 0;

        // Give player 0 three open wind pon melds (E=27, S=28, W=29).
        state.players[0].melds = vec![
            Meld {
                meld_type: MeldType::Pon,
                tiles: vec![108, 109, 110], // E (27*4=108)
                opened: true,
                from_who: 1,
                called_tile: Some(108),
            },
            Meld {
                meld_type: MeldType::Pon,
                tiles: vec![112, 113, 114], // S (28*4=112)
                opened: true,
                from_who: 2,
                called_tile: Some(112),
            },
            Meld {
                meld_type: MeldType::Pon,
                tiles: vec![116, 117, 118], // W (29*4=116)
                opened: true,
                from_who: 3,
                called_tile: Some(116),
            },
        ];

        // Player 2 discards N (30*4=120). Player 0 calls daiminkan.
        let discarder: u8 = 2;
        state.last_discard = Some((discarder, 120));

        let action = Action::new(
            ActionType::Daiminkan,
            Some(120),
            vec![121, 122, 123],
            Some(pid),
        );

        state._resolve_kan(pid, action);

        // PAO should be set: yaku 50 (daisuushii) → discarder 2
        assert_eq!(
            state.players[0].pao.get(&50),
            Some(&discarder),
            "Daisuushii PAO should point to discarder {}",
            discarder
        );
    }

    #[test]
    fn test_daiminkan_no_pao_insufficient_melds() {
        use crate::action::{Action, ActionType};
        use crate::types::{Meld, MeldType};

        let mut state = create_test_state(2);
        state._initialize_next_round(true, false);

        let pid: u8 = 0;

        // Give player 0 only ONE dragon pon meld (haku=31).
        state.players[0].melds = vec![Meld {
            meld_type: MeldType::Pon,
            tiles: vec![124, 125, 126], // haku (31*4=124)
            opened: true,
            from_who: 1,
            called_tile: Some(124),
        }];

        // Player 1 discards hatsu (32*4=128). Player 0 calls daiminkan.
        // After this, player 0 has 2 dragon melds — NOT 3, so no PAO.
        let discarder: u8 = 1;
        state.last_discard = Some((discarder, 128));

        let action = Action::new(
            ActionType::Daiminkan,
            Some(128),
            vec![129, 130, 131],
            Some(pid),
        );

        state._resolve_kan(pid, action);

        // No PAO should be set (only 2 dragon melds, need 3)
        assert!(
            state.players[0].pao.is_empty(),
            "PAO should NOT be set with only 2 dragon melds, got: {:?}",
            state.players[0].pao
        );
    }

    #[test]
    fn test_ron_pao_50_50_split() {
        // Ron with PAO: pao player and discarder split total 50/50.
        // score / 2 goes to pao player, remainder to discarder.

        // Single yakuman ron by ko: 32000
        let score: i32 = 32000;
        let pao_amt = score / 2; // 16000
        let discarder_amt = score - pao_amt; // 16000
        assert_eq!(pao_amt, 16000);
        assert_eq!(discarder_amt, 16000);
        assert_eq!(pao_amt + discarder_amt, score, "Must sum to total score");

        // Double yakuman ron by ko: 64000
        let score2: i32 = 64000;
        let pao_amt2 = score2 / 2;
        let discarder_amt2 = score2 - pao_amt2;
        assert_eq!(pao_amt2, 32000);
        assert_eq!(discarder_amt2, 32000);
        assert_eq!(pao_amt2 + discarder_amt2, score2);

        // Single yakuman ron by oya: 48000
        let score3: i32 = 48000;
        let pao_amt3 = score3 / 2;
        let discarder_amt3 = score3 - pao_amt3;
        assert_eq!(pao_amt3, 24000);
        assert_eq!(discarder_amt3, 24000);
        assert_eq!(pao_amt3 + discarder_amt3, score3);

        // Simulate delta computation (as in state/mod.rs ron PAO logic):
        // winner w, pao_payer p, discarder d (all different)
        let (w, p, d) = (0usize, 1usize, 2usize);
        let mut deltas = [0i32; 4];
        deltas[w] += score;
        deltas[p] -= pao_amt;
        deltas[d] -= score - pao_amt;
        assert_eq!(deltas.iter().sum::<i32>(), 0, "Deltas must be zero-sum");
    }

    #[test]
    fn test_tsumo_pao_non_pao_split() {
        // Mixed PAO/non-PAO tsumo scoring (e.g. daisangen from PAO + another yakuman self-drawn).
        // pao_yakuman_val = 1 (daisangen), non_pao_yakuman_val = 1 (e.g. suuankou),
        // total_yakuman_val = 2.

        let np: usize = 4;
        let pao_yakuman_val: i32 = 1;
        let non_pao_yakuman_val: i32 = 1;

        // Case 1: Ko winner (pid=0, oya=1)
        {
            let pid = 0usize;
            let oya = 1usize;
            let pp = 3usize; // pao player

            let unit: i32 = 32000; // ko winner unit
            let honba_total: i32 = 0;
            let pao_amt = pao_yakuman_val * unit + honba_total; // 32000
            let oya_pay = non_pao_yakuman_val * 16000; // 16000
            let ko_pay = non_pao_yakuman_val * 8000; // 8000

            let mut deltas = vec![0i32; np];

            // PAO player pays pao portion
            deltas[pp] -= pao_amt;

            // Non-PAO split among other players
            for (i, delta) in deltas.iter_mut().enumerate().take(np) {
                if i != pid {
                    if i == oya {
                        *delta -= oya_pay;
                    } else if i != pp {
                        // non-oya, non-pao ko player
                        *delta -= ko_pay;
                    } else {
                        // pao player also pays ko share for non-pao part
                        *delta -= ko_pay;
                    }
                }
            }

            let total_win: i32 = -deltas.iter().filter(|&&d| d < 0).sum::<i32>();
            deltas[pid] += total_win;

            assert_eq!(
                deltas.iter().sum::<i32>(),
                0,
                "Ko winner tsumo PAO deltas must be zero-sum"
            );
            // PAO player pays: 32000 (pao) + 8000 (ko share of non-pao) = 40000
            assert_eq!(deltas[pp], -40000, "PAO player should pay 40000 total");
            // Oya pays: 16000 (oya share of non-pao)
            assert_eq!(deltas[oya], -16000, "Oya should pay 16000");
            // Remaining ko pays: 8000 (ko share of non-pao)
            let other_ko = (0..np).find(|&i| i != pid && i != oya && i != pp).unwrap();
            assert_eq!(deltas[other_ko], -8000, "Other ko should pay 8000");
            // Winner gets: 40000 + 16000 + 8000 = 64000
            assert_eq!(deltas[pid], 64000, "Winner should receive 64000");
        }

        // Case 2: Oya winner (pid=0, oya=0)
        {
            let pid = 0usize;
            let pp = 2usize; // pao player

            let unit: i32 = 48000; // oya winner unit
            let pao_amt = pao_yakuman_val * unit; // 48000
            let ko_share = non_pao_yakuman_val * 16000; // 16000 per non-oya player

            let mut deltas = vec![0i32; np];

            // PAO player pays pao portion
            deltas[pp] -= pao_amt;

            // Non-PAO split: each non-oya pays 16000
            for (i, delta) in deltas.iter_mut().enumerate().take(np) {
                if i != pid {
                    *delta -= ko_share;
                }
            }

            let total_win: i32 = -deltas.iter().filter(|&&d| d < 0).sum::<i32>();
            deltas[pid] += total_win;

            assert_eq!(
                deltas.iter().sum::<i32>(),
                0,
                "Oya winner tsumo PAO deltas must be zero-sum"
            );
            // PAO player pays: 48000 (pao) + 16000 (ko share) = 64000
            assert_eq!(deltas[pp], -64000, "PAO player should pay 64000 total");
            // Other ko players each pay 16000
            for (i, &delta) in deltas.iter().enumerate().take(np) {
                if i != pid && i != pp {
                    assert_eq!(delta, -16000, "Ko player {} should pay 16000", i);
                }
            }
            // Winner gets: 64000 + 16000 + 16000 = 96000
            assert_eq!(deltas[pid], 96000, "Oya winner should receive 96000");
        }
    }

    #[test]
    fn test_tenhou_tsumo_pao_composite() {
        // Tenhou rule (yakuman_pao_is_liability_only = false):
        // Double yakuman tsumo (1x PAO + 1x non-PAO) by ko player.
        // PAO pays ALL yakuman: total_yakuman_val * unit = 2 * 32000 = 64000.
        // Other players pay 0.

        let np: usize = 4;
        let pid = 0usize; // ko winner
        let oya = 1usize;
        let pp = 3usize; // pao player
        let total_yakuman_val: i32 = 2;
        let unit: i32 = 32000; // ko
        let honba_total: i32 = 0;

        // Tenhou: PAO pays everything
        let full_amt = total_yakuman_val * unit + honba_total; // 64000
        let mut deltas = vec![0i32; np];
        deltas[pp] -= full_amt;
        let total_win = full_amt;
        deltas[pid] += total_win;

        assert_eq!(deltas.iter().sum::<i32>(), 0, "Deltas must be zero-sum");
        assert_eq!(deltas[pp], -64000, "PAO player pays all 64000");
        assert_eq!(deltas[oya], 0, "Oya pays nothing under Tenhou PAO");
        let other_ko = (0..np).find(|&i| i != pid && i != oya && i != pp).unwrap();
        assert_eq!(
            deltas[other_ko], 0,
            "Other ko pays nothing under Tenhou PAO"
        );
        assert_eq!(deltas[pid], 64000, "Winner receives 64000");
    }

    #[test]
    fn test_tenhou_ron_pao_composite() {
        // Tenhou rule (yakuman_pao_is_liability_only = false):
        // Triple yakuman ron (2x PAO + 1x non-PAO) by oya.
        // Total: 3 * 48000 = 144000. Split 50/50 between PAO and discarder.
        // PAO pays 72000, discarder pays 72000.

        let np: usize = 4;
        let w_pid = 0usize; // oya winner
        let pao_payer = 1usize;
        let discarder = 2usize;
        let total_yakuman_val: i32 = 3;
        let unit: i32 = 48000; // oya
        let honba_ron: i32 = 0;

        // Tenhou: total score split 50/50
        let total_base = total_yakuman_val * unit; // 144000
        let pao_amt = total_base / 2 + honba_ron; // 72000
        let score = total_base + honba_ron; // 144000
        let discarder_amt = score - pao_amt; // 72000

        let mut deltas = vec![0i32; np];
        deltas[w_pid] += score;
        deltas[pao_payer] -= pao_amt;
        deltas[discarder] -= discarder_amt;

        assert_eq!(deltas.iter().sum::<i32>(), 0, "Deltas must be zero-sum");
        assert_eq!(pao_amt, 72000, "PAO pays half of total (72000)");
        assert_eq!(discarder_amt, 72000, "Discarder pays half of total (72000)");
        assert_eq!(deltas[w_pid], 144000, "Winner receives 144000");
    }

    #[test]
    fn test_mjsoul_3p_ron_pao_composite() {
        // 3P MjSoul rule (yakuman_pao_is_liability_only = true):
        // Double yakuman ron (1x PAO daisangen + 1x non-PAO tsuuiisou) by oya.
        // Total: 2 * 48000 = 96000. Only PAO portion split 50/50.
        // PAO pays 24000 (= 48000/2), discarder pays 72000 (= 48000/2 + 48000).

        let np: usize = 3;
        let w_pid = 0usize; // oya winner
        let pao_payer = 1usize;
        let discarder = 2usize;
        let pao_yakuman_val: i32 = 1;
        let total_yakuman_val: i32 = 2;
        let unit: i32 = 48000; // oya
        let honba_ron: i32 = 0;

        // 3P MjSoul: only PAO portion split 50/50
        let split_base = pao_yakuman_val * unit; // 48000
        let pao_amt = split_base / 2 + honba_ron; // 24000
        let score = total_yakuman_val * unit + honba_ron; // 96000
        let discarder_amt = score - pao_amt; // 72000

        let mut deltas = vec![0i32; np];
        deltas[w_pid] += score;
        deltas[pao_payer] -= pao_amt;
        deltas[discarder] -= discarder_amt;

        assert_eq!(deltas.iter().sum::<i32>(), 0, "Deltas must be zero-sum");
        assert_eq!(pao_amt, 24000, "PAO pays half of PAO portion (24000)");
        assert_eq!(
            discarder_amt, 72000,
            "Discarder pays half of PAO portion + full non-PAO (72000)"
        );
        assert_eq!(deltas[w_pid], 96000, "Winner receives 96000");
    }

    #[test]
    fn test_mjsoul_4p_ron_pao_composite() {
        // 4P MjSoul rule (yakuman_pao_is_liability_only = true):
        // Double yakuman ron (1x PAO daisangen + 1x non-PAO tsuuiisou) by ko.
        // Total: 2 * 32000 = 64000. PAO portion only split 50/50.
        // split_base = pao_yakuman_val * unit = 1 * 32000 = 32000
        // PAO pays 16000 (= 32000/2), discarder pays 48000 (= 64000 - 16000).

        let np: usize = 4;
        let w_pid = 0usize; // ko winner
        let pao_payer = 1usize;
        let discarder = 2usize;
        let total_yakuman_val: i32 = 2;
        let pao_yakuman_val: i32 = 1;
        let unit: i32 = 32000; // ko
        let honba_ron: i32 = 0;

        // 4P MjSoul: PAO portion only split 50/50 for Ron
        let split_base = pao_yakuman_val * unit; // 32000
        let pao_amt = split_base / 2 + honba_ron; // 16000
        let score = total_yakuman_val * unit + honba_ron; // 64000
        let discarder_amt = score - pao_amt; // 48000

        let mut deltas = vec![0i32; np];
        deltas[w_pid] += score;
        deltas[pao_payer] -= pao_amt;
        deltas[discarder] -= discarder_amt;

        assert_eq!(deltas.iter().sum::<i32>(), 0, "Deltas must be zero-sum");
        assert_eq!(pao_amt, 16000, "PAO pays half of PAO portion (16000)");
        assert_eq!(discarder_amt, 48000, "Discarder pays remainder (48000)");
        assert_eq!(deltas[w_pid], 64000, "Winner receives 64000");
    }

    /// When handle_kita receives an Action with tile=None, it must still
    /// find and remove the correct North tile from the hand instead of
    /// defaulting to tile 0 (1m).
    #[test]
    fn test_kita_tile_none_removes_north_from_hand() {
        use crate::action::{Action, ActionType};

        let mut state = create_sanma_test_state(5);
        state._initialize_round(0, 0, 0, 0, None, None);
        // Fix rinshan draw to a non-North tile so the assertion is deterministic.
        state.wall.tiles = (36u8..56).collect(); // 1p-5p: all sanma-legal

        // Hand: 1p(36) 2p(40) 3p(44) 4p(48) 5p(52) 6p(56) 7p(60) 8p(64) 9p(68)
        //       1s(72) 2s(76) 3s(80) N(120)
        // drawn_tile = N(120) — player just drew North
        state.players[0].hand = vec![36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 120];
        state.drawn_tile = Some(120);
        state.current_player = 0;
        state.phase = Phase::WaitAct;
        state.active_players = vec![0];
        state.needs_tsumo = false;

        let hand_size_before = state.players[0].hand.len();
        let has_north_before = state.players[0].hand.iter().any(|&t| t / 4 == 30);
        assert!(
            has_north_before,
            "Hand should contain North tile before kita"
        );

        // Create a Kita action with tile=None (the bug scenario from issue #179)
        let kita_action = Action::new(ActionType::Kita, None, vec![], Some(0));
        state.handle_kita(0, &kita_action);

        // After kita, the North tile must be removed from the hand
        let has_north_after = state.players[0].hand.iter().any(|&t| t / 4 == 30);
        assert!(
            !has_north_after,
            "North tile must be removed from hand after kita, \
             even when the action has tile=None"
        );

        // The North tile must be in kita_tiles
        assert_eq!(
            state.players[0].kita_tiles.len(),
            1,
            "Kita tiles should contain exactly one tile"
        );
        assert_eq!(
            state.players[0].kita_tiles[0] / 4,
            30,
            "Kita tile must be a North tile"
        );

        // After rinshan draw, hand size should be same as before
        // (removed 1 North + gained 1 rinshan = net 0 change)
        assert_eq!(
            state.players[0].hand.len(),
            hand_size_before,
            "Hand size must remain the same after kita + rinshan draw"
        );
    }

    /// After kita with tile=None during riichi, ankan must still appear
    /// in legal_actions on the subsequent turn.
    #[test]
    fn test_sanma_ankan_available_after_kita_in_riichi() {
        use crate::action::{Action, ActionType};
        use crate::state_3p::legal_actions::GameState3PLegalActions;
        use std::collections::HashMap;

        let mut state = create_sanma_test_state(5);
        state._initialize_round(0, 0, 0, 0, None, None);
        // Keep rinshan deterministic and leave enough wall for post-kita legal actions.
        state.wall.tiles = (36u8..56).collect(); // 1p-5p: all sanma-legal

        // Player 1 hand: 2p(40,41) 4p(48,49,50) 5p(52) 6p(56) 7p(60)
        //                 1s(72,73,74) 8s(100) 9s(104)
        // 13 tiles, player is in riichi
        state.players[1].hand = vec![40, 41, 48, 49, 50, 52, 56, 60, 72, 73, 74, 100, 104];
        state.players[1].riichi_declared = true;
        state.players[1].riichi_declaration_index = Some(0);
        state.players[1].discards.push(131); // F discarded for riichi
        state.players[1].discard_from_hand.push(true);
        state.players[1].discard_is_riichi.push(true);

        // Simulate: player 1 draws North tile
        let north_tile: u8 = 121;
        state.players[1].hand.push(north_tile);
        state.drawn_tile = Some(north_tile);
        state.current_player = 1;
        state.phase = Phase::WaitAct;
        state.active_players = vec![1];
        state.needs_tsumo = false;

        // Player 1 does kita with tile=None (the bug scenario)
        let kita_action = Action::new(ActionType::Kita, None, vec![], Some(1));
        state.handle_kita(1, &kita_action);

        // After kita + rinshan draw, verify North was removed
        assert!(
            !state.players[1].hand.iter().any(|&t| t / 4 == 30),
            "North must be removed from hand after kita (tile=None)"
        );
        // Hand should have 14 tiles (had 14, removed N, gained rinshan = 14)
        assert_eq!(
            state.players[1].hand.len(),
            14,
            "Hand should have 14 tiles after kita rinshan draw (before discard)"
        );

        // Now player 1 discards the rinshan tile (tsumogiri)
        let rinshan_tile = state.drawn_tile.unwrap();
        let discard = Action::new(ActionType::Discard, Some(rinshan_tile), vec![], Some(1));
        let mut actions = HashMap::new();
        actions.insert(1u8, discard);
        state.step(&actions);

        // Hand should be 13 tiles after discard (14 - 1 discard = 13)
        assert_eq!(
            state.players[1].hand.len(),
            13,
            "Hand should have 13 tiles after discard"
        );

        // Simulate: eventually player 1 draws 4p (the 4th copy)
        let fourth_4p: u8 = 51; // tile 51 = 4p
        state.players[1].hand.push(fourth_4p);
        state.drawn_tile = Some(fourth_4p);
        state.current_player = 1;
        state.phase = Phase::WaitAct;
        state.active_players = vec![1];
        state.needs_tsumo = false;

        // Verify hand has 4x 4p
        let count_4p = state.players[1]
            .hand
            .iter()
            .filter(|&&t| t / 4 == 12)
            .count();
        assert_eq!(count_4p, 4, "Player 1 should have 4 copies of 4p");

        // Verify legal_actions includes Ankan
        let legal = state._get_legal_actions_internal(1);
        let has_ankan = legal.iter().any(|a| a.action_type == ActionType::Ankan);
        assert!(
            has_ankan,
            "Ankan should be available after riichi when holding 4 copies of drawn tile. \
             Legal actions: {:?}",
            legal
                .iter()
                .map(|a| format!("{:?}(tile={:?})", a.action_type, a.tile))
                .collect::<Vec<_>>()
        );

        // Verify Kita is NOT available (no North tiles in hand)
        let has_kita = legal.iter().any(|a| a.action_type == ActionType::Kita);
        assert!(
            !has_kita,
            "Kita should NOT be available when no North tiles are in hand"
        );
    }

    /// Verify that kita with a correct tile (Some(120)) still works.
    #[test]
    fn test_kita_with_correct_tile_still_works() {
        use crate::action::{Action, ActionType};

        let mut state = create_sanma_test_state(5);
        state._initialize_round(0, 0, 0, 0, None, None);
        state.wall.tiles = (36u8..56).collect(); // 1p-5p: all sanma-legal

        state.players[0].hand = vec![36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 120];
        state.drawn_tile = Some(120);
        state.current_player = 0;
        state.phase = Phase::WaitAct;
        state.active_players = vec![0];
        state.needs_tsumo = false;

        // Kita with correct tile=Some(120) — should work as before
        let kita_action = Action::new(ActionType::Kita, Some(120), vec![], Some(0));
        state.handle_kita(0, &kita_action);

        assert!(
            !state.players[0].hand.contains(&120),
            "Tile 120 must be removed from hand"
        );
        assert_eq!(state.players[0].kita_tiles, vec![120]);
        assert_eq!(
            state.players[0].hand.len(),
            13,
            "Hand should have 13 tiles after kita + rinshan"
        );
    }

    /// After kita with tile=None, reach must remain available on subsequent turns.
    #[test]
    fn test_sanma_reach_available_after_kita() {
        use crate::action::{Action, ActionType};
        use crate::state_3p::legal_actions::GameState3PLegalActions;
        use std::collections::HashMap;

        let mut state = create_sanma_test_state(5);
        state._initialize_round(0, 0, 0, 0, None, None);
        // Riichi remains legal only while wall.tiles.len() > 14, so use a fixed long wall.
        state.wall.tiles = (36u8..56).collect(); // 1p-5p: all sanma-legal

        // Player 1: hand in tenpai (waiting on 7s)
        // 2p(40) 3p(44) 4p(48) 5p(52) 6p(56) 7p(60)
        // 2s(76) 3s(80) 4s(84) 5s(88) 6s(92) 7s(96) 8s(100)
        state.players[1].hand = vec![40, 44, 48, 52, 56, 60, 76, 80, 84, 88, 92, 96, 100];
        state.players[1].score = 35000;
        state.current_player = 1;
        state.phase = Phase::WaitAct;
        state.active_players = vec![1];
        state.needs_tsumo = false;

        // Player 1 draws North
        let north_tile: u8 = 122;
        state.players[1].hand.push(north_tile);
        state.drawn_tile = Some(north_tile);

        // Player 1 does kita with tile=None
        let kita_action = Action::new(ActionType::Kita, None, vec![], Some(1));
        state.handle_kita(1, &kita_action);

        // Discard rinshan tile
        let rinshan_tile = state.drawn_tile.unwrap();
        let discard = Action::new(ActionType::Discard, Some(rinshan_tile), vec![], Some(1));
        let mut actions = HashMap::new();
        actions.insert(1u8, discard);
        state.step(&actions);

        // Now player 1 draws a tile that maintains tenpai — say 9p(68)
        state.players[1].hand.push(68);
        state.drawn_tile = Some(68);
        state.current_player = 1;
        state.phase = Phase::WaitAct;
        state.active_players = vec![1];
        state.needs_tsumo = false;

        let legal = state._get_legal_actions_internal(1);
        let has_reach = legal.iter().any(|a| a.action_type == ActionType::Riichi);
        assert!(
            has_reach,
            "Reach should be available when hand is tenpai. \
             Hand size: {}, Legal actions: {:?}",
            state.players[1].hand.len(),
            legal
                .iter()
                .map(|a| format!("{:?}(tile={:?})", a.action_type, a.tile))
                .collect::<Vec<_>>()
        );
    }

    /// After kita with tile=None, tsumo must remain available on subsequent turns.
    #[test]
    fn test_sanma_tsumo_available_after_kita() {
        use crate::action::{Action, ActionType};
        use crate::state_3p::legal_actions::GameState3PLegalActions;

        let mut state = create_sanma_test_state(5);
        state._initialize_round(0, 0, 0, 0, None, None);
        state.wall.tiles = (36u8..56).collect(); // 1p-5p: all sanma-legal

        // Player 2: tenpai hand (tanki wait on E)
        // 2p(40) 3p(44) 4p(48)  5p(52) 6p(56) 7p(60)
        // 2s(76) 3s(80) 4s(84)  5s(88) 6s(92) 7s(96)
        // E(108)
        // Winning tile: E(109) → 4 sequences + E pair = menzen tsumo
        state.players[2].hand = vec![40, 44, 48, 52, 56, 60, 76, 80, 84, 88, 92, 96, 108];
        state.current_player = 2;
        state.phase = Phase::WaitAct;
        state.active_players = vec![2];
        state.needs_tsumo = false;

        // Player 2 draws North
        let north_tile: u8 = 123;
        state.players[2].hand.push(north_tile);
        state.drawn_tile = Some(north_tile);

        // Player 2 does kita with tile=None
        let kita_action = Action::new(ActionType::Kita, None, vec![], Some(2));
        state.handle_kita(2, &kita_action);

        // After kita, hand should have 14 tiles (had 14, -N, +rinshan = 14)
        assert_eq!(state.players[2].hand.len(), 14);

        // Discard the rinshan tile, then draw the winning tile
        let rinshan_tile = state.drawn_tile.unwrap();
        if let Some(idx) = state.players[2]
            .hand
            .iter()
            .position(|&t| t == rinshan_tile)
        {
            state.players[2].hand.remove(idx);
        }
        state.players[2].discards.push(rinshan_tile);
        state.drawn_tile = None;

        // Draw winning tile E(109) → completes E pair
        state.players[2].hand.push(109);
        state.drawn_tile = Some(109);
        state.current_player = 2;
        state.phase = Phase::WaitAct;
        state.active_players = vec![2];

        let legal = state._get_legal_actions_internal(2);
        let has_tsumo = legal.iter().any(|a| a.action_type == ActionType::Tsumo);
        assert!(
            has_tsumo,
            "Tsumo should be available when hand is a winning hand. \
             Hand size: {}, Legal actions: {:?}",
            state.players[2].hand.len(),
            legal
                .iter()
                .map(|a| format!("{:?}(tile={:?})", a.action_type, a.tile))
                .collect::<Vec<_>>()
        );
    }
}
