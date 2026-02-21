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
        use crate::yaku::{calculate_yaku, YakuContext};
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
        use crate::yaku::{calculate_yaku, YakuContext};
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
        use crate::yaku::{calculate_yaku, YakuContext};
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
        assert_eq!((0u8 + 1) % np, 1, "Next player after 0 should be 1");
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
        assert!(Action::new(ActionType::Kita, None, vec![], None)
            .encode()
            .is_err());
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
        assert_eq!(obs.hands.len(), 3, "Observation should have 3 hands");
        assert_eq!(obs.scores.len(), 3, "Observation should have 3 scores");
        assert_eq!(obs.discards.len(), 3, "Observation should have 3 discards");
        assert_eq!(obs.melds.len(), 3, "Observation should have 3 melds");
    }
}
