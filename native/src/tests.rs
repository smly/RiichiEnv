#[cfg(test)]
mod unit_tests {
    use crate::action::Phase;
    use crate::agari::{is_agari, is_chiitoitsu, is_kokushi};
    use crate::env::RiichiEnv;
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

        let score = calculate_score(4, 30, false, true, 0); // Ko Tsumo

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

    // --- Helper for creating RiichiEnv in tests ---
    fn create_test_env(game_type: u8) -> RiichiEnv {
        // Construct directly using new API or manual GameState construction
        RiichiEnv {
            state: crate::state::GameState::new(
                game_type,
                false, // skip_mjai_logging
                None,  // seed
                0,     // round_wind
                crate::rule::GameRule::default(),
            ),
        }
    }

    #[test]
    fn test_seeded_shuffle_changes_between_rounds() {
        let mut env = create_test_env(2);
        env.state.seed = Some(42);

        env.state._initialize_next_round(true, false);
        let digest1 = env.state.wall.wall_digest.clone();

        env.state._initialize_next_round(true, false);
        let digest2 = env.state.wall.wall_digest.clone();

        assert_ne!(
            digest1, digest2,
            "Wall digest should differ between rounds when seed is fixed"
        );
    }

    #[test]
    fn test_sudden_death_hanchan_logic() {
        use serde_json::Value;

        // 4-player Hanchan (game_type 2)
        // Scores < 30000. Round South 4 (Round Wind 1, Kyoku 3).
        // Trigger Ryukyoku.
        // Expect: Next round is West 1 (Round Wind 2, Kyoku 0). Game NOT done.

        let mut env = create_test_env(2);
        env.state.round_wind = 1;
        env.state.kyoku_idx = 3;
        env.state.oya = 3;
        for i in 0..4 {
            env.state.players[i].score = 25000;
            env.state.players[i].nagashi_eligible = false;
        }
        // We also need to set needs_initialize_next_round to false initially
        env.state.needs_initialize_next_round = false;

        // Trigger Ryukyoku (draw)
        env.state._trigger_ryukyoku("exhaustive_draw");
        // This sets needs_initialize_next_round = true, pending_oya_won = false (if nouten), pending_is_draw = true

        // Simulate step calling initialize_next_round
        if env.state.needs_initialize_next_round {
            env.state
                ._initialize_next_round(env.state.pending_oya_won, env.state.pending_is_draw);
            env.state.needs_initialize_next_round = false;
        }

        assert!(
            !env.state.is_done,
            "Game should not be done (Sudden Death should trigger)"
        );
        assert_eq!(env.state.round_wind, 2, "Should enter West round");
        assert_eq!(env.state.kyoku_idx, 0, "Should be West 1 (Kyoku 0)");
        assert_eq!(env.state.oya, 0, "Oya should rotate to player 0");

        // Now set scores > 30000 and trigger draw again.
        // West 1. Oya is 0.
        let new_scores = [31000, 25000, 24000, 20000];
        for (player, &score) in env.state.players.iter_mut().zip(new_scores.iter()) {
            player.score = score;
        }

        env.state._trigger_ryukyoku("exhaustive_draw");
        if env.state.needs_initialize_next_round {
            env.state
                ._initialize_next_round(env.state.pending_oya_won, env.state.pending_is_draw);
            env.state.needs_initialize_next_round = false;
        }

        assert!(
            env.state.is_done,
            "Game should be done (Score >= 30000 in West)"
        );

        // Verify MJAI Event order
        // Check logs for last sequence
        let logs = &env.state.mjai_log;
        let event_types: Vec<String> = logs
            .iter()
            .filter_map(|s| {
                let v: Value = serde_json::from_str(s).ok()?;
                v.get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t.to_string())
            })
            .collect();

        // Expect ryukyoku -> end_kyoku -> end_game
        let last_event = event_types.last().expect("Should have events");
        assert_eq!(last_event, "end_game");

        // Check if ryukyoku is recently before it
        assert!(event_types.contains(&"ryukyoku".to_string()));
    }

    #[test]
    fn test_is_tenpai() {
        use crate::agari_calculator::AgariCalculator;
        // 111,222,333m, 444p, 11s (Tenpai on 1s)
        let hand = vec![0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 72];
        let calc = AgariCalculator::new(hand, Vec::new());
        assert!(calc.is_tenpai());
        let waits = calc.get_waits_u8();
        assert!(waits.contains(&18)); // 1s
    }

    #[test]
    fn test_kuikae_deadlock_repro() {
        use crate::action::{Action, ActionType};
        use std::collections::HashMap;

        let mut env = create_test_env(4);
        let pid = 0;

        // Hand: 4m, 5m, 6m, 6m. (12, 16, 20, 21)
        // 3m is 8.
        env.state.players[pid as usize].hand = vec![12, 16, 20, 21];

        // Setup P3 (Kamicha of P0)
        env.state.current_player = 3;
        env.state.phase = Phase::WaitAct;
        env.state.active_players = vec![3];
        env.state.players[3].hand.push(8); // Give 3m

        // Action: P3 discards 3m
        let mut actions = HashMap::new();
        actions.insert(3, Action::new(ActionType::Discard, Some(8), vec![]));

        env.state.step(&actions);

        env.state.step(&actions);

        assert_eq!(
            env.state.phase,
            Phase::WaitAct,
            "Should proceed to WaitAct as deadlock Chi is filtered out"
        );
        assert_eq!(env.state.current_player, 0, "Should be P0's turn");

        // Verify current_claims is empty or does not contain 0
        if let Some(claims) = env.state.current_claims.get(&0) {
            assert!(claims.is_empty(), "P0 should have no legal claims");
        }
    }
    #[test]
    fn test_match_84_agari_check() {
        use crate::agari_calculator::AgariCalculator;
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

        let calc = AgariCalculator::new(tiles, Vec::new());

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
            kyoutaku: 0,
            tsumi: 0,
        };

        // 1. Check 6p (14 -> 56)
        let res6p = calc.calc(56, vec![], vec![], Some(cond.clone()));
        println!(
            "6p Result: Agari={}, Shape={}, Han={}, Yaku={:?}",
            res6p.agari, res6p.has_agari_shape, res6p.han, res6p.yaku
        );
        assert!(!res6p.agari, "6p should NOT be Agari (No Yaku)");
        assert!(res6p.has_agari_shape, "6p should have Agari Shape");
        assert_eq!(res6p.han, 0, "6p should have 0 Han");

        // 2. Check 9p (17 -> 68)
        let res9p = calc.calc(68, vec![], vec![], Some(cond));
        println!(
            "9p Result: Agari={}, Han={}, Yaku={:?}",
            res9p.agari, res9p.han, res9p.yaku
        );
        assert!(res9p.agari, "9p should be Agari");
        assert!(res9p.han >= 3, "9p should be Junchan (>= 3 Han)"); // Junchan (3)
    }
}
