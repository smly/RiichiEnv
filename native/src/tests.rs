#[cfg(test)]
mod unit_tests {
    use crate::agari::{is_agari, is_chiitoitsu, is_kokushi};
    use crate::env::{Phase, RiichiEnv};
    use crate::score::calculate_score;
    use crate::types::Hand;
    use std::collections::HashMap;

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

        let score = calculate_score(4, 30, false, true); // Ko Tsumo

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
        // Construct directly since fields are pub
        RiichiEnv {
            wall: Vec::new(),
            hands: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            melds: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            discard_from_hand: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            current_player: 0,
            turn_count: 0,
            is_done: false,
            needs_tsumo: false,
            needs_initialize_next_round: false,
            pending_oya_won: false,
            pending_is_draw: false,
            scores: [25000; 4],
            score_deltas: [0; 4],
            riichi_sticks: 0,
            riichi_declared: [false; 4],
            riichi_stage: [false; 4],
            double_riichi_declared: [false; 4],
            phase: Phase::WaitAct,
            active_players: vec![0],
            last_discard: None,
            current_claims: HashMap::new(),
            pending_kan: None,
            oya: 0,
            honba: 0,
            kyoku_idx: 0,
            dora_indicators: Vec::new(),
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            is_rinshan_flag: false,
            is_first_turn: true,
            missed_agari_riichi: [false; 4],
            missed_agari_doujun: [false; 4],
            riichi_pending_acceptance: None,
            nagashi_eligible: [true; 4],
            drawn_tile: None,
            wall_digest: String::new(),
            salt: String::new(),
            agari_results: HashMap::new(),
            last_agari_results: HashMap::new(),
            round_end_scores: None,
            mjai_log: Vec::new(),
            mjai_log_per_player: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            player_event_counts: [0; 4],
            round_wind: 0,
            ippatsu_cycle: [false; 4],
            game_mode: game_type,
            skip_mjai_logging: false,
            seed: None,
            hand_index: 0,
            forbidden_discards: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            rule: crate::rule::GameRule::default(),
            pao: [
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            ],
            discard_is_riichi: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            riichi_declaration_index: [None; 4],
        }
    }

    #[test]
    fn test_seeded_shuffle_changes_between_rounds() {
        let mut env = create_test_env(2);
        env.seed = Some(42);

        env._initialize_next_round(true, false);
        let digest1 = env.wall_digest.clone();

        env._initialize_next_round(true, false);
        let digest2 = env.wall_digest.clone();

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
        env.round_wind = 1;
        env.kyoku_idx = 3;
        env.oya = 3;
        env.scores = [25000, 25000, 25000, 25000];
        // We also need to set needs_initialize_next_round to false initially
        env.needs_initialize_next_round = false;
        env.nagashi_eligible = [false; 4];

        // Trigger Ryukyoku (draw)
        env._trigger_ryukyoku("exhaustive_draw");
        // This sets needs_initialize_next_round = true, pending_oya_won = false (if nouten), pending_is_draw = true

        // Simulate step calling initialize_next_round
        if env.needs_initialize_next_round {
            env._initialize_next_round(env.pending_oya_won, env.pending_is_draw);
            env.needs_initialize_next_round = false;
        }

        assert!(
            !env.is_done,
            "Game should not be done (Sudden Death should trigger)"
        );
        assert_eq!(env.round_wind, 2, "Should enter West round");
        assert_eq!(env.kyoku_idx, 0, "Should be West 1 (Kyoku 0)");
        assert_eq!(env.oya, 0, "Oya should rotate to player 0");

        // Now set scores > 30000 and trigger draw again.
        // West 1. Oya is 0.
        env.scores = [31000, 25000, 24000, 20000];

        env._trigger_ryukyoku("exhaustive_draw");
        if env.needs_initialize_next_round {
            env._initialize_next_round(env.pending_oya_won, env.pending_is_draw);
            env.needs_initialize_next_round = false;
        }

        assert!(env.is_done, "Game should be done (Score >= 30000 in West)");

        // Verify MJAI Event order
        // Check logs for last sequence
        let logs = &env.mjai_log;
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
}
