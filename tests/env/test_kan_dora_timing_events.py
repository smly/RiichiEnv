"""Test kan dora reveal timing event ordering"""

from riichienv import (
    Action,
    ActionType,
    GameRule,
    GameType,
    Meld,
    MeldType,
    Phase,
    RiichiEnv,
)


class TestKanDoraTimingEvents:
    def test_ankan_dora_before_rinshan(self):
        """
        Ankan reveals dora before rinshan tsumo.
        Expected order: ankan → dora → tsumo
        """
        rule = GameRule.default_tenhou()

        env = RiichiEnv(seed=42, game_mode=0, rule=rule)
        env.reset()

        # Setup: Give player 0 four 1m tiles (tids 0,1,2,3)
        hands = env.hands
        hands[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env.hands = hands
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.needs_tsumo = False
        env.drawn_tile = 3

        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]

        # Find ankan action
        ankan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.ANKAN]
        assert len(ankan_actions) > 0, "Should have ANKAN action available"

        # Execute ankan
        env.step({player_id: ankan_actions[0]})

        # Check event order
        events = [ev["type"] for ev in env.mjai_log]
        ankan_idx = events.index("ankan")
        dora_indices = [i for i, t in enumerate(events) if t == "dora"]
        tsumo_indices = [i for i, t in enumerate(events) if t == "tsumo"]

        # Find the dora and tsumo after the ankan
        dora_after_ankan = [i for i in dora_indices if i > ankan_idx]
        tsumo_after_ankan = [i for i in tsumo_indices if i > ankan_idx]

        assert len(dora_after_ankan) > 0, "Should have dora event after ankan"
        assert len(tsumo_after_ankan) > 0, "Should have tsumo event after ankan"

        # Verify order: ankan → dora → tsumo
        assert dora_after_ankan[0] < tsumo_after_ankan[0], (
            f"Dora should come before tsumo. "
            f"Order: ankan@{ankan_idx}, dora@{dora_after_ankan[0]}, "
            f"tsumo@{tsumo_after_ankan[0]}"
        )

    def test_kakan_dora_before_discard(self):
        """
        Kakan reveals dora before discard.
        Expected order: kakan → tsumo → dora → dahai
        """
        rule = GameRule.default_tenhou()

        env = RiichiEnv(seed=42, game_mode=0, rule=rule)
        env.reset()

        # Setup: Give player 0 a pon of 1m (tiles 0,1,2) and the 4th tile (tile 3) in hand
        hands = env.hands
        hands[0] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 60]  # Include tile 3 (the 4th 1m)
        env.hands = hands
        melds = env.melds
        melds[0] = [Meld(MeldType.Pon, tiles=[0, 1, 2], opened=True)]
        env.melds = melds
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.needs_tsumo = False
        env.drawn_tile = 3  # The 4th 1m tile is the drawn tile

        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]

        # Find kakan action
        kakan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.KAKAN]
        assert len(kakan_actions) > 0, "Should have KAKAN action available"

        # Execute kakan
        env.step({player_id: kakan_actions[0]})

        # Now we need to discard
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]
        discard_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.DISCARD]
        assert len(discard_actions) > 0, "Should have DISCARD action available"

        # Execute discard
        env.step({player_id: discard_actions[0]})

        # Check event order
        events = [ev["type"] for ev in env.mjai_log]
        kakan_idx = events.index("kakan")
        dora_indices = [i for i, t in enumerate(events) if t == "dora"]
        tsumo_indices = [i for i, t in enumerate(events) if t == "tsumo"]
        dahai_indices = [i for i, t in enumerate(events) if t == "dahai"]

        # Find events after kakan
        dora_after_kakan = [i for i in dora_indices if i > kakan_idx]
        tsumo_after_kakan = [i for i in tsumo_indices if i > kakan_idx]
        dahai_after_kakan = [i for i in dahai_indices if i > kakan_idx]

        assert len(dora_after_kakan) > 0, "Should have dora event after kakan"
        assert len(tsumo_after_kakan) > 0, "Should have tsumo event after kakan"
        assert len(dahai_after_kakan) > 0, "Should have dahai event after kakan"

        # Verify order: kakan → tsumo → dora → dahai
        assert tsumo_after_kakan[0] < dora_after_kakan[0], (
            f"Tsumo should come before dora. "
            f"Order: kakan@{kakan_idx}, tsumo@{tsumo_after_kakan[0]}, "
            f"dora@{dora_after_kakan[0]}"
        )
        assert dora_after_kakan[0] < dahai_after_kakan[0], (
            f"Dora should come before dahai. "
            f"Order: kakan@{kakan_idx}, tsumo@{tsumo_after_kakan[0]}, "
            f"dora@{dora_after_kakan[0]}, dahai@{dahai_after_kakan[0]}"
        )

    def test_daiminkan_dora_before_discard(self):
        """
        Daiminkan reveals dora before discard.
        Expected order: daiminkan → tsumo → dora → dahai
        """
        rule = GameRule.default_tenhou()

        env = RiichiEnv(seed=42, game_mode=0, rule=rule)
        env.reset()

        # Setup: Player 0 has tile 75 (4th copy of 1s) and will discard it.
        # Player 1 has tiles 72,73,74 (other three 1s copies) for daiminkan.
        hands = env.hands
        hands[0] = [75, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        hands[1] = [72, 73, 74, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        env.hands = hands
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.needs_tsumo = False
        env.drawn_tile = 75

        # Step 1: Player 0 discards tile 75
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=75)})

        # Step 2: Player 1 calls DAIMINKAN
        assert 1 in obs_dict, "Player 1 should be active"
        obs = obs_dict[1]
        kan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.DAIMINKAN]
        assert len(kan_actions) > 0, "P1 should have DAIMINKAN option"
        env.step({1: kan_actions[0]})

        # Step 3: Player 1 discards after rinshan draw
        player_id = 1
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]
        discard_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.DISCARD]
        assert len(discard_actions) > 0, "Should have DISCARD action available"
        env.step({player_id: discard_actions[0]})

        # Check event order
        events = [ev["type"] for ev in env.mjai_log]
        daiminkan_idx = events.index("daiminkan")
        dora_indices = [i for i, t in enumerate(events) if t == "dora"]
        tsumo_indices = [i for i, t in enumerate(events) if t == "tsumo"]
        dahai_indices = [i for i, t in enumerate(events) if t == "dahai"]

        # Find events after daiminkan
        dora_after = [i for i in dora_indices if i > daiminkan_idx]
        tsumo_after = [i for i in tsumo_indices if i > daiminkan_idx]
        dahai_after = [i for i in dahai_indices if i > daiminkan_idx]

        assert len(dora_after) > 0, "Should have dora event after daiminkan"
        assert len(tsumo_after) > 0, "Should have tsumo event after daiminkan"
        assert len(dahai_after) > 0, "Should have dahai event after daiminkan"

        # Verify order: daiminkan → tsumo → dora → dahai
        assert tsumo_after[0] < dora_after[0], (
            f"Tsumo should come before dora. "
            f"Order: daiminkan@{daiminkan_idx}, tsumo@{tsumo_after[0]}, "
            f"dora@{dora_after[0]}"
        )
        assert dora_after[0] < dahai_after[0], (
            f"Dora should come before dahai. "
            f"Order: daiminkan@{daiminkan_idx}, tsumo@{tsumo_after[0]}, "
            f"dora@{dora_after[0]}, dahai@{dahai_after[0]}"
        )


class TestKanDoraTimingEvents3P:
    """3P (sanma) regression tests for kan dora timing event ordering."""

    def test_ankan_dora_before_rinshan_3p(self):
        """
        Ankan reveals dora before rinshan tsumo in 3P.
        Expected order: ankan → dora → tsumo
        """
        rule = GameRule.default_tenhou()

        env = RiichiEnv(seed=42, game_mode=GameType.SAN_IKKYOKU, rule=rule)
        env.reset()

        # 3P uses the standard 0-135 tile ID scheme; 2m-8m (IDs 4-31) are excluded.
        # 1p tiles are IDs 36-39. Give player 0 four 1p tiles for ankan.
        hands = env.hands
        hands[0] = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        env.hands = hands
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.needs_tsumo = False
        env.drawn_tile = 39

        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]

        ankan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.ANKAN]
        assert len(ankan_actions) > 0, "Should have ANKAN action available"

        env.step({player_id: ankan_actions[0]})

        events = [ev["type"] for ev in env.mjai_log]
        ankan_idx = events.index("ankan")
        dora_indices = [i for i, t in enumerate(events) if t == "dora"]
        tsumo_indices = [i for i, t in enumerate(events) if t == "tsumo"]

        dora_after_ankan = [i for i in dora_indices if i > ankan_idx]
        tsumo_after_ankan = [i for i in tsumo_indices if i > ankan_idx]

        assert len(dora_after_ankan) > 0, "Should have dora event after ankan"
        assert len(tsumo_after_ankan) > 0, "Should have tsumo event after ankan"

        # Verify order: ankan → dora → tsumo
        assert dora_after_ankan[0] < tsumo_after_ankan[0], (
            f"Dora should come before tsumo. "
            f"Order: ankan@{ankan_idx}, dora@{dora_after_ankan[0]}, "
            f"tsumo@{tsumo_after_ankan[0]}"
        )

    def test_kakan_dora_before_discard_3p(self):
        """
        Kakan reveals dora before discard in 3P.
        Expected order: kakan → tsumo → dora → dahai
        """
        rule = GameRule.default_tenhou()

        env = RiichiEnv(seed=42, game_mode=GameType.SAN_IKKYOKU, rule=rule)
        env.reset()

        # 3P uses the standard 0-135 tile ID scheme; 1p tiles are 36-39.
        # Give player 0 a pon of 1p (tiles 36,37,38) and the 4th tile (39) in hand.
        hands = env.hands
        hands[0] = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        env.hands = hands
        melds = env.melds
        melds[0] = [Meld(MeldType.Pon, tiles=[36, 37, 38], opened=True)]
        env.melds = melds
        env.active_players = [0]
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.needs_tsumo = False
        env.drawn_tile = 39

        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]

        kakan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.KAKAN]
        assert len(kakan_actions) > 0, "Should have KAKAN action available"

        env.step({player_id: kakan_actions[0]})

        # Now discard after rinshan draw
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]
        discard_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.DISCARD]
        assert len(discard_actions) > 0, "Should have DISCARD action available"
        env.step({player_id: discard_actions[0]})

        events = [ev["type"] for ev in env.mjai_log]
        kakan_idx = events.index("kakan")
        dora_indices = [i for i, t in enumerate(events) if t == "dora"]
        tsumo_indices = [i for i, t in enumerate(events) if t == "tsumo"]
        dahai_indices = [i for i, t in enumerate(events) if t == "dahai"]

        dora_after_kakan = [i for i in dora_indices if i > kakan_idx]
        tsumo_after_kakan = [i for i in tsumo_indices if i > kakan_idx]
        dahai_after_kakan = [i for i in dahai_indices if i > kakan_idx]

        assert len(dora_after_kakan) > 0, "Should have dora event after kakan"
        assert len(tsumo_after_kakan) > 0, "Should have tsumo event after kakan"
        assert len(dahai_after_kakan) > 0, "Should have dahai event after kakan"

        # Verify order: kakan → tsumo → dora → dahai
        assert tsumo_after_kakan[0] < dora_after_kakan[0], (
            f"Tsumo should come before dora. "
            f"Order: kakan@{kakan_idx}, tsumo@{tsumo_after_kakan[0]}, "
            f"dora@{dora_after_kakan[0]}"
        )
        assert dora_after_kakan[0] < dahai_after_kakan[0], (
            f"Dora should come before dahai. "
            f"Order: kakan@{kakan_idx}, tsumo@{tsumo_after_kakan[0]}, "
            f"dora@{dora_after_kakan[0]}, dahai@{dahai_after_kakan[0]}"
        )
