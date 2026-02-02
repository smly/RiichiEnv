"""Test kan dora reveal timing event ordering"""

from riichienv import (
    ActionType,
    GameRule,
    KanDoraTimingMode,
    Meld,
    MeldType,
    Phase,
    RiichiEnv,
)


class TestKanDoraTimingEvents:
    def test_tenhou_ankan_dora_before_rinshan(self):
        """
        Tenhou: Ankan reveals dora before rinshan tsumo
        Expected order: ankan → dora → tsumo
        """
        rule = GameRule.default_tenhou()
        assert rule.kan_dora_timing == KanDoraTimingMode.TenhouImmediate

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
        ankan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.Ankan]
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

    def test_tenhou_kakan_dora_before_discard(self):
        """
        Tenhou: Kakan reveals dora before discard (after rinshan tsumo)
        Expected order: kakan → tsumo → dahai → dora
        """
        rule = GameRule.default_tenhou()
        assert rule.kan_dora_timing == KanDoraTimingMode.TenhouImmediate

        env = RiichiEnv(seed=42, game_mode=0, rule=rule)
        env.reset()

        # Setup: Give player 0 a pon of 1m (tiles 0,1,2) and the 4th tile (tile 3) in hand
        hands = env.hands
        hands[0] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 60]  # Include tile 3 (the 4th 1m)
        env.hands = hands
        melds = env.melds
        melds[0] = [Meld(MeldType.Peng, tiles=[0, 1, 2], opened=True)]
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
        kakan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.Kakan]
        assert len(kakan_actions) > 0, "Should have KAKAN action available"

        # Execute kakan
        env.step({player_id: kakan_actions[0]})

        # Now we need to discard
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]
        discard_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.Discard]
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

        # Verify order: kakan → tsumo → dahai → dora
        assert tsumo_after_kakan[0] < dahai_after_kakan[0], (
            f"Tsumo should come before dahai. "
            f"Order: kakan@{kakan_idx}, tsumo@{tsumo_after_kakan[0]}, "
            f"dahai@{dahai_after_kakan[0]}"
        )
        assert dahai_after_kakan[0] < dora_after_kakan[0], (
            f"Dahai should come before dora. "
            f"Order: kakan@{kakan_idx}, tsumo@{tsumo_after_kakan[0]}, "
            f"dahai@{dahai_after_kakan[0]}, dora@{dora_after_kakan[0]}"
        )

    def test_majsoul_ankan_dora_immediate(self):
        """
        Majsoul: Ankan reveals dora immediately after kan declaration
        Expected order: ankan → dora → tsumo
        """
        rule = GameRule.default_mjsoul()
        assert rule.kan_dora_timing == KanDoraTimingMode.MajsoulImmediate

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
        ankan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.Ankan]
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

        # Verify order: ankan → dora → tsumo (same as Tenhou for ankan)
        assert dora_after_ankan[0] < tsumo_after_ankan[0], (
            f"Dora should come before tsumo. "
            f"Order: ankan@{ankan_idx}, dora@{dora_after_ankan[0]}, "
            f"tsumo@{tsumo_after_ankan[0]}"
        )
