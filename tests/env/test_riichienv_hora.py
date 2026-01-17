from riichienv import Action, ActionType, Meld, MeldType, Phase, RiichiEnv


class TestRiichiEnv:
    def test_chankan_ron_detection(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        # P1 Wait for 7p (Kanchan wait 6p-8p)
        # 1m, 2m, 3m (0, 4, 8)
        # 2p, 3p, 4p (40, 44, 48)
        # 1s, 2s, 3s (72, 76, 80)
        # 6p, 8p (56, 64) - Wait 7p (60)
        # 9s, 9s (104, 105) - Head
        p1_hand = [0, 4, 8, 40, 44, 48, 56, 64, 72, 76, 80, 104, 105]
        h = env.hands
        h[1] = p1_hand
        env.hands = h

        # P0 has a Pon of 7p
        # Tiles 60, 61, 62 in meld
        p0_meld = Meld(MeldType.Peng, [60, 61, 62], True)
        m = env.melds
        m[0] = [p0_meld]
        env.melds = m

        # P0 hand has 63 (the 4th 7p)
        h = env.hands
        h[0] = [63] + list(range(120, 132))  # Honors
        h[0].sort()
        env.hands = h

        # Setup state to simulate P0 drawing 63
        env.current_player = 0
        env.drawn_tile = 63
        env.active_players = [0]
        env.phase = Phase.WaitAct

        # P0 performs Kakan
        env.needs_tsumo = False
        print(
            f"DEBUG: P0 Legals: {[(a.action_type, a.tile, a.consume_tiles) for a in env.get_observations([0])[0].legal_actions()]}"
        )
        obs_dict = env.step({0: Action(ActionType.Kakan, tile=63)})

        # Verify P1 can Ron
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players
        obs1 = obs_dict[1]
        legals = obs1.legal_actions()
        ron = [a for a in legals if a.action_type == ActionType.Ron]
        assert len(ron) == 1

        # Execute Ron
        env.step({1: ron[0]})
        ev_types = [ev["type"] for ev in env.mjai_log[-3:]]
        assert "hora" in ev_types

        # 3: 槍槓. Should be the ONLY yaku.
        assert env.agari_results[1].yaku == [3]

        # Init Round: South 1. Bakaze=South (1).
        env.reset(bakaze=1, oya=0, honba=0, kyotaku=0)

        # Ensure P2 has tile 33 (9p) for discard
        h = env.hands
        h[2] = [33] + list(range(40, 52))  # P2 Hand
        h[3] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 32]  # P3 Hand (was pid=3)
        env.hands = h

        pid = 3
        # North player in South 1.

        # Melds (3 tiles): Pon South (112, 113, 114)
        m1 = Meld(MeldType.Peng, [112, 113, 114], True)

        melds = env.melds
        melds[pid] = [m1]
        env.melds = melds

        # Trigger Ron on 9p (33)
        discard_tile = 33
        env.current_player = 2
        env.phase = Phase.WaitAct
        env.active_players = [2]

        # Step: P2 discards 9p
        env.needs_tsumo = False
        obs_dict = env.step({2: Action(ActionType.Discard, tile=discard_tile)})

        assert pid in obs_dict, "P3 should receive an observation after discard."
        legals = obs_dict[pid].legal_actions()
        has_ron = any(a.action_type == ActionType.Ron for a in legals)
        assert has_ron, "Ron should be legal because South (Bakaze) is a Yaku in a South round."
        obs_dict = env.step({pid: Action(ActionType.Ron, tile=discard_tile)})
        # 11: 役牌:場風牌 => 南
        # 21: 対々和
        # 22: 三暗刻
        # 27: 混一色
        assert env.agari_results[pid].yaku == [11, 21, 22, 27]
