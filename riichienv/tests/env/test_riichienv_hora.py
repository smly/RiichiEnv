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
        obs_dict = env.step({0: Action(ActionType.Kakan, tile=63)})

        # Verify P1 can Ron
        assert env.phase == Phase.WaitResponse
        assert 1 in env.active_players
        obs1 = obs_dict[1]
        legals = obs1.legal_actions()
        ron = [a for a in legals if a.type == ActionType.Ron]
        assert len(ron) == 1

        # Execute Ron
        env.step({1: ron[0]})
        ev_types = [ev["type"] for ev in env.mjai_log[-3:]]
        assert "hora" in ev_types

        # 3: 槍槓. Should be the ONLY yaku.
        assert env.agari_results[1].yaku == [3]
