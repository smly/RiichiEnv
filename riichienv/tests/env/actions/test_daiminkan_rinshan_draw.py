import riichienv.convert as cvt
from riichienv import Action, ActionType, Phase, RiichiEnv


class TestDaiminkan:
    def test_daiminkan_rinshan_draw(self):
        """
        Verify that DAIMINKAN (Open Kan) triggers a Rinshan draw.
        """
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup Player 1 with a triplet of 2s
        p1 = 1
        tid_2s_1 = cvt.mjai_to_tid("2s")  # First 2s
        # Get 3 copies for hand
        triplet = [tid_2s_1, tid_2s_1 + 1, tid_2s_1 + 2]
        # Fill rest of hand to 13 total (3 + 10)
        # Just use some honors or manzu
        misc = [i for i in range(10)]
        h = env.hands
        h[p1] = sorted(triplet + misc)
        env.hands = h

        # Player 0 discards the 4th 2s
        p0 = 0
        discard_tile = tid_2s_1 + 3

        env.current_player = p0
        env.active_players = [p0]
        env.phase = Phase.WaitAct
        # Ensure P0 has the tile to discard
        # Hand size must be 14 (13+1) for discard

        # Actually RiichiEnv usually has 14 tiles if drawn_tile is None but hand has 14
        # OR 13 tiles + drawn_tile set.
        # But here we simulate "Already Drew".
        # Let's set drawn_tile to discard_tile for simplicity or put in hand.
        env.drawn_tile = discard_tile
        # 13 tiles in hand
        h = env.hands
        h[p0] = sorted([i for i in range(13)])
        env.hands = h

        # P0 Discards
        env.step({p0: Action(ActionType.Discard, tile=discard_tile)})

        # P1 Should have legal action DAIMINKAN
        obs = env.get_observations([p1])[p1]
        legal_actions = obs.legal_actions()
        kakan_actions = [a for a in legal_actions if a.action_type == ActionType.DAIMINKAN]

        assert len(kakan_actions) > 0, "P1 should have DAIMINKAN option"
        daiminkan_action = kakan_actions[0]

        # Capture wall length
        wall_len_before = len(env.wall)

        # Execute DAIMINKAN
        env.step({p1: daiminkan_action})

        # Verification
        # 1. P1 current player
        assert env.current_player == p1

        # 2. Drawn tile (Rinshan) should be set
        assert env.drawn_tile is not None

        # 3. Wall decreased by 1
        assert len(env.wall) == wall_len_before - 1

        # 4. Tsumo event logged
        last_event = env.mjai_log[-1]
        assert last_event["type"] == "tsumo"
        assert last_event["actor"] == p1
        assert last_event["pai"] == cvt.tid_to_mjai(env.drawn_tile)
