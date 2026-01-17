from riichienv import Action, ActionType, Phase, RiichiEnv


class TestIllegalActions:
    def test_illegal_discard_penalty(self) -> None:
        env = RiichiEnv(game_mode="4p-red-east", seed=42)
        env.reset()

        # Player 0 Turn
        assert env.current_player == 0
        assert env.phase == Phase.WaitAct

        # P0 attempts to discard a tile they definitely don't have.
        # Standard tile IDs 0-135.
        # Let's ensure P0 doesn't have tile 135.
        p0_hand = env.hands[0]
        invalid_tile = 0
        while invalid_tile in p0_hand:
            invalid_tile += 1

        # Perform illegal action
        action = Action(ActionType.Discard, tile=invalid_tile)
        empty_dict = env.step({0: action})

        # NOTE: should return empty_dict?
        assert empty_dict == {}
        assert not env.done()

        # Verify Penalty
        # 1. MJAI Log should have ryukyoku with reason Error
        assert env.mjai_log[-1]["type"] == "end_kyoku"
        assert env.mjai_log[-2]["type"] == "ryukyoku"
        assert "Error: Illegal Action" in env.mjai_log[-2]["reason"]
        assert env.mjai_log[-2]["deltas"] == [-12000, 4000, 4000, 4000]

        # 2. Score Check
        # Oya (P0) Chombo: -12000 total (+4000 for others)
        scores = env.scores()
        assert scores[0] == 25000 - 12000
        assert scores[1] == 25000 + 4000
        assert scores[2] == 25000 + 4000
        assert scores[3] == 25000 + 4000

        # 3. Game Continuation (Renchan)
        # We need to step once to trigger the next round initialization
        env.step({})
        assert not env.done()

        assert env.oya == 0
        assert env.honba == 1
        assert env.kyoku_idx == 0

        assert len(env.hands[0]) == 14

    def test_illegal_response_penalty(self) -> None:
        # Scenario where response is illegal
        env = RiichiEnv(game_mode="4p-red-east", seed=42)
        env.reset()

        # P0 Turn
        assert env.current_player == 0

        # P1 sends action during P0's turn (Illegal: Out of Turn)
        # P0 sends VALID action.
        p0_hand = env.hands[0]
        valid_tile = p0_hand[-1]

        actions = {
            0: Action(ActionType.Discard, tile=valid_tile),  # Valid
            1: Action(ActionType.Discard, tile=0),  # Illegal (Out of turn)
        }
        env.step(actions)

        # Verify Penalty on P1
        ryukyoku_ev = None
        for ev in reversed(env.mjai_log):
            if ev["type"] == "ryukyoku":
                ryukyoku_ev = ev
                break

        assert ryukyoku_ev is not None
        assert "Error: Illegal Action" in ryukyoku_ev["reason"]

        # P1 (Ko) Chombo: -8000. Oya(P0)+4000. P2,P3 +2000.
        scores = env.scores()
        assert scores[1] == 25000 - 8000
        assert scores[0] == 25000 + 4000
        assert scores[2] == 25000 + 2000

        # Renchan (Abortive Draw)
        env.step({})
        assert env.honba == 1
        assert env.oya == 0

    def test_lower_id_penalty_priority(self) -> None:
        # Both P1 and P2 send illegal actions
        env = RiichiEnv(game_mode="4p-red-east", seed=42)
        env.reset()

        # P0 Valid discard
        # P1, P2 Illegal

        p0_hand = env.hands[0]
        valid_tile = p0_hand[-1]

        actions = {
            0: Action(ActionType.Discard, tile=valid_tile),
            1: Action(ActionType.Discard, tile=0),
            2: Action(ActionType.Discard, tile=0),
        }

        env.step(actions)

        # Expect P1 punished
        scores = env.scores()
        assert scores[1] == 25000 - 8000
        assert scores[2] == 25000 + 2000  # P2 gets points!
