from riichienv import Action, ActionType, Phase

from ..helper import helper_setup_env


class TestClaimPriority:
    """
    Verify claim priority: Pon > Chi.
    """

    def test_pon_priority_over_chi(self):
        """
        Setup scenario:

        1. Player 0 discards a tile.
        2. Player 1 (Next) wants to Chi.
        3. Player 2 (Opposite/Other) wants to Pon.
        4. Verify claim priority: Pon > Chi.
        """
        env = helper_setup_env(
            seed=1,
            hands=[
                [57] + [2] * 12,  # P0
                [62, 65] + [0] * 11,  # P1
                [56, 58] + [1] * 11,  # P2
                [12, 16, 19, 21, 48, 59, 64, 77, 81, 89, 104, 130, 133],  # P3
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=100,
        )

        # P0 Discards 57
        env.step({0: Action(ActionType.Discard, tile=57)})
        assert env.phase == Phase.WaitResponse

        # We expect P1 and P2 to be in active_players if they have legal actions.
        assert env.active_players == [1, 2]

        # Submit Actions: P1 Chi, P2 Pon
        action_chi = Action(ActionType.Chi, tile=57, consume_tiles=[62, 65])
        action_pon = Action(ActionType.Pon, tile=57, consume_tiles=[56, 58])
        actions = {1: action_chi, 2: action_pon}

        # Step
        env.step(actions)

        # Expectation: Pon wins.
        assert env.phase == Phase.WaitAct
        assert env.active_players == [2]

        # Check Log
        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "pon"
