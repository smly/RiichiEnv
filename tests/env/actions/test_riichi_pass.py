from riichienv import Action, ActionType, Phase

from ..helper import helper_setup_env


class TestRiichiPassAction:
    """
    Tests that the player can pass after declaring Riichi and not win.
    """

    def test_riichi_pass_action(self) -> None:
        env = helper_setup_env(
            seed=42,
            riichi_declared=[True, False, False, False],
            current_player=0,
            phase=Phase.WaitAct,
            hands=[
                list(range(13)),
                [],
                [],
                [],
            ],
            melds=[
                [],
                [],
                [],
                [],
            ],
            drawn_tile=10,
        )
        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]

        # expected lecal actions => [ActionType.Discard, ActionType.Tsumo]
        assert ActionType.Discard in [a.action_type for a in obs.legal_actions()], (
            "DISCARD should be available in Riichi"
        )
        assert ActionType.Pass not in [a.action_type for a in obs.legal_actions()], (
            "PASS should NOT be available in Riichi (WaitAct)"
        )
        assert ActionType.Tsumo in [a.action_type for a in obs.legal_actions()], "Tsumo should be available in Riichi"

        act = Action(ActionType.Discard, 10)
        obs = env.step({player_id: act})
        assert 1 in obs, "Should be player 1's turn. Ensure the game has not been aborted."
        assert env.phase == Phase.WaitAct
        assert env.current_player == 1
        assert env.last_discard == (0, 10)
        assert env.mjai_log[-2]["type"] == "dahai"
        assert env.mjai_log[-2]["actor"] == 0
        assert env.mjai_log[-2]["tsumogiri"] is True
