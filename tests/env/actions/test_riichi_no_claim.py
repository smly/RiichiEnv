from riichienv import Action, ActionType, Phase

from ..helper import helper_setup_env


class TestRiichiNoClaim:
    """
    Verify that Riichi player does not offer Pon or Chi.
    """

    def test_riichi_no_claim(self):
        # P0 in Riichi
        env = helper_setup_env(
            hands=[
                [0, 1, 4, 5, 8, 20, 20, 20, 20, 20, 20, 20, 20],
                [23, 35, 38, 61, 69, 70, 79, 83, 98, 123, 127, 128, 130],
                [1, 4, 5, 8, 12, 17, 56, 59, 81, 94, 101, 106, 122],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            ],
            current_player=3,
            active_players=[3],
            riichi_declared=[True, False, False, False],
            phase=Phase.WaitAct,
            drawn_tile=2,
        )
        env.step({3: Action(ActionType.Discard, 2)})

        # Turned into P0's turn implies PON opportunity was skipped
        assert env.phase == Phase.WaitAct and env.active_players == [0]

        # Test Chi
        env = helper_setup_env(
            hands=[
                [0, 1, 4, 5, 8, 20, 20, 20, 20, 20, 20, 20, 20],
                [23, 35, 38, 61, 69, 70, 79, 83, 98, 123, 127, 128, 130],
                [1, 4, 5, 8, 12, 17, 56, 59, 81, 94, 101, 106, 122],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11],
            ],
            current_player=3,
            active_players=[3],
            riichi_declared=[True, False, False, False],
            phase=Phase.WaitAct,
            drawn_tile=11,
        )
        obs = env.step({3: Action(ActionType.Discard, 11)})
        assert env.phase == Phase.WaitResponse and env.active_players == [0]
        actions = obs[0].legal_actions()
        assert not any(ac.action_type == ActionType.Chi for ac in actions)
        assert any(ac.action_type == ActionType.Ron for ac in actions)

    def test_riichi_rule_violation(self) -> None:
        pass
