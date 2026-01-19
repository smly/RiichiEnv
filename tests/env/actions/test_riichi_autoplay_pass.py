import pytest

from riichienv import Action, ActionType, parse_hand, parse_tile

from ..helper import helper_setup_env


class TestRiichiAutoPlayAfterPass:
    """
    Tests that the player can autoplay after declaring Riichi and not win.
    """

    def test_riichi_setup(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("111222333444m1z")[0],
                parse_hand("111222333444p2z")[0],
                parse_hand("111222333444s3z")[0],
                parse_hand("555666777888m4z")[0],
            ],
            current_player=0,
            drawn_tile=parse_tile("5p"),
        )
        obs = env.get_observations([0])
        obs = env.step({0: Action(ActionType.Riichi)})
        legal_actions = obs[0].legal_actions()

        # All legal actions should be Discard
        assert all(a.action_type == ActionType.Discard for a in legal_actions)
        # "5p" and "1z" should be the only tiles to discard in this case.
        legal_tiles = {a.tile for a in legal_actions}
        assert legal_tiles == {parse_tile("5p"), parse_tile("1z")}

    @pytest.mark.skip(reason="Too complex to test")
    def test_riichi_autoplay_after_pass(self):
        """
        Test scenario:
        1. P0 Riichi
        2. P1 Riichi
        3. P3 Riichi
        4. P2's discard cannot be called (chi) by P3. P3 is Riichi.
        5. P2's discard cannot be called (pon) by P0. P0 is Riichi.
        6. P2's discard can be called (ron) by P1. However, P1 passes.
        7. P2's discard cannot be called (ron) by P1.
        8. P1's discard can be called (ron) by P0.
        """
        pass
