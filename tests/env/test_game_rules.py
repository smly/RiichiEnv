from riichienv import Action, ActionType, GameType, Wind, parse_hand

from .helper import helper_setup_env


class TestGameRule:
    """
    TODO: Add more test scenarios.
    """

    def test_one_round_with_abortive_draw(self) -> None:
        env = helper_setup_env(
            game_mode=GameType.YON_IKKYOKU,
            hands=[
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
                parse_hand("19m19p19s1234567z")[0],
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=1,
        )
        assert env.oya == 0
        assert env._custom_round_wind == Wind.East
        assert env.honba == 0

        obs = env.step({0: Action(ActionType.Discard, tile=80)})
        assert 1 in obs
        assert any(a for a in obs[1].legal_actions() if a.action_type == ActionType.KyushuKyuhai)
        env.step({1: Action(ActionType.KyushuKyuhai)})
        assert env.done()

    def test_tonpuu_ranchan_transitions(self) -> None:
        """
        Test transition: East 1 Honba 0 -> East 1 Honba 1
        """
        env = helper_setup_env(
            game_mode=GameType.YON_TONPUSEN,
            hands=[
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
                parse_hand("19m19p19s1234567z")[0],
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=1,
        )
        assert env.oya == 0
        assert env._custom_round_wind == Wind.East
        assert env.honba == 0

        obs = env.step({0: Action(ActionType.Discard, tile=80)})
        assert 1 in obs
        assert any(a for a in obs[1].legal_actions() if a.action_type == ActionType.KyushuKyuhai)
        obs = env.step({1: Action(ActionType.KyushuKyuhai)})
        assert not env.done()

        assert 0 in obs
        assert env.oya == 0
        assert env._custom_round_wind == Wind.East
        assert env.honba == 1
        assert env.mjai_log[-2]["type"] == "start_kyoku"
        assert env.mjai_log[-2]["oya"] == 0
        assert env.mjai_log[-2]["honba"] == 1

    def test_tonpuu_transitions(self) -> None:
        """
        Test transition: East 1 -> East 2
        """
        env = helper_setup_env(
            game_mode=GameType.YON_TONPUSEN,
            hands=[
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
                parse_hand("19m19p19s1234567z")[0],
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
                [0, 4, 8, 12, 36, 40, 44, 48, 72, 76, 80, 108, 112],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=1,
        )
        assert env.oya == 0
        assert env._custom_round_wind == Wind.East
        assert env.honba == 0

        obs = env.step({0: Action(ActionType.Discard, tile=80)})
        assert 1 in obs
        assert any(a for a in obs[1].legal_actions() if a.action_type == ActionType.KyushuKyuhai)
        obs = env.step({1: Action(ActionType.KyushuKyuhai)})
        assert not env.done()

        assert 0 in obs
        assert env.oya == 0
        assert env._custom_round_wind == Wind.East
        assert env.honba == 1
        assert env.mjai_log[-2]["type"] == "start_kyoku"
        assert env.mjai_log[-2]["oya"] == 0
        assert env.mjai_log[-2]["honba"] == 1
