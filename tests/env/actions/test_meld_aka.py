from riichienv import Action, ActionType, Phase, parse_hand
from riichienv import convert as cvt

from ..helper import helper_setup_env


class TestMeldWithAkaDora:
    """
    Tests for melding with aka-dora
    """

    def test_can_upper_chi_5m(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("344m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=17,
        )
        env.step({0: Action(ActionType.Discard, 17, [])})
        assert env.phase == Phase.WaitResponse
        # 3m(8), 4m (12), 5m(17)
        env.step({1: Action(ActionType.Chi, tile=17, consume_tiles=[8, 12])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "chi"
        assert env.active_players == [1]

    def test_can_upper_chi_red_5m(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("344m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=16,
        )
        env.step({0: Action(ActionType.Discard, 16, [])})
        assert env.phase == Phase.WaitResponse
        # 3m(8), 4m (12), 0m(16)
        env.step({1: Action(ActionType.Chi, tile=16, consume_tiles=[8, 12])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "chi"
        assert env.active_players == [1]

    def test_can_middle_chi_5m(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("446m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=17,
        )
        env.step({0: Action(ActionType.Discard, 17, [])})
        assert env.phase == Phase.WaitResponse
        # 4m (12), 5m(17), 6m (20)
        env.step({1: Action(ActionType.Chi, tile=17, consume_tiles=[12, 20])})
        assert env.phase == Phase.WaitAct
        assert env.active_players == [1]
        assert env.mjai_log[-1]["type"] == "chi"

    def test_can_middle_chi_red_5m(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("446m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=16,
        )
        env.step({0: Action(ActionType.Discard, 16, [])})
        assert env.phase == Phase.WaitResponse
        # 4m (12), 0m(16), 6m (20)
        env.step({1: Action(ActionType.Chi, tile=16, consume_tiles=[12, 20])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "chi"
        assert env.active_players == [1]

    def test_can_lower_chi_5m(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("776m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=17,
        )
        env.step({0: Action(ActionType.Discard, 17, [])})
        assert env.phase == Phase.WaitResponse
        # 4m (12), 5m(17), 6m (20)
        env.step({1: Action(ActionType.Chi, tile=17, consume_tiles=[20, 24])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "chi"
        assert env.active_players == [1]

    def test_can_lower_chi_red_5m(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("776m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=16,
        )
        env.step({0: Action(ActionType.Discard, 16, [])})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [1]
        assert 20 in env.hands[1]  # 6m
        assert 24 in env.hands[1]  # 7m (first 7m)
        env.step({1: Action(ActionType.Chi, tile=16, consume_tiles=[20, 24])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "chi"
        assert env.active_players == [1]

    def test_can_lower_chi_red_5m_different_canonical_id(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("776m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=16,
        )
        env.step({0: Action(ActionType.Discard, 16, [])})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [1]
        assert 20 in env.hands[1]  # 6m
        assert 25 in env.hands[1]  # 7m (second 7m)
        assert cvt.tid_to_mpsz(25) == "7m"
        # NOTE: Verify relaxed check for chi
        env.step({1: Action(ActionType.Chi, tile=16, consume_tiles=[20, 25])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "chi"
        assert env.active_players == [1]

    def test_can_pon_5m_aka(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("556m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=16,
        )
        env.step({0: Action(ActionType.Discard, 16, [])})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [1]
        assert 17 in env.hands[1]
        assert 18 in env.hands[1]
        env.step({1: Action(ActionType.Pon, tile=16, consume_tiles=[17, 18])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "pon"
        assert env.active_players == [1]

    def test_can_pon_5m_normal(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("556m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=19,
        )
        env.step({0: Action(ActionType.Discard, 19, [])})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [1]
        assert 17 in env.hands[1]
        assert 18 in env.hands[1]
        env.step({1: Action(ActionType.Pon, tile=19, consume_tiles=[17, 18])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "pon"
        assert env.active_players == [1]

    def test_can_pon_red_5m_in_hand(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                parse_hand("123456789m1234p")[0],
                parse_hand("056m123456789p3s")[0],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=18,
        )
        env.step({0: Action(ActionType.Discard, 18, [])})
        assert env.phase == Phase.WaitResponse
        assert env.active_players == [1]
        assert 16 in env.hands[1]
        assert 17 in env.hands[1]
        env.step({1: Action(ActionType.Pon, tile=18, consume_tiles=[16, 17])})
        assert env.phase == Phase.WaitAct
        assert env.mjai_log[-1]["type"] == "pon"
        assert env.active_players == [1]
