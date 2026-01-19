from riichienv import Action, ActionType, Meld, MeldType, Phase, parse_hand

from ..helper import helper_setup_env


class TestKyushuKyuhai:
    def test_kyushu_kyuhai_abortive_draw(self) -> None:
        """
        Verify that KYUSHU_KYUHAI (9 types of terminals and honors) triggers an abortive draw.
        """
        # Setup player 0 with 9 unique terminals/honors
        env = helper_setup_env(
            game_mode=0,
            seed=42,
            hands=[
                parse_hand("19m19p19s123z2222m")[0],
                [],
                [],
                [],
            ],
            drawn_tile=12,
            current_player=0,
            active_players=[0],
            phase=Phase.WaitAct,
            needs_tsumo=False,
        )
        obs = env.get_observations([0])[0]
        legals = [a.action_type for a in obs.legal_actions()]
        assert ActionType.KYUSHU_KYUHAI in legals

        # Execute it
        env.step({0: Action(ActionType.KYUSHU_KYUHAI)})

        # Check if ryukyoku is logged
        found_ryukyoku = False
        for log in env.mjai_log:
            if log["type"] == "ryukyoku" and log.get("reason") == "kyushu_kyuhai":
                found_ryukyoku = True
                break

        assert found_ryukyoku
        assert env.done()

    def test_kyushu_kyuhai_not_available_after_meld(self) -> None:
        """
        Verify that KYUSHU_KYUHAI is not available if the player has any melds.
        """
        env = helper_setup_env(
            game_mode=0,
            seed=42,
            hands=[
                parse_hand("19m19p19s123z2222m")[0],
                [],
                [],
                [],
            ],
            melds=[
                [],
                [Meld(MeldType.Peng, [20, 21, 22], True)],
                [],
                [],
            ],
            drawn_tile=12,
            current_player=0,
            active_players=[0],
            phase=Phase.WaitAct,
            needs_tsumo=False,
        )
        obs = env.get_observations([0])[0]
        legals = [a.action_type for a in obs.legal_actions()]

        # Should NOT be available because there's a meld
        assert ActionType.KYUSHU_KYUHAI not in legals
