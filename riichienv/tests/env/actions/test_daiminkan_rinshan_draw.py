import riichienv.convert as cvt
from riichienv import Action, ActionType, Phase, parse_hand

from ..helper import helper_setup_env


class TestDaiminkan:
    def test_daiminkan_rinshan_draw(self):
        """
        Verify that DAIMINKAN (Open Kan) triggers a Rinshan draw.
        """
        env = helper_setup_env(
            hands=[
                list(range(13)),
                list(parse_hand("222s")[0]) + list(range(10)),
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            phase=Phase.WaitAct,
            drawn_tile=cvt.mpsz_to_tid("2s"),
        )
        obs_dict = env.step({0: Action(ActionType.Discard, tile=cvt.mpsz_to_tid("2s"))})
        assert 1 in obs_dict, "Player 1 should be active"
        assert env.phase == Phase.WaitResponse, f"Phase should be WaitResponse, got {env.phase}"

        # P1 Should have legal action DAIMINKAN
        obs = obs_dict[1]
        legal_actions = obs.legal_actions()
        kan_actions = [a for a in legal_actions if a.action_type == ActionType.DAIMINKAN]
        assert len(kan_actions) > 0, "P1 should have DAIMINKAN option"
        daiminkan_action = kan_actions[0]

        # Capture wall length
        wall_len_before = len(env.wall)

        # Execute DAIMINKAN
        env.step({1: daiminkan_action})

        # Verification
        # 1. P1 current player
        assert env.current_player == 1
        assert env.phase == Phase.WaitAct, f"Phase should be WaitAct, got {env.phase}"
        # 2. Drawn tile (Rinshan) should be set
        assert env.drawn_tile is not None
        # 3. Wall decreased by 1
        assert len(env.wall) == wall_len_before - 1
        # 4. Tsumo event logged
        last_event = env.mjai_log[-1]
        assert last_event["type"] == "tsumo"
        assert last_event["actor"] == 1
        assert last_event["pai"] == cvt.tid_to_mjai(env.drawn_tile)
