from riichienv import ActionType, Phase

from ..helper import helper_setup_env


class TestValidAnkanRiichiLegality:
    def test_valid_ankan_riichi_legality(self) -> None:
        env = helper_setup_env(
            seed=42,
            hands=[
                [0] * 13,
                [0] * 13,
                [4, 5, 68, 69, 71, 73, 74, 75, 88, 89, 90, 97, 98],
                [0] * 13,
            ],
            current_player=2,
            active_players=[2],
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=70,
            riichi_declared=[False, False, True, False],
        )
        obs_dict = env.get_observations()
        obs = obs_dict[2]
        legals = obs.legal_actions()

        ankan = [a for a in legals if a.action_type == ActionType.Ankan]
        assert len(ankan) == 1, f"Ankan should be LEGAL as it does NOT change waits. Legals: {legals}"
