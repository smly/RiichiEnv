import json

from riichienv import Action, ActionType, RiichiEnv

from .env.helper import helper_setup_env


class TestMjaiProtocol:
    def test_action_to_mjai_red_fives(self):
        # Test red fives mapping to 5mr, 5pr, 5sr
        act_m = Action(ActionType.Discard, tile=16)
        assert '"pai":"5mr"' in act_m.to_mjai()

        act_p = Action(ActionType.Discard, tile=52)
        assert '"pai":"5pr"' in act_p.to_mjai()

        act_s = Action(ActionType.Discard, tile=88)
        assert '"pai":"5sr"' in act_s.to_mjai()

    def test_select_action_from_mjai_discard(self):
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()
        obs = obs_dict[0]

        # Get a legal discard
        legal_discards = [a for a in obs.legal_actions() if a.action_type == ActionType.Discard]
        target_act = legal_discards[0]
        mjai_resp = json.loads(target_act.to_mjai())

        # Select from MJAI
        selected = obs.select_action_from_mjai(mjai_resp)
        assert selected is not None
        assert selected.action_type == ActionType.Discard
        assert selected.tile == target_act.tile

    def test_select_action_from_mjai_chi(self):
        env = helper_setup_env(
            seed=42,
            hands=[
                [0, 1, 2, 9, 16, 17, 18, 19, 32, 33, 34, 35, 48, 52, 108],
                [3, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 49],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=108,
            wall=list(range(136)),
        )

        # Setup for Chi
        # P0 discards 3m.
        # Manually trigger discard and claim update.
        env.current_player = 0
        obs_dict = env.step({0: Action(ActionType.Discard, tile=9)})

        obs1 = obs_dict[1]
        chi_acts = [a for a in obs1.legal_actions() if a.action_type == ActionType.Chi]
        assert len(chi_acts) > 0

        target_act = chi_acts[0]
        mjai_resp = json.loads(target_act.to_mjai())

        selected = obs1.select_action_from_mjai(mjai_resp)
        assert selected is not None
        assert selected.action_type == ActionType.Chi
        assert set(selected.consume_tiles) == set(target_act.consume_tiles)

    def test_select_action_from_mjai_none(self):
        env = helper_setup_env(
            seed=42,
            hands=[
                [0, 1, 2, 9, 16, 17, 18, 19, 32, 33, 34, 35, 48, 52, 108],
                [3, 4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 49],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=108,
            wall=list(range(136)),
        )
        # tile 9 is in P0 hand (3m)
        obs_dict = env.step({0: Action(ActionType.Discard, tile=9)})
        obs1 = obs_dict[1]

        selected = obs1.select_action_from_mjai({"type": "none"})
        assert selected is not None
        assert selected.action_type == ActionType.Pass
