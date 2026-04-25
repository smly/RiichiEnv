import json

from riichienv import Action, ActionType, RiichiEnv

from .env.helper import helper_setup_env


class TestMjaiProtocol:
    def test_action_to_mjai_red_fives(self):
        # Test red fives mapping to 5mr, 5pr, 5sr
        act_m = Action(ActionType.DISCARD, tile=16)
        assert '"pai":"5mr"' in act_m.to_mjai()

        act_p = Action(ActionType.DISCARD, tile=52)
        assert '"pai":"5pr"' in act_p.to_mjai()

        act_s = Action(ActionType.DISCARD, tile=88)
        assert '"pai":"5sr"' in act_s.to_mjai()

    def test_select_action_from_mjai_discard(self):
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()
        obs = obs_dict[0]

        # Get a legal discard
        legal_discards = [a for a in obs.legal_actions() if a.action_type == ActionType.DISCARD]
        target_act = legal_discards[0]
        mjai_resp = json.loads(target_act.to_mjai())

        # Select from MJAI
        selected = obs.select_action_from_mjai(mjai_resp)
        assert selected is not None
        assert selected.action_type == ActionType.DISCARD
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
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=9)})

        obs1 = obs_dict[1]
        chi_acts = [a for a in obs1.legal_actions() if a.action_type == ActionType.CHI]
        assert len(chi_acts) > 0

        target_act = chi_acts[0]
        mjai_resp = json.loads(target_act.to_mjai())

        selected = obs1.select_action_from_mjai(mjai_resp)
        assert selected is not None
        assert selected.action_type == ActionType.CHI
        assert set(selected.consume_tiles) == set(target_act.consume_tiles)

    def test_select_action_from_mjai_dahai_tsumogiri(self):
        # Player has two non-red 5m in the existing hand and draws a third 5m.
        # mjai's tsumogiri flag must disambiguate which Discard action is picked.
        # Tile ids 17, 18 are non-red 5m; 16 is red 5m. Drawn tile = 19 (non-red 5m).
        env = helper_setup_env(
            seed=42,
            hands=[
                [0, 4, 17, 18, 36, 40, 44, 48, 72, 76, 80, 84, 108],
                [3, 7, 11, 15, 23, 27, 31, 35, 39, 43, 47, 51, 109],
                [],
                [],
            ],
            current_player=0,
            active_players=[0],
            drawn_tile=19,
        )
        obs = env.get_observation(0)

        # Tsumogiri=True must select the action whose tile is the drawn tile (19).
        selected_t = obs.select_action_from_mjai({"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True})
        assert selected_t is not None
        assert selected_t.action_type == ActionType.DISCARD
        assert selected_t.tile == 19

        # Tsumogiri=False must select an action whose tile is NOT the drawn tile.
        selected_f = obs.select_action_from_mjai({"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False})
        assert selected_f is not None
        assert selected_f.action_type == ActionType.DISCARD
        assert selected_f.tile != 19
        assert selected_f.tile in (17, 18)

    def test_select_action_from_mjai_pon_consumed(self):
        # Player has 5m red, 5m, 5m in hand and someone discards a 5m → two
        # Pon actions exist (with and without the red 5m). The `consumed` field
        # must disambiguate between them.
        env = helper_setup_env(
            seed=42,
            hands=[
                [0, 4, 8, 12, 16, 17, 18, 36, 40, 44, 48, 72, 76],  # P0: has 5mr,5m,5m
                [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 108],  # P1
                [],
                [],
            ],
            current_player=1,
            active_players=[1],
            drawn_tile=108,
        )
        # P1 discards a non-red 5m (tile id 19).
        env.step({1: Action(ActionType.DISCARD, tile=19)})
        obs0 = env.get_observation(0)

        pon_acts = [a for a in obs0.legal_actions() if a.action_type == ActionType.PON]
        assert len(pon_acts) >= 2  # both with and without aka

        # With red 5m in consumed
        sel_aka = obs0.select_action_from_mjai(
            {
                "type": "pon",
                "actor": 0,
                "target": 1,
                "pai": "5m",
                "consumed": ["5mr", "5m"],
            }
        )
        assert sel_aka is not None
        assert sel_aka.action_type == ActionType.PON
        assert 16 in sel_aka.consume_tiles  # red 5m

        # Without red 5m in consumed
        sel_plain = obs0.select_action_from_mjai(
            {
                "type": "pon",
                "actor": 0,
                "target": 1,
                "pai": "5m",
                "consumed": ["5m", "5m"],
            }
        )
        assert sel_plain is not None
        assert sel_plain.action_type == ActionType.PON
        assert 16 not in sel_plain.consume_tiles  # no red 5m

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
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=9)})
        obs1 = obs_dict[1]

        selected = obs1.select_action_from_mjai({"type": "none"})
        assert selected is not None
        assert selected.action_type == ActionType.PASS
