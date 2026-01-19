from riichienv import ActionType, Meld, MeldType, Phase

from ..helper import helper_setup_env


class TestKakan:
    def test_kakan_action_from_tsumo(self):
        """
        Verifies that KAKAN (Added Kan) actions are:
        1. Correctly generated in legal_actions when a player has a Pon and the 4th tile.
        2. Correctly executed, upgrading the Pon to an Addgang (Kakan) meld.
        """
        # Case1
        env = helper_setup_env(
            hands=[
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 60],
                [],
                [],
                [],
            ],
            melds=[
                [Meld(MeldType.Peng, tiles=[0, 1, 2], opened=True)],
                [],
                [],
                [],
            ],
            active_players=[0],
            current_player=0,
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=2,  # Use duplicate tid to verify relaxed check
        )
        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]
        assert any([a.action_type == ActionType.Kakan for a in obs.legal_actions()]), (
            "Should have KAKAN action available"
        )

    def test_kakan_action_from_hand(self):
        # Case2
        env = helper_setup_env(
            hands=[
                [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [],
                [],
                [],
            ],
            melds=[
                [Meld(MeldType.Peng, tiles=[0, 1, 2], opened=True)],
                [],
                [],
                [],
            ],
            active_players=[0],
            current_player=0,
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=13,
        )
        player_id = 0
        obs_dict = env.get_observations([player_id])
        obs = obs_dict[player_id]
        assert any([a.action_type == ActionType.Kakan for a in obs.legal_actions()]), (
            "Should have KAKAN action available"
        )

        kakan_actions = [a for a in obs.legal_actions() if a.action_type == ActionType.Kakan]
        assert len(kakan_actions) > 0, "Should have KAKAN action available"

        k_action = kakan_actions[0]
        assert k_action.tile == 3, "Should be able to Kakan with tile 3 (1m)"
        assert k_action.consume_tiles == [0, 1, 2], "Should consume the 3 tiles in the Pon"

        env.step({player_id: k_action})
        assert len(env.melds[player_id]) == 1
        new_meld = env.melds[player_id][0]
        assert new_meld.meld_type == MeldType.Addgang
        assert sorted(new_meld.tiles) == [0, 1, 2, 3]

        # Hand check
        current_hand = env.hands[player_id]
        assert 3 not in current_hand, "Tile 3 should be removed from hand"
        assert any([ev["type"] == "kakan" for ev in env.mjai_log])
