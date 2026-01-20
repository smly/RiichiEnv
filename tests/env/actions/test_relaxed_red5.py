from riichienv import Action, ActionType, MeldType

from ..helper import helper_setup_env


class TestRelaxedRed5Check:
    def test_chi_red5_valid(self) -> None:
        """
        Verify that Chi can use a Red 5 specifically when the player holds it.
        Setup: Player 0 has [4m, 0m(Red 5m)] + filler. Player 3 discards 3m. P0 calls Chi.
        """
        # 0m(Red 5m), 4m. Hand needs to be 13 tiles.
        # Filler: 1z,1z,1z, 2z,2z,2z, 3z,3z,3z, 4z,4z
        hand = [16, 12] + [108, 109, 110, 112, 113, 114, 116, 117, 118, 120, 121]

        env = helper_setup_env(
            hands=[hand, [], [], []],
            current_player=3,  # P3 acts
            drawn_tile=8,  # 3m
            active_players=[3],
            phase=0,  # Discard
        )

        # Player 3 discards 3m
        obs = env.step({3: Action(ActionType.Discard, tile=8)})

        if 0 in obs:
            # P0 should have legal actions.
            legal_types = [a.action_type for a in obs[0].legal_actions()]
            print(f"DEBUG: P0 Legal Action Types: {legal_types}")

        # Player 0 Chi: [0m, 4m, 3m] -> consume [16, 12]
        action = Action(ActionType.Chi, tile=8, consume_tiles=[16, 12])

        obs = env.step({0: action})

        assert env.melds[0][0].meld_type == MeldType.Chi
        assert 16 in env.melds[0][0].tiles

    def test_chi_red5_relaxed_match_non_red(self) -> None:
        """
        Verify strict checking for Red 5 matching.
        """
        hand = [16, 12] + [108, 109, 110, 112, 113, 114, 116, 117, 118, 120, 121]
        env = helper_setup_env(hands=[hand, [], [], []], current_player=3, drawn_tile=8, active_players=[3], phase=0)
        env.step({3: Action(ActionType.Discard, tile=8)})

        # Action specifies Normal 5m (20) but we have Red 5m (16).
        action = Action(ActionType.Chi, tile=8, consume_tiles=[20, 12])

        env.step({0: action})

        has_meld = len(env.melds[0]) > 0
        assert not has_meld

    def test_chi_normal5_relaxed(self) -> None:
        """
        Verify relaxed checking for Normal 5s.
        """
        # Case 1: Hand=[17, 12], Action=[18, 12]. Should FAIL.
        # 17 (5m_1), 12 (4m). Discard 8 (3m). Valid Chi (345).
        # Action asks for 18 (5m_2, Normal). Hand has 17.
        # Relax check: 18 matches 17 (Same type, Non-Red).
        hand = [17, 12] + [108, 109, 110, 112, 113, 114, 116, 117, 118, 120, 121]
        env = helper_setup_env(hands=[hand, [], [], []], current_player=3, drawn_tile=8, active_players=[3], phase=0)
        env.step({3: Action(ActionType.Discard, tile=8)})

        action = Action(ActionType.Chi, tile=8, consume_tiles=[18, 12])
        env.step({0: action})
        assert len(env.melds[0]) == 0, "Strict hand check should prevent using 18 when holding 17"

    def test_chi_normal5_relaxed_pass(self) -> None:
        # Case 2: Hand=[17, 12], Action=[17, 12]. Should PASS.
        hand = [17, 12] + [108, 109, 110, 112, 113, 114, 116, 117, 118, 120, 121]
        env = helper_setup_env(hands=[hand, [], [], []], current_player=3, drawn_tile=8, active_players=[3], phase=0)
        env.step({3: Action(ActionType.Discard, tile=8)})

        action = Action(ActionType.Chi, tile=8, consume_tiles=[17, 12])
        env.step({0: action})
        assert len(env.melds[0]) > 0
        assert 17 in env.melds[0][0].tiles
