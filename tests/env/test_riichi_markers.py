from riichienv import ActionType, RiichiEnv


class TestRiichiMarker:
    def test_riichi_declaration_markers(self) -> None:
        """
        Test that discard_is_riichi and riichi_declaration_index are correctly updated.
        """
        env = RiichiEnv(seed=42)
        env.reset()
        # Manual setup for Tenpai (Shanpon wait)
        # 123m, 456m, 789m, 2p(3), East(2) -> Discard 2p -> Wait 2p/East
        # 1m: 0,1,2. 4m: 12,13,14. 7m: 24,25,26. 2p: 40,41,42. East: 108,109.
        hands = env.hands
        hands[0] = [0, 1, 2, 12, 13, 14, 24, 25, 26, 40, 41, 42, 108, 109]
        env.hands = hands
        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        riichi_actions = [a for a in legals if a.action_type == ActionType.RIICHI]
        assert len(riichi_actions) > 0, "Manual setup did not provide Riichi opportunity"

        # 1. Declare Riichi
        # Use the first available Riichi action (doesn't matter which tile is discarded later)
        env.step({0: riichi_actions[0]})

        # Verify Riichi Stage set
        assert env.riichi_stage[0]
        assert not env.riichi_declared[0]

        # Determine what to discard
        # Riichi action doesn't specify discard tile, it just enters stage.
        # Now we must discard. LEGAL discards are restricted.
        # We must pick a legal discard.
        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        discards = [a for a in legals if a.action_type == ActionType.DISCARD]
        assert len(discards) > 0
        discard_action = discards[0]

        # 2. Discard (Riichi Declaration)
        env.step({0: discard_action})

        # Check discard_is_riichi
        # P0 should have 1 discard (assuming it's first turn)
        num_discards = len(env.discards[0])
        assert num_discards > 0
        riichi_discard_tile = env.discards[0][-1]
        assert riichi_discard_tile == discard_action.tile

        # The boolean array should reflect this
        riichi_flags = env.discard_is_riichi[0]
        assert len(riichi_flags) == num_discards
        assert riichi_flags[-1]
        assert not env.riichi_stage[0]
        assert env.riichi_declared[0]

        # Check declaration index
        riichi_idx = env.riichi_declaration_index[0]
        assert riichi_idx == num_discards - 1

        # 3. Next Step (verify state persists or resets)
        # Just verify other players don't have it set
        assert env.riichi_declaration_index[1] is None

    def test_no_riichi(self) -> None:
        env = RiichiEnv(seed=42)
        env.reset()

        # P0 just discards naturally
        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        discards = [a for a in legals if a.action_type == ActionType.DISCARD]

        if discards:
            env.step({0: discards[0]})
            assert not env.discard_is_riichi[0][-1]
            assert env.riichi_declaration_index[0] is None

    def test_reset_clears_markers(self) -> None:
        """
        Verify that reset() clears the riichi markers.
        """
        env = RiichiEnv(seed=42)
        env.reset()
        # Manual setup for Tenpai
        hands = env.hands
        hands[0] = [0, 1, 2, 12, 13, 14, 24, 25, 26, 40, 41, 42, 108, 109]
        env.hands = hands
        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        riichi_actions = [a for a in legals if a.action_type == ActionType.RIICHI]
        assert riichi_actions

        env.step({0: riichi_actions[0]})

        # Discard
        obs = env.get_observations([0])[0]
        legals = obs.legal_actions()
        discards = [a for a in legals if a.action_type == ActionType.DISCARD]
        env.step({0: discards[0]})

        # Check markers set
        assert env.riichi_declaration_index[0] is not None
        assert True in env.discard_is_riichi[0]

        # RESET
        env.reset()

        # Check markers cleared
        assert env.riichi_declaration_index[0] is None

        # discards should be empty, so discard_is_riichi should be empty
        assert len(env.discard_is_riichi[0]) == 0
