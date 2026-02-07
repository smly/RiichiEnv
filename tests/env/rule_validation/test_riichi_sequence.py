"""
Test for Riichi action sequence handling.

This test verifies that RiichiEnv correctly handles the MJAI reach + dahai sequence
that Mortal (and other MJAI-based bots) use.

Problem scenario:
1. Player has 4 tiles of the same kind (can ankan) and is tenpai (can riichi)
2. Bot chooses to riichi
3. After riichi declaration, ankan should no longer be available (unless wait is unchanged)
4. The legal_actions should reflect the riichi_stage state

The issue was that RiichiEnv's legal_actions was computed before the riichi_stage
was set, causing a mismatch with Mortal's internal state.
"""

from riichienv import ActionType, Phase

from ..helper import helper_setup_env


class TestRiichiSequenceHandling:
    """Test riichi declaration sequence handling."""

    def test_riichi_stage_disables_ankan(self) -> None:
        """
        Test that when riichi_stage is True (reach declared, awaiting discard),
        ankan is NOT available.

        riichi_stage is the intermediate state between declaring reach and choosing
        the discard tile. At this point only discard actions are offered.
        The Tenhou "wait-unchanged" ankan rule applies later, after riichi_accepted.

        Hand: 333346m 23477p 345s (tenpai, waiting for 5m)
        With 3m as drawn tile (4 tiles of 3m in hand).

        Before riichi: can_ankan = True (4 tiles of 3m)
        After riichi_stage = True: can_ankan = False
        """
        # Tile IDs for 333346m 23477p 345s
        # 3m = tile_id 8,9,10,11 (0-indexed: 3m is index 2, so 2*4=8, 2*4+1=9, 2*4+2=10, 2*4+3=11)
        # 4m = tile_id 12,13,14,15
        # 6m = tile_id 20,21,22,23
        # 2p = tile_id 40,41,42,43
        # 3p = tile_id 44,45,46,47
        # 4p = tile_id 48,49,50,51
        # 7p = tile_id 60,61,62,63
        # 3s = tile_id 80,81,82,83
        # 4s = tile_id 84,85,86,87
        # 5s = tile_id 88,89,90,91

        hand = [
            8,
            9,
            10,  # 3m x3
            12,  # 4m
            20,  # 6m
            40,
            44,
            48,  # 2p, 3p, 4p
            60,
            61,  # 7p x2
            80,
            84,
            88,  # 3s, 4s, 5s
        ]
        drawn_tile = 11  # 3m (4th tile, completing the set for ankan)

        env = helper_setup_env(
            seed=42,
            hands=[
                [0] * 13,
                [0] * 13,
                [0] * 13,
                hand,  # Player 3
            ],
            current_player=3,
            active_players=[3],
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=drawn_tile,
            riichi_declared=[False, False, False, False],
            points=[25000, 25000, 25000, 25000],
        )

        # Verify initial state: should have both riichi and ankan available
        obs_dict = env.get_observations()
        obs = obs_dict[3]
        legals = obs.legal_actions()

        riichi_actions = [a for a in legals if a.action_type == ActionType.Riichi]
        ankan_actions = [a for a in legals if a.action_type == ActionType.Ankan]

        assert len(riichi_actions) > 0, "Should be able to riichi when tenpai and menzen"
        assert len(ankan_actions) > 0, "Should be able to ankan with 4 tiles of 3m"

        # Now set riichi_stage = True (simulating after reach declaration)
        env.riichi_stage = [False, False, False, True]

        # Get new observations - legal_actions should now exclude ankan
        # (or include only if it doesn't change waits)
        obs_dict = env.get_observations()
        obs = obs_dict[3]
        legals_after_reach = obs.legal_actions()

        # During riichi_stage (reach declared, awaiting discard), ankan is not offered.
        # Only discard actions are available at this point.
        ankan_after_reach = [a for a in legals_after_reach if a.action_type == ActionType.Ankan]
        assert len(ankan_after_reach) == 0, "Ankan should NOT be available during riichi_stage"

        discard_actions = [a for a in legals_after_reach if a.action_type == ActionType.Discard]
        assert len(discard_actions) > 0, "Should be able to discard after reach"

        # Riichi should NOT be available anymore (already in riichi_stage)
        riichi_after_reach = [a for a in legals_after_reach if a.action_type == ActionType.Riichi]
        assert len(riichi_after_reach) == 0, "Riichi should not be available when already in riichi_stage"

    def test_mjai_reach_then_dahai_sequence(self) -> None:
        """
        Test the actual MJAI sequence: reach event followed by dahai event.

        This simulates what Mortal does:
        1. Returns {"type": "reach", "actor": 3}
        2. Then returns {"type": "dahai", "actor": 3, "pai": "6m", "tsumogiri": false}

        RiichiEnv should process both correctly.
        """
        hand = [
            8,
            9,
            10,  # 3m x3
            12,  # 4m
            20,  # 6m
            40,
            44,
            48,  # 2p, 3p, 4p
            60,
            61,  # 7p x2
            80,
            84,
            88,  # 3s, 4s, 5s
        ]
        drawn_tile = 11  # 3m

        env = helper_setup_env(
            seed=42,
            hands=[
                [0] * 13,
                [0] * 13,
                [0] * 13,
                hand,
            ],
            current_player=3,
            active_players=[3],
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=drawn_tile,
            riichi_declared=[False, False, False, False],
            points=[25000, 25000, 25000, 25000],
        )

        # Get initial observations
        obs_dict = env.get_observations()
        obs = obs_dict[3]

        # Simulate Mortal returning reach action
        reach_response = {"type": "reach", "actor": 3}
        reach_action = obs.select_action_from_mjai(reach_response)

        assert reach_action is not None, "Should be able to select reach action"
        assert reach_action.action_type == ActionType.Riichi, "Action type should be Riichi"

        # Step with reach action (no tile specified - this is the issue!)
        # RiichiEnv's Riichi action can optionally include a tile
        actions = {3: reach_action}
        new_obs_dict = env.step(actions)

        # After reach action (without tile), riichi_stage should be True
        assert env.riichi_stage[3] is True, "riichi_stage should be True after reach action"
        assert env.riichi_declared[3] is False, "riichi_declared should still be False (not yet accepted)"

        # Now get new observations - this is where the fix matters
        new_obs = new_obs_dict[3]
        new_legals = new_obs.legal_actions()

        # Should only have discard actions available (and possibly ankan if wait unchanged)
        # But should NOT have Riichi available
        riichi_in_new_legals = [a for a in new_legals if a.action_type == ActionType.Riichi]
        assert len(riichi_in_new_legals) == 0, (
            f"Riichi should not be available after reach declaration. Legals: {new_legals}"
        )

        # Should have discard actions
        discard_in_new_legals = [a for a in new_legals if a.action_type == ActionType.Discard]
        assert len(discard_in_new_legals) > 0, "Should have discard actions available"

        # Now simulate the dahai action
        # Mortal wants to discard 6m (tile_id 20)
        dahai_response = {"type": "dahai", "actor": 3, "pai": "6m", "tsumogiri": False}
        dahai_action = new_obs.select_action_from_mjai(dahai_response)

        assert dahai_action is not None, f"Should be able to select dahai action. Legals: {new_legals}"
        assert dahai_action.action_type == ActionType.Discard, "Action type should be Discard"

        # Step the env with the dahai action to complete the reach→dahai sequence
        env.step({3: dahai_action})

        # After the discard following reach, riichi should now be fully declared
        assert env.riichi_declared[3] is True, "riichi_declared should be True after reach+dahai"
        assert env.riichi_stage[3] is False, "riichi_stage should be False after dahai completes"

        # Riichi deposit: 1000 pts deducted
        scores = env.scores()
        assert scores[3] == 24000, f"Score should be 24000 after 1000pt riichi deposit, got {scores[3]}"

    def test_legal_actions_consistency_with_mortal_state(self) -> None:
        """
        Test that legal_actions is consistent with what Mortal expects after reach.

        After reach declaration:
        - can_discard = True
        - can_riichi = False
        - can_ankan = only if doesn't change wait (using Tenhou rules)
        - can_kakan = False (during riichi_stage, before acceptance)
        """
        hand = [
            8,
            9,
            10,  # 3m x3
            12,  # 4m
            20,  # 6m
            40,
            44,
            48,  # 2p, 3p, 4p
            60,
            61,  # 7p x2
            80,
            84,
            88,  # 3s, 4s, 5s
        ]
        drawn_tile = 11  # 3m

        env = helper_setup_env(
            seed=42,
            hands=[
                [0] * 13,
                [0] * 13,
                [0] * 13,
                hand,
            ],
            current_player=3,
            active_players=[3],
            phase=Phase.WaitAct,
            needs_tsumo=False,
            drawn_tile=drawn_tile,
            riichi_declared=[False, False, False, False],
            points=[25000, 25000, 25000, 25000],
        )

        # Set riichi_stage directly to simulate post-reach state
        env.riichi_stage = [False, False, False, True]

        obs_dict = env.get_observations()
        obs = obs_dict[3]
        legals = obs.legal_actions()

        # Check consistency with expected Mortal state after reach
        can_discard = any(a.action_type == ActionType.Discard for a in legals)
        can_riichi = any(a.action_type == ActionType.Riichi for a in legals)

        assert can_discard is True, "Should be able to discard during riichi_stage"
        assert can_riichi is False, "Should NOT be able to riichi when already in riichi_stage"
