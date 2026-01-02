from riichienv import Meld, MeldType, RiichiEnv
from riichienv.action import ActionType


class TestKakan:
    def test_kakan_action_generation_and_execution(self):
        """
        Verifies that KAKAN (Added Kan) actions are:
        1. Correctly generated in legal_actions when a player has a Pon and the 4th tile.
        2. Correctly executed, upgrading the Pon to an Addgang (Kakan) meld.
        """
        env = RiichiEnv(seed=42)
        env.reset()

        # Setup Player 0
        # Player 0 has a Pon of 1m (tiles 0, 1, 2)
        # Player 0 has the 4th 1m (tile 3) in hand

        # Manually inject state
        player_id = 0
        env.hands[player_id] = [3, 4, 8, 12, 16, 20, 24, 28, 32, 36]  # 10 tiles + 3 in meld = 13.
        # Ensure hand is sorted for consistency, though Env usually handles it.
        env.hands[player_id].sort()

        # Create Pon Meld
        # Note: RiichiEnv Melds often store tiles as list of integers
        pon_meld = Meld(MeldType.Peng, tiles=[0, 1, 2], opened=True)
        env.melds[player_id] = [pon_meld]

        env.active_players = [player_id]
        env.current_player = player_id
        env.phase = 0  # Phase.WAIT_ACT

        # Force drawn tile to be something unrelated, or just ensure hand has correct count.
        # If it's WAIT_ACT, usually we just drew a tile.
        # Let's verify _get_legal_actions logic.
        # It usually expects 14 tiles total (hand + drawn) or 13 + drawn.
        # env.hands[0] is 10. + 1 drawn? That's 11. + 3 melded = 14.
        # So we need 13 tiles in hand if no drawn tile, or 12 + 1 drawn.
        # Let's make logic simple: 10 in hand, 3 in meld. We need 1 more to simulate "After Draw".

        env.hands[player_id] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 10 tiles
        env.drawn_tile = 13  # 11th tile. Total 11 + 3 = 14. Correct.

        # Get Observations
        obs_dict = env._get_observations([player_id])
        obs = obs_dict[player_id]

        # Check Legal Actions
        legal_actions = obs.legal_actions()
        kakan_actions = [a for a in legal_actions if a.type == ActionType.KAKAN]

        assert len(kakan_actions) > 0, "Should have KAKAN action available"

        # Verify action details
        k_action = kakan_actions[0]
        assert k_action.tile == 3, "Should be able to Kakan with tile 3 (1m)"
        assert k_action.consume_tiles == [3], "Should consume tile 3"

        # Execute Kakan
        env.step({player_id: k_action})

        # Verify State Update
        # 1. Meld should be upgraded to Addgang (Kakan)
        assert len(env.melds[player_id]) == 1
        new_meld = env.melds[player_id][0]
        assert new_meld.meld_type == MeldType.Addgang
        assert sorted(list(new_meld.tiles)) == [0, 1, 2, 3]

        # 2. Hand should not contain tile 3 anymore
        # Also drawn_tile should be consumed (or effectively merged then removed)
        # RiichiEnv step merges drawn_tile into hand before action execution if current player.
        # Hand count should be original 10 + 1 (drawn) - 1 (kakan) = 10?
        # Wait, Kakan uses 1 tile from hand to add to 3 existing.
        # After Kakan, we have 4 in meld. Total tiles 14.
        # We need to draw a Replacement Tile (Rinshan).
        # RiichiEnv usually handles Rinshan logic in step?
        # Or does it just set phase to WAIT_ACT again and draw?
        # Let's check env.py logic for KAKAN execution flow if needed,
        # but for this test we primarily care about the successful execution of the action.

        # Hand check
        current_hand = env.hands[player_id]
        assert 3 not in current_hand, "Tile 3 should be removed from hand"

        # MJAI Log check
        # After Kakan, a Rinshan replacement tile is usually drawn, generating a 'tsumo' event.
        # So the Kakan event should be the second to last event (or earlier if other logic exists).

        # Check last few events for Kakan
        found_kakan = False
        for ev in env.mjai_log[-3:]:  # check last 3
            # RiichiEnv currently logs KAKAN as type="kakan" with 1 consumed tile (the added tile).
            # ANKAN consumes 4, DAIMINKAN consumes 3.
            if ev["type"] == "kakan" and len(ev.get("consumed", [])) == 1:
                found_kakan = True
                break

        assert found_kakan, "KAKAN event should be logged (type='kakan' with 1 consumed tile)"
