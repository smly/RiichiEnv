from riichienv import Action, ActionType, Phase, RiichiEnv


class TestDiscardType:
    def test_discard_type_tracking(self) -> None:
        """
        Verify that `discard_from_hand` correctly tracks whether a discard was
        Tedashi (True) or Tsumogiri (False).
        """
        env = RiichiEnv(seed=42)
        env.reset()

        # Force a known state
        # P0 Hands
        # Make P0 turn
        env.current_player = 0
        env.phase = Phase.WaitAct

        # Case 1: Tsumogiri (Discard Drawn Tile)
        drawn_tile = 10  # 3m
        env.drawn_tile = drawn_tile
        # Ensure tile is in hand (simulate draw)
        # MUST assign back to env.hands to persist change to Rust struct
        h = env.hands
        h[0].append(drawn_tile)
        env.hands = h

        # Action: Discard the drawn tile
        env.step({0: Action(ActionType.Discard, tile=drawn_tile)})

        # Verify: Last discard for P0 should be Tsumogiri (from_hand=False)
        assert len(env.discard_from_hand[0]) == 1
        assert len(env.discards[0]) == 1
        assert env.discards[0][-1] == drawn_tile
        assert env.discard_from_hand[0][-1] is False, "Should be Tsumogiri (False)"

        # Case 2: Tedashi (Discard From Hand)
        # Advance/Reset to P0 again or just force state
        env.current_player = 0
        env.phase = Phase.WaitAct
        env.active_players = [0]
        env.needs_tsumo = False  # Already "drew" manually below

        new_drawn = 20  # 6m
        env.drawn_tile = new_drawn

        h = env.hands
        h[0].append(new_drawn)
        env.hands = h

        # Pick a different tile to discard (Tedashi)
        # P0 hand has 10 (from previous if not claimed) or other initial tiles.
        hand_tile = env.hands[0][0]
        if hand_tile == new_drawn:
            hand_tile = env.hands[0][1]

        env.step({0: Action(ActionType.Discard, tile=hand_tile)})

        # Verify
        assert len(env.discard_from_hand[0]) == 2
        assert env.discard_from_hand[0][-1] is True, "Should be Tedashi (True)"
