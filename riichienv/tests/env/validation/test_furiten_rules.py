from riichienv import Action, ActionType, GameType, Phase, RiichiEnv


def test_furiten_ron():
    """
    Test that a player cannot Ron on a tile they have previously discarded (Furiten),
    even if the waiting tile matches.
    """
    env = RiichiEnv(game_type=GameType.YON_HANCHAN)
    env.reset()

    # Manually setup state for Player 0
    # Hand: 13m 22p 567s 888p ... waiting on 2m (Ryanmen 1-4m or Kanchan 2m?
    # Let's make it simple:
    # Hand: 1, 3m (Kanchan 2m), 555s, 666p, 777z, 11z.
    # Wait is 2m.
    # 34-tile IDs: 1m=0, 2m=1, 3m=2.
    # 136 IDs:
    # 1m: 0
    # 3m: 8
    # 5s: 36,37,38
    # 6p: 20,21,22
    # ...

    # Cleaner approach: Use a standard wait like 23m (Ryanmen 1m, 4m).
    # Hand: 2m, 3m, 55s, 666p, 777z, 11z. (11 tiles) + 3 more? No, needs 13 tiles.
    # Hand: 2m, 3m, 55s, 666p, 888s, 11z, 11z. (13 tiles).
    # Waits: 1m (0), 4m (3).

    # Player 0 discards 1m previously.
    # Player 1 discards 4m.
    # Player 0 cannot Ron on 4m due to Furiten (1m is in discards and is a wait).

    player_id = 0

    # 2m=4, 3m=8
    # 5s=52,53 (pair)
    # 6p=60,61,62 (triplet)
    # 8s=92,93,94 (triplet)
    # 1z=108,109,110 (triplet)
    hand_tiles = [4, 8, 52, 53, 60, 61, 62, 92, 93, 94, 108, 109, 110]  # 23m + pair + 3 trip

    # Manually set hand for Player 0
    # NOTE: env.hands returns a copy, so we must assign back.
    hands = env.hands
    hands[player_id] = hand_tiles
    env.hands = hands

    # Set Discards: Player 0 discarded 1m (0)
    discards = env.discards
    discards[player_id].append(0)  # 1m
    env.discards = discards

    # Trigger check: Player 1 discards 4m (12)
    # This tile completes the 23m sequence (1 or 4 wait).
    discard_tile = 12  # 4m
    target_player = 1

    # Mocking the action flow is hard without calling step.
    # We can invoke internal `_get_legal_actions_internal` or `_update_claims`.
    # `_update_claims` populates `current_claims`.

    # However, `RiichiEnv` logic for claims happens inside `step`.
    # Manually set hand for Player 0

    # Force P1 turn and state
    env.current_player = target_player
    env.phase = Phase.WaitAct

    # We must ensure P1 can discard.
    # Just setting phase/current_player might be enough for step() to accept the action.
    # Note: step() validates action against legal_actions usually?
    # But usually checks if legal. If we force it, it might work or we might need to update legal actions explicitly?
    # RiichiEnv calculates legal actions on step entry or lazily?
    # step() -> _step_wait_act -> ... checks action validity.
    # We need to simulate that P1 has drawn a tile or has 14 tiles.
    # P1 Hand length: 13.
    # Add discard_tile to P1 hand to make 14 (so they can discard it).
    hands = env.hands
    hands[target_player].append(discard_tile)
    env.hands = hands

    # Perform discard step
    actions = {target_player: Action(ActionType.Discard, tile=discard_tile)}
    obs_dict = env.step(actions)

    # Should now be in WaitResponse phase (if Ron/Pon/Chi is possible)
    # Check if P0 has observation.
    # Since P0 is Furiten, they should NOT be able to Ron.
    # They also cannot Chi from P1 (Shimoccha/Toimen relation? 1->0 is not Kamicha).
    # And no Pon/Kan on matching tiles since they don't have pair/trip of 4m.
    # So P0 should have NO legal actions and thus NOT be in obs_dict.

    assert 0 not in obs_dict, "Player 0 should not be active (Furiten -> No Ron, and no other calls)"


def test_ron_allowed_without_furiten():
    # Counter-test for Furiten test
    env = RiichiEnv(game_type=GameType.YON_HANCHAN)
    env.reset()

    player_id = 0
    # Same hand: 23m + pair + 3 trip. Waiting 1-4m.
    hand_tiles = [4, 8, 52, 53, 60, 61, 62, 92, 93, 94, 108, 109, 110]
    hands = env.hands
    hands[player_id] = hand_tiles
    env.hands = hands

    # NO Discards for P0.
    discards = env.discards
    discards[player_id] = []
    env.discards = discards

    discard_tile = 12  # 4m
    target_player = 1

    env.current_player = target_player
    env.phase = Phase.WaitAct

    hands = env.hands
    hands[target_player].append(discard_tile)
    env.hands = hands

    actions = {target_player: Action(ActionType.Discard, tile=discard_tile)}
    obs_dict = env.step(actions)

    # P0 should now be able to Ron.
    assert 0 in obs_dict, "Player 0 should be active (No Furiten -> Ron allowed)"
    obs0 = obs_dict[0]
    ron_moves = [a for a in obs0.legal_actions() if a.action_type == ActionType.Ron]
    assert len(ron_moves) > 0, "Ron should be allowed when not in Furiten"
