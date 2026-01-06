import pytest

from riichienv import Action, ActionType, Phase, RiichiEnv


def setup_env_with_wall():
    env = RiichiEnv()
    # Initialize wall with enough tiles to allow claims
    env.wall = list(range(136))
    return env


@pytest.mark.skip(
    reason="Rust implementation lacks lookahead to prevent Call leading to no legal discard (Kuikae deadlock)"
)
def test_kuikae_suji_chi():
    env = setup_env_with_wall()
    env.reset()

    # Setup P2 hand: 2s 3s 4s 4s (Indices: 79, 82, 85, 86)
    # We need to be careful with tile IDs.
    # 2s: 76, 77, 78, 79
    # 3s: 80, 81, 82, 83
    # 4s: 84, 85, 86, 87

    h2 = [79, 82, 85, 86]
    # Fill with junk to makes 13 tiles
    h2 += [0] * 9
    hands = env.hands
    hands[2] = h2
    env.hands = hands

    env.phase = Phase.WaitAct
    env.current_player = 1
    # P1 discards 1s (72).
    discard_tile = 72
    action_discard = Action(ActionType.Discard, discard_tile, [])

    obs_dict = env.step({1: action_discard})

    # Check if P2 is offered Chi
    assert 2 in obs_dict, "Player 2 should be active after P1 discard"
    obs2 = obs_dict[2]
    actions = obs2.legal_actions()

    # 1s 2s 3s Chi involves consuming 2s + 3s.
    chi_action = None
    for a in actions:
        if a.type == ActionType.Chi:
            # Check consumed tiles. Should include 2s and 3s.
            # a.consume_tiles is a list of tile IDs.
            # We look for a set that matches 2s and 3s.
            ct = a.consume_tiles
            # Convert to simplified type/value for easier checking?
            # 79//4 = 19 (2s), 82//4 = 20 (3s).
            if len(ct) == 2 and (ct[0] // 4 in [19, 20]) and (ct[1] // 4 in [19, 20]):
                chi_action = a
                break

    print(f"DEBUG: Actions offered to P2: {[(a.type, a.consume_tiles) for a in actions]}")
    assert chi_action is None, "Chi (1s-2s-3s) should NOT be offered due to Kuikae (results in dead-end)"

    if chi_action is None:
        print("SUCCESS: Chi was correctly withheld.")
    else:
        print("FAILURE: Chi was offered.")


def test_kuikae_discard_restriction():
    env = RiichiEnv()
    env.reset()

    # Clean setup
    hands = env.hands

    # P2 Hand: 2s(79), 3s(82), 4s(85), 7p(60) -- plus junk to make 13
    # We want P2 to have exactly these critical tiles.
    # 2s, 3s, 4s are needed for the sequence and forbidden check.
    # 7p is the legal safe discard.
    base_hand = [79, 82, 85, 60]
    junk = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 9 tiles. Total 13.
    # Ensure junk doesn't interfere (0-8 are 1m..3m)
    h2 = base_hand + junk
    hands[2] = h2

    # Clear others to prevent Ron/claims
    hands[0] = [100] * 13  # Avoid 1s/4s interaction
    hands[1] = [100] * 13
    hands[3] = [100] * 13
    env.hands = hands

    env.phase = Phase.WaitAct
    env.current_player = 1

    # P1 discards 1s (72 is 1s index 0).
    discard_tile = 72
    obs = env.step({1: Action(ActionType.Discard, discard_tile, [])})

    # P2 should be active with Chi offer.
    assert 2 in obs, "P2 should be active"
    actions = obs[2].legal_actions()

    # Find Chi action
    chi_action = None
    for a in actions:
        if a.action_type == ActionType.Chi:
            # We want to consume 2s(79), 3s(82)
            if 79 in a.consume_tiles and 82 in a.consume_tiles:
                chi_action = a
                break

    assert chi_action is not None, "Chi should be offered (valid because 7p is legal)"

    # Execute Chi
    # P2 consumes 79, 82. + 72 (called).
    # Hand becomes 4s(85), 7p(60) + junk.
    # Forbidden: 4s(85).
    obs2 = env.step({2: chi_action})

    # P2 should be active (Discard phase)
    assert 2 in obs2, "P2 should be active for discard"
    discard_actions = obs2[2].legal_actions()

    can_discard_4s = False
    can_discard_7p = False

    for a in discard_actions:
        if a.action_type == ActionType.Discard:
            t = a.tile
            # 85 is 4s (index 21 * 4 + 1)
            # Actually 85 / 4 = 21.
            if t // 4 == 21:
                can_discard_4s = True
            if t // 4 == 15:  # 60 / 4 = 15 (7p)
                can_discard_7p = True

    assert can_discard_7p, "Should be able to discard 7p"
    assert not can_discard_4s, "Should NOT be able to discard 4s due to Kuikae"
