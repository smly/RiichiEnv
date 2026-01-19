import pytest

from riichienv import Action, ActionType, Phase

from ..helper import helper_setup_env


@pytest.mark.skip(
    reason="Rust implementation lacks lookahead to prevent Call leading to no legal discard (Kuikae deadlock)"
)
def test_kuikae_suji_chi():
    h = [[0] * 13 for _ in range(4)]
    # Setup P2 hand: 2s 3s 4s 4s (Indices: 79, 82, 85, 86)
    # Junk: 0*9 tiles = 13.
    h[2] = [79, 82, 85, 86] + [0] * 9

    env = helper_setup_env(
        hands=h,
        current_player=1,
        phase=Phase.WaitAct,
        needs_tsumo=False,
        wall=list(range(136)),
    )

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
        if a.action_type == ActionType.Chi:
            ct = a.consume_tiles
            if len(ct) == 2 and (ct[0] // 4 in [19, 20]) and (ct[1] // 4 in [19, 20]):
                chi_action = a
                break

    print(f"DEBUG: Actions offered to P2: {[(a.action_type, a.consume_tiles) for a in actions]}")
    assert chi_action is None, "Chi (1s-2s-3s) should NOT be offered due to Kuikae (results in dead-end)"


def test_kuikae_discard_restriction():
    h = [list(range(13)) for _ in range(4)]
    # P1 discards 72, ensure P1 has 72.
    h[1] = [72] + list(range(12))  # Ensure P1 has 72

    # P2 Hand: 2s(79), 3s(82), 4s(85), 7p(60) -- plus junk to make 13
    base_hand = [79, 82, 85, 60]
    junk = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 9 tiles. Total 13.
    h[2] = base_hand + junk

    env = helper_setup_env(
        hands=h,
        current_player=1,
        phase=Phase.WaitAct,
        needs_tsumo=False,
        wall=list(range(136)),
    )

    # P1 discards 1s (72).
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
    obs2 = env.step({2: chi_action})

    # P2 should be active (Discard phase)
    assert 2 in obs2, "P2 should be active for discard"
    discard_actions = obs2[2].legal_actions()

    can_discard_4s = False
    can_discard_7p = False

    for a in discard_actions:
        if a.action_type == ActionType.Discard:
            t = a.tile
            if t // 4 == 21:  # 4s
                can_discard_4s = True
            if t // 4 == 15:  # 7p
                can_discard_7p = True

    assert can_discard_7p, "Should be able to discard 7p"
    assert not can_discard_4s, "Should NOT be able to discard 4s due to Kuikae"
