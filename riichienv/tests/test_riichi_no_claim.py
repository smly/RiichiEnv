from riichienv.action import ActionType
from riichienv.env import Phase, RiichiEnv


def test_riichi_no_claim():
    env = RiichiEnv(seed=42)
    env.reset()

    # P0 in Riichi
    pid = 0
    env.riichi_declared[pid] = True

    # Give P0 a pair of 1m (tid 0, 1) to enable potential Pon
    # And 2m, 3m for potential Chi? (Chi only from left player)
    env.hands[0] = [0, 1, 4, 5, 8] + [20] * 8  # 1m, 1m, 2m, 2m, 3m...

    # P3 discards 1m (tid 2) -> P0 could Pon
    # P0 is next to P3? No, P3->P0. So P0 can also Chi from P3.
    # P3 discards 3m (tid 11) -> P0 could Chi (1m, 2m + 3m)

    env.hands[3] = [0] * 12 + [2]  # P3 has 1m (tid 2)
    env.current_player = 3

    # Execute discard 1m from P3
    # P0 has pair of 1m. Normal P0 could Pon.
    # But P0 is Riichi. Should not offer Pon.

    # Ensure no Ron (P0 not Tenpai or 1m not Agari)

    # Mock _get_ron_potential to return empty just to be safe strictly about Claims?
    # Or just rely on hands not being Tenpai.

    obs = env._perform_discard(2)

    # Check legal actions for P0
    # If P0 is active:
    # - If WAIT_RESPONSE: Only RON allowed.
    # - If WAIT_ACT: Means claim was skipped. Success.

    if 0 in obs:
        if env.phase == Phase.WAIT_RESPONSE:
            actions = obs[0].legal_actions()
            types = [a.type for a in actions]
            assert ActionType.PON not in types
            assert ActionType.CHI not in types
            assert ActionType.DAIMINKAN not in types
            # Must strictly contain only RON and PASS (pass on ron)
            for t in types:
                assert t in [ActionType.RON, ActionType.PASS]
        elif env.phase == Phase.WAIT_ACT:
            # Success: Turned into P0's turn implies PON/CHI opportunity was skipped
            assert env.current_player == 0

    # Test Chi
    # P3 discards 3m (tid 11). P0 has 1m(0), 2m(4). -> Chi possible.
    env.current_player = 3
    env.hands[3] = [0] * 12 + [11]
    obs = env._perform_discard(11)

    if 0 in obs:
        if env.phase == Phase.WAIT_RESPONSE:
            actions = obs[0].legal_actions()
            types = [a.type for a in actions]
            assert ActionType.CHI not in types
        elif env.phase == Phase.WAIT_ACT:
            assert env.current_player == 0


def test_pon_available_without_riichi():
    env = RiichiEnv(seed=42)
    env.reset()
    pid = 0
    # No Riichi
    env.hands[0] = [0, 1, 4, 5, 8] + [20] * 8
    env.hands[3] = [0] * 12 + [2]  # 1m
    env.current_player = 3

    obs = env._perform_discard(2)

    # Should get WAIT_RESPONSE with PON
    assert env.phase == Phase.WAIT_RESPONSE
    assert 0 in obs
    types = [a.type for a in obs[0].legal_actions()]
    assert ActionType.PON in types
