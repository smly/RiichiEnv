from riichienv import Action, ActionType, Meld, MeldType, Phase, RiichiEnv


def test_ron_mismatch_after_call():
    """
    Reproduce Match 263 Step 781 inconsistency.
    P3 in temporary furiten, calls Pon/Chi, discards.
    Temporary furiten should be cleared, allowing Ron on next discard.
    """
    env = RiichiEnv(seed=42)
    env.reset()

    # Setup P3 to be tenpai on 7s (ID 92)
    # Hand: 1m, 1m, 1m, 7s
    # Melds: Chi(7m,8m,9m), Chi(7p,8p,9p), Pon(P,P,P)
    # 1m: 0, 1, 2
    # 7s: 92, 93, 94, 95. We use 92.
    # P: 136-139 (White Dragon)
    # 1m=0..3, 1p=36..39, 1s=72..75, E=108, S=112, W=116, N=120, P=124, F=128, C=132

    # P3 Hand: [0, 1, 2, 92]
    hands = env.hands
    hands[3] = [0, 1, 2, 92]

    # P3 Melds
    # White Pon (P) = 124, 125, 126
    # 789m = (6*4=24, 7*4=28, 8*4=32)
    # 789p = (15*4=60, 16*4=64, 17*4=68)
    melds = env.melds
    melds[3] = [
        Meld(MeldType.Pon, [124, 125, 126], True),
        Meld(MeldType.Chi, [24, 28, 32], True),
        Meld(MeldType.Chi, [60, 64, 68], True),
    ]
    env.melds = melds
    env.hands = hands

    # Manually set temporary furiten for P3
    mad = env.missed_agari_doujun
    mad[3] = True
    env.missed_agari_doujun = mad

    # Now P3 calls someone.
    # P1 discards 1p (36). P3 calls Pon.
    hands = env.hands
    hands[3] = [92, 37, 38, 40]  # 7s, 1p, 1p, 2p (4 tiles for 3 melds)
    # Ensure P1 has 36 (1p)
    hands[1] = [36] + hands[1][1:]
    env.hands = hands

    env.current_player = 1
    env.phase = Phase.WaitAct
    # P1 discards 1p (36).
    obs = env.step({1: Action(ActionType.Discard, 36)})

    # P3 calls Pon
    # (Consume 37, 38)
    env.step({3: Action(ActionType.Pon, 36, [37, 38])})

    # P3 discards dummy (2p - 40).
    obs = env.step({3: Action(ActionType.Discard, 40)})

    # Now P3 should NO LONGER be in temporary furiten.
    # P2 discards 7s (93).
    h = env.hands
    h[2] = [93] + h[2][1:]
    env.hands = h

    env.current_player = 2
    env.phase = Phase.WaitAct
    obs = env.step({2: Action(ActionType.Discard, 93)})

    # P3 should be offered Ron.
    assert 3 in obs, "P3 should be active for Ron"
    actions = obs[3].legal_actions()
    action_types = [a.action_type for a in actions]
    assert ActionType.Ron in action_types, f"P3 should have Ron offered. Actions: {action_types}"
