from riichienv import ActionType, Phase, RiichiEnv


def test_valid_ankan_riichi_legality():
    env = RiichiEnv(seed=42)
    env.reset()

    # Setup a hand that is Tenpai and has 4 of a tile whose ankan doesn't change waits.
    # Hand: 1m 1m 1m 2m 3m 4m 5m 6m 7m 8m 9m 9m 9m (9-gate wait for Chuuren, but let's be simpler)
    # Simpler: 2m 2m 2m 3m 4m 5m 6m 7m 8m 1z 1z 1z 2z + Draw 2m
    # Waits for 2m 2m 3m 4m 5m 6m 7m 8m 1z 1z 1z 2z are 2m, 5m, 8m.
    # If we ankan 2m 2m 2m 2m, the hand becomes 3m 4m 5m 6m 7m 8m 1z 1z 1z 2z.
    # Waits for 3m 4m 5m 6m 7m 8m 1z 1z 1z 2z are still 2m, 5m, 8m?
    # No, it's (345)(678) 1z1z1z 2z wait 2z. Wait changed.

    # Better: 2m 3m 4m 5m 6m 7m 8m 9m 9m 9m 1z 1z 1z + Draw 9m
    # Hand: 2m 3m 4m 5m 6m 7m 8m (9m 9m 9m) 1z 1z 1z
    # Wait is 2m-5m-8m or 1m-4m-7m...
    # Let's use a very standard one:
    # 1m 1m 1m 2m 3m, 5z 5z, 6z 6z 6z, 7z 7z 7z
    # Draw 1m. Ankan 1m.
    # Before: (111) 23 (55) (666) (777) -> Wait 1m, 4m.
    # After: Kan(1111) 23 (55) (666) (777) -> Wait 1m, 4m.
    # Wait does not change!

    # Exact Match 2 Hand: [4, 5, 68, 69, 71, 73, 74, 75, 88, 89, 90, 97, 98, 70]
    # 2m: 4, 5
    # 9p: 68, 69, 71, 70 (dt=70)
    # 1s: 73, 74, 75
    # 5s: 88, 89, 90 (0s is 88? 0s, 5s, 5s)
    # 7s: 97, 98
    p2_hand = [4, 5, 68, 69, 71, 73, 74, 75, 88, 89, 90, 97, 98, 70]
    p2_hand.sort()

    h = env.hands
    h[2] = p2_hand
    env.hands = h

    # Phase: P2 Turn (Drawn tile 70)
    env.current_player = 2
    env.phase = Phase.WaitAct
    env.drawn_tile = 70
    rd = list(env.riichi_declared)
    rd[2] = True
    env.riichi_declared = rd

    # Check legal actions
    obs = env.get_observations()
    obs2 = obs[2]
    legals = obs2.legal_actions()

    ankan = [a for a in legals if a.action_type == ActionType.Ankan]
    print(f"Legals: {legals}")
    assert len(ankan) == 1, f"Ankan should be LEGAL as it does NOT change waits. Legals: {legals}"
