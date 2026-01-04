from riichienv import RiichiEnv
from riichienv.action import Action, ActionType


class TestClaimPriority:
    def test_pon_priority_over_chi(self):
        env = RiichiEnv(seed=1)
        env.reset()

        # Setup scenario:
        # Player 0 discards a tile.
        # Player 1 (Next) wants to CHI.
        # Player 2 (Opposite/Other) wants to PON.
        # PON > CHI.

        # Tile to be discarded: 57 (4s?) - let's verify tile equivalence if needed,
        # but for claim logic ID matching is enough.

        # Player 1 (Chi-er) needs tiles to chi 57.
        # e.g. 57 is 4s (approx). Needs 5s, 6s or 2s, 3s.
        # Let's say we give P1: 62 (5s), 65 (6s)
        # 57, 62, 65 forms a sequence.

        # Player 2 (Ponner) needs pair of 57.
        # Give P2: 56, 58 (which are equivalent to 57 in value, e.g. 4s)
        # Actually 57 is likely 5 bamboos?
        # 0-35: Man
        # 36-71: Pin
        # 72-107: Sou
        # Wait, 57 is in the middle of 36-71 (Pin).
        # 36 + 4*9 = 72.
        # 57 is likely 5p or 6p?
        # 36 (1p), 40 (2p), 44 (3p), 48 (4p), 52 (5p), 56 (6p).
        # So 56, 57, 58, 59 are 6p.

        # Target tile 57 is 6p.

        # P1 wants Chi. Needs to make sequence with 6p.
        # If P1 has 7p (60..63) and 8p (64..67).
        # Let's use 62 (7p) and 65 (assuming 8p? Wait 60 is 7p. 64 is 8p.)
        # 56(6p), 60(7p), 64(8p).
        # Wrapper uses 136 format.

        # Let's trust the IDs from the script reference if they worked there.
        # Script had: Chi tile=57, consume=[62, 65].
        # Pon tile=57, consume=[56, 58].

        # Force Hands
        h = env.hands
        # P1 needs 62, 65.
        h[1] = [62, 65] + [0] * 11  # Filler
        h[1].sort()

        # P2 needs 56, 58.
        h[2] = [56, 58] + [1] * 11  # Filler
        h[2].sort()

        # P0 needs to discard 57.
        h[0] = [57] + [2] * 12
        env.hands = h

        # Set turn to P0
        env.current_player = 0
        env.phase = 0  # WaitAct
        env.active_players = [0]
        env.drawn_tile = 100  # Irrelevant

        # P0 Discards 57
        env.step({0: Action(ActionType.DISCARD, tile=57)})

        # Now should be WaitResponse
        assert env.phase == 1

        # Check active players.
        # P1 (Next) can Chi?
        # P2 can Pon?
        # Ideally P1, P2 should be active.
        # Note: Depending on P3's hand, P3 might be active too if we didn't clear it, but default reset random hands.
        # Let's check who is active.
        print("Active players:", env.active_players)

        # We expect P1 and P2 to be in active_players if they have legal actions.
        assert 1 in env.active_players
        assert 2 in env.active_players

        # Submit Actions
        # P1 Chi
        action_chi = Action(type=ActionType.CHI, tile=57, consume_tiles=[62, 65])
        # P2 Pon
        action_pon = Action(type=ActionType.PON, tile=57, consume_tiles=[56, 58])

        # If P3 is also active (maybe random hand has something), we need to handle it.
        # For this test, ensure P3 is NOT active or provide dummy pass.
        # Simplest: Force P3 hand to be empty or garbage that can't claim 6p.
        # 57 is 6p.
        # P3 hand: [130]*13 (West/North/Etc that doesn't match 6p)
        if 3 in env.active_players:
            # Just pass for P3 if active
            pass

        actions = {1: action_chi, 2: action_pon}

        # If P3 active, add PASS
        if 3 in env.active_players:
            actions[3] = Action(ActionType.PASS)

        # Step
        env.step(actions)

        # Expectation:
        # Pon wins.
        # Current player becomes P2 (Ponner).
        # Phase becomes WaitAct (P2 must discard).

        print(f"DEBUG: Before Step. Phase is now {env.phase}")
        assert env.phase == 0  # Phase.WaitAct is 0 in Rust
        assert 2 in env.active_players

        # Check Log
        last_ev = env.mjai_log[-1]
        assert last_ev["type"] == "pon"
