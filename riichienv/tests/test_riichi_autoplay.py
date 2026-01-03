from riichienv.env import RiichiEnv


def test_riichi_autoplay():
    env = RiichiEnv(seed=42)
    env.reset()

    # P0 in Riichi
    pid = 0
    env.riichi_declared[pid] = True

    # Previous player (P3) discards
    # P0 draws
    # If P0 has no Tsumo/Ankan/Kakan, it should auto-discard (PASS)
    # and P1 should become active

    # Force Env state
    env.current_player = 3
    env.discards[3].append(0)  # dummy
    # Mock P0 hand to be garbage (no wins)
    env.hands[0] = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

    # We call _perform_discard(tile) for P3's discard (e.g. tile 100)
    # This triggers P0 draw.
    # We ensure P0's drawn tile doesn't complete anything.
    # The wall needs to be set up so P0 draws a safe tile.

    # Let's mock _perform_discard to set up P0 state explicitly or call it.
    # Calling env._perform_discard(100) on P3's turn:
    # 1. P3 discards 100.
    # 2. Check Ron/Pon/Chi... assume none.
    # 3. Next player P0 draws.
    # 4. P0 in Riichi. Checks legal actions.
    # 5. If PASS is only action, recurses.
    # 6. P0 discards drawn tile.
    # 7. Next player P1 draws.
    # 8. P1 not in Riichi. Returns obs for P1.

    # We need to make sure 100 is not claimable by anyone.
    # Safest is to empty hands of others or give them non-matching tiles.
    env.hands[1] = [0] * 13
    env.hands[2] = [0] * 13
    # P3 needs 100 in hand
    env.hands[3] = [0] * 12 + [100]

    # P3 discards 100
    # P0 draws 101 (from wall)
    # P0 discards 101 (auto)
    # P1 draws 102
    # Active player should be [1]

    env.wall = [101, 102, 103, 104] * 10  # Enough tiles (40 > 14)

    # Needs to set tsumogiri or drawn_tile properly.
    # If drawn_tile=None, assumes manual discard from hand.
    env.drawn_tile = None

    obs = env._perform_discard(100)  # P3 discards 100

    # Check if P1 is active
    assert list(obs.keys()) == [1]
    assert env.current_player == 1

    # Verify P0 events in log
    # Should see Tsumo P0 -> Dahai P0 (Tsumogiri) -> Tsumo P1

    # Get relevant events descending
    events = env.mjai_log

    # Last event should be Tsumo P1
    assert events[-1]["type"] == "tsumo"
    assert events[-1]["actor"] == 1

    # Before that, Dahai P0
    assert events[-2]["type"] == "dahai"
    assert events[-2]["actor"] == 0
    assert events[-2]["tsumogiri"] is True

    # Before that, Tsumo P0
    assert events[-3]["type"] == "tsumo"
    assert events[-3]["actor"] == 0
