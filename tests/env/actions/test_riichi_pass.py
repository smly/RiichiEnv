from riichienv import Action, ActionType, Phase, RiichiEnv


def test_riichi_pass_action():
    env = RiichiEnv(seed=42)
    env.reset()

    # Manually setup Riichi state for player 0
    pid = 0
    rd = env.riichi_declared
    rd[pid] = True
    env.riichi_declared = rd
    env.current_player = pid
    env.phase = Phase.WaitAct

    # Give a drawn tile
    env.drawn_tile = 10  # Some tile ID
    # Ensure this tile is NOT in hand (it's drawn)
    # And hand has 13 tiles
    env.hands[pid] = list(range(13))  # Dummy hand

    # Check legal actions
    obs = env.get_observations([pid])[pid]
    actions = obs.legal_actions()

    types = [a.action_type for a in actions]
    assert ActionType.DISCARD in types, "DISCARD (tsumogiri) should be available in Riichi"
    for a in actions:
        if a.action_type == ActionType.DISCARD:
            # Must be tsumogiri
            pass  # We don't check tile equality here since env.drawn_tile is set manually.
            # But we can check if it aligns with what we expect.
            # In Rust, if drawn_tile is set, it allows discarding it.
            pass
    assert ActionType.PASS not in types, "PASS should NOT be available in Riichi (WaitAct)"

    # Execute DISCARD (Tsumogiri) which is the valid move
    # env.drawn_tile is 10.
    act = Action(ActionType.Discard, 10)
    obs = env.step({pid: act})

    # Verify discard happened
    assert env.last_discard is not None
    # Rust last_discard is tuple (seat, tile)
    assert env.last_discard == (pid, 10)
    # Drawn tile is cleared
    assert env.drawn_tile is None or env.current_player != pid
    # Turn advanced (unless mid-game logic keeps it, but here it should advance)
    # Check mjai log for dahai tsumogiri
    env.mjai_log[-2]  # -1 might be tsumo for next player
    # Actually wait, step appends dahai, then checks claims, then maybe tsumo for next.
    # Find the dahai event
    dahai_ev = next((e for e in reversed(env.mjai_log) if e["type"] == "dahai" and e["actor"] == pid), None)
    assert dahai_ev is not None
    assert dahai_ev["tsumogiri"] is True
