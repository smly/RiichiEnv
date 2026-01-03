from riichienv.action import Action, ActionType
from riichienv.env import Phase, RiichiEnv


def test_riichi_autoplay_after_pass():
    env = RiichiEnv(seed=42)
    env.reset()

    # P0 in Riichi
    pid = 0
    env.riichi_declared[pid] = True

    # Setup P1 with a Pon chance from P3
    # P3 discards 1s (tid 36). P1 has pair of 1s (tid 37, 38).
    env.hands[1] = [37, 38] + [20] * 11

    # Setup P0 hand to be garbage, no wins.
    env.hands[0] = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]  # wait, 37 is used by P1.
    env.hands[0] = [4, 5, 6, 12, 13, 14, 20, 21, 22, 28, 29, 30, 40]

    # P3 discards 1s (36)
    env.current_player = 3
    env.hands[3] = [0] * 12 + [36]
    env.drawn_tile = None

    env.wall = [101, 102, 103, 104] * 10

    # Step 1: P3 discards 36.
    obs = env._perform_discard(36)

    # Should be WaitResponse for P1 (Pon)
    assert env.phase == Phase.WaitResponse
    assert 1 in obs
    assert ActionType.PON in [a.type for a in obs[1].legal_actions()]

    # Step 2: P1 passes.
    # Logic should proceed to P0 Tsumo.
    # P0 in Riichi -> Auto-play.
    # Next active should be P1 draws (since P0 auto-played).

    pass_action = Action(ActionType.PASS)
    obs = env.step({1: pass_action})

    # Assert active player is [1] (P1's turn)
    # If P0 was active, key would be 0.
    assert list(obs.keys()) == [1]
    assert env.current_player == 1

    # Verify P0 events
    # Tsumo P0 -> Dahai P0 (True) -> Tsumo P1
    events = env.mjai_log
    assert events[-1]["type"] == "tsumo"
    assert events[-1]["actor"] == 1
    assert events[-2]["type"] == "dahai"
    assert events[-2]["actor"] == 0
    assert events[-2]["tsumogiri"] is True
    assert events[-3]["type"] == "tsumo"
    assert events[-3]["actor"] == 0
