from riichienv.action import Action, ActionType
from riichienv.env import Phase, RiichiEnv


def test_riichi_pass_action():
    env = RiichiEnv(seed=42)
    env.reset()

    # Manually setup Riichi state for player 0
    pid = 0
    env.riichi_declared[pid] = True
    env.current_player = pid
    env.phase = Phase.WAIT_ACT

    # Give a drawn tile
    env.drawn_tile = 10  # Some tile ID
    # Ensure this tile is NOT in hand (it's drawn)
    # And hand has 13 tiles
    env.hands[pid] = list(range(13))  # Dummy hand

    # Check legal actions
    obs = env.get_observations([pid])[pid]
    actions = obs.legal_actions()

    types = [a.type for a in actions]
    assert ActionType.DISCARD not in types, "DISCARD should not be available in Riichi"
    assert ActionType.PASS in types, "PASS should be available in Riichi"

    # Execute PASS
    pass_action = Action(ActionType.PASS)
    env.step({pid: pass_action})

    # Verify Tsumogiri happened
    # Last discard should be 10
    assert env.discards[pid][-1] == 10
    # Drawn tile is cleared
    assert env.drawn_tile is None or env.current_player != pid
    # Turn advanced (unless mid-game logic keeps it, but here it should advance)
    # Check mjai log for dahai tsumogiri
    last_log = env.mjai_log[-2]  # -1 might be tsumo for next player
    # Actually wait, step appends dahai, then checks claims, then maybe tsumo for next.
    # Find the dahai event
    dahai_ev = next((e for e in reversed(env.mjai_log) if e["type"] == "dahai" and e["actor"] == pid), None)
    assert dahai_ev is not None
    assert dahai_ev["tsumogiri"] is True
