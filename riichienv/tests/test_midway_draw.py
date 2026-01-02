import pytest
from riichienv.env import RiichiEnv, Action, ActionType, Phase
from riichienv._riichienv import Meld, MeldType


def test_sufuurenta():
    env = RiichiEnv()
    env.reset()

    # Wind tiles (East) for 4 players
    # TIDs for East: 108, 109, 110, 111
    tiles = [108, 109, 110, 111]

    # Set hands to scattered tiles to prevent any claims
    # Using 1, 4, 7 for each suit.
    scattered = [0, 4, 8, 36, 40, 44, 72, 76, 80, 112, 116, 120, 124]

    for i in range(4):
        p = env.current_player
        for j in range(4):
            env.hands[j] = scattered[:]
        # Put the wind tile in hand and discard it
        env.hands[p][0] = tiles[i]
        env.drawn_tile = tiles[i]
        env.step({p: Action(ActionType.DISCARD, tile=tiles[i])})
        if i < 3:
            assert not env.is_done

    assert env.is_done
    assert any(e["reason"] == "sufuurenta" for e in env.mjai_log if e["type"] == "ryukyoku")


def test_suurechi():
    env = RiichiEnv()
    env.reset()

    # Ensure no Ron potential
    scattered = [0, 4, 8, 36, 40, 44, 72, 76, 80, 112, 116, 120, 124]
    for i in range(4):
        env.hands[i] = scattered[:]

    for i in range(4):
        p = env.current_player
        env.riichi_declared[p] = True
        env.current_player = (env.current_player + 1) % 4

    p = env.current_player
    env.riichi_declared[p] = True

    env.hands[p] = scattered[:]
    env.drawn_tile = 100
    env.step({p: Action(ActionType.DISCARD, tile=100)})

    assert env.is_done
    assert any(e["reason"] == "suurechi" for e in env.mjai_log if e["type"] == "ryukyoku")


def test_suukansansen():
    env = RiichiEnv()
    env.reset()

    # Ensure no Ron potential
    scattered = [0, 4, 8, 36, 40, 44, 72, 76, 80, 112, 116, 120, 124]
    for i in range(4):
        env.hands[i] = scattered[:]

    env.melds[0] = [Meld(MeldType.Angang, [0, 1, 2, 3], False), Meld(MeldType.Angang, [4, 5, 6, 7], False)]
    env.melds[1] = [Meld(MeldType.Angang, [8, 9, 10, 11], False), Meld(MeldType.Angang, [12, 13, 14, 15], False)]

    p = env.current_player
    env.hands[p] = scattered[:]
    env.drawn_tile = 108
    env.step({p: Action(ActionType.DISCARD, tile=108)})

    assert env.is_done
    assert any(e["reason"] == "suukansansen" for e in env.mjai_log if e["type"] == "ryukyoku")
