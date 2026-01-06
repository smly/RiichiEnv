import pytest

from riichienv import Action, ActionType, Meld, MeldType

from .env.helper import helper_setup_env


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_sufuurenta():
    # Wind tiles (East) for 4 players: 108, 109, 110, 111
    tiles = [108, 109, 110, 111]
    scattered = [0, 4, 8, 36, 40, 44, 72, 76, 80, 112, 116, 120, 124]

    env = helper_setup_env(
        hands=[scattered[:] for _ in range(4)],
        wall=list(range(136)),
    )

    for i in range(4):
        p = env.current_player
        # Put the wind tile in hand and discard it
        h = env.hands
        h[p][0] = tiles[i]
        env.hands = h
        env.drawn_tile = tiles[i]
        env.step({p: Action(ActionType.Discard, tile=tiles[i])})
        if i < 3:
            assert not env.done()

    assert env.done()
    assert any(e["reason"] == "sufuurenta" for e in env.mjai_log if e["type"] == "ryukyoku")


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_suurechi():
    scattered = [0, 4, 8, 36, 40, 44, 72, 76, 80, 112, 116, 120, 124]

    env = helper_setup_env(
        hands=[scattered[:] for _ in range(4)],
        riichi_declared=[True, True, False, False],
        current_player=1,
        drawn_tile=100,
        wall=list(range(136)),
    )

    p = env.current_player
    env.step({p: Action(ActionType.Discard, tile=100)})

    assert env.done()
    assert any(e["reason"] == "suurechi" for e in env.mjai_log if e["type"] == "ryukyoku")


@pytest.mark.skip(reason="Legacy python test - Rust implementation pending or parity missing")
def test_suukansansen():
    scattered = [0, 4, 8, 36, 40, 44, 72, 76, 80, 112, 116, 120, 124]

    env = helper_setup_env(
        hands=[scattered[:] for _ in range(4)],
        melds=[
            [Meld(MeldType.Angang, [0, 1, 2, 3], False), Meld(MeldType.Angang, [4, 5, 6, 7], False)],
            [Meld(MeldType.Angang, [8, 9, 10, 11], False), Meld(MeldType.Angang, [12, 13, 14, 15], False)],
            [],
            [],
        ],
        current_player=1,
        drawn_tile=108,
        wall=list(range(136)),
    )

    p = env.current_player
    env.step({p: Action(ActionType.Discard, tile=108)})

    assert env.done()
    assert any(e["reason"] == "suukansansen" for e in env.mjai_log if e["type"] == "ryukyoku")
