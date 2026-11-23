"""Observation.drawn_tile belongs to the player who drew it, never to an opponent."""

import pytest

from riichienv import Action, ActionType, Meld, MeldType, Observation, Observation3P, Phase, RiichiEnv

# Seat 0 has pon of 4p, keeps the fourth 4p and has just drawn haku (124).
DRAWER = [0, 32, 51, 60, 64, 68, 73, 89, 108, 112, 124]
# 23p55p123456789s waits on 1p/4p, so seat 1 may rob an added kan of 4p.
ROBBER = [40, 44, 53, 54, 72, 76, 80, 84, 88, 92, 96, 100, 104]
BYSTANDERS = [
    [2, 33, 37, 42, 46, 56, 62, 66, 69, 74, 78, 82, 86],
    [3, 34, 38, 43, 47, 58, 63, 67, 70, 75, 79, 83, 87],
]


def drawn_tiles(env):
    observations = env.get_observations()
    return [observations[seat].drawn_tile for seat in sorted(observations)]


@pytest.mark.parametrize("players", [3, 4])
def test_only_the_dealer_sees_its_first_draw(players):
    env = RiichiEnv(f"{players}p-red-single", seed=42)
    observations = env.reset()
    assert observations[0].drawn_tile == env.drawn_tile is not None
    assert drawn_tiles(env) == [env.drawn_tile] + [None] * (players - 1)


@pytest.mark.parametrize("players", [3, 4])
def test_player_asked_to_rob_an_added_kan_does_not_see_the_draw(players):
    env = RiichiEnv(f"{players}p-red-single", seed=42)
    env.reset()
    env.hands = [DRAWER, ROBBER, *BYSTANDERS[: players - 2]]
    env.melds = [[Meld(MeldType.Pon, [48, 49, 50], True, 2, 50)]] + [[] for _ in range(players - 1)]
    env.drawn_tile = 124
    env.current_player = 0
    env.phase = Phase.WaitAct
    env.active_players = [0]

    kakan = next(a for a in env._get_legal_actions(0) if a.action_type == ActionType.KAKAN)
    observations = env.step({0: kakan})
    assert env.phase == Phase.WaitResponse
    assert list(observations) == [1]
    robber = observations[1]
    assert any(a.action_type == ActionType.RON and a.tile == 51 for a in robber.legal_actions())
    assert robber.drawn_tile is None
    cls = Observation3P if players == 3 else Observation
    assert cls.deserialize_from_base64(robber.serialize_to_base64()).drawn_tile is None
    # The kan maker still holds haku and still sees it.
    assert drawn_tiles(env) == [124] + [None] * (players - 1)

    observations = env.step({1: Action(ActionType.PASS)})
    assert list(observations) == [0]
    assert observations[0].drawn_tile == env.drawn_tile is not None
    assert drawn_tiles(env) == [env.drawn_tile] + [None] * (players - 1)
