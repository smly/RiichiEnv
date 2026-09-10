"""Reach daiminkan -> kakan from a complete wall through legal actions only."""

import pytest

from riichienv import Action, ActionType, GameRule, Phase, RiichiEnv


def assert_inventory(env):
    meld_tiles = [tile for melds in env.melds for meld in melds for tile in meld.tiles]
    tiles = env.wall + meld_tiles + [tile for hand in env.hands for tile in hand]
    tiles += [tile for discards in env.discards for tile in discards if tile not in meld_tiles]
    assert sorted(tiles) == list(range(136))


def reach_chankan(rule):
    # P1 starts one tile short of 234p234m23456s EE, a souzu wait.
    ready = [40, 44, 48, 4, 8, 12, 89, 92, 76, 80, 84, 108, 109]
    hands = [[], [t for t in ready if t != 48] + [132], [], []]
    reserved = set(hands[1] + [36, 37, 38, 39, 96, 97, 98, 99, 48, 0, 1, 2, 3, 120])
    pool = [t for t in range(136) if t not in reserved]
    hands[2] = [37, 38, 39, 97, 98] + pool[:8]
    del pool[:8]
    discard_after_pon = hands[2][-1]
    hands[0] = [96] + pool[:12]
    del pool[:12]
    hands[3] = pool[:13]
    del pool[:13]
    deal = [t for start in (0, 4, 8) for hand in hands for t in hand[start : start + 4]]
    deal.extend(hand[12] for hand in hands)
    wall = deal + [120, 0, 1, 48, 2, 3, 36] + pool + [99]
    assert sorted(wall) == list(range(136))
    env = RiichiEnv("4p-red-single", seed=42, rule=rule)
    env.reset(wall=wall, oya=0)

    def step(actions):
        observations = env.step(actions)
        assert not env.is_done
        assert observations
        assert_inventory(env)

    def discard(pid, tile):
        if env.phase == Phase.WaitResponse:
            step({p: Action(ActionType.PASS) for p in env.active_players})
        assert env.current_player == pid
        action = next(a for a in env._get_legal_actions(pid) if a.action_type == ActionType.DISCARD and a.tile == tile)
        step({pid: action})

    def call(pid, kind):
        action = next(a for a in env._get_legal_actions(pid) if a.action_type == kind)
        step({p: action if p == pid else Action(ActionType.PASS) for p in env.active_players})

    discard(0, 96)
    call(2, ActionType.PON)
    discard(2, discard_after_pon)
    discard(3, 0)
    discard(0, 1)
    discard(1, 132)  # P1 drew 4p and is now waiting on souzu.
    discard(2, 2)
    discard(3, 3)
    discard(0, 36)
    stale_chi = next(a for a in env._get_legal_actions(1) if a.action_type == ActionType.CHI)
    assert not any(a.action_type == ActionType.RON for a in env._get_legal_actions(1))
    call(2, ActionType.DAIMINKAN)
    assert env.drawn_tile == 99
    kakan = next(a for a in env._get_legal_actions(2) if a.action_type == ActionType.KAKAN and a.tile == 99)
    step({2: kakan})
    assert env.phase == Phase.WaitResponse
    assert env.active_players == [1]
    return env, stale_chi


@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
def test_chankan_offers_only_current_actions(preset):
    env, _ = reach_chankan(getattr(GameRule, f"default_{preset}")())
    actions = env.get_observation(1).legal_actions()
    assert {a.action_type for a in actions} == {ActionType.RON, ActionType.PASS}
    for action in actions:
        alternate = env.clone()
        alternate.step({1: action})
        assert_inventory(alternate)
        if action.action_type == ActionType.RON:
            assert alternate.is_done
            assert 1 in alternate.win_results
        else:
            assert not alternate.is_done
            assert alternate.phase == Phase.WaitAct
            assert alternate.current_player == 2
            assert alternate.rinshan_draw_count == 2
            assert alternate.missed_agari_doujun[1]


@pytest.mark.parametrize("preset", ["mjsoul", "tenhou"])
def test_stale_chi_is_rejected_without_duplicating_tiles(preset):
    env, stale_chi = reach_chankan(getattr(GameRule, f"default_{preset}")())
    env.step({1: stale_chi})
    assert_inventory(env)
    assert any(e.get("reason") == "Error: Illegal Action by Player 1" for e in env.mjai_log)
