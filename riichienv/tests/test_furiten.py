from riichienv.action import Action, ActionType
from riichienv.env import RiichiEnv


def _clear_other_hands(env, actor_to_keep):
    # Use unique honors for each player to avoid any Pon/Chi
    # P0: East(108..), P1: South(112..), P2: West(116..), P3: North(120..)
    tiles = [108, 112, 116, 120]
    for i in range(4):
        if i != actor_to_keep:
            env.hands[i] = [tiles[i]] * 13


def test_furiten_discard():
    env = RiichiEnv(seed=42)
    env.reset()
    # Tanyao hand waiting on 8p (64).
    # 234m, 234m, 567s, 888s, 8p wait.
    env.hands[0] = [4, 8, 12, 5, 9, 13, 88, 92, 96, 100, 101, 102, 64]
    env.discards[0] = [65]  # 8p discarded
    _clear_other_hands(env, 0)

    env.current_player = 3
    ronners = env._get_ron_potential(66, is_chankan=False)
    assert 0 not in ronners


def test_furiten_temporary():
    env = RiichiEnv(seed=42)
    env.reset()
    # P1 has the wait
    env.hands[1] = [4, 8, 12, 5, 9, 13, 88, 92, 96, 100, 101, 102, 64]
    env.discards[1] = []
    _clear_other_hands(env, 1)

    # P3 discards 8p (65)
    env.current_player = 3
    env.hands[3] = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 65]
    obs = env._perform_discard(65)
    assert 1 in obs

    # P1 passes
    env.step({1: Action(ActionType.PASS)})
    # P1 should be in temporary furiten
    assert env.missed_agari_doujun[1] is True

    # After everyone passed P3 discard, it should advance to P0 draw.
    assert env.current_player == 0

    # Check if Ron potential for P1 is blocked when P0 discards 8p (66)
    ronners = env._get_ron_potential(66, is_chankan=False)
    assert 1 not in ronners

    # P0 draws (done in step above) and discards something else.
    env.hands[0] = [132, 133, 134, 135, 124, 125, 126, 127, 128, 129, 130, 131, 108]
    env._perform_discard(108)  # P0 discards -> Moves to P1
    assert env.current_player == 1
    assert env.missed_agari_doujun[1] is False


def test_furiten_riichi():
    env = RiichiEnv(seed=42)
    env.reset()
    env.hands[0] = [4, 8, 12, 5, 9, 13, 88, 92, 96, 100, 101, 102, 64]
    env.riichi_declared[0] = True
    _clear_other_hands(env, 0)
    env.hands[3] = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 65]

    env.current_player = 3
    env._perform_discard(65)
    env.step({0: Action(ActionType.PASS)})
    assert env.missed_agari_riichi[0] is True

    # Check if Ron potential for P0 is blocked
    env.current_player = 1
    assert 0 not in env._get_ron_potential(66, is_chankan=False)

    # After P3 pass, it advanced to P0 draw.
    # But P0 is in Riichi, so it auto-played and advanced to P1.
    assert env.current_player == 1

    # Check if permanent furiten still holds
    assert env.missed_agari_riichi[0] is True

    # Even after more turns, it should hold
    # Transition P3 -> P0 -> P1. (P0 in Riichi)
    env.current_player = 3
    env.hands[3] = [120] * 12 + [132]
    env.wall = [133] * 20
    env._perform_discard(132)  # P3 discards -> P0 draws 133 and auto-discards -> Moves to P1
    assert env.current_player == 1
    assert env.missed_agari_riichi[0] is True


def test_no_yaku_ron():
    env = RiichiEnv(seed=42)
    env.reset()
    # 123m, 456p, 111s, 222s, 2z(South). (No Ittsu, No Tanyao, No Yakuhai if P0=East)
    # 123m: 0, 4, 8. 456p: 48, 53, 56. 111s: 72, 73, 74. 222s: 76, 77, 78. 2z: 112.
    env.hands[0] = [0, 4, 8, 48, 53, 56, 72, 73, 74, 76, 77, 78, 112]
    _clear_other_hands(env, 0)

    env.current_player = 3
    env.hands[3] = [120] * 12 + [113]
    ronners = env._get_ron_potential(113, is_chankan=False)
    assert 0 not in ronners
