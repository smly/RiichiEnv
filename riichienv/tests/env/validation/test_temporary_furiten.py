from riichienv import Action, ActionType, Phase, RiichiEnv


def setup_env_with_wall():
    env = RiichiEnv(seed=42)
    # Initialize wall with enough tiles
    env.wall = list(range(136))
    return env


def test_temporary_furiten_chankan():
    env = setup_env_with_wall()
    env.reset()

    # h3 simple Tanyao.
    # 234m: 4, 8, 12
    # 234p: 40, 44, 48
    # 234s: 76, 80, 84
    # 66s: 92, 93 (Pair)
    # 67s: 94, 96 (Wait 5s(88) or 8s(100))
    h3 = [4, 8, 12, 40, 44, 48, 76, 80, 84, 92, 93, 94, 96]

    # Correctly update Rust hands
    hands = env.hands
    hands[3] = h3
    env.hands = hands

    # Setup P1 to discard 5s (88).
    env.current_player = 1
    env.active_players = [1]
    env.phase = Phase.WaitAct
    discard_tile_5s = 88

    obs_dict = env.step({1: Action(ActionType.Discard, discard_tile_5s, [])})

    # P3 should be offered Ron.
    assert 3 in obs_dict
    acts = [a for a in obs_dict[3].legal_actions() if a.action_type == ActionType.Ron]
    assert len(acts) > 0, "P3 should be offered Ron on 5s originally"

    # P3 PASSES.
    env.step({3: Action(ActionType.Pass, None, [])})

    # Check if missed_agari_doujun is set for P3.
    assert env.missed_agari_doujun[3], "P3 should be in temporary furiten after passing Ron"

    # P2 Discards 5s (88) again.
    # P3 should NOT be offered Ron due to missed_agari_doujun.
    env.current_player = 2
    env.active_players = [2]
    env.phase = Phase.WaitAct

    obs_dict_2 = env.step({2: Action(ActionType.Discard, discard_tile_5s, [])})

    if 3 in obs_dict_2:
        acts_2 = [a for a in obs_dict_2[3].legal_actions() if a.action_type == ActionType.Ron]
        assert len(acts_2) == 0, "P3 should NOT be offered Ron due to Temporary Furiten"
    else:
        # If P3 not in obs, it means no actions offered (correct).
        pass
