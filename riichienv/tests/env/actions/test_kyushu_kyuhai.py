from riichienv import Action, ActionType, Meld, MeldType, RiichiEnv


def test_kyushu_kyuhai():
    # Use YON_IKKYOKU (0) to ensure game ends after one kyoku
    env = RiichiEnv(game_type=0, mjai_mode=True)
    env.reset()

    # Setup player 0 with 9 unique terminals/honors
    # TIDs = tt * 4
    # 1m, 9m, 1p, 9p, 1s, 9s, E, S, W
    # Setup player 0 with 9 unique terminals/honors + 5 others = 14 tiles
    # TIDs = tt * 4
    # 1m, 9m, 1p, 9p, 1s, 9s, E, S, W
    terminal_tids = [0, 8 * 4, 9 * 4, 17 * 4, 18 * 4, 26 * 4, 27 * 4, 28 * 4, 29 * 4]

    # Other 5 tiles (4 in hand + 1 drawn)
    other_tids = [4, 5, 6, 7, 12]  # 2m, 2m, 2m, 2m, 4m

    # Hand should be 13 tiles, plus 1 drawn tile
    env.hands = [terminal_tids + other_tids[:4], [], [], []]
    env.drawn_tile = other_tids[4]

    env.current_player = 0
    env.needs_tsumo = False

    obs = env.get_observations([0])[0]
    legals = [a.action_type for a in obs.legal_actions()]

    assert ActionType.KYUSHU_KYUHAI in legals

    # Execute it
    env.step({0: Action(ActionType.KYUSHU_KYUHAI)})

    # Check if ryukyoku is logged
    found_ryukyoku = False
    for log in env.mjai_log:
        if log["type"] == "ryukyoku" and log.get("reason") == "kyushu_kyuhai":
            found_ryukyoku = True
            break

    assert found_ryukyoku
    assert env.needs_initialize_next_round or env.is_done


def test_kyushu_kyuhai_not_available_after_meld():
    env = RiichiEnv(mjai_mode=True)
    env.reset()

    terminal_tids = [0, 8 * 4, 9 * 4, 17 * 4, 18 * 4, 26 * 4, 27 * 4, 28 * 4, 29 * 4]
    other_tids = [4, 5, 6, 7, 12]
    env.hands = [terminal_tids + other_tids[:4], [], [], []]
    env.drawn_tile = other_tids[4]

    m = Meld(MeldType.Peng, [20, 21, 22], True)
    env.melds = [[], [m], [], []]

    env.current_player = 0
    env.needs_tsumo = False

    obs = env.get_observations([0])[0]
    legals = [a.action_type for a in obs.legal_actions()]

    # Should NOT be available because there's a meld
    assert ActionType.KYUSHU_KYUHAI not in legals
