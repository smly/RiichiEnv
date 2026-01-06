from riichienv import Action, ActionType, Phase

from .helper import helper_setup_env


def test_no_chi_during_riichi():
    h = [[0] * 13 for _ in range(4)]
    h[2] = [76, 80, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    env = helper_setup_env(
        hands=h,
        riichi_declared=[False, False, True, False],
        current_player=1,
        drawn_tile=72,  # 1s
        phase=Phase.WaitAct,
        needs_tsumo=False,
        wall=list(range(136)),
    )

    action = Action(ActionType.Discard, 72, [])
    obs_dict = env.step({1: action})

    assert 2 in obs_dict, "Player 2 should be active (Draw/Tsumo) but with NO claims offered"
    obs2 = obs_dict[2]
    actions = obs2.legal_actions()

    chi_actions = [a for a in actions if a.action_type == ActionType.Chi]
    pon_actions = [a for a in actions if a.action_type == ActionType.Pon]

    assert len(chi_actions) == 0, f"Chi should NOT be offered during Riichi! Offered: {chi_actions}"
    assert len(pon_actions) == 0, f"Pon should NOT be offered during Riichi! Offered: {pon_actions}"


def test_chi_offered_when_not_in_riichi():
    h = [[0] * 13 for _ in range(4)]
    h[2] = [76, 80, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    env = helper_setup_env(
        hands=h,
        riichi_declared=[False, False, False, False],
        current_player=1,
        drawn_tile=72,  # 1s
        phase=Phase.WaitAct,
        needs_tsumo=False,
        wall=list(range(136)),
    )

    action = Action(ActionType.Discard, 72, [])
    obs_dict = env.step({1: action})

    assert 2 in obs_dict, f"Player 2 should be in active players when NOT in Riichi, but obs_dict was {obs_dict.keys()}"
    obs2 = obs_dict[2]
    actions = obs2.legal_actions()

    chi_actions = [a for a in actions if a.action_type == ActionType.Chi]
    assert len(chi_actions) > 0, "Player 2 should have Chi actions when NOT in Riichi"


def test_no_pon_during_riichi():
    h = [[0] * 13 for _ in range(4)]
    h[2] = [76, 77, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    env = helper_setup_env(
        hands=h,
        riichi_declared=[False, False, True, False],
        current_player=1,
        drawn_tile=78,  # 2s (76, 77, 78, 79 are all 2s)
        phase=Phase.WaitAct,
        needs_tsumo=False,
        wall=list(range(136)),
    )

    action = Action(ActionType.Discard, 78, [])
    obs_dict = env.step({1: action})

    assert 2 in obs_dict, "Player 2 should be active (Draw/Tsumo) but with NO claims offered"
    obs2 = obs_dict[2]
    actions = obs2.legal_actions()

    chi_actions = [a for a in actions if a.action_type == ActionType.Chi]
    pon_actions = [a for a in actions if a.action_type == ActionType.Pon]

    assert len(chi_actions) == 0, f"Chi should NOT be offered during Riichi! Offered: {chi_actions}"
    assert len(pon_actions) == 0, f"Pon should NOT be offered during Riichi! Offered: {pon_actions}"
