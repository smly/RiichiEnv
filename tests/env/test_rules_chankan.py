from riichienv import Action, ActionType, GameRule, Meld, MeldType, Phase, RiichiEnv


def setup_kokushi_scenario(env: RiichiEnv):
    """
    Sets up a scenario where:
    - Player 0 has 4x 9p (ready to Ankan).
    - Player 1 has Kokushi Tenpai waiting on 9p.
    """
    # Build hands locally first
    hands = [[], [], [], []]

    # Player 0: 4x 9p (9p is tile 35 -> 35*4 = 140, 141, 142, 143. wait, 9p is 8 + 9 + 9 = 26? no.
    # 0-8: 1-9m
    # 9-17: 1-9p
    # 18-26: 1-9s
    # 27-33: z
    # 9p is index 17. 17*4 = 68. 68, 69, 70, 71.

    # Player 0 hand: 9p 9p 9p 9p ... (need 14 tiles effectively after draw)
    # We will force Player 0 to be current player and have drawn one of the 9p.
    p0_hand = [68, 69, 70]  # 3x 9p
    # Fill rest with random (0-9)
    for i in range(10):
        p0_hand.append(i)
    p0_hand.append(71)  # 4th 9p (drawn)
    hands[0] = sorted(p0_hand)

    # Player 1: Kokushi Tenpai waiting on 9p
    # 13 tiles, waiting for 9p.
    terminals = [0, 8, 9, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    p1_hand = terminals + [0]  # Double 1m
    p1_hand_136 = [t * 4 for t in p1_hand]
    hands[1] = sorted(p1_hand_136)

    # Assign complete structure back to env
    env.hands = hands

    env.current_player = 0
    env.drawn_tile = 71  # 4th 9p
    env.phase = Phase.WaitAct
    env.needs_tsumo = False  # Already "drawn"


def test_chankan_kokushi_tenhou_default():
    # Default (or explicit Tenhou) -> False
    rule = GameRule.default_tenhou()
    env = RiichiEnv(rule=rule)
    setup_kokushi_scenario(env)

    # Player 0 performs Ankan with 9p
    action = Action(ActionType.ANKAN, 71, [68, 69, 70, 71])

    env.step({0: action})

    # Expect: Ankan executed successfully.
    # Since checking is disabled, it should NOT enter WaitResponse for Chankan.
    # It should implicitly continue (rinshan draw).

    assert env.current_player == 0
    assert env.phase == Phase.WaitAct, f"Phase {env.phase} != WaitAct. Drawn: {env.drawn_tile}"
    assert env.drawn_tile is not None


def test_chankan_kokushi_mjsoul():
    # MJSoul -> True
    rule = GameRule.default_mjsoul()
    env = RiichiEnv(rule=rule)
    setup_kokushi_scenario(env)

    # Player 0 performs Ankan with 9p
    action = Action(ActionType.ANKAN, 71, [68, 69, 70, 71])

    obs_dict = env.step({0: action})

    # Expect: Chankan interruption.
    # Phase -> WaitResponse
    # Active Players -> [1] (Player 1 has Ron)

    assert env.phase == Phase.WaitResponse, f"Phase {env.phase} != WaitResponse (Players: {env.active_players})"
    assert 1 in env.active_players

    p1_obs = obs_dict[1]
    legal_actions = p1_obs.legal_actions()
    ron_actions = [a for a in legal_actions if a.action_type == ActionType.RON]
    assert len(ron_actions) > 0
    assert ron_actions[0].tile == 71  # Ron on the Ankan tile


def setup_kakan_scenario(env: RiichiEnv):
    """
    Sets up a scenario where:
    - Player 0 has a Pon of 9p (open) and draws the 4th 9p.
    - Player 1 has a standard Tenpai waiting on 9p (e.g. 8p 8p pair wait? or 78p).
    """
    # Build hands locally first
    hands = [[], [], [], []]
    melds = [[], [], [], []]

    # Player 0: Has Pon of 9p (3 tiles)
    # 9p = 71 (we used 68, 69, 70 in Pon)
    # Tiles in hand: random, + drawn 71
    # Pon: 68, 69, 70

    # P0 Hand: 10 tiles (random) + 1 drawn
    p0_hand = []
    for i in range(10):
        p0_hand.append(i)
    p0_hand.append(71)  # 4th 9p
    hands[0] = sorted(p0_hand)

    # P0 Meld: Pon 9p
    m = Meld(MeldType.Peng, [68, 69, 70], True)
    melds[0] = [m]

    # Player 1: Tenpai checking 9p.
    p1_hand = []
    # 1m (0), 2m (4), 3m (8) -> Triplets (9 tiles)
    for t in [0, 4, 8]:
        p1_hand.extend([t * 4, t * 4 + 1, t * 4 + 2])

    # 4m (12) -> Pair (2 tiles)
    # Total so far: 11
    p1_hand.extend([12 * 4, 12 * 4 + 1])

    # 7p(60), 8p(64) -> 2 tiles
    # Total: 13.
    p1_hand.append(60)  # 7p
    p1_hand.append(64)  # 8p

    hands[1] = sorted(p1_hand)

    # Assign structure
    env.hands = hands
    env.melds = melds

    env.current_player = 0
    env.drawn_tile = 71
    env.phase = Phase.WaitAct
    env.needs_tsumo = False


def test_standard_chankan_kakan():
    # Use Tenhou rules (Kokushi Ankan Ron DISABLED)
    # Result: Standard Chankan should STILL WORK.
    rule = GameRule.default_tenhou()
    env = RiichiEnv(rule=rule)
    setup_kakan_scenario(env)

    # P0 performs Kakan (Added Kan) on 9p
    action = Action(ActionType.KAKAN, 71, [71])

    obs_dict = env.step({0: action})

    # Expect: WaitResponse (P1 can Ron)
    assert env.phase == Phase.WaitResponse, f"Should allow standard Chankan. Phase={env.phase}"
    assert 1 in env.active_players

    p1_obs = obs_dict[1]
    legal = p1_obs.legal_actions()
    ron = [a for a in legal if a.action_type == ActionType.RON]
    assert len(ron) > 0
    assert ron[0].tile == 71
