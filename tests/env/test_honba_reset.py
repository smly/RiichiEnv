from riichienv import Action, ActionType, Phase, RiichiEnv


def test_honba_reset_on_ko_win():
    # Initialize env
    env = RiichiEnv(game_mode="4p-red-half", seed=42)
    # Start with Oya 0, Honba 5
    env.reset(oya=0, honba=5)

    # Force a win for Player 1 (Ko) via Ron
    # Player 0 discards 7m, Player 1 calls Ron
    # Setup Hands
    # Player 1: Chiitoitsu tenpai on 7m - 1m1m2m2m3m3m4m4m5m5m6m6m7m
    # Using correct tile IDs (each tile has unique ID)
    h1 = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24]
    # Player 0 has Tid 25 (7m) to discard, 14 tiles for Oya
    h0 = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 25]

    hands = env.hands
    hands[0] = h0
    hands[1] = h1
    env.hands = hands

    # Set high scores to prevent tobi (bankruptcy) from ending the game
    env.set_scores([60000, 25000, 25000, 25000])

    # 0 discards 25 (7m)
    env.current_player = 0
    env.needs_tsumo = False
    env.phase = Phase.WaitAct
    env.active_players = [0, 1, 2, 3]

    # Step where 0 discards
    env.step({0: Action(ActionType.DISCARD, 25, [])})

    # Now it should be WaitResponse phase, Player 1 can Ron
    assert env.phase == Phase.WaitResponse
    assert env.last_discard == (0, 25)  # (pid, tile)

    # Player 1 Rons
    env.step({1: Action(ActionType.RON, 25, [])})

    # Transition to end state
    env.step({})

    assert env.honba == 0, f"Honba should reset to 0 after Ko Ron win, but got {env.honba}"
    assert env.oya == 1, f"Oya should rotate to 1, got {env.oya}"


def test_honba_increment_on_oya_win():
    # Initialize env
    env = RiichiEnv(game_mode="4p-red-half", seed=42)
    env.reset()

    # Set initial state: Oya is Player 0, Honba is 5
    env.set_state(oya=0, honba=5)

    # Force a win for Player 0 (Oya)
    env.current_player = 0
    h0 = [0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24]
    hands = env.hands
    hands[0] = h0
    env.hands = hands
    env.drawn_tile = 24
    env.needs_tsumo = False
    env.phase = Phase.WaitAct

    env.step({0: Action(ActionType.TSUMO)})

    env.step({})

    assert env.honba == 6, f"Honba should increment to 6 after Oya win, but got {env.honba}"
    assert env.oya == 0, f"Oya should remain 0, got {env.oya}"
