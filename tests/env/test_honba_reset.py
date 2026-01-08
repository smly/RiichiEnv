from riichienv import Action, ActionType, Phase, RiichiEnv


def test_honba_reset_on_ko_win():
    # Initialize env
    env = RiichiEnv(game_mode="4p-red-half", seed=42)
    env.reset()

    # Set initial state: Oya is Player 0, Honba is 5
    env.set_state(oya=0, honba=5)
    # Force a win for Player 1 (Ko/Non-dealer)
    # We can use step() with a Tsumo or Ron, but easier to mimic the end_kyoku logic or induce a win.

    # Let's set up a Tsumo win for Player 1.
    env.current_player = 1
    # Give Player 1 a winning hand (Tenpai + drawn tile)
    # Simple Tenpai: 1s1s 2s2s 3s3s 4s4s 5s5s 6s6s 7s
    # Draw 7s
    h1 = [0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24]  # 1s-7s pairs
    hands = env.hands
    hands[1] = h1
    env.hands = hands
    env.drawn_tile = 24  # 7s
    env.needs_tsumo = False  # We already "drew"
    env.phase = Phase.WaitAct

    # Action Tsumo
    env.step({1: Action(ActionType.Tsumo, None, [])})

    # Now env should have proceeded to next round via _end_kyoku -> _initialize_next_round
    # Since Player 1 (Ko) won, Honba should be 0.
    # New Oya should be 1.

    # Need to call step() again to trigger _initialize_next_round?
    # reset/step usually handles it. _end_kyoku sets needs_initialize_next_round=True.
    # The NEXT step() call triggers initialization.

    print(f"DEBUG: Honba before next step: {env.honba}")

    # Step with empty actions to trigger initialization
    env.step({})

    print(f"DEBUG: Honba after win: {env.honba}")

    assert env.honba == 0, f"Honba should reset to 0 after Ko win, but got {env.honba}"
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

    env.step({0: Action(ActionType.Tsumo)})

    # Trigger initialization
    env.step({})

    print(f"DEBUG: Honba after Oya win: {env.honba}")

    assert env.honba == 6, f"Honba should increment to 6 after Oya win, but got {env.honba}"
    assert env.oya == 0, f"Oya should remain 0, got {env.oya}"


if __name__ == "__main__":
    test_honba_reset_on_ko_win()
    test_honba_increment_on_oya_win()
