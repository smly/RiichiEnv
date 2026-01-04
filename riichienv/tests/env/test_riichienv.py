from riichienv import RiichiEnv, Observation, Action, ActionType, Phase


class TestRiichiEnv:
    def test_initialization(self) -> None:
        env = RiichiEnv(seed=42)
        assert env.wall == [], "Wall should be empty at the start"
        obs_dict = env.reset()

        # NOTE: 136 total tiles - (13 tiles * 4 players + 1 initial draw) = 83.
        # The remainder consists of the draw pile (70 - 1 = 69 tiles) and the dead wall (14 tiles) = 83 tiles.
        # This is the state of env.wall immediately after reset().
        assert len(env.wall) == 83, "Wall should have 83 tiles after reset"

        assert len(env.hands) == 4, "Hands should have 4 players at the start"
        assert len(env.hands[0]) == 14, "Dealer's hands should have 14 tiles at the start"
        assert len(env.hands[1]) == 13, "Player 1's hands should have 13 tiles at the start"
        assert len(env.hands[2]) == 13, "Player 2's hands should have 13 tiles at the start"
        assert len(env.hands[3]) == 13, "Player 3's hands should have 13 tiles at the start"

        assert len(env.melds) == 4, "Melds should have 4 players at the start"
        assert env.melds[0] == [], "Dealer's melds should be empty at the start"

        assert len(env.discards) == 4, "Discards should have 4 players at the start"
        assert env.discards[0] == [], "Dealer's discards should be empty at the start"

        assert env.current_player == 0, "Current player should be 0 at the start"
        assert env.turn_count == 0, "Turn count should be 0 at the start"
        assert env.is_done is False, "is_done should be False at the start"
        assert env.needs_tsumo is False, "needs_tsumo should be False at the start"

        # First `obs_dict` should be the dealer's observation
        assert list(obs_dict.keys()) == [0]

        first_dealer_obs = obs_dict[0]
        assert isinstance(first_dealer_obs, Observation)
        assert first_dealer_obs.player_id == 0
        assert len(first_dealer_obs.hand) == 14, "Dealer's hand should have 14 tiles at the start"
        assert len(first_dealer_obs.events) == 3, "First events: start_game, start_kyoku, tsumo"
        assert first_dealer_obs.events[0]["type"] == "start_game"
        assert first_dealer_obs.events[1]["type"] == "start_kyoku"
        assert first_dealer_obs.events[2]["type"] == "tsumo"

        # Test `Observation.new_events()`
        assert len(first_dealer_obs.new_events()) == 3, "Same as events"

        # Test `Observation.legal_actions()`
        assert len(first_dealer_obs.legal_actions()) == 14

        # Test `Observation.to_dict()`
        obs_data = first_dealer_obs.to_dict()
        assert obs_data["legal_actions"][0]["type"] == 0
        assert obs_data["legal_actions"][0]["tile"] == 9
        assert obs_data["legal_actions"][0]["consume_tiles"] == []

        # Test `Observation.select_action_from_mjai()`
        assert first_dealer_obs.select_action_from_mjai({"type": "dahai", "pai": "1m"}) is None
        assert first_dealer_obs.select_action_from_mjai({"type": "dahai", "pai": "3m"}) is not None

    def test_basic_step_processing(self) -> None:
        # env.phase should be either Phase.WaitAct (player's turn action phase)
        # or Phase.WaitResponse (waiting for responses like Chi, Pon, or Ron).
        # Responses such as Chi, Pon, and Ron are handled in Phase.WaitResponse.
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()

        # Only the current player can act in the WaitAct phase
        assert env.phase == Phase.WaitAct
        assert env.current_player == 0
        assert list(obs_dict.keys()) == [0]
        obs = obs_dict[0]
        tile_to_discard = obs.hand[-1]
        obs_dict = env.step({0: Action(ActionType.Discard, tile=tile_to_discard)})

        # Players who can act are stored in `env.active_players`
        # Multiple players may be able to act in the `Phase.WaitResponse` phase
        # If no one can act, the `Phase.WaitResponse` phase is skipped
        while env.phase == Phase.WaitResponse:
            actions = {pid: Action(ActionType.Pass) for pid in env.active_players}
            obs_dict = env.step(actions)

        # WaitAct phase for the next player
        assert env.phase == Phase.WaitAct
        assert env.current_player == 1
        assert list(obs_dict.keys()) == [1]

        # Game end condition is checked via `env.done()`
        assert not env.done()
        assert len(env.hands[1]) == 14

        obs = obs_dict[1]
        assert len(obs.hand) == 14

        # TODO: `env`` is invisible to Agent, so should be passed in obs
        assert env.drawn_tile is not None
        assert obs.events[0]["type"] == "start_game"
        assert obs.events[1]["type"] == "start_kyoku"
        # Other players' hands are not visible
        assert obs.events[1]["tehais"][0][0] == "?"
        assert obs.events[1]["tehais"][1][0] != "?"
        assert obs.events[1]["tehais"][2][0] == "?"
        assert obs.events[1]["tehais"][3][0] == "?"
        assert obs.events[2]["type"] == "tsumo"
        assert obs.events[2]["actor"] == 0
        # Other players' tsumo tiles are not visible
        assert obs.events[2]["pai"] == "?"

        # mjai log is recorded in env
        assert len(env.mjai_log) == 5 # start_game, start_kyoku, tsumo, dahai, tsumo
        assert env.mjai_log[0]["type"] == "start_game"
        assert env.mjai_log[1]["type"] == "start_kyoku"
        assert env.mjai_log[2]["type"] == "tsumo"
        assert env.mjai_log[3]["type"] == "dahai"
        assert env.mjai_log[4]["type"] == "tsumo"
        # NOTE: All players' discards and hands are visible from `env`.
        assert env.mjai_log[2]["actor"] == 0
        assert env.mjai_log[2]["pai"] != "?"
        assert env.mjai_log[3]["actor"] == 0
        assert env.mjai_log[3]["pai"] != "?"
        assert env.mjai_log[4]["actor"] == 1
        assert env.mjai_log[4]["pai"] != "?"
