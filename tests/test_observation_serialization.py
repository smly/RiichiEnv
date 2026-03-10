import base64

import pytest

from riichienv import Action, ActionType, Observation, Phase, RiichiEnv


def _get_initial_obs(seed=42):
    """Helper: create env, reset, return the dealer's initial observation."""
    env = RiichiEnv(seed=seed)
    obs_dict = env.reset()
    return env, obs_dict[0]


def _assert_obs_fields_equal(original, restored):
    """Assert all serialised fields of two Observations match."""
    assert restored.player_id == original.player_id
    assert restored.hands == original.hands
    assert restored.discards == original.discards
    assert restored.dora_indicators == original.dora_indicators
    assert restored.scores == original.scores
    assert restored.riichi_declared == original.riichi_declared
    assert restored.events == original.events
    assert restored.honba == original.honba
    assert restored.riichi_sticks == original.riichi_sticks
    assert restored.round_wind == original.round_wind
    assert restored.oya == original.oya


def _assert_legal_actions_equal(original, restored):
    """Assert legal_actions match between original and restored observations."""
    orig_actions = original.legal_actions()
    rest_actions = restored.legal_actions()
    assert len(rest_actions) == len(orig_actions)
    for a, b in zip(orig_actions, rest_actions):
        assert a.action_type == b.action_type
        assert a.tile == b.tile
        assert a.consume_tiles == b.consume_tiles


class TestObservationSerialization:
    def test_round_trip_initial(self):
        """Serialize then deserialize an initial observation; fields must match."""
        _, obs = _get_initial_obs()
        encoded = obs.serialize_to_base64()
        restored = Observation.deserialize_from_base64(encoded)
        _assert_obs_fields_equal(obs, restored)

    def test_legal_actions_preserved_after_deserialize(self):
        """legal_actions should be preserved after round-trip."""
        _, obs = _get_initial_obs()
        assert len(obs.legal_actions()) > 0, "Original should have legal actions"
        restored = Observation.deserialize_from_base64(obs.serialize_to_base64())
        _assert_legal_actions_equal(obs, restored)

    def test_round_trip_after_actions(self):
        """Serialize after progressing the game a few turns."""
        env = RiichiEnv(seed=42)
        obs_dict = env.reset()

        for _ in range(4):
            if env.is_done:
                break
            # WaitAct phase: current player discards
            assert env.phase == Phase.WaitAct
            pid = env.current_player
            obs = obs_dict[pid]
            tile = obs.hand[-1]
            obs_dict = env.step({pid: Action(ActionType.Discard, tile=tile)})
            # Handle WaitResponse phase (pass all claims)
            while env.phase == Phase.WaitResponse:
                actions = {p: Action(ActionType.Pass) for p in env.active_players}
                obs_dict = env.step(actions)

        if not env.is_done:
            pid = env.current_player
            obs = env.get_observations([pid])[pid]
            encoded = obs.serialize_to_base64()
            restored = Observation.deserialize_from_base64(encoded)
            _assert_obs_fields_equal(obs, restored)
            _assert_legal_actions_equal(obs, restored)

    def test_round_trip_preserves_incremental_events_for_same_player(self):
        """Serialized observations should preserve per-player unseen-event deltas."""
        env = RiichiEnv(seed=9)
        obs_dict = env.reset()

        hands = env.hands
        hands[1] = []
        hands[2] = []
        hands[3] = []
        env.hands = hands

        dealer_obs = obs_dict[0]
        assert [event["type"] for event in dealer_obs.events] == [
            "start_game",
            "start_kyoku",
            "tsumo",
        ]

        discard = dealer_obs.hand[0]
        obs_dict = env.step({0: Action(ActionType.Discard, tile=discard)})

        for pid in [1, 2, 3]:
            obs = obs_dict[pid]
            discard = obs.hand[0]
            obs_dict = env.step({pid: Action(ActionType.Discard, tile=discard)})
            if env.phase == Phase.WaitResponse:
                obs_dict = env.step(
                    {player: Action(ActionType.Pass) for player in env.active_players}
                )

        dealer_obs_2 = obs_dict[0]
        assert [event["type"] for event in dealer_obs_2.events] == [
            "dahai",
            "tsumo",
            "dahai",
            "tsumo",
            "dahai",
            "tsumo",
            "dahai",
            "tsumo",
        ]

        restored = Observation.deserialize_from_base64(
            dealer_obs_2.serialize_to_base64()
        )
        assert [event["type"] for event in restored.events] == [
            "dahai",
            "tsumo",
            "dahai",
            "tsumo",
            "dahai",
            "tsumo",
            "dahai",
            "tsumo",
        ]
        assert restored.new_events() == dealer_obs_2.new_events()

    def test_invalid_base64_raises(self):
        """Invalid base64 input must raise ValueError."""
        with pytest.raises(ValueError):
            Observation.deserialize_from_base64("!!!not-base64!!!")

    def test_invalid_json_raises(self):
        """Valid base64 but invalid JSON must raise ValueError."""
        bad_json = base64.b64encode(b"not json").decode()
        with pytest.raises(ValueError):
            Observation.deserialize_from_base64(bad_json)

    @pytest.mark.parametrize("seed", [0, 1, 99, 12345, 999999])
    def test_round_trip_multiple_seeds(self, seed):
        """Round-trip works across different RNG seeds."""
        _, obs = _get_initial_obs(seed=seed)
        encoded = obs.serialize_to_base64()
        restored = Observation.deserialize_from_base64(encoded)
        _assert_obs_fields_equal(obs, restored)
        _assert_legal_actions_equal(obs, restored)
