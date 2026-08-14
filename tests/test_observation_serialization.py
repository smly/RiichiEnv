import base64
import json
from pathlib import Path

import pytest

from riichienv import Action, ActionType, Observation, Observation3P, Phase, RiichiEnv

CONTRACT_DIR = Path(__file__).parent / "contracts" / "v0.4.8"


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
    assert restored.tsumogiri_flags == original.tsumogiri_flags
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
    @pytest.mark.parametrize(
        ("filename", "observation_type", "game_mode"),
        [
            ("observation_4p.b64", Observation, "4p-red-single"),
            ("observation_3p.b64", Observation3P, "3p-red-single"),
        ],
    )
    def test_v048_wire_fixture_remains_decodable_and_stable(self, filename, observation_type, game_mode):
        fixture = (CONTRACT_DIR / filename).read_text(encoding="ascii").strip()
        restored = observation_type.deserialize_from_base64(fixture)

        assert restored.player_id == 0
        assert restored.serialize_to_base64() == fixture

        current = RiichiEnv(game_mode=game_mode, seed=42).reset()[0]
        assert current.public_history_complete is True
        assert current.discard_actor_history == b""
        assert all(not flags for flags in current.tsumogiri_flags)
        assert all(not flags for flags in current.public_tsumogiri_history)
        assert current.public_temporary_safe_masks == [0] * len(current.hands)
        assert current.serialize_to_base64() == fixture

        # DREV-v2 history is a runtime sidecar and intentionally stays out of
        # the frozen Observation base64 wire.
        current_restored = observation_type.deserialize_from_base64(fixture)
        assert current_restored.public_history_complete is False
        assert current_restored.discard_actor_history == b""
        assert current_restored.public_temporary_safe_masks == [0] * len(current_restored.hands)
        with pytest.raises(ValueError, match="complete runtime public-history sidecar"):
            current_restored.encode_drev_v2()

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
            obs_dict = env.step({pid: Action(ActionType.DISCARD, tile=tile)})
            # Handle WaitResponse phase (pass all claims)
            while env.phase == Phase.WaitResponse:
                actions = {p: Action(ActionType.PASS) for p in env.active_players}
                obs_dict = env.step(actions)

        if not env.is_done:
            pid = env.current_player
            obs = env.get_observations([pid])[pid]
            assert obs.public_history_complete is True
            assert len(obs.discard_actor_history) == sum(map(len, obs.discards))
            assert obs.resolved_discard_count == len(obs.discard_actor_history)
            for seat in range(4):
                assert len(obs.public_tsumogiri_history[seat]) == len(obs.discards[seat])
                assert obs.tsumogiri_flags[seat] == []
                assert len(obs.discard_is_riichi[seat]) == len(obs.discards[seat])
                assert obs.discard_actor_history.count(seat) == len(obs.discards[seat])
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
        obs_dict = env.step({0: Action(ActionType.DISCARD, tile=discard)})

        for pid in [1, 2, 3]:
            obs = obs_dict[pid]
            discard = obs.hand[0]
            obs_dict = env.step({pid: Action(ActionType.DISCARD, tile=discard)})
            if env.phase == Phase.WaitResponse:
                obs_dict = env.step({player: Action(ActionType.PASS) for player in env.active_players})

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

        restored = Observation.deserialize_from_base64(dealer_obs_2.serialize_to_base64())
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

    @pytest.mark.parametrize(
        ("filename", "observation_type", "invalid_player_id"),
        [
            ("observation_4p.b64", Observation, 4),
            ("observation_3p.b64", Observation3P, 3),
        ],
    )
    def test_out_of_range_player_id_is_rejected(self, filename, observation_type, invalid_player_id):
        fixture = (CONTRACT_DIR / filename).read_text(encoding="ascii").strip()
        payload = json.loads(base64.b64decode(fixture))
        payload["player_id"] = invalid_player_id
        invalid = base64.b64encode(json.dumps(payload).encode()).decode()

        with pytest.raises(ValueError, match="player_id"):
            observation_type.deserialize_from_base64(invalid)

    @pytest.mark.parametrize(
        ("filename", "observation_type"),
        [
            ("observation_4p.b64", Observation),
            ("observation_3p.b64", Observation3P),
        ],
    )
    @pytest.mark.parametrize(
        "mutation",
        [
            lambda payload: (
                payload["hands"].__setitem__(0, []),
                payload.__setitem__("_legal_actions", []),
            ),
            lambda payload: payload["hands"].__setitem__(0, [0] * 8 + [4] * 5),
            lambda payload: payload["melds"].__setitem__(
                1,
                [
                    {
                        "meld_type": "Pon",
                        "tiles": [255, 255, 255],
                        "opened": True,
                        "from_who": 0,
                        "called_tile": 255,
                    }
                ],
            ),
            lambda payload: payload["melds"].__setitem__(
                1,
                [
                    {
                        "meld_type": "Pon",
                        "tiles": [],
                        "opened": True,
                        "from_who": 0,
                        "called_tile": 0,
                    }
                ],
            ),
        ],
        ids=["empty-hand", "duplicate-hand-tiles", "invalid-meld-tile", "empty-meld"],
    )
    def test_malformed_observation_payload_is_rejected_without_panic(self, filename, observation_type, mutation):
        fixture = (CONTRACT_DIR / filename).read_text(encoding="ascii").strip()
        payload = json.loads(base64.b64decode(fixture))
        mutation(payload)
        invalid = base64.b64encode(json.dumps(payload).encode()).decode()

        with pytest.raises(ValueError):
            observation_type.deserialize_from_base64(invalid)

    @pytest.mark.parametrize(
        ("filename", "observation_type"),
        [
            ("observation_4p.b64", Observation),
            ("observation_3p.b64", Observation3P),
        ],
    )
    def test_action_selection_only_observation_wire_remains_decodable(self, filename, observation_type):
        fixture = (CONTRACT_DIR / filename).read_text(encoding="ascii").strip()
        payload = json.loads(base64.b64decode(fixture))
        payload["hands"][0] = []
        encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()

        restored = observation_type.deserialize_from_base64(encoded)

        assert restored.hand == []
        assert restored.legal_actions()
        with pytest.raises(ValueError, match="effective self hand length"):
            restored.encode()
        with pytest.raises(ValueError, match="effective self hand length"):
            restored.encode_extended()

    @pytest.mark.parametrize("seed", [0, 1, 99, 12345, 999999])
    def test_round_trip_multiple_seeds(self, seed):
        """Round-trip works across different RNG seeds."""
        _, obs = _get_initial_obs(seed=seed)
        encoded = obs.serialize_to_base64()
        restored = Observation.deserialize_from_base64(encoded)
        _assert_obs_fields_equal(obs, restored)
        _assert_legal_actions_equal(obs, restored)
