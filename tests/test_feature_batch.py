from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import cast

import numpy as np

from riichienv import (
    Observation,
    Observation3P,
    RiichiEnv,
    encode_base_batch,
    encode_base_batch_3p,
    encode_drev_batch,
    encode_drev_batch_3p,
    encode_drev_v2_batch,
    encode_drev_v2_batch_3p,
    encode_extended_batch,
    encode_extended_batch_3p,
    encode_extended_with_sp_batch,
    encode_extended_with_sp_batch_3p,
    encode_extended_with_sp_drev_v2_batch,
    encode_extended_with_sp_drev_v2_batch_3p,
    encode_sp_batch,
    encode_sp_batch_3p,
)


def test_four_player_batch_encoders_match_single_observation_bytes():
    observations = [RiichiEnv(seed=seed).reset()[0] for seed in (7, 8)]

    assert encode_base_batch(observations) == b"".join(observation.encode() for observation in observations)
    assert encode_extended_batch(observations) == b"".join(
        observation.encode_extended() for observation in observations
    )
    assert encode_sp_batch(observations) == b"".join(observation.encode_sp() for observation in observations)
    assert encode_drev_batch(observations) == b"".join(observation.encode_drev() for observation in observations)
    assert encode_drev_v2_batch(observations) == b"".join(observation.encode_drev_v2() for observation in observations)
    assert encode_extended_with_sp_batch(observations) == b"".join(
        observation.encode_extended_with_sp() for observation in observations
    )
    assert encode_extended_with_sp_drev_v2_batch(observations) == b"".join(
        observation.encode_extended_with_sp_drev_v2() for observation in observations
    )

    values = np.frombuffer(encode_extended_batch(observations), dtype=np.float32)
    assert values.shape == (2 * 215 * 34,)


def test_three_player_batch_encoders_match_single_observation_bytes():
    observations = [cast(Observation3P, RiichiEnv(game_mode="3p-red-single", seed=seed).reset()[0]) for seed in (7, 8)]

    assert encode_base_batch_3p(observations) == b"".join(observation.encode() for observation in observations)
    assert encode_extended_batch_3p(observations) == b"".join(
        observation.encode_extended() for observation in observations
    )
    assert encode_sp_batch_3p(observations) == b"".join(observation.encode_sp() for observation in observations)
    assert encode_drev_batch_3p(observations) == b"".join(observation.encode_drev() for observation in observations)
    assert encode_drev_v2_batch_3p(observations) == b"".join(
        observation.encode_drev_v2() for observation in observations
    )
    assert encode_extended_with_sp_batch_3p(observations) == b"".join(
        observation.encode_extended_with_sp() for observation in observations
    )
    assert encode_extended_with_sp_drev_v2_batch_3p(observations) == b"".join(
        observation.encode_extended_with_sp_drev_v2() for observation in observations
    )

    values = np.frombuffer(encode_extended_with_sp_batch_3p(observations), dtype=np.float32)
    assert values.shape == (2 * 402 * 27,)


def test_batch_encoders_accept_empty_batches():
    assert encode_base_batch([]) == b""
    assert encode_extended_batch_3p([]) == b""
    assert encode_sp_batch_3p([]) == b""
    assert encode_drev_batch_3p([]) == b""
    assert encode_drev_v2_batch([]) == b""
    assert encode_drev_v2_batch_3p([]) == b""
    assert encode_extended_with_sp_batch_3p([]) == b""
    assert encode_extended_with_sp_drev_v2_batch([]) == b""
    assert encode_extended_with_sp_drev_v2_batch_3p([]) == b""


def _observation_with_called_meld(filename: str, observation_type):
    path = Path(__file__).parent / "contracts" / "v0.4.8" / filename
    payload = json.loads(base64.b64decode(path.read_text()).decode())
    payload["discards"][0] = [52]
    payload["melds"][1] = [
        {
            "meld_type": "Pon",
            "tiles": [52, 53, 54],
            "opened": True,
            "from_who": 0,
            "called_tile": 52,
        }
    ]
    encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return observation_type.deserialize_from_base64(encoded)


def test_base_batch_preserves_legacy_called_tile_counting():
    observation_4p = _observation_with_called_meld("observation_4p.b64", Observation)
    observation_3p = _observation_with_called_meld("observation_3p.b64", Observation3P)

    assert encode_base_batch([observation_4p]) == observation_4p.encode()
    assert encode_base_batch_3p([observation_3p]) == observation_3p.encode()
