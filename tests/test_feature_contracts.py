from __future__ import annotations

import hashlib

import numpy as np
import pytest

from riichienv import RiichiEnv


def _quantized_hash(raw: bytes, dtype: type[np.float32] | type[np.uint8]) -> tuple[int, str]:
    values = np.frombuffer(raw, dtype=dtype)
    if dtype is np.float32:
        # Quantization protects the semantic ABI while avoiding platform-libm
        # last-bit differences in decay/probability features.
        canonical = np.rint(values.astype(np.float64) * 1_000_000).astype("<i8").tobytes()
    else:
        canonical = values.tobytes()
    return values.size, hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    ("game_mode", "method", "dtype", "expected_length", "expected_hash"),
    [
        (
            "4p-red-single",
            "encode",
            np.float32,
            74 * 34,
            "67d0e084db0403ce93181f33b804b4d4282c77615240dfaa53c71fb873672c85",
        ),
        (
            "4p-red-single",
            "encode_extended",
            np.float32,
            215 * 34,
            "aeb1edf73a85462916da62371f46e1aa70ffcd42c1b6cb06a7d8c2bf2275ce48",
        ),
        ("4p-red-single", "mask", np.uint8, 82, "f74474a7dffdf5a1a134c0797bb0ef71cd105db952b6446c31d3fbe924650c2f"),
        (
            "4p-red-single",
            "encode_sp",
            np.float32,
            178 * 34,
            "163b947b70add97a3336642468ce208657955758b9de004347374433c0cb926f",
        ),
        (
            "4p-red-single",
            "encode_drev",
            np.float32,
            9 * 34,
            "b950d7a5a24da82ff7cbb5fc041dccee7bf6e22c70b4d0ae2b8889b9a96f92a9",
        ),
        (
            "4p-red-single",
            "encode_extended_with_sp",
            np.float32,
            (215 + 178 + 9) * 34,
            "47d10458bc439d1f86292b90dd556c0c8aa670c449389ac3137724ab1a4a9dfb",
        ),
        (
            "3p-red-single",
            "encode",
            np.float32,
            74 * 27,
            "2e1a722a7b5f6dd002a5779bdce371812cfa3ea963bbefc92593de56f6837969",
        ),
        (
            "3p-red-single",
            "encode_extended",
            np.float32,
            215 * 27,
            "b70c0cd30a48db6f1cab97126560a3470830c5420393ac929f4e793c79a9236f",
        ),
        (
            "3p-red-single",
            "encode_sp",
            np.float32,
            178 * 27,
            "7984e88bde2aaa0f961bdb5a9c59a3fcf846545e4e227ff9cb441843acbf5783",
        ),
        (
            "3p-red-single",
            "encode_drev",
            np.float32,
            9 * 27,
            "46be13d2dc57d15a3505198ac0dcc7da2f1ec2f739e73f9789e9cb5bebfa34ab",
        ),
        (
            "3p-red-single",
            "encode_extended_with_sp",
            np.float32,
            (215 + 178 + 9) * 27,
            "65f6f6a79b1c7e52e6e4b03b14ad00281d5a3df397c34e3794e9cfbde388ec3e",
        ),
        ("3p-red-single", "mask", np.uint8, 60, "e11ac5811436ca4cad52839a7e9f543b310215c7213cb1e9cd6f7be3768f1433"),
    ],
)
def test_v0_feature_and_action_mask_semantics(
    game_mode: str,
    method: str,
    dtype: type[np.float32] | type[np.uint8],
    expected_length: int,
    expected_hash: str,
):
    observation = RiichiEnv(game_mode=game_mode, seed=42).reset()[0]
    actual_length, actual_hash = _quantized_hash(getattr(observation, method)(), dtype)

    assert actual_length == expected_length
    assert actual_hash == expected_hash
