from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_drev_replays.py"
MANIFEST_PATH = Path(__file__).parent / "data" / "external_replay" / "manifest.json"
TENHOU_FIXTURE = MANIFEST_PATH.parent / "tenhou_4p_ranked_excerpt.jsonl"
COMPLETE_FIXTURE = MANIFEST_PATH.parent / "tenhou_4p_honba_kyotaku_excerpt.jsonl"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_drev_replays", VALIDATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_manifest(
    directory: Path,
    fixture_bytes: bytes,
    *,
    fixture_sha256: str | None = None,
    expected: dict | None = None,
    include_validation_block: bool = False,
    allow_trailing_start_kyoku: bool | None = None,
) -> Path:
    fixture = directory / "fixture.jsonl"
    fixture.write_bytes(fixture_bytes)
    entry: dict[str, object] = {
        "id": "test-fixture",
        "fixture": fixture.name,
        "fixture_sha256": fixture_sha256 or hashlib.sha256(fixture_bytes).hexdigest(),
        "rule": "tenhou",
    }
    if expected is not None or include_validation_block or allow_trailing_start_kyoku is not None:
        drev_validation: dict[str, object] = {}
        if expected is not None:
            drev_validation["expected"] = expected
        if allow_trailing_start_kyoku is not None:
            drev_validation["allow_trailing_start_kyoku"] = allow_trailing_start_kyoku
        entry["drev_validation"] = drev_validation
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "fixtures": [entry]}), encoding="utf-8")
    return manifest


def _tenhou_expected() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["fixtures"][0]["drev_validation"]["expected"]


def test_committed_manifest_passes_and_writes_machine_readable_report(tmp_path):
    report_path = tmp_path / "drev-report.json"
    result = _run_validator("--report", str(report_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 5 DREV replay fixtures" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["fixture_count"] == 5
    assert len(report["fixtures"]) == 5
    assert all(item["report"]["hard_safe_false_positives"] == 0 for item in report["fixtures"])


def test_fixture_sha256_tampering_returns_nonzero(tmp_path):
    fixture_bytes = TENHOU_FIXTURE.read_bytes()
    manifest = _write_manifest(tmp_path, fixture_bytes, fixture_sha256="0" * 64)

    result = _run_validator("--no-committed", "--allow-unfrozen", str(manifest))

    assert result.returncode == 1
    assert "fixture SHA-256 mismatch" in result.stdout


def test_manifest_expected_mismatch_returns_nonzero(tmp_path):
    fixture_bytes = TENHOU_FIXTURE.read_bytes()
    expected = _tenhou_expected()
    expected["rounds"] = 0
    manifest = _write_manifest(
        tmp_path,
        fixture_bytes,
        expected=expected,
        allow_trailing_start_kyoku=True,
    )

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "report.rounds differs from manifest" in result.stdout


def test_invalid_jsonl_returns_nonzero(tmp_path):
    manifest = _write_manifest(tmp_path, b"{not-json}\n")

    result = _run_validator("--no-committed", "--allow-unfrozen", str(manifest))

    assert result.returncode == 1
    assert "FAILED" in result.stdout


def test_frozen_manifest_rejects_missing_drev_validation_block(tmp_path):
    manifest = _write_manifest(tmp_path, COMPLETE_FIXTURE.read_bytes())

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "drev_validation block is required" in result.stdout


def test_frozen_manifest_rejects_missing_expected_key(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        COMPLETE_FIXTURE.read_bytes(),
        include_validation_block=True,
    )

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "drev_validation.expected key is required" in result.stdout


def test_frozen_manifest_rejects_partial_expected_object(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        COMPLETE_FIXTURE.read_bytes(),
        expected={"semantic_sha256": "0" * 64},
    )

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "must contain exactly the frozen key set" in result.stdout
    assert "missing=" in result.stdout


def test_frozen_manifest_requires_semantic_digest_in_complete_key_set(tmp_path):
    expected = _tenhou_expected()
    expected.pop("semantic_sha256")
    manifest = _write_manifest(tmp_path, TENHOU_FIXTURE.read_bytes(), expected=expected)

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "must contain exactly the frozen key set" in result.stdout
    assert "semantic_sha256" in result.stdout


def test_exploratory_manifest_requires_explicit_unfrozen_opt_out(tmp_path):
    manifest = _write_manifest(tmp_path, COMPLETE_FIXTURE.read_bytes())

    frozen = _run_validator("--no-committed", str(manifest))
    exploratory = _run_validator("--no-committed", "--allow-unfrozen", str(manifest))

    assert frozen.returncode == 1
    assert exploratory.returncode == 0, exploratory.stdout + exploratory.stderr
    assert "Validated 1 DREV replay fixtures" in exploratory.stdout


def test_trailing_start_kyoku_requires_explicit_manifest_opt_in(tmp_path):
    manifest = _write_manifest(tmp_path, TENHOU_FIXTURE.read_bytes(), expected=_tenhou_expected())

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "native DREV validation failed" in result.stdout


def test_trailing_start_kyoku_opt_in_rejects_one_following_action(tmp_path):
    fixture_bytes = TENHOU_FIXTURE.read_bytes() + b'{"type":"tsumo","actor":1,"pai":"9m"}\n'
    manifest = _write_manifest(
        tmp_path,
        fixture_bytes,
        expected=_tenhou_expected(),
        allow_trailing_start_kyoku=True,
    )

    result = _run_validator("--no-committed", str(manifest))

    assert result.returncode == 1
    assert "native DREV validation failed" in result.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        {"hard_safe_false_positives": 1},
        {"legal_ron_zero_probability": 1},
        {"yaku_impossible_conflicts": 1},
        {"yaku_confirmed_conflicts": 1},
        {"loss_lower_bound_violations": 1},
        {"feature_invariant_failures": 1},
        {"decisions": 0},
        {"opponent_candidate_cells": 0},
        {"violations": [{"kind": "test"}]},
    ],
)
def test_native_correctness_failures_are_hard_failures(mutation):
    validator = _load_validator()
    report = {
        "schema_version": 1,
        "drev_schema_id": "riichienv.drev_v2.81ch.v1",
        "variant": "4p",
        "rounds": 1,
        "decisions": 1,
        "candidate_tile_types": 1,
        "physical_discard_candidates": 1,
        "opponent_candidate_cells": 3,
        "shape_wait_cells": 0,
        "yaku_valid_win_cells": 0,
        "legal_ron_cells": 0,
        "hard_safe_cells": 1,
        "hard_safe_false_positives": 0,
        "legal_ron_zero_probability": 0,
        "yaku_impossible_conflicts": 0,
        "yaku_confirmed_conflicts": 0,
        "loss_lower_bound_violations": 0,
        "feature_invariant_failures": 0,
        "semantic_sha256": "0" * 64,
        "violations": [],
        "metrics": {
            "wait_brier": 0.0,
            "wait_log_loss": 0.0,
            "wait_average_precision": 0.0,
            "wait_ece_10": 0.0,
            "ron_brier": 0.0,
            "ron_log_loss": 0.0,
            "ron_average_precision": 0.0,
            "ron_ece_10": 0.0,
            "legal_ron_loss_mae": 0.0,
            "legal_ron_loss_bias": 0.0,
        },
        "yaku_support": {},
    }
    report.update(mutation)

    with pytest.raises(validator.DrevReplayValidationError):
        validator.validate_report(report, "test")
