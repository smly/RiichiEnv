#!/usr/bin/env python3
"""Validate DREV v2 against full-information MJAI replay fixtures.

The native validator is the only component allowed to inspect concealed
hands.  This wrapper keeps corpus selection, fixture integrity, frozen
expectations, failure policy, and machine-readable reporting in one place.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from riichienv import DREV_V2_SCHEMA_ID, GameRule
from riichienv._riichienv import validate_drev_replay_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    ROOT / "tests" / "data" / "external_replay" / "manifest.json",
    ROOT / "tests" / "data" / "drev_replay" / "manifest.json",
]

FAILURE_COUNTERS = (
    "hard_safe_false_positives",
    "legal_ron_zero_probability",
    "yaku_impossible_conflicts",
    "yaku_confirmed_conflicts",
    "loss_lower_bound_violations",
    "feature_invariant_failures",
)
GOLDEN_COUNT_KEYS = (
    "rounds",
    "decisions",
    "candidate_tile_types",
    "physical_discard_candidates",
    "opponent_candidate_cells",
    "shape_wait_cells",
    "yaku_valid_win_cells",
    "legal_ron_cells",
    "hard_safe_cells",
)
COUNT_KEYS = (
    *GOLDEN_COUNT_KEYS,
    *FAILURE_COUNTERS,
)
EXPECTED_KEYS = (
    "variant",
    *GOLDEN_COUNT_KEYS,
    "semantic_sha256",
    "yaku_support",
)
EXPECTED_KEY_SET = frozenset(EXPECTED_KEYS)
METRIC_KEYS = (
    "wait_brier",
    "wait_log_loss",
    "wait_average_precision",
    "wait_ece_10",
    "ron_brier",
    "ron_log_loss",
    "ron_average_precision",
    "ron_ece_10",
    "legal_ron_loss_mae",
    "legal_ron_loss_bias",
)


class DrevReplayValidationError(AssertionError):
    """Raised when a fixture, native report, or frozen expectation fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DrevReplayValidationError(message)


def _require_sha256(value: Any, context: str) -> str:
    is_lower_hex = (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )
    _require(is_lower_hex, f"{context}: expected 64-character lowercase SHA-256")
    return value


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DrevReplayValidationError(f"{context}: invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{context}: expected a JSON object")
    return value


def _read_jsonl(raw: bytes, fixture: Path) -> str:
    try:
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise DrevReplayValidationError(f"{fixture}: cannot decode MJAI JSONL: {error}") from error


def _game_rule(name: Any, context: str) -> GameRule:
    _require(name in {"tenhou", "mjsoul"}, f"{context}: invalid rule preset {name!r}")
    return GameRule.default_mjsoul() if name == "mjsoul" else GameRule.default_tenhou()


def _nonnegative_int(value: Any, context: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{context}: expected a non-negative integer",
    )
    return value


def validate_report(report: Mapping[str, Any], context: str) -> None:
    """Validate the native schema and enforce all exact-correctness gates."""

    _require(report.get("schema_version") == 1, f"{context}: unsupported native report schema")
    _require(
        report.get("drev_schema_id") == DREV_V2_SCHEMA_ID,
        f"{context}: report.drev_schema_id does not match {DREV_V2_SCHEMA_ID!r}",
    )
    for key in COUNT_KEYS:
        _nonnegative_int(report.get(key), f"{context}: report.{key}")
    variant = report.get("variant")
    _require(variant in {"4p", "3p"}, f"{context}: report.variant must be '4p' or '3p'")
    _require(report["decisions"] > 0, f"{context}: report contains no DREV decisions")
    _require(
        report["opponent_candidate_cells"] > 0,
        f"{context}: report contains no opponent/candidate labels",
    )
    _require(
        report["physical_discard_candidates"] >= report["candidate_tile_types"],
        f"{context}: physical candidate count is smaller than tile-type count",
    )
    active_opponents = 3 if variant == "4p" else 2
    _require(
        report["opponent_candidate_cells"] == report["physical_discard_candidates"] * active_opponents,
        f"{context}: opponent/candidate cell count is inconsistent with the variant",
    )
    _require(
        report["shape_wait_cells"] >= report["yaku_valid_win_cells"] >= report["legal_ron_cells"],
        f"{context}: wait/yaku/legal-Ron counts are inconsistent",
    )
    _require(
        report["hard_safe_cells"] <= report["opponent_candidate_cells"],
        f"{context}: hard-safe count exceeds the labelled cells",
    )

    semantic_sha = _require_sha256(report.get("semantic_sha256"), f"{context}: report.semantic_sha256")
    _require(bool(semantic_sha), f"{context}: empty semantic digest")
    metrics_value = report.get("metrics")
    _require(isinstance(metrics_value, dict), f"{context}: report.metrics must be an object")
    metrics = cast(dict[str, Any], metrics_value)
    for key in METRIC_KEYS:
        value = metrics.get(key)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"{context}: report.metrics.{key} must be finite",
        )
    yaku_support_value = report.get("yaku_support")
    _require(isinstance(yaku_support_value, dict), f"{context}: report.yaku_support must be an object")
    yaku_support = cast(dict[Any, Any], yaku_support_value)
    _require(
        all(isinstance(key, str) and key for key in yaku_support),
        f"{context}: yaku support keys must be non-empty strings",
    )
    for key, value in yaku_support.items():
        _nonnegative_int(value, f"{context}: report.yaku_support.{key}")

    violations_value = report.get("violations")
    _require(isinstance(violations_value, list), f"{context}: report.violations must be a list")
    violations = cast(list[Any], violations_value)
    _require(
        all(isinstance(item, dict) for item in violations),
        f"{context}: every report violation must be an object",
    )

    failures = {key: report[key] for key in FAILURE_COUNTERS if report[key] != 0}
    _require(not failures, f"{context}: non-zero DREV correctness failures: {failures}")
    _require(not violations, f"{context}: native validator reported {len(violations)} violation(s)")


def _validate_expected(expected: Any, context: str) -> dict[str, Any]:
    _require(isinstance(expected, dict), f"{context}: drev_validation.expected must be an object")
    expected = cast(dict[str, Any], expected)
    keys = set(expected)
    missing = [key for key in EXPECTED_KEYS if key not in keys]
    unexpected = sorted(keys - EXPECTED_KEY_SET)
    _require(
        not missing and not unexpected,
        f"{context}: drev_validation.expected must contain exactly the frozen key set; "
        f"missing={missing}, unexpected={unexpected}",
    )
    _require(expected["variant"] in {"4p", "3p"}, f"{context}: expected.variant must be '4p' or '3p'")
    for key in GOLDEN_COUNT_KEYS:
        _nonnegative_int(expected[key], f"{context}: expected.{key}")
    _require_sha256(expected["semantic_sha256"], f"{context}: expected.semantic_sha256")
    yaku_support = expected["yaku_support"]
    _require(isinstance(yaku_support, dict), f"{context}: expected.yaku_support must be an object")
    _require(
        all(isinstance(key, str) and key for key in yaku_support),
        f"{context}: expected yaku support keys must be non-empty strings",
    )
    for key, value in yaku_support.items():
        _nonnegative_int(value, f"{context}: expected.yaku_support.{key}")
    return expected


def _validation_policy(
    entry: Mapping[str, Any], context: str, *, require_expected: bool
) -> tuple[dict[str, Any] | None, bool]:
    if "drev_validation" not in entry:
        _require(
            not require_expected,
            f"{context}: drev_validation block is required for frozen validation",
        )
        return None, False

    drev_validation = entry["drev_validation"]
    _require(isinstance(drev_validation, dict), f"{context}: drev_validation must be an object")
    allow_trailing_start_kyoku = drev_validation.get("allow_trailing_start_kyoku", False)
    _require(
        isinstance(allow_trailing_start_kyoku, bool),
        f"{context}: drev_validation.allow_trailing_start_kyoku must be a boolean",
    )
    if "expected" not in drev_validation:
        _require(
            not require_expected,
            f"{context}: drev_validation.expected key is required for frozen validation",
        )
        return None, allow_trailing_start_kyoku
    return _validate_expected(drev_validation["expected"], context), allow_trailing_start_kyoku


def _compare_expected(report: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    for key in EXPECTED_KEYS:
        value = expected[key]
        _require(key in report, f"{context}: expected report key {key!r} is missing")
        _require(
            report[key] == value,
            f"{context}: report.{key} differs from manifest: expected {value!r}, got {report[key]!r}",
        )


def validate_fixture(manifest_path: Path, entry: Mapping[str, Any], *, require_expected: bool = True) -> dict[str, Any]:
    fixture_id = entry.get("id")
    _require(isinstance(fixture_id, str) and fixture_id, f"{manifest_path}: fixture lacks id")
    context = f"{manifest_path}: fixture {fixture_id}"
    expected, allow_trailing_start_kyoku = _validation_policy(
        entry,
        context,
        require_expected=require_expected,
    )
    fixture_name_value = entry.get("fixture")
    _require(
        isinstance(fixture_name_value, str) and fixture_name_value,
        f"{context}: fixture path is missing",
    )
    fixture_name = cast(str, fixture_name_value)
    fixture = (manifest_path.parent / fixture_name).resolve()

    try:
        raw = fixture.read_bytes()
    except OSError as error:
        raise DrevReplayValidationError(f"{context}: cannot read {fixture}: {error}") from error
    expected_sha = _require_sha256(entry.get("fixture_sha256"), f"{context}: fixture SHA-256")
    actual_sha = hashlib.sha256(raw).hexdigest()
    _require(actual_sha == expected_sha, f"{fixture}: fixture SHA-256 mismatch")

    rule_name = entry.get("rule")
    rule = _game_rule(rule_name, context)
    try:
        report_text = validate_drev_replay_jsonl(
            _read_jsonl(raw, fixture),
            rule,
            allow_trailing_start_kyoku=allow_trailing_start_kyoku,
        )
    except Exception as error:
        raise DrevReplayValidationError(f"{context}: native DREV validation failed: {error}") from error
    _require(isinstance(report_text, str), f"{context}: native validator did not return JSON text")
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise DrevReplayValidationError(f"{context}: native validator returned invalid JSON: {error}") from error
    _require(isinstance(report, dict), f"{context}: native report must be a JSON object")

    validate_report(report, context)
    if expected is not None:
        _compare_expected(report, expected, context)

    return {
        "id": fixture_id,
        "fixture": fixture_name,
        "fixture_sha256": actual_sha,
        "rule": rule_name,
        "report": report,
    }


def validate_manifest(manifest_path: Path, *, require_expected: bool = True) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json_object(manifest_path, str(manifest_path))
    _require(manifest.get("schema_version") == 1, f"{manifest_path}: unsupported manifest schema")
    fixtures_value = manifest.get("fixtures")
    _require(isinstance(fixtures_value, list), f"{manifest_path}: fixtures must be a list")
    fixtures = cast(list[Any], fixtures_value)
    _require(bool(fixtures), f"{manifest_path}: manifest contains no fixtures")
    _require(all(isinstance(entry, dict) for entry in fixtures), f"{manifest_path}: invalid fixture entry")
    return [validate_fixture(manifest_path, entry, require_expected=require_expected) for entry in fixtures]


def _write_report(path: Path, manifests: list[Path], fixtures: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "manifests": [str(path) for path in manifests],
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DREV v2 against full-information replay manifests")
    parser.add_argument("manifests", nargs="*", type=Path, help="Additional opt-in corpus manifests")
    parser.add_argument("--no-committed", action="store_true", help="Skip the committed CI manifests")
    parser.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="Allow positional exploratory manifests to omit drev_validation.expected",
    )
    parser.add_argument("--report", type=Path, help="Write all native fixture reports as JSON")
    args = parser.parse_args(argv)
    committed_manifests = [] if args.no_committed else DEFAULT_MANIFESTS
    manifests = committed_manifests + args.manifests
    if not manifests:
        parser.error("no manifests selected")

    try:
        resolved = [path.resolve() for path in manifests]
        committed_paths = {path.resolve() for path in DEFAULT_MANIFESTS}
        fixtures = [
            fixture
            for path in resolved
            for fixture in validate_manifest(
                path,
                require_expected=path in committed_paths or not args.allow_unfrozen,
            )
        ]
        if args.report is not None:
            _write_report(args.report, resolved, fixtures)
    except Exception as error:
        print(f"FAILED: {error}")
        return 1

    print(f"Validated {len(fixtures)} DREV replay fixtures from {len(manifests)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
