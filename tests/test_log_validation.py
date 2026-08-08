from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_logs.py"
FIXTURE_PATH = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_logs", VALIDATOR_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_mjai_fixture_satisfies_full_replay_contract():
    validator = _load_validator()
    validator.validate_tenhou_log(FIXTURE_PATH, rule="tenhou")


def test_validator_cli_returns_nonzero_when_a_log_fails(tmp_path):
    invalid_log = tmp_path / "invalid.jsonl"
    invalid_log.write_text("{not-json}\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), str(invalid_log)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "FAILED" in result.stdout
