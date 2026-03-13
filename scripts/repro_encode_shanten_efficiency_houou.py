#!/usr/bin/env python3
"""Reproduce the Houou shanten-efficiency panic from a Tenhou JSON paipu."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import riichienv

DEFAULT_LOG_ID = "2024010100gm-00a9-0000-1d4dbec0"
DEFAULT_PAIPU_DIR = (
    Path(__file__).resolve().parents[2] / "tenhou-analysis" / "tenhou-data-extractor" / "paipu"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one Tenhou JSON replay with mjai-reviewer and scan replay decisions "
            "until encode_shanten_efficiency() panics."
        )
    )
    parser.add_argument(
        "--paipu-dir",
        type=Path,
        default=DEFAULT_PAIPU_DIR,
        help=f"Directory containing Tenhou JSON logs (default: {DEFAULT_PAIPU_DIR})",
    )
    parser.add_argument(
        "--log-id",
        default=DEFAULT_LOG_ID,
        help=f"Replay log id without extension (default: {DEFAULT_LOG_ID})",
    )
    parser.add_argument(
        "--round-index",
        type=int,
        default=None,
        help="Optional zero-based round index to scan. Defaults to all rounds.",
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=None,
        help="Optional seat filter for replay decisions.",
    )
    parser.add_argument(
        "--decision-index",
        type=int,
        default=None,
        help="Optional zero-based decision index within each scanned round.",
    )
    parser.add_argument(
        "--mjai-reviewer-cmd",
        default="mjai-reviewer",
        help="Command prefix for mjai-reviewer (default: mjai-reviewer)",
    )
    parser.add_argument(
        "--mjai-out",
        type=Path,
        default=None,
        help="Optional path to keep the converted MJAI log instead of using a temp file.",
    )
    parser.add_argument(
        "--expect-panic",
        action="store_true",
        help="Exit non-zero if the panic is not reproduced.",
    )
    return parser.parse_args()


def _resolve_command(command_text: str) -> list[str]:
    command = shlex.split(command_text)
    if not command:
        raise SystemExit("mjai-reviewer command must not be empty")

    executable = command[0]
    if shutil.which(executable) is None and not Path(executable).exists():
        raise SystemExit(f"mjai-reviewer executable not found: {executable}")
    return command


def _convert_to_mjai(
    tenhou_json_path: Path,
    mjai_out: Path,
    mjai_reviewer_cmd: str,
) -> None:
    command = [
        *_resolve_command(mjai_reviewer_cmd),
        "--no-review",
        "-i",
        str(tenhou_json_path),
        "--mjai-out",
        str(mjai_out),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        detail = stderr or stdout or f"exit code {exc.returncode}"
        raise SystemExit(f"mjai-reviewer failed: {' '.join(command)}\n{detail}") from exc


def _iter_steps(kyoku, seat: int | None):
    if seat is None:
        for decision_index, (pid, obs, act) in enumerate(kyoku.steps(skip_single_action=False)):
            yield decision_index, pid, obs, act
        return

    for decision_index, (obs, act) in enumerate(kyoku.steps(seat=seat, skip_single_action=False)):
        yield decision_index, seat, obs, act


def _hand_to_tiles(hand: list[int]) -> list[str]:
    return [riichienv.convert.tid_to_mjai(tile) for tile in hand]


def main() -> int:
    args = _parse_args()

    tenhou_json_path = args.paipu_dir / f"{args.log_id}.json"
    if not tenhou_json_path.exists():
        print(f"Tenhou JSON not found: {tenhou_json_path}", file=sys.stderr)
        return 2

    if args.mjai_out is not None:
        mjai_path = args.mjai_out
        mjai_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="riichienv-shanten-repro-")
        mjai_path = Path(cleanup.name) / f"{args.log_id}.mjson"

    try:
        _convert_to_mjai(tenhou_json_path, mjai_path, args.mjai_reviewer_cmd)
        replay = riichienv.MjaiReplay.from_jsonl(str(mjai_path))

        panic_found = False
        steps_scanned = 0
        for round_index, kyoku in enumerate(replay.take_kyokus()):
            if args.round_index is not None and round_index != args.round_index:
                continue

            for decision_index, pid, obs, act in _iter_steps(kyoku, args.seat):
                if args.decision_index is not None and decision_index != args.decision_index:
                    continue

                steps_scanned += 1
                try:
                    obs.encode_shanten_efficiency()
                except BaseException as exc:  # PanicException inherits BaseException
                    panic_found = True
                    print("encode_shanten_efficiency() panicked")
                    print(f"log_id={args.log_id}")
                    print(f"round_index={round_index}")
                    print(f"decision_index={decision_index}")
                    print(f"seat={pid}")
                    print(f"action_type={act.action_type}")
                    print(f"hand={' '.join(_hand_to_tiles(obs.hand))}")
                    print(f"error={type(exc).__name__}: {exc}")
                    break

            if panic_found:
                break

        if not panic_found:
            print(
                "No panic reproduced while scanning "
                f"{steps_scanned} decision(s) from {args.log_id}."
            )
            return 1 if args.expect_panic else 0

        return 0
    finally:
        if cleanup is not None:
            cleanup.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
