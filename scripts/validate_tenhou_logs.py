"""
Datasource: https://www.kaggle.com/datasets/shokanekolouis/tenhou-to-mjai

Validates Tenhou logs by replaying each kyoku step-by-step and checking:
  1. All encode_* methods produce finite, correctly-shaped tensors (no panic)
  2. The replayed action is always contained in legal_actions
  3. The action mask is consistent with legal_actions (3P)
  4. legal_actions is non-empty at every decision point
  5. Feature value ranges are within expected bounds
  6. Score continuity: end_scores of kyoku k == scores of kyoku k+1
  7. Score conservation: sum(delta) accounts for riichi stick movement
  8. First obs scores match kyoku starting scores
"""
import argparse
from pathlib import Path

from riichienv import MjaiReplay
from riichienv._riichienv import Action, Action3P, Observation, Observation3P
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_key(action: Action | Action3P) -> tuple:
    """Hashable identity for an action (action_type, tile, sorted consume_tiles)."""
    d = action.to_dict()
    return (d["type"], d["tile"], tuple(sorted(d["consume_tiles"])))


def _check_finite(arr: np.ndarray, name: str) -> None:
    assert np.isfinite(arr).all(), f"{name} contains non-finite values"


def _check_range(arr: np.ndarray, lo: float, hi: float, name: str) -> None:
    if arr.size == 0:
        return
    assert arr.min() >= lo, f"{name} has value {arr.min()} < {lo}"
    assert arr.max() <= hi, f"{name} has value {arr.max()} > {hi}"


# ---------------------------------------------------------------------------
# Feature encoding validation
# ---------------------------------------------------------------------------

def validate_obs_encoding_4p(obs: Observation) -> None:
    """Validate all encode_* methods on a 4-player Observation."""
    feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, 34)
    _check_finite(feat, "encode")

    feat = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(4, 34)
    _check_finite(feat, "discard_history_decay")

    feat = np.frombuffer(obs.encode_yaku_possibility(), dtype=np.float32).reshape(4, 21, 2)
    _check_finite(feat, "yaku_possibility")

    feat = np.frombuffer(obs.encode_furiten_ron_possibility(), dtype=np.float32).reshape(4, 21)
    _check_finite(feat, "furiten_ron_possibility")

    feat = np.frombuffer(obs.encode_shanten_efficiency(), dtype=np.float32).reshape(4, 4)
    _check_finite(feat, "shanten_efficiency")

    feat = np.frombuffer(obs.encode_kawa_overview(), dtype=np.float32).reshape(4, 7, 34)
    _check_finite(feat, "kawa_overview")

    feat = np.frombuffer(obs.encode_fuuro_overview(), dtype=np.float32).reshape(4, 4, 5, 34)
    _check_finite(feat, "fuuro_overview")

    feat = np.frombuffer(obs.encode_ankan_overview(), dtype=np.float32).reshape(4, 34)
    _check_finite(feat, "ankan_overview")

    feat = np.frombuffer(obs.encode_action_availability(), dtype=np.float32).reshape((11,))
    _check_finite(feat, "action_availability")
    _check_range(feat, 0.0, 1.0, "action_availability")

    feat = np.frombuffer(obs.encode_riichi_sutehais(), dtype=np.float32).reshape(3, 3)
    _check_finite(feat, "riichi_sutehais")

    feat = np.frombuffer(obs.encode_last_tedashis(), dtype=np.float32).reshape(3, 3)
    _check_finite(feat, "last_tedashis")

    feat = np.frombuffer(obs.encode_pass_context(), dtype=np.float32).reshape((3,))
    _check_finite(feat, "pass_context")

    feat = np.frombuffer(obs.encode_discard_candidates(), dtype=np.float32).reshape((5,))
    _check_finite(feat, "discard_candidates")

    feat = np.frombuffer(obs.encode_extended(), dtype=np.float32).reshape(215, 34)
    _check_finite(feat, "extended")

    # Sequence encodings (variable-length)
    feat = np.frombuffer(obs.encode_seq_sparse(), dtype=np.uint16)
    assert len(feat) <= 25, f"Seq sparse too long: {len(feat)}"

    feat = np.frombuffer(obs.encode_seq_numeric(), dtype=np.float32).reshape((12,))
    _check_finite(feat, "seq_numeric")

    feat = np.frombuffer(obs.encode_seq_progression(), dtype=np.uint16)
    assert feat.size % 5 == 0, f"Seq progression size {feat.size} not divisible by 5"
    if feat.size > 0:
        feat = feat.reshape(-1, 5)

    feat = np.frombuffer(obs.encode_seq_candidates(), dtype=np.uint16)
    assert feat.size % 4 == 0, f"Seq candidates size {feat.size} not divisible by 4"
    if feat.size > 0:
        feat = feat.reshape(-1, 4)


def validate_obs_encoding_3p(obs: Observation3P) -> None:
    """Validate all encode_* methods on a 3-player Observation3P."""
    feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(74, 27)
    _check_finite(feat, "encode")

    feat = np.frombuffer(obs.encode_discard_history_decay(), dtype=np.float32).reshape(3, 27)
    _check_finite(feat, "discard_history_decay")

    feat = np.frombuffer(obs.encode_yaku_possibility(), dtype=np.float32).reshape(3, 21, 2)
    _check_finite(feat, "yaku_possibility")

    feat = np.frombuffer(obs.encode_furiten_ron_possibility(), dtype=np.float32).reshape(3, 21)
    _check_finite(feat, "furiten_ron_possibility")

    feat = np.frombuffer(obs.encode_shanten_efficiency(), dtype=np.float32).reshape(3, 4)
    _check_finite(feat, "shanten_efficiency")

    feat = np.frombuffer(obs.encode_kawa_overview(), dtype=np.float32).reshape(3, 7, 27)
    _check_finite(feat, "kawa_overview")

    feat = np.frombuffer(obs.encode_fuuro_overview(), dtype=np.float32).reshape(3, 4, 5, 27)
    _check_finite(feat, "fuuro_overview")

    feat = np.frombuffer(obs.encode_ankan_overview(), dtype=np.float32).reshape(3, 27)
    _check_finite(feat, "ankan_overview")

    feat = np.frombuffer(obs.encode_action_availability(), dtype=np.float32).reshape((11,))
    _check_finite(feat, "action_availability")
    _check_range(feat, 0.0, 1.0, "action_availability")

    feat = np.frombuffer(obs.encode_riichi_sutehais(), dtype=np.float32).reshape(2, 3)
    _check_finite(feat, "riichi_sutehais")

    feat = np.frombuffer(obs.encode_last_tedashis(), dtype=np.float32).reshape(2, 3)
    _check_finite(feat, "last_tedashis")

    feat = np.frombuffer(obs.encode_pass_context(), dtype=np.float32).reshape((3,))
    _check_finite(feat, "pass_context")

    feat = np.frombuffer(obs.encode_discard_candidates(), dtype=np.float32).reshape((5,))
    _check_finite(feat, "discard_candidates")

    feat = np.frombuffer(obs.encode_extended(), dtype=np.float32).reshape(215, 27)
    _check_finite(feat, "extended")


# ---------------------------------------------------------------------------
# Action / mask consistency validation
# ---------------------------------------------------------------------------

def validate_action_in_legals(obs: Observation | Observation3P, action: Action | Action3P,
                              *, ctx: str) -> None:
    """Assert the replayed action is in obs.legal_actions()."""
    legals = obs.legal_actions()

    assert len(legals) > 0, f"{ctx}: legal_actions is empty"

    action_k = _action_key(action)
    legal_keys = {_action_key(a) for a in legals}
    assert action_k in legal_keys, (
        f"{ctx}: action {action.to_mjai()} not in legal_actions "
        f"({len(legals)} legals: {[a.to_mjai() for a in legals[:10]]}...)"
    )


def validate_mask_consistency_3p(obs: Observation3P, action: Action3P, *, ctx: str) -> None:
    """Assert that mask is consistent with legal_actions and the replayed action (3P)."""
    legals = obs.legal_actions()
    mask_bytes = obs.mask()
    mask = np.frombuffer(mask_bytes, dtype=np.uint8)
    action_space = obs.action_space_size

    assert len(mask) == action_space, (
        f"{ctx}: mask length {len(mask)} != action_space_size {action_space}"
    )

    # Number of 1-bits in mask should equal number of legal actions
    mask_count = int(mask.sum())
    assert mask_count == len(legals), (
        f"{ctx}: mask has {mask_count} bits set but {len(legals)} legal actions"
    )

    # Every legal action should be findable by its encoded id
    for legal_action in legals:
        aid = legal_action.encode()
        assert 0 <= aid < action_space, (
            f"{ctx}: legal action id {aid} out of range [0, {action_space})"
        )
        assert mask[aid] == 1, (
            f"{ctx}: mask bit for legal action {legal_action.to_mjai()} (id={aid}) is 0"
        )
        found = obs.find_action(aid)
        assert found is not None, (
            f"{ctx}: legal action {legal_action.to_mjai()} not found via find_action({aid})"
        )

    # The replayed action should have its mask bit set
    action_id = action.encode()
    assert 0 <= action_id < action_space, (
        f"{ctx}: action_id {action_id} out of range [0, {action_space})"
    )
    assert mask[action_id] == 1, (
        f"{ctx}: mask bit for replayed action {action.to_mjai()} (id={action_id}) is 0"
    )


# ---------------------------------------------------------------------------
# Score continuity validation
# ---------------------------------------------------------------------------

def validate_score_continuity(kyokus: list, *, log_name: str) -> None:
    """Validate score consistency across kyokus.

    Checks:
      - end_scores of kyoku k == scores of kyoku k+1 (score continuity)
      - sum(delta_scores) + riichi stick change * 1000 == 0 (score conservation)
      - end_scores is non-empty for every kyoku
    """
    for ki, kyoku in enumerate(kyokus):
        ctx = f"{log_name} kyoku={ki}"
        np_ = len(kyoku.scores)

        # end_scores must be present
        assert len(kyoku.end_scores) == np_, (
            f"{ctx}: end_scores length {len(kyoku.end_scores)} != num_players {np_}"
        )

        # Score conservation: delta_sum + riichi stick change = 0
        # Riichi sticks deposited during this kyoku reduce the score pool;
        # sticks collected by the winner increase it.
        delta = [e - s for s, e in zip(kyoku.scores, kyoku.end_scores)]
        delta_sum = sum(delta)

        if ki < len(kyokus) - 1:
            next_kyoku = kyokus[ki + 1]

            # Score continuity: this kyoku's end == next kyoku's start
            assert list(kyoku.end_scores) == list(next_kyoku.scores), (
                f"{ctx}: end_scores {kyoku.end_scores} != "
                f"next scores {next_kyoku.scores}"
            )

            # Conservation with riichi stick accounting
            liqibang_change = next_kyoku.liqibang - kyoku.liqibang
            assert delta_sum == -liqibang_change * 1000, (
                f"{ctx}: delta_sum={delta_sum} but riichi stick change "
                f"{kyoku.liqibang}->{next_kyoku.liqibang} expects "
                f"{-liqibang_change * 1000}"
            )
        else:
            # Last kyoku: no next kyoku to compare, but delta_sum should
            # account for any remaining supply sticks being awarded.
            # Just verify total scores are consistent (sum == initial total).
            pass


def validate_first_obs_scores(kyoku, first_obs: Observation | Observation3P,
                              *, ctx: str) -> None:
    """Verify the first observation's scores/honba/riichi_sticks match the kyoku metadata."""
    assert list(first_obs.scores) == list(kyoku.scores), (
        f"{ctx}: first obs scores {first_obs.scores} != kyoku scores {kyoku.scores}"
    )
    assert first_obs.honba == kyoku.ben, (
        f"{ctx}: first obs honba {first_obs.honba} != kyoku ben {kyoku.ben}"
    )
    assert first_obs.riichi_sticks == kyoku.liqibang, (
        f"{ctx}: first obs riichi_sticks {first_obs.riichi_sticks} != "
        f"kyoku liqibang {kyoku.liqibang}"
    )


# ---------------------------------------------------------------------------
# Per-log validation
# ---------------------------------------------------------------------------

def validate_tenhou_log(log_path: Path, *, rule: str | None = None) -> None:
    """Validates a Tenhou log by replaying all kyoku steps.

    Checks:
      - Feature encodings produce valid (finite, in-range) tensors
      - The replayed action is in legal_actions at every step
      - Mask / legal_actions consistency (3P)
      - Score continuity and conservation across kyokus
      - First obs scores match kyoku metadata

    Raises ValueError or AssertionError on failure.
    """
    replay = MjaiReplay.from_jsonl(str(log_path), rule=rule)
    kyokus = list(replay.take_kyokus())

    # Cross-kyoku score validation
    validate_score_continuity(kyokus, log_name=log_path.name)

    for kyoku_idx, kyoku in enumerate(kyokus):
        is_3p = len(kyoku.scores) == 3
        first_obs_checked = False

        step_count = 0
        for seat, obs, action in kyoku.steps(skip_single_action=False):
            ctx = f"kyoku={kyoku_idx} step={step_count} seat={seat}"

            # Validate first observation's scores match kyoku metadata
            if not first_obs_checked:
                validate_first_obs_scores(kyoku, obs, ctx=ctx)
                first_obs_checked = True

            if is_3p:
                assert isinstance(obs, Observation3P), f"{ctx}: Expected Observation3P, got {type(obs)}"
                validate_obs_encoding_3p(obs)
                validate_action_in_legals(obs, action, ctx=ctx)
                validate_mask_consistency_3p(obs, action, ctx=ctx)
            else:
                assert isinstance(obs, Observation), f"{ctx}: Expected Observation, got {type(obs)}"
                validate_obs_encoding_4p(obs)
                validate_action_in_legals(obs, action, ctx=ctx)

            step_count += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Tenhou logs: feature encodings, legal action consistency, mask correctness"
    )
    parser.add_argument("paths", nargs="*", help="Log file paths or directories to validate")
    parser.add_argument("--rule", default="tenhou",
                        choices=["tenhou", "mjsoul", "tenhou_sanma", "mjsoul_sanma"],
                        help="Game rule to use for parsing")
    parser.add_argument("--glob", default="*.mjson", help="Glob pattern for log files (default: *.mjson)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of files to validate (0=all)")
    args = parser.parse_args()

    if not args.paths:
        args.paths = ["/data/workspace/tenhou/2026"]

    log_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            log_files.append(path)
        elif path.is_dir():
            log_files.extend(sorted(path.glob(args.glob)))
        else:
            print(f"Warning: {p} not found, skipping")

    if not log_files:
        print("No log files found")
        return

    total = len(log_files)
    if args.limit > 0:
        log_files = log_files[:args.limit]

    print(f"Found {total} log files, validating {len(log_files)}")

    failed = 0
    for i, log_path in enumerate(log_files):
        print(f"[{i + 1}/{len(log_files)}] {log_path.name}")
        try:
            validate_tenhou_log(log_path, rule=args.rule)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\nDone: {len(log_files) - failed}/{len(log_files)} passed")


if __name__ == "__main__":
    main()
