#!/usr/bin/env python3
"""Generate replay_manifest.json listing every jsonl.gz under a data dir.

The replay_viewer.html dropdown reads this manifest to populate its file
picker. Run this after adding or removing replays:

    python3 scripts/build_replay_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = [
    "data-mjsoul-4p-2026-01",
    "data-mjsoul-3p-2026-01",
]


def collect(dirs: list[Path]) -> list[str]:
    out: list[str] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.jsonl.gz")):
            out.append(str(p.relative_to(REPO_ROOT)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        action="append",
        default=None,
        help="Repository-relative data directory (repeatable).",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(REPO_ROOT / "replay_manifest.json"),
        help="Destination manifest JSON.",
    )
    ap.add_argument(
        "--per-dir",
        type=int,
        default=20,
        help="Maximum files included per leaf directory (default 20). "
        "Use 0 for no limit.",
    )
    args = ap.parse_args()

    dirs = [REPO_ROOT / d for d in (args.dir or DEFAULT_DIRS)]
    if args.per_dir > 0:
        paths: list[str] = []
        for d in dirs:
            if not d.exists():
                continue
            for sub in sorted(p for p in d.iterdir() if p.is_dir()):
                taken = 0
                for p in sorted(sub.rglob("*.jsonl.gz")):
                    if taken >= args.per_dir:
                        break
                    paths.append(str(p.relative_to(REPO_ROOT)))
                    taken += 1
            # Also include any *.jsonl.gz directly under d.
            for p in sorted(d.glob("*.jsonl.gz")):
                paths.append(str(p.relative_to(REPO_ROOT)))
    else:
        paths = collect(dirs)

    out = {"files": paths}
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(paths)} entries to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
