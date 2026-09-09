# Validating the simulator against Mahjong Soul records

`scripts/validate_mjsoul_corpus.py` replays raw Mahjong Soul `.bin.xz`
records through `RiichiEnv.step()` with the Mahjong Soul rule preset. It
initializes each round from its recorded wall, scores, dealer, round wind,
honba, and riichi deposits. It does not overwrite the simulator state with
recorded draws or use reconstructed replay scores as the oracle.

The checker runs each round in both single-round and match modes. Single-round
mode preserves the completed hand and win results; match mode independently
checks whether the match ends and which round would follow. The next round is
then initialized from its own recorded wall.

Final-hand scores and final-match scores have separate oracles: the round's
payment record excludes unclaimed deposits, while the original game header
includes their award to the top player. Single-round mode preserves the former;
match mode settles the latter. A header-mutation test verifies this distinction.

Checks cover:

- All initial hands and dora indicators, including red fives.
- Every recorded draw, remaining wall count, and revealed dora indicator.
- Every played action's membership in the legal actions, including exact
  consumed tiles for calls and all claimants in multiple ron.
- Conservation of physical tiles and points, including riichi deposits, after
  each simulator step. Called discards are counted once, and extracted North
  tiles are counted separately from kans.
- Winning hands, han, applicable fu, positive-valued yaku, winning seats, and
  final scores against the original provider records.
- Abortive/exhaustive draw reasons, next-round dealer, wind, honba, deposits,
  score continuity, match termination, and final game scores from the record header.

Record format normalization is explicit: empty exhaustive-draw delta vectors
mean zero transfers; zero-valued dora entries are not yaku; fu is not compared
for yakuman, where it does not determine the payment.
For an immediate dealer heavenly-hand win, the provider's named winning tile
can differ from the wall's 14th tile. The complete 14-tile hand is still compared
exactly, and the simulator's own legal tsumo action is used.

## Running locally

The decoder is the `mjsoul-parser` package provided by the separate local
validation project. Install it into this checkout's environment:

```bash
uv sync --package riichienv --dev
uv pip install --python .venv/bin/python /data/gitws/riichienv-validation/packages/mjsoul-parser
.venv/bin/python scripts/validate_mjsoul_corpus.py \
  --root /data/mjsoul --sample 20000 --workers 8 \
  --output /data/riichienv-validation-runs/my-run
```

The default sample is balanced across three/four players and Jade/Throne
tables, spread across dates, and deduplicated by record filename. It is a
stratified sample, not a uniform random sample of all files. Record ordering
and selection are reproducible with `--seed` (default `20260909`).

An output directory contains the selected paths (`manifest.json`), per-file
source hashes, counters and failures (`results.jsonl`), and an aggregate
summary with the simulator revision and validator hash (`summary.json`).
Any mismatch or decode failure makes the process exit nonzero. No failures
are counted as successes or silently skipped.

Use `--manifest PATH` to replay an identical sample, or `--files PATH...` for
individual reproductions. Raw account/profile information is not included in
the output. The test fixture retains only gameplay fields and source identity.

## Limits

Passing a corpus is evidence about the exercised games, not a proof of every
rule or reachable state. The checker verifies recorded choices, not equality
of the complete legal-action sets. Unchosen illegal actions, rare interactions,
other rule presets, hidden-information APIs, and malformed/untrusted inputs
need separate tests. Match validation compares transitions between rounds;
it supplies each subsequent recorded wall rather than validating random wall
generation across an entire game.

The committed mutation tests show that changing an oracle draw, red tile,
score, or next-round honba is detected. Keep small rule regression tests
alongside corpus checks, especially for cases too rare to appear in a sample.
