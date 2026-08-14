# External replay golden fixtures

These small, anonymized excerpts are committed solely for deterministic replay
validation. `manifest.json` records the exact upstream URL and commit, data
license, transformation, upstream source SHA-256, derived fixture SHA-256,
normalized action digest, result, score transition, and next-round header for
each fixture.

- `tenhou_4p_ranked_excerpt.jsonl` is derived from the CC BY 4.0 Tenhou fixture
  in `NikkeTryHard/tenhou-to-mjai`.
- `tenhou_4p_dealer_tsumo_excerpt.jsonl` adds the same source game's dealer
  tsumo and split-payment oracle.
- `tenhou_4p_honba_kyotaku_excerpt.jsonl` is a terminal Tenhou round with
  non-zero honba and kyotaku, also from the CC BY 4.0 converter test data.
- `mjsoul_3p_double_ron_excerpt.jsonl` is derived from the Apache-2.0 Mahjong
  Soul three-player fixture in the `3p` branch of `hidacow/mjai-reviewer3p`.

Where an upstream next round exists, the excerpt stops after that round's
header. This preserves the provider's authoritative transition metadata
without redistributing a full game. The honba/kyotaku fixture instead ends at
the game's terminal round so the result delta itself is the only end-score
oracle. Only complete round indices listed in the manifest are validated.

For a larger local corpus, create manifests with the same schema outside the
repository and opt in explicitly:

```bash
uv run python scripts/validate_external_replays.py --no-committed /path/to/corpus-manifest.json
```

The same fixtures also carry a `drev_validation.expected` block. It freezes
the number of replayed discard decisions, tile-type and physical discard
candidates, candidate/opponent labels, exact wait and legal-Ron support,
hard-safe proofs, public-yaku support, and a digest over predictions plus
hidden-state labels. Run the DREV validator with:

```bash
uv run python scripts/validate_drev_replays.py
uv run python scripts/validate_drev_replays.py --report /tmp/drev-report.json
uv run python scripts/validate_drev_replays.py --no-committed /path/to/corpus-manifest.json
uv run python scripts/validate_drev_replays.py --no-committed --allow-unfrozen /path/to/exploratory.json
```

Only the native label oracle may inspect concealed hands. The encoded DREV
input remains the original masked public observation. Exact semantic failures
make the command exit non-zero; calibration metrics are reported rather than
treated as correctness assertions on this small CI corpus. Frozen manifests
must include the complete expected-key set and semantic digest. The three
excerpts which intentionally retain the following round's empty
`start_kyoku` header opt in explicitly; a non-empty trailing round is always
rejected.

The default command also loads `../drev_replay/manifest.json`, whose larger
tracked replay stresses riichi and discard-history reconstruction. It is kept
separate because it is not used as an attributed external score oracle.
