# Refactor Validation Strategy

This document defines the evidence required before replacing RiichiEnv's legacy state, replay, observation, or binding paths. Passing unit tests alone is not sufficient because replay order and feature planes are training ABIs, while riichilab consumes Rust internals directly.

## Test pyramid

### 1. Pure unit and property tests

Run on every change:

- tile, scoring, rule, action encoding, and state-transition units;
- `EventJournal` boundary, cursor, unfinished-EOF, and censorship properties;
- feature buffer length and single-row/batch-row equality;
- 4P/3P action masks and typed game-mode validation;
- reset rejection before mutation, deterministic reseeding, complete pending-action batches, and terminal decision suppression.

Important journal invariant for every appended prefix:

```text
safe_prefix(delay=1)
  ends before a withheld start_game/end_game or exactly at end_kyoku,
  never contains the current start_kyoku,
  is monotonic and idempotent,
  concatenated deltas equal the same prefix,
  and malformed/unframed input permanently fails closed.
```

Cover normal wins, exhaustive and abortive draws, chombo, kan/chankan, kita, double/triple ron, and 3P boundaries.

### 2. External compile contracts

Compile tests must import RiichiEnv as a downstream crate rather than using private modules.

- Current riichilab consumer: `GameState`, `GameState3P`, Observation types, Action/Phase, constructors, methods, and public status fields.
- New consumer: `GameEngine`, `EngineConfig`, snapshots, action masks, batch decisions, and `EventJournal`.
- Python stub and package-root import snapshot.
- WASM build plus generated TypeScript declarations and typed-array feature lengths.

The legacy compile contract stays until riichilab has migrated and its minimum dependency version has advanced.

### 3. Deterministic lifecycle tests

For fixed seeds in both variants, record and compare after every decision:

- phase, current and active players;
- legal action IDs and selected action ID;
- appended canonical events;
- scores, round metadata, wall count, and terminal state;
- each seat's `new_events` cursor behavior;
- next-kyoku initialization and `end_game` ordering;
- illegal-action legacy reason and score/chombo result.

WaitResponse cases must submit a complete multi-player action map. The new and legacy paths must produce identical traces.

### 4. Wire and model-ABI goldens

Committed, reviewable fixtures are required for:

- `contracts/v0.4.8/observation_{4p,3p}.b64` decode and byte-for-byte legacy re-encode;
- all ActionType → MJAI forms, actors, red fives, consumed-tile order, parser mapping, and current `reach` omission of `pai`;
- action masks and every action ID mapping (82/60);
- base, extended, SP, DREV, and sequence shapes/dtypes;
- selected semantic feature cells plus a deterministic full-vector digest.
- the historical base-v0/extended-v0 channel-30 difference after a called meld.

New encoders must compare every float or documented-tolerance value against v0. Shape-only checks do not protect trained models.

### 5. Replay corpus validation

The small tracked MJAI corpus runs in CI. For every replay decision it checks:

- the recorded action is legal and selected by the observation;
- mask and legal actions agree;
- every feature encoder has the documented shape and finite values;
- win actions agree with waits and shanten;
- MJAI action round-trips;
- scores are continuous and conserved across hands.

The CLI must return non-zero when any file fails. Larger Tenhou/MJSoul corpora remain an opt-in pre-release job and should report decision count, first mismatch, and a stable trace digest.

Add small redistributable fixtures for 4P, 3P, reach splitting, calls, kan/kita, multiple ron, chombo, and truncated/live logs. MJSoul conversion needs its own explicit loader tests rather than relying on MJAI fixtures.

### 6. Cross-language integration

- Rust → Python and Rust → WASM Observation payload decode.
- Python/WASM/TypeScript journals receive the same event prefixes and produce identical cursor positions and completed-kyoku spans.
- Journal fixtures cover decimal/exponent integer forms, JavaScript safe-integer bounds, missing/null keys, recursive duplicate keys, and untrusted completion markers.
- A browser test uses the real WASM package at least once; mock-only viewer tests are insufficient.
- Core, Python, WASM, and TypeScript schema-v1 deltas agree that `events` is raw JSON and use the same camelCase envelope.
- riichilab smoke test starts a game, exchanges actions, publishes safe replay deltas, completes the game, and falls back from active to persisted log without a 404 race.

### 7. Performance gates

Performance comparisons use release builds, warm/cold results where relevant, and fixed fixtures whose post-discard shanten is asserted before measurement.

Record:

- SP s0/s1/s2/s3 and approximation paths;
- closed/open, aka, dora, kan, and replay-derived positions;
- allocation count and requested bytes per call;
- Observation → base/extended/SP/DREV combined features;
- complete engine decision cycles and batches 1/16/256;
- steps/s and p50/p95 latency.

As of 2026-08-08 on arm64, short diagnostic measurements show roughly 17.6 µs (s0), 20.8 µs (s1), 136 µs (s2), and 6.6 ms (s3). These are diagnostic baselines, not portable pass/fail thresholds. The corrected benchmark fixtures and their shanten assertions are the contract.

The initial borrowed `FeatureContext` benchmark on the same machine takes
about 254 ns to validate and build. Constructing SP and DREV inputs through one
shared context took about 241 ns versus 264 ns through two independent
preprocessing passes. This is a small but measurable reduction; SP recursion,
especially true s2/s3 positions, remains the dominant cost.

Optimization order is:

1. borrowed evaluator calls and shared `FeatureContext`;
2. count-delta discard candidates and caller-owned buffers;
3. reusable SP probability/cache/arena workspace;
4. action-ID batch stepping with the Python GIL released;
5. parallel scheduling only after deterministic serial batch parity.

## Merge gates by migration phase

| Phase | Required evidence |
|---|---|
| Additive facade/journal | full Rust/Python/UI suite; pure-core build; riichilab compile contract; live censorship tests |
| Binding adoption | Python/WASM/TS parity; generated declarations/stubs; real WASM integration |
| Borrowed feature path | complete v0 vector differential; replay corpus digest; allocation benchmark |
| Replay cursor/decision trace replacement | decision-by-decision legacy differential on 4P/3P corpus |
| Player-state consolidation | full lifecycle differential, scoring/rule suite, replay trace digest |
| Legacy deprecation/removal | riichilab migrated; documented minimum versions; at least one deprecation release |

## Foundation currently present

- pure Rust base/extended/SP/DREV Observation entry points;
- explicit v0/v1 `FeatureSpec` constants and caller-buffer batch writers;
- typed `GameEngine`/`BatchGameEngine` facade;
- pure append-only `EventJournal` and schema-v1 cursor envelope;
- synthetic journal censorship/property tests;
- fail-closed malformed-stream and current-draw concealment regressions;
- v0.4.8 4P/3P Observation wire fixtures;
- called-meld feature differential and reset/step validation regressions;
- external riichilab-style compile contract;
- tracked replay validator test and failure exit status;
- corrected SP benchmark fixtures with post-discard shanten assertions;
- Python, WASM typed-array, and TypeScript adapters over the new seams.
- generated and package-exported TypeScript declarations for the browser APIs.
- real-browser smoke coverage for the bundled web-target WASM loader.
- pure Rust `ReplayLog`/borrowed kyoku cursor with tracked-corpus coverage;
- shared 4P/3P `FeatureContext` preprocessing for combined extended/SP/DREV;
- 3P SP/DREV contracts for the compact 27-tile axis, two-opponent DREV
  normalization, absent-manzu suppression, and sanma dora wrapping.
- public-`GameEngine` rule regressions for 4P chi/pon and 3P pon, including
  `kuikae_forbidden` ON/OFF and live/replay parity;
- public-`GameEngine` kan lifecycle coverage for ankan, daiminkan, and kakan in
  both variants, including rinshan draws, dora timing, ippatsu cancellation,
  the post-rinshan discard boundary, and kakan interrupted by chankan;
- rule-edge integration matrices for kokushi chankan on ankan, riichi-time
  wait-preserving ankan, fourth-kan ownership, and red-five chi/pon kuikae;
- complete multi-responder action batches for 4P triple ron (including the
  `sanchaho_is_draw` switch), 3P double ron, and 4P ron/pon/chi priority;
- public-`GameEngine` pao establishment when the final dragon or wind set is
  completed by daiminkan, for daisangen/daisuushii in both variants;
- last-live-tile call coverage from pon through the final draw and discard,
  including called-discard nagashi invalidation, ordinary exhaustive draw,
  `end_kyoku`, and next-kyoku initialization ordering;
- 4P/3P nagashi mangan checks for eligible terminal/honor discards and
  invalidation by a non-terminal discard or a call;
- two-kyoku 4P/3P JSONL integration coverage shared by `ReplayLog` and
  `EventJournal`, including score continuity, typed action order, cursor seek,
  completed spans, and whole-kyoku spectator delay;
- table-driven 4P/3P exhaustive-draw settlement for every tenpai count and
  both dealer-continuation branches, with exact deltas, honba, kyotaku, and
  next-kyoku assertions;
- composite Daisangen + Tsuuiisou Pao settlement across 4P/3P, Ron/Tsumo, and
  Tenhou/Mahjong Soul liability policies, including honba and kyotaku;
- configurable nagashi-mangan round semantics for win-style and draw-style
  rules, plus last-live-draw Ankan/Kakan prohibition and the tedashi Ron-tile
  regression;
- fixed-seed 4P/3P East and half-game simulations that run to `end_game`
  twice, compare every pending action mask and appended event suffix, then
  round-trip the complete log through `ReplayLog` and `EventJournal` with
  action-order, score-continuity, truncated-prefix, and corrupt-actor checks;
- Python and real Node/WASM fixed-seed East-game smoke tests that exercise the
  shipped engine facade through completion and validate replay/journal output;
- differential SP yaku/fu checks against `HandEvaluator` over the committed
  winning-hand corpus, explicit kan/open-hand fallback checks, and exact
  without-replacement probability oracles for tenpai and one-shanten paths.

## Latest replay-corpus feature validation

On 2026-08-09, `scripts/validate_sp_features.py` was run against both local
MjSoul corpora after rebuilding the Python extension. The validator checks the
compact/canonical tile mapping, all SP shape/yaku/point ranges, binary planes,
probability monotonicity, `win <= tenpai`, shanten lower bounds, drawable waits,
min/mean/max and point-threshold ordering, DREV range, and exact standalone vs
combined block equality.

| Variant | Files | Sampled observations | Issues |
|---|---:|---:|---:|
| 4P | 17 | 500 | 0 |
| 3P | 25 | 474 | 0 |

Commands used `--sample-every 20`, `--max-files 25`, and
`--max-observations 500` for each corpus. This is a broad semantic sample, not
a replacement for full-corpus validation before a release.

The next integration priorities are engine-generated policies that deliberately
exercise calls, riichi, kan/kita, and multiple-Ron paths inside a full game;
external Tenhou/Mahjong Soul result differentials; and deeper SP s2/s3
probability oracles. These remaining items are migration gates, not permission
to remove legacy APIs.
