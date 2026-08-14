# Refactor Validation Strategy

This document defines the evidence required before replacing RiichiEnv's legacy state, replay, observation, or binding paths. Passing unit tests alone is not sufficient because replay order and feature planes are training ABIs, while riichilab consumes Rust internals directly.

## Test pyramid

### 1. Pure unit and property tests

Run on every change:

- tile, scoring, rule, action encoding, and state-transition units;
- `EventJournal` boundary, cursor, unfinished-EOF, and censorship properties;
- feature buffer length and single-row/batch-row equality;
- DREV-v2 hard-safe proofs, per-seat temporary-furiten expiry, unresolved-
  response exclusion, public-history completeness, and public-only input
  invariants;
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
- base, extended, SP, DREV, combined, and sequence shapes/dtypes;
- the DREV-v2 schema ID, all 81 ordered channel names, relative-opponent slot
  order, and the inactive sanma slot across Rust/Python/WASM;
- selected semantic feature cells plus a deterministic full-vector digest;
- the historical base-v0/extended-v0 channel-30 difference after a called meld.

Frozen encoders must compare every float or documented-tolerance value against
their committed v0/v1 goldens. New versioned encoders need independent semantic
oracles and their own digests; comparing DREV v2 to v1 would preserve the v1
defects rather than validate the new contract. Shape-only checks do not protect
trained models.

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
- Observation → base/extended/SP/DREV and the 402/474-channel combined
  features;
- DREV v1/v2 separately, including 4P/3P, history length, batch 1/16/256, and
  caller-buffer reuse;
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
- explicit v0/v1/v2 `FeatureSpec` constants and caller-buffer batch writers;
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
- versioned 81-channel 4P/3P DREV-v2 encoders with full tedashi/tsumogiri
  discard history, resolved same-turn and post-riichi safety proofs, soft-only
  suji/kabe evidence, 19 evidence heads per relative opponent, and explicit
  474-channel combined bundles;
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
  `sanchaho_is_draw` switch), 3P double ron, and 4P ron/pon/chi priority, with
  the multi-ron terminal logs round-tripped through `ReplayLog` and
  `EventJournal`;
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
- fixed-seed forced-policy 4P/3P East games that prefer legal scoring and call
  actions over discard/pass, guaranteeing coverage for reach/reach-accepted,
  pon, every kan form, 4P chi, and 3P kita while checking exact called-tile
  ownership, reach-discard flags, dora order, score continuity, and replay
  determinism;
- Python and real Node/WASM fixed-seed East-game smoke tests that exercise the
  shipped engine facade through completion and validate replay/journal output;
- differential SP yaku/fu checks against `HandEvaluator` over the committed
  winning-hand corpus, explicit kan/open-hand fallback checks, and exact
  without-replacement probability oracles for tenpai and one-shanten paths;
- four attributed external-result differentials: Tenhou 4P call/Ron, dealer
  tsumo, and terminal honba/kyotaku rounds plus a Mahjong Soul 3P
  Kita/double-Ron round. They cover normalized actions, winner/target/tile,
  han/fu/yaku values, Ron and split-tsumo payments, recorded deltas, end scores,
  next-round metadata, and the no-next-round terminal path;
- independent physical-copy, without-replacement SP oracles for 4P/3P s2 and
  s3 small walls. Every miss is removed from the wall and every discard is
  enumerated; all reachable tenpai/win prefixes are compared with the
  versioned prefix-corrected calculator, while the frozen ABI's terminal cell
  is checked against the same oracle.

## Latest replay-corpus feature validation

On 2026-08-09, `scripts/validate_sp_features.py` was run against both local
MjSoul corpora after rebuilding the Python extension. The validator checks the
compact/canonical tile mapping, all SP shape/yaku/point ranges, binary planes,
probability monotonicity, `win <= tenpai`, shanten lower bounds, drawable waits,
min/mean/max and point-threshold ordering, frozen DREV-v1 range, and exact
standalone versus 402-channel combined block equality. It predates DREV v2 and
is not evidence for the 81-channel contract.

| Variant | Files | Sampled observations | Issues |
|---|---:|---:|---:|
| 4P | 17 | 500 | 0 |
| 3P | 25 | 474 | 0 |

Commands used `--sample-every 20`, `--max-files 25`, and
`--max-observations 500` for each corpus. This is a broad semantic sample, not
a replacement for full-corpus validation before a release.

## External result and deeper SP validation

The committed external replay manifest lives under
`tests/data/external_replay/`. It records immutable upstream revisions,
licenses, transformations, source hashes, normalized action hashes, provider
results, score transitions, and next-round metadata. CI validates four small
anonymized excerpts; larger local manifests remain an explicit pre-release
job:

```bash
uv run python scripts/validate_external_replays.py
uv run python scripts/validate_external_replays.py --no-committed /path/to/manifest.json
```

The replay path now retains external yaku values and Mahjong Soul terminal
score deltas, selects `HandEvaluator` or `HandEvaluator3P` from the variant,
and rejects unknown yaku labels instead of silently dropping them. The
validator composes separately emitted `reach_accepted` deposits into the round
delta and checks the resulting in-hand kyotaku count against the calculator
context. The committed fixtures exercise Tenhou 4P call/Ron, dealer tsumo
with split payments, and a terminal non-zero-honba/kyotaku Ron result, plus a
Mahjong Soul 3P Kita/double-Ron result. They are intentionally not a claim of
exhaustive platform parity; pao, abortive draws, non-dealer tsumo, and rare
limit-hand results remain appropriate additions to the opt-in corpus.

Adapter regressions additionally require every winner in a double/triple-Ron
batch to produce a decision sample, distinguish declaration from accepted
riichi deposits, preserve exclusive double-riichi/ippatsu/ura state in 4P and
3P, and score Mahjong Soul ron-on-Kita as Ron rather than Tsumo. Public native
replay input rejects empty rounds/results, inconsistent player counts,
out-of-range seats/rounds/winds, malformed call cardinality, and invalid tile
tokens before state replay; these cases must raise a typed error rather than a
Rust panic.

The s2/s3 oracle exposed a historical SP series property: for the frozen 4P
v0 and 3P v1 layouts, the terminal cell is the DP endpoint, but earlier cells
are tail-indexed and can depend on the total horizon. Changing those values in
place would break trained-model ABI. `calculate_sp_v1` and
`calculate_sp_3p_v2` therefore return distinct, low-level
`Sp4PV1Result`/`Sp3PV2Result` diagnostics with version-matched encoding methods,
while regular Observation and batch encoders remain frozen. The corrected path
is substantially slower and retains the documented high-shanten and
missed-tile-identity approximations; it is a correctness reference and feature
experiment, not yet the default inference path.

The next SP implementation task is to build the state-transition graph once
and evaluate all deadlines over it, then consider end-to-end versioned
Observation/batch bindings only after correctness goldens and replay-derived
performance gates are green. Legacy APIs remain supported throughout that
migration.

## DREV v2 correctness and quality gates

DREV v2 is a public-information feature, while a correctness oracle may inspect
the complete game state only to produce labels. Tests must keep this boundary
explicit.

Current deterministic regressions cover both variants where applicable:

- a decoded v0.4.x Observation has no runtime history and DREV v2 fails closed;
- engine observations carry aligned chronological actor, riichi-discard, and
  tedashi/tsumogiri histories;
- the current unresolved normal discard is excluded until the response window
  closes;
- an opponent's own river is permanently hard-safe;
- a resolved normal discard is temporarily hard-safe against each non-
  discarder, remains so while other players act, and expires only when the
  protected seat next draws or calls;
- a resolved post-riichi pass remains hard-safe after later draws because the
  wait is fixed;
- adding or removing a hard-safe proof changes `ron_prob`, but does not erase
  or renormalize the structural `wait_prob`;
- suji and kabe reduce a soft prior but never produce hard-safe zero;
- completed kans, honor triplets, sanma 1m/9m triplets, and four-meld
  pair-wait copy exhaustion produce hard-safe zero, while an ordinary 4P
  numeric Pon alone does not;
- a called physical tile is not double-counted into a false wall;
- an externally supplied opponent concealed hand is rejected;
- the 3P absent opponent slot, including all yaku planes, is zero;
- post-riichi Kita can only extract the just-drawn North;
- all 19 evidence heads use the documented order and ordinal states, including
  normal-yaku/value groups and Kokushi, Daisangen, wind yakuman, Suuankou,
  Tsuuiisou, Ryuuiisou, Chinroutou, Chuuren, and Suukantsu;
- three public kans remain strong Suukantsu evidence, while Chi and exhausted
  dragon/wind blockers produce genuinely impossible yakuman-family states;
  the offered discard is removed from those blocker counts before declaring
  the family impossible;
- three public wind triplets give the missing-wind candidate a confirmed
  yakuman loss floor in both 4P and 3P without broadcasting confirmation to
  unrelated tiles;
- four compatible public melds give only matching pair candidates confirmed
  Tsuuiisou, Ryuuiisou, or Chinroutou loss floors, and compatible confirmed
  families stack;
- candidate-conditioned ordinary-yaku floors cover conclusive flush,
  terminal/honor, Tanyao, and Shousangen shapes, while public Sanankou,
  Sankantsu, and Sanshoku doukou are counted independently;
- replay action dora snapshots are visible in the immediately following 4P/3P
  Observation, including kan/Kita snapshots and the legacy incremental
  singular-marker input;
- cumulative dora snapshots cannot shrink or rewrite an earlier indicator;
  malformed updates leave the prior public indicator list unchanged and make
  DREV v2 fail closed;
- positive-but-unconfirmed evidence does not act as a probability, add han, or
  lift yakuman loss; only public confirmation and exact public bonus counts do;
- the 81-channel standalone and 474-channel combined layouts match their
  exported schema metadata and single-row/batch output;
- caller-owned DREV and combined feature buffers are fully cleared before
  reuse, so no positive cell survives from a prior observation.
- the hidden-state exact-zero integration oracle runs four deterministic seeds
  for both 4P and 3P, enumerates every legal physical discard against every
  opponent, and currently checks 17,920 candidate/opponent cells including
  3,871 hard-safe cells with zero false positives.

The seeded integration oracle validates the exact-zero rule claim, not
statistical quality or rare-event coverage. The corpus-quality oracle must
extend the same hidden-state boundary to record exact waits, legal Ron,
canonical yaku and yakuman IDs/units, han/fu, bonus components, points, and
player deltas. Only the labels may inspect concealed hands, wall state, ura
indicators, or legal responses; the encoded input must be generated from the
original public Observation. Train/calibration/test splits are by game, not by
observation, to avoid adjacent-position leakage.

Promotion from experimental use requires that labelled corpus and the following
reported metrics:

1. `hard_safe_zero` false-positive count, which must be exactly zero;
2. legal-Ron log loss, Brier score, AUPRC, and adaptive ECE;
3. conditional-loss and expected-loss MAE, bias, and tail-weighted error;
4. within-decision candidate ranking regret and paired-seed policy return;
5. separate 4P/3P, riichi/dama/open, turn, dealer, red, Kakan/Ankan/Kita, pao,
   and multi-Ron slices;
6. per-family support and recall for rare yakuman rather than only a micro
   average.

Normal-discard same-turn-furiten epochs are now represented by per-seat runtime
masks. The next history gate is a versioned public-event reducer shared by live
and replay paths. It must type Draw, Call, Kakan, Ankan, Kita, DoraFlip,
ReachAccepted, and OfferResolved without exposing private response state. Only
then may DREV claim source-specific Kakan/Ankan/Kita response proofs. See ADR
0002.
