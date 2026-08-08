# ADR 0001: Stage Engine, Replay, and Feature Boundaries

- Status: Accepted
- Date: 2026-08-08
- Deciders: RiichiEnv maintainers

## Context

RiichiEnv currently serves several workloads through overlapping types:

- deterministic 4-player and 3-player game simulation;
- online MJAI bots and the riichilab game server;
- reinforcement-learning rollout and feature generation;
- static replay parsing and training decision traces;
- Python, Rust, WASM, and browser viewers.

The concrete `GameState` and `GameState3P` types expose mutable internals. Python owns another dispatch layer, replay construction depends on Python-only code paths, and the browser has a separate state reducer. Observation JSON/base64 fields, action IDs, feature channel order, and MJAI JSONL are already external model and server ABIs even though most are not versioned.

riichilab 0.4.8 directly uses concrete state fields, constructor side effects, per-seat `new_events` cursors, `Action::to_mjai()`, and the legacy illegal-action chombo message. A refactor that only passes RiichiEnv's unit tests can therefore still break a live server or silently change a trained model's inputs.

Live spectating adds a security boundary: while kyoku N is in progress, spectators may see data only through the `end_kyoku` of kyoku N-1. The current hand's `start_kyoku`, concealed hands, draw, and wall must never enter the public prefix.

## Decision drivers

- Preserve current Rust, Python, and riichilab behavior during migration.
- Make the normal RL path typed, deterministic, batchable, and able to disable logging.
- Make replay parsing and live append independent of Python and filesystem paths.
- Use one canonical MJAI event stream across server, replay, stats, and viewer code.
- Treat action IDs, serialized observations, and feature planes as versioned ABIs.
- Avoid a big-bang generic rewrite of the different 4P and 3P state machines.
- Measure hot paths before changing SP/DREV algorithms or adding parallelism.

## Considered options

### A. Replace both state machines with one generic engine immediately

This could eventually remove substantial duplication. It also combines rule migration, state-machine migration, replay changes, feature changes, and binding changes into one correctness boundary. Existing tests do not yet provide enough cross-consumer or decision-by-decision evidence for that risk.

### B. Add stable seams and migrate consumers incrementally

Keep the concrete state machines as compatibility implementations, then introduce a typed engine facade, pure feature encoders, and an append-only event journal. Bindings become adapters over those seams. Common player/domain components can move only after differential contracts are green.

### C. Improve only Python and JavaScript wrappers

This is initially cheap but leaves replay and feature semantics duplicated across languages. It also cannot provide a clean Rust or WASM API and does not remove cloning from the hot path.

## Decision

Choose option B.

The target dependency direction is:

```text
4P state ─┐
          ├─> GameEngine / GameSnapshot / StepOutcome
3P state ─┘                    │
                              ├─> versioned feature encoders
                              └─> canonical MJAI events
                                          │
MJAI bytes/events ─> EventJournal ─────────┼─> replay cursor / decision trace
                                          └─> spectator delay policy

                  Rust / Python / WASM / JS adapters
```

### 1. Engine facade

New consumers use `engine::GameEngine` with a validated `GameMode` and `EngineConfig`. The facade owns 4P/3P dispatch and returns:

- variant-independent observations and action masks;
- `GameSnapshot` rather than public-field polling;
- `StepOutcome` with appended events and a typed legacy error;
- an on-demand `EventJournal` when logging is enabled.

`EventLogPolicy::Off` removes MJAI serialization from rollout hot paths. `BatchGameEngine` initially provides deterministic serial orchestration and flat `(environment, player)` decisions. Parallel scheduling remains a caller/worker concern until serial batch semantics and measurements are stable.

Reset inputs are validated before either state machine is mutated: dealer and round ranges, score count, and the exact physical 4P/3P wall permutation. Supplying a seed restarts the wall sequence deterministically. `GameEngine::step` requires exactly one action for every pending player; the legacy implicit-pass behavior remains available only through the concrete compatibility states.

`GameState`, `GameState3P`, their public fields, and the existing Python `RiichiEnv` remain compatibility APIs. They are not removed in this decision.

### 2. Feature ABI

Observation feature calculation is pure Rust and available without the Python feature flag. `features` publishes immutable `FeatureSpec` values and caller-buffer batch writers. Existing layouts are explicitly `v0`:

| Layout | Shape per observation | Action space |
|---|---:|---:|
| 4P base | `74 × 34` | 82 |
| 4P extended | `215 × 34` | 82 |
| 4P SP | `178 × 34` | 82 |
| 4P DREV | `9 × 34` | 82 |
| 4P extended + SP + DREV | `402 × 34` | 82 |
| 3P base | `74 × 27` | 60 |
| 3P extended | `215 × 27` | 60 |
| 3P SP | `178 × 27` | 60 |
| 3P DREV | `9 × 27` | 60 |
| 3P extended + SP + DREV | `402 × 27` | 60 |

Changing a channel's meaning, order, normalization, tile axis, or dtype requires a new feature version. Existing model ABI is not changed in place.

Public single-row and batch encoders validate externally constructible Observation DTOs and return `RiichiResult`; private unchecked writers are used only after validation. A malformed player ID, tile/meld shape, or impossible hand therefore fails at the API boundary instead of panicking inside a feature loop.

The 3P SP/DREV calculators retain canonical 34-tile IDs internally, exclude
2m through 8m from wall/progression/risk calculations, and compact results to
the same 27-column tile axis as the other 3P feature blocks. Sanma DREV uses
two active-opponent slots and leaves the third slot zero. Because the frozen
3P Observation payload has no kita count, SP score projection cannot include
nukidora or subtract set-aside North tiles from the unseen wall until a
versioned observation schema adds that field.

One historical inconsistency is itself part of v0: after a called meld, base channel 30 counts the called tile in both the discard and meld, while the first 74 channels of extended-v0 subtract that duplicate. The dedicated base encoder preserves this difference and contract tests include a called-meld fixture.

The compatibility Observation remains an owned DTO. A borrowed, per-decision
`FeatureContext` now shares hand counts, visible-tile counts, red-five flags,
and discard candidates across the 4P and 3P extended/SP/DREV combined paths. Base, SP,
and DREV remain separate encoders over that context, and complete v0 vector
tests protect their existing meanings. A fuller borrowed `PlayerView` can be
introduced later when state-owned observations can avoid DTO construction too.

### 3. Replay and live journal

`replay::EventJournal` is the canonical append-only MJAI JSONL boundary. It accepts text, any `BufRead`, or individual raw events; preserves unknown JSON fields and original event text; indexes only explicit `start_kyoku`/`end_kyoku` pairs; and never treats an unfinished EOF as a completed hand.

Cursor envelopes use schema version 1 and half-open event positions. Filesystem paths, gzip, HTTP caching, and WebSocket transport belong in adapters.

For a spectator delay of one kyoku:

- before the first `end_kyoku`, expose only safe prelude metadata such as `start_game`;
- while kyoku N is running, expose exactly through kyoku N-1's `end_kyoku`;
- between hands, continue withholding the newly completed hand until the next `start_kyoku` establishes the one-hand delay;
- after a structurally valid `end_game`, expose the full log;
- never include the withheld `start_kyoku` or anything after it.

The full-log release applies only to a stream with exactly one initial `start_game`, at least one completed `start_kyoku`/`end_kyoku` pair, and no permanent censorship violation. Standard shuffle seeds are retained but withheld until that completion. Missing or duplicate game starts, unknown gameplay outside a kyoku, recursive duplicate JSON keys, unsafe/unknown `start_game` fields, and unknown `end_game` extensions fail closed. Known public final `scores`/`ranks` vectors remain allowed. A later event merely named `end_game` cannot turn a malformed or truncated feed into a concealed-information disclosure.

JSON metadata follows JavaScript safe-integer semantics in every adapter: integral lexical forms such as `1.0` and `2e0` are accepted, while values outside ±(2^53−1) are not trusted. Raw accepted text remains unchanged. Live adapters must still feed the journal only authoritative engine/server events; structural validation cannot authenticate an attacker who forges an otherwise complete game transcript.

The engine state and the spectator policy are separate. Server code must publish a journal prefix, never a `GameSnapshot`, as the spectator replay payload.

### 4. Bindings

- Python exposes `GameEngine`, serial `BatchGameEngine`, contiguous batch feature buffers, and the same journal cursor semantics while retaining `RiichiEnv`, `MjaiReplay`, aliases, and current dict returns. Rust transitions run with the Python GIL released.
- WASM exposes the stateful engine through pending action IDs only. It resolves IDs against cached observations and returns base/extended `Float32Array` features plus a `Uint8Array` mask; JavaScript never constructs raw Rust actions.
- Core, Python, WASM, and TypeScript schema-v1 deltas use camelCase envelope fields and canonical raw JSON strings in `events`. TypeScript additionally exposes a parsed convenience view.
- TypeScript provides a high-level append-only timeline with the same fail-closed censorship policy, and `LiveViewer` accepts the shared raw-event delta. The npm package emits and exports declarations for these APIs.

### 5. Compatibility policy

The following are frozen until explicit versioned replacements and migration gates exist:

- action IDs 0–81 (4P) and 0–59 (3P);
- v0 feature shapes, order, values, and float32/uint8 dtypes;
- Observation base64 as base64(JSON serde), including current private-looking field names;
- per-seat `get_observation`/`new_events` cursor behavior;
- canonical MJAI JSONL event names and fields;
- `Action::to_mjai()` output, including the current omission of `pai` on `reach`;
- legacy illegal action conversion to a chombo with `Error: Illegal Action by Player N`;
- replay decision ordering, synthetic pass behavior, and `skip_single_action` defaults.

The versionless Observation wire is fragile. New wire protocols should use an envelope and schema version; old payload generation and decoding remain available as `v0.4.8` compatibility behavior.

The new action-ID facade deduplicates collisions and selects an equivalent physical action with the lowest red-five consumption cost. This policy is additive and does not alter the concrete Observation methods. The already-published `Observation::find_action` first-match behavior and the repository's current non-red discard preference differ for some saved observations; reproducing old decision traces therefore requires an explicitly versioned selector rather than another silent change.

## Consequences

### Positive

- Rust, Python, and WASM can share replay and feature semantics.
- RL callers gain strict modes, typed snapshots/errors, disabled logging, and batch-oriented buffers without waiting for state-machine unification.
- riichilab can migrate off public internals incrementally.
- Live replay censorship becomes a testable pure function over an append-only log.
- Existing models and replay tools get explicit regression fixtures.

### Negative

- Legacy and new APIs coexist for multiple releases.
- 4P/3P dispatch and some player-state duplication remain temporarily.
- On-demand journal construction reparses retained engine events; a synchronized journal sink may be introduced after semantics are stable.
- The first batch facade still constructs owned observations and is not the final maximum-throughput API.

### Risks

- A legacy state mutation through `state_mut()` can bypass facade invariants.
- Raw MJAI is forward-compatible at the journal layer, but typed replay parsers may still reject future fields/types.
- A fast borrowed feature path can diverge from the owned Observation encoder unless differential tests compare every row.

## Follow-up actions

1. Keep compile contracts for riichilab's current imports and public fields.
2. Add deterministic 4P/3P lifecycle and multi-response differential traces.
3. Keep committed Observation, action, feature, and replay fixtures.
4. Move the remaining Python-only decision-step iterator behind the pure
   `ReplayLog`/`ReplayCursor` without changing training semantics.
5. Expand `FeatureContext` into a state-borrowed `PlayerView` only after full
   engine-to-feature differential traces exist.
6. Add reusable `SpWorkspace` only after allocation and replay-derived SP benchmarks are established.
7. Add action-ID/caller-buffer batch APIs and release the Python GIL around pure batch computation.
8. Migrate riichilab to the facade/journal, then deprecate direct public-field access in a later major release.

## Open questions

- Whether a versioned legal-action wire should include the discard tile on `reach`; the current v0 output must not change silently.
- The exact API/name for a v0.4.8 first-match action-ID selector needed by historical decision traces.
- Capacity bounds and ownership model for reusable SP workspaces after pathological 3-shanten calls.
