# ADR 0002: Version DREV Around Public History and Risk Evidence

- Status: Accepted for experimental use
- Date: 2026-08-14
- Deciders: RiichiEnv maintainers

## Context

DREV v1 is a frozen nine-plane heuristic. It is inexpensive, but several of
its quantities are not strong enough to be interpreted as deal-in
probabilities or expected loss:

- suji and ryanmen kabe can reduce the aggregate threat to zero even though
  tanki, shanpon, kanchan, chiitoitsu, and kokushi waits remain possible;
- the tenpai prior is hand-written and uncalibrated;
- a called discard can be counted once in the river and once in the caller's
  meld, creating a false four-tile wall;
- the sanma Kita correction changes the visible North count, but the v1 kabe
  calculation only reads numbered-suit counts, so it does not express the
  opponent's additional value;
- neither the complete chronological tedashi/tsumogiri history nor resolved
  no-Ron opportunities are available to the encoder.

The strongest exact inference available to a public-information agent is not a
statistical suji rule. It is a furiten proof. A tile in an opponent's own river
cannot be Ron'd by that opponent. Likewise, after an opponent has established
riichi, a later public discard that passes a complete Ron response window is
safe against that opponent for the rest of the hand, because the wait is
fixed. Before riichi, the same resolved pass proves safety only until that
opponent next changes its hand by drawing or calling. Before the response
window resolves, no such conclusion is allowed.

At the same time, a risk feature needs value information. A low-probability
yakuman deal-in is not interchangeable with a low-value hand. Public calls,
riichi, dora, Kita, and visible blockers provide useful evidence for common
yaku and yakuman families, although they do not reveal the opponent's exact
hand.

Observation base64 and DREV v1 are already model and wire ABIs. They cannot be
silently redefined.

## Decision drivers

- Never turn suji, kabe, or a learned prior into a claim of zero Ron risk.
- Use every public tedashi/tsumogiri record without reading concealed hands,
  the wall, ura indicators, or opponent legal actions.
- Keep v0.4.x Observation base64 and DREV v1 byte/shape compatible.
- Fail closed when complete public history is unavailable.
- Give yaku and yakuman evidence explicit, reviewable semantics.
- Support the same contract in 4P and 3P, including Kita value.
- Keep the first implementation cheap enough for rollout feature generation,
  while leaving calibration and a richer event model as separate work.

## Decision

### 1. Freeze DREV v1 and add DREV v2

DREV v1 remains `9 x 34` in 4P and `9 x 27` in 3P. DREV v2 is a distinct
feature specification and API:

| Variant | FeatureSpec | Shape |
|---|---|---:|
| 4P | `DREV_4P_V2` | `81 x 34` |
| 3P | `DREV_3P_V2` | `81 x 27` |
| 4P combined | `EXTENDED_SP_DREV_4P_V2` | `474 x 34` |
| 3P combined | `EXTENDED_SP_DREV_3P_V2` | `474 x 27` |

The 81 channels comprise 24 risk channels and three relative-opponent blocks
of 19 public yaku/value/yakuman evidence channels. The stable schema identifier
is `riichienv.drev_v2.81ch.v1`. Ordered channel names and the opponent-slot
order are exposed by Rust, Python, and WASM so bindings do not duplicate the
layout by hand. Relative slot N maps to `(observer + N) % num_players`; the
third slot is an all-zero compatibility slot in sanma.

The existing 402-channel combined layouts remain frozen. Consumers opt in to
the new 474-channel bundle explicitly: 4P uses extended-v0 + SP-v0 + DREV-v2,
and 3P uses extended-v0 + Kita-aware SP-v1 + DREV-v2. This preserves existing
models while still providing one-call and batch encoders for new models.

### 2. Separate hard proof from soft estimation

Each of three relative-opponent slots has six risk channels:

1. `hard_safe_zero`;
2. `wait_prob`;
3. `ron_prob`;
4. conditional loss normalized by 100,000 points;
5. conditional loss normalized by 30,000 points;
6. uncertainty.

The next six channels aggregate active opponents: all-safe, maximum Ron prior,
two summed expected-loss scales, the minimum-loss legal-discard marker, and
maximum uncertainty. The third opponent slot is all zero in sanma.

`hard_safe_zero` is set only by a rule proof:

- the tile type is in that opponent's own river; or
- the tile passed a resolved normal-discard Ron window after that opponent's
  riichi declaration had resolved; or
- the tile passed a resolved normal-discard Ron window without a win and that
  opponent has not drawn or called since. This is a same-turn-furiten proof and
  may apply before riichi;
- a completed public kan has consumed all four copies, or a public triplet
  cannot reuse that type in a sequence (honors in both variants and 1m/9m in
  sanma); or
- after four public melds, at least three non-offered copies are visible, so
  the opponent cannot already hold the matching tile needed for the only
  remaining pair wait.

An unresolved latest discard is never marked safe. Suji, kabe, recency, and
tedashi/tsumogiri history modify only the soft wait prior. A kabe therefore
reduces ryanmen likelihood but leaves a positive risk for other wait shapes.
Hard safety does not remove the tile from wait-prior normalization and does not
zero `wait_prob`; it sets `ron_prob` to zero. This preserves the distinction
between a structurally plausible wait and a rule-proven inability to win by
Ron at this decision. A public numeric Pon is not generally sufficient in 4P:
the fourth copy can still complete a sequence in the concealed part, so it
remains soft risk unless another proof above applies.

The temporary mask is cleared independently for a seat on its next draw or
call, including hand-changing kan/Kita transitions. A post-riichi resolved
pass is also reconstructed as a persistent mask because the wait is fixed.
Kakan, Ankan, and Kita response windows are not themselves promoted to hard
proof: source and rule differences around chankan need a richer typed public-
event reducer. Omitting those proofs increases uncertainty but cannot create a
false-safe tile.

### 3. Track complete public discard history outside the frozen wire

The live 4P/3P states retain, for the whole hand:

- every discard actor in chronological order;
- every per-seat tedashi/tsumogiri flag;
- every riichi-declaration discard flag;
- the number of normal-discard response windows that have resolved;
- each seat's current same-turn-furiten safety mask; and
- a validity/completeness flag that is cleared on an inconsistent transition.

These fields are attached to an engine-produced Observation as a runtime
sidecar and are skipped by serde. Consequently, current v0.4.x Observation
base64 re-serializes byte-for-byte, and a decoded legacy payload explicitly has
`public_history_complete = false`. DREV v2 rejects such a payload rather than
reconstructing chronology from rivers. DREV v1 remains available for it.

The legacy Rust `GameState` structs expose every field publicly. Retaining the
new runtime reducer in those concrete compatibility states therefore adds
fields to their pre-1.0 struct-literal source surface. The fields remain public
so downstream code can migrate with struct update syntax, while constructors
and the `GameEngine` facade remain the supported creation paths. This is a
Rust source migration for exhaustive literals, not a change to Observation
wire data or to existing public-field reads; removing or privatizing the
legacy fields is deferred to a later major boundary.

Because exhaustive downstream `GameState`/`GameState3P` literals must account
for the added runtime reducer fields, publishing this work requires a
coordinated `0.5.0` version across the Rust/Python/WASM packages. Publishing it
as `0.4.9` would incorrectly place a Rust source migration inside Cargo's
`^0.4.8` compatibility range.

MJAI and Mahjong Soul replay conversion retain the provider's explicit
tsumogiri/moqie flag in replay-private metadata, without adding a required
field to the public legacy action enum. A missing source flag always
invalidates v2 history, even when the physical tile happens to equal the drawn
tile. When the explicit flag is present, a provider's alternate 136-ID for the
same tile type may be used to remove the physical copy because hard safety is
defined on tile types. A contradictory source flag or a missing tile type also
invalidates the runtime sidecar, and DREV v2 then fails closed.

Authoritative cumulative dora snapshots carried by Mahjong Soul draw,
discard, kan, and Kita actions are applied at the same replay boundary as the
action through replay-private metadata. The legacy singular `dora_marker`
compatibility input is treated as an incremental reveal and canonicalized to
the same cumulative form. The next Observation therefore sees the same public
indicators as the live state; malformed, overlong, or sanma-impossible
indicator snapshots fail closed. Because provider snapshots are cumulative,
every update must also retain the complete previous prefix and may never
shrink. A prefix rewrite invalidates v2 history without mutating the already
visible indicators. Legacy records that put the complete round-level list in
`NewRound` but also provide chronological action snapshots use only that
list's first marker as the validation baseline, matching replay initialization.

### 4. Use public-only visible counts

DREV v2 counts the observing player's hand and public rivers, calls, dora
indicators, and Kita tiles. It rejects an Observation that contains an
opponent concealed hand. The called physical tile is omitted from the meld
count because it remains represented in the source river. This prevents three
visible copies from becoming a false four-tile wall.

### 5. Expose major yaku and yakuman evidence

Each opponent has nineteen tile-broadcast evidence channels. Except for public
bonus value, their ordinal meaning is:

```text
0.00 = publicly impossible
0.50 = not ruled out
0.75 = positive public evidence
1.00 = publicly confirmed
```

They are evidence states, not calibrated probabilities.

The normal-yaku/value heads, in schema order, are:

- riichi;
- tanyao;
- yakuhai, including seat and round wind;
- honitsu and chinitsu;
- toitoi;
- chiitoitsu;
- terminal/honor families such as chanta, junchan, and honroutou;
- sequence-family possibility, with fully public ittsu or sanshoku doujun as
  the only current confirmation path;
- publicly visible aka/dora/Kita value.

Closed-hand possibilities such as pinfu and iipeikou are not independently
identified from public state. They remain within the broad sequence-family
possibility head until a hidden-state label schema is introduced.

The nine Ron-capable yakuman-family heads are:

- kokushi;
- daisangen;
- shousuushii/daisuushii as a wind-yakuman family;
- suuankou;
- visible-composition evidence for tsuuiisou, chinroutou, and ryuuiisou;
- chuuren;
- suukantsu.

Kokushi and Chuuren can normally only be marked possible or impossible from
public state. Daisangen, the wind family, Suuankou through four concealed kans,
and Suukantsu have public confirmation paths. Three public kans are strong
Suukantsu evidence rather than an impossible state; any Chi proves it
impossible. Publicly exhausted dragon/wind copies can likewise prove the
corresponding family impossible even after two or three relevant calls. A
candidate tile in the observer's hand is put back before this blocker test:
that copy becomes available to complete the opponent's hand when it is
discarded, so it cannot itself prove the candidate safe from a dragon or wind
yakuman.
Composition heads become strong when every public meld is compatible, but do
not claim broadcast confirmation while concealed tiles remain.

The evidence heads are tile-broadcast, while loss remains tile-conditioned.
In particular, three public wind triplets are only `strong` in the broadcast
wind-family channel, but discarding the missing fourth wind necessarily
completes either Shousuushii or Daisuushii. That candidate therefore receives
a confirmed one-yakuman loss floor and a full Ron-yaku factor without claiming
the same certainty for unrelated tiles. Likewise, four compatible public
melds leave only the pair: a compatible candidate confirms Tsuuiisou,
Ryuuiisou, or Chinroutou for that candidate and compatible families stack.

The ordinary-yaku loss floor is also candidate-conditioned where public state
is conclusive. Four public melds can prove open Honitsu/Chinitsu,
Chanta/Junchan/Honroutou, Tanyao, or Shousangen from the offered pair tile.
Public Sanankou, Sankantsu, Sanshoku doukou, Yakuhai, Toitoi, and fully exposed
Ittsu/Sanshoku doujun contribute deterministic lower-bound han independently.
These exact lower bounds are separate from the ordinal evidence values.

Evidence ordinals never feed the loss calculation as fractional probabilities
or han. Conditional loss is raised only by publicly confirmed ordinary yaku,
exact public bonus tiles, candidate dora, and publicly confirmed yakuman
families. Compatible but unconfirmed yakuman evidence does not increase the
loss estimate. Multiple independently confirmed families contribute one
yakuman unit each; rule-dependent double-yakuman variants and pao liability are
not inferred from this Observation contract. Counted yakuman is reached only
through the ordinary han path and uses the evaluator's single kazoe cap.

Tenhou and chiihou are excluded because they are not Ron risks. Exact
distinctions within grouped families are also outside v2. A future supervised
label schema will use canonical yaku IDs, yakuman IDs/units, han/fu, dora
components, and engine-exact settlement rather than these public evidence
groups.

### 6. Preserve the fixed-wait premise in sanma

After riichi, Kita is limited to the just-drawn North. Extracting a North that
was already held could change the wait and invalidate the persistent no-Ron
proof. A separately signalled riichi declaration must first be completed by
its declaration discard, so Kita is unavailable during that stage.

Sanma keeps the common 81-channel contract, removes 2m through 8m only from
the compact 27-column tile axis, and leaves `relative_3` zero. Public Kita
counts contribute both to visible North and to exact public bonus value; a
West indicator may make a Kita tile both nukidora and ordinary dora. Public
meld evidence also treats a manzu Chinitsu as impossible under the sanma tile
set.

### 7. Keep inference priors explicitly experimental

The v2 wait, Ron, and conditional-loss values are deterministic public-state
priors. They have not been calibrated against an out-of-sample replay corpus
and are not advertised as probabilities or expected-value estimates. The
uncertainty channel and evidence heads make that limitation visible to a model.

The loss head is also intentionally a public lower-information heuristic. It
does not reproduce hidden fu, ura dora, rule-dependent double yakuman, pao, or
multi-Ron settlement. Publicly confirmed yaku and bonus counts prevent ordinal
evidence from being mistaken for value, but they do not make the head engine-
exact.

The exact-zero contract has an engine-labelled integration oracle: across
seeded 4P/3P games it enumerates every legal physical discard candidate and
checks each public hard-safe cell against the authoritative hidden-state Ron
legality. Before v2 is used as a default policy input, that harness must grow
into a replay corpus which also labels canonical yaku/yakuman IDs, points, and
the observing player's engine-exact settlement. The oracle may use hidden
state for labels only; model inputs remain public.

## Consequences

### Positive

- Resolved normal discards enlarge exact same-turn safe sets, and resolved
  post-riichi discards remain safe under the fixed-wait premise, without
  assuming that an opponent behaves rationally.
- Full tedashi/tsumogiri history contributes soft evidence instead of being
  reduced to the last discard.
- Suji and kabe can no longer manufacture a false zero-risk claim.
- Major hand-value and nine Ron-capable yakuman families are represented
  independently of the Ron prior.
- 4P and 3P share one channel contract, with an explicit inactive sanma slot.
- Existing observations and trained v1 models remain usable.

### Negative

- Engine-produced observations carry runtime history that is intentionally not
  portable through the legacy base64 payload.
- The first v2 scans the discard history per observation and uses heuristic
  priors; it is a correctness-oriented intermediate step rather than the final
  sequence model.
- Tile-type planes merge red and normal five candidates even though their deal-
  in values can differ by one han.
- Grouped yaku evidence loses distinctions that an exact supervised target can
  retain.

## Rejected alternatives

### Change DREV v1 in place

Rejected because trained models depend on its exact nine-plane meanings.

### Treat suji or a full wall as completely safe

Rejected because non-ryanmen and special-hand waits remain possible.

### Derive v2 from a decoded legacy Observation

Rejected because per-discard order, response completion, and reliable
tedashi/tsumogiri information are absent. Guessing would turn missing evidence
into false confidence.

### Use opponent concealed hands during inference

Rejected as an information leak. Hidden state is permitted only in offline
labels and differential assertions.

## Follow-up actions

1. Replace the discard-only sidecar with a versioned `PublicEventV1` reducer
   shared by live 4P/3P and replay paths. Include Draw, Call, Kakan, Ankan,
   Kita, DoraFlip, ReachAccepted, and OfferResolved events.
2. Evaluate source-specific Kakan/Ankan/Kita response proofs only after the
   reducer has fail-closed transition tests for chankan and hand-change epochs.
3. Move from tile-type planes to physical action candidates so red fives,
   Kakan, Ankan, and Kita can have distinct risks.
4. Extract one engine-parity Ron/settlement oracle and publish exact structural
   yaku IDs, yakuman IDs/units, han/fu, dora components, points, and player
   deltas as training labels.
5. Calibrate wait, Ron, and loss heads on split replay data; report log loss,
   Brier score, ECE, AUPRC, expected-loss error, candidate ranking regret, and
   rare-yakuman recall separately for 4P/3P and key rule/action slices.
6. Share an incremental bitmask/history workspace if profiling shows the
   current scan materially affects rollout throughput.
