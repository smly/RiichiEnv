# Benchmark Results

This document records the benchmark design and the measurements taken after the
engine/replay/feature-boundary refactor. The two primary targets are the winning
hand/yaku pipeline and model feature encoding.

## Measurement environment

| Item | Value |
|---|---|
| Date | 2026-08-08 (JST) |
| Source revision | `3f4a3839c85736ac60770126d37eb78419db533a`, plus the benchmark-only `feature_bench` harness |
| Machine | MacBook Air, Apple M3, 8 cores (4 performance + 4 efficiency), 24 GB RAM |
| OS | macOS 26.6 (arm64) |
| Rust | `rustc 1.92.0`, LLVM 21.1.3 |
| Cargo | `cargo 1.92.0` |
| Profile | Criterion release/optimized benchmark profile |

The benchmarks ran sequentially to avoid benchmark-to-benchmark CPU
contention. Criterion used its default 3 second warm-up and 100 samples over an
approximately 5 second measurement window. Values below are Criterion's
central time estimate with its 95% confidence interval: the regression slope
when Criterion could compute it, otherwise the sample mean. The results are
wall-clock measurements for this machine, not portable pass/fail thresholds.

Commands:

```bash
cargo bench -p riichienv-core --bench agari_bench -- --noplot
cargo bench -p riichienv-core --bench feature_bench -- --noplot
cargo bench -p riichienv-core --bench sp_bench -- --noplot
```

## Benchmark design

### Winning hand and yaku evaluation

`agari_bench` uses the same committed, correctness-checked fixtures as
`tests/agari_correctness.rs`:

- 816 valid 4-player winning hands;
- 402 valid 3-player winning hands, including kita/nukidora cases;
- 200 non-winning 13-tile hands;
- 28 representative score calculations from low-value hands through multiple
  yakuman and both 4P/3P payments.

Inputs are parsed and `HandEvaluator` instances are constructed before the
timed loop. The benchmark then separates shape detection (`is_agari`), tenpai
search, division enumeration, and the public end-to-end `HandEvaluator::calc`
path. The end-to-end path includes shape detection, dora/aka/ura processing,
division enumeration, yaku and fu evaluation, and final payment calculation.
Because `calc` consumes its indicator vectors and conditions, the repeatable
fixture loop also includes cloning those small inputs.

Each Criterion iteration processes the entire named corpus. The normalized
column divides the corpus estimate by its hand count.

| Benchmark | Work per iteration | Estimate | 95% CI | Normalized |
|---|---:|---:|---:|---:|
| `is_agari/positive` | 816 hands | 29.373 us | 28.916–30.151 us | 36.0 ns/hand |
| `is_agari/negative` | 200 hands | 6.796 us | 6.579–7.059 us | 34.0 ns/hand |
| `is_tenpai` | 200 hands | 130.437 us | 128.976–132.347 us | 652 ns/hand |
| `find_divisions` | 816 hands | 92.525 us | 92.215–92.904 us | 113 ns/hand |
| `hand_evaluator/calc_4p` | 816 hands | 483.579 us | 478.126–491.049 us | 0.593 us/hand |
| `hand_evaluator/calc_3p` | 402 hands | 245.757 us | 243.110–249.077 us | 0.611 us/hand |
| `calculate_score` | 28 score cases | 94.813 ns | 91.552–98.707 ns | 3.39 ns/case |

The end-to-end 4P and 3P yaku/scoring paths are both about 0.6 us per winning
hand on this machine. The 3P normalized result is approximately 3% slower,
consistent with the additional sanma dora and kita handling. These figures do
not include parsing a textual hand or constructing a new evaluator.

### Feature encoding

`feature_bench` obtains observations through the public `GameEngine` rather
than constructing synthetic DTOs. It uses deterministic seeds beginning at
10,000, disables event logging, advances the game by selecting the lowest legal
action ID, and retains 16 progressed observations for each variant. Fixture
construction happens outside the timed loop.

Every measurement encodes the same 16-observation batch and includes public
Observation validation. Two output ownership modes are compared:

- `allocate`: allocate and return a new contiguous `Vec<f32>`;
- `into`: reuse a correctly sized caller-owned buffer.

The covered v0 layouts are:

| Variant | Layout | Shape per observation |
|---|---|---:|
| 4P | base | `74 x 34` |
| 4P | extended | `215 x 34` |
| 4P | SP | `178 x 34` |
| 4P | DREV | `9 x 34` |
| 4P | extended + SP + DREV | `402 x 34` |
| 3P | base | `74 x 27` |
| 3P | extended | `215 x 27` |
| 3P | SP | `178 x 27` |
| 3P | DREV | `9 x 27` |
| 3P | extended + SP + DREV | `402 x 27` |

The timing table below predates the 3P SP/DREV addition and measures the 4P
batch path only. A later benchmark run should add replay-derived 3P positions
without mixing those results into the existing 4P baseline.

#### 4-player feature batch

| Layout / ownership | Batch estimate | 95% CI | Per observation | `into` time reduction |
|---|---:|---:|---:|---:|
| base / allocate | 12.242 us | 12.155–12.331 us | 0.765 us | — |
| base / into | 9.975 us | 9.889–10.090 us | 0.623 us | 18.51% |
| extended / allocate | 704.350 us | 692.561–718.564 us | 44.022 us | — |
| extended / into | 673.193 us | 671.802–674.552 us | 42.075 us | 4.42% |
| SP / allocate | 7.186 ms | 7.156–7.218 ms | 449.101 us | — |
| SP / into | 7.215 ms | 7.178–7.254 ms | 450.908 us | -0.40% |
| DREV / allocate | 9.381 us | 9.356–9.408 us | 0.586 us | — |
| DREV / into | 9.288 us | 9.257–9.323 us | 0.581 us | 0.99% |
| combined / allocate | 7.894 ms | 7.863–7.928 ms | 493.385 us | — |
| combined / into | 7.871 ms | 7.839–7.906 ms | 491.917 us | 0.30% |

#### 3-player feature batch

| Layout / ownership | Batch estimate | 95% CI | Per observation | `into` time reduction |
|---|---:|---:|---:|---:|
| base / allocate | 9.172 us | 9.113–9.251 us | 0.573 us | — |
| base / into | 6.900 us | 6.877–6.927 us | 0.431 us | 24.77% |
| extended / allocate | 647.686 us | 643.786–653.572 us | 40.480 us | — |
| extended / into | 642.606 us | 641.044–644.317 us | 40.163 us | 0.78% |

Caller-owned buffers materially help the lightweight base layouts. They make
little latency difference once shanten/SP computation dominates, although they
still avoid output allocation and reduce allocator pressure in long-running RL
workers.

### SP complexity and encoding overhead

The standalone SP benchmark asserts each fixture's best post-discard shanten
before measurement. This prevents benchmark labels from silently drifting away
from the recursive path they are intended to cover.

| SP benchmark | Estimate | 95% CI |
|---|---:|---:|
| closed, 0-shanten | 17.781 us | 17.536–18.208 us |
| closed, 1-shanten | 21.014 us | 20.883–21.239 us |
| closed, 2-shanten | 129.275 us | 128.886–129.725 us |
| closed, 3-shanten | 6.677 ms | 6.652–6.709 ms |
| closed, 5-shanten light path | 13.672 us | 13.558–13.884 us |
| open 0-shanten with aka and dora | 72.172 us | 71.667–72.931 us |
| encode result, allocate | 1.206 us | 1.200–1.212 us |
| encode result, caller buffer | 0.861 us | 0.859–0.864 us |
| calculate + allocate encode | 18.729 us | 18.695–18.766 us |
| build shared `FeatureContext` | 250.259 ns | 249.095–251.455 ns |
| build SP and DREV inputs independently | 258.107 ns | 256.732–259.487 ns |
| build SP and DREV inputs from one context | 241.913 ns | 240.892–243.040 ns |

The 3-shanten recursive case is about 52 times slower than the 2-shanten case
and 318 times slower than the 1-shanten case. In the progressed 4P feature
corpus, SP accounts for approximately 92% of combined encode time. SP
calculation is therefore the primary feature-generation optimization target;
output allocation and the shared preprocessing context are secondary.

The caller-buffer SP result encoder is 28.56% faster than the allocating result
encoder, but result encoding itself is only about 1 us. Sharing one
`FeatureContext` reduces SP+DREV input-preparation time by 6.27%, also a small
absolute improvement compared with the recursive SP calculation.

## Interpretation and limitations

- Correctness is a prerequisite: all fixture han, fu, yaku, masks, feature
  vectors, and single-row/batch parity tests passed before timing.
- The yaku corpus is broad and correctness-labelled, but its reported number is
  corpus throughput rather than a production latency percentile.
- The feature corpus is deterministic and reproducible, not a frequency-weighted
  sample of real games. In particular, rare 3-shanten positions can dominate
  tail latency even when the corpus average is much lower.
- These benchmarks measure pure Rust. Python/WASM boundary conversion, model
  inference, engine stepping, replay I/O, and thread scheduling are out of scope.
- No allocation counter is installed in this run; `allocate` versus `into`
  measures elapsed-time impact, not the number or size of allocations.
- Comparisons across machines or toolchains should save a named Criterion
  baseline and rerun under similar thermal and power conditions.

## Recommended performance gates

Track the following as the stable, reviewable performance surface:

1. 4P/3P `HandEvaluator::calc` normalized per winning hand;
2. base and extended batch encoders in both allocation modes;
3. combined 4P encoder, with SP reported separately;
4. true SP s0/s1/s2/s3 fixtures and the open aka/dora fixture;
5. caller-buffer SP encoding and shared `FeatureContext` preprocessing.

Do not gate on a single absolute number across heterogeneous CI runners. Use a
fixed runner or a named local baseline, and treat a statistically significant
change as actionable only when it also exceeds a practical threshold (for
example 5%).
