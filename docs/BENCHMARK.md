# Benchmark Results

This document records the benchmark design and the measurements taken after the
engine/replay/feature-boundary refactor. The two primary targets are the winning
hand/yaku pipeline and model feature encoding.

## Measurement environment

| Item | Value |
|---|---|
| Date | 2026-08-09 (JST) |
| Source revision | `dbb4b512624246e7c5881e7eaabcd36775caad99`, plus the reviewed working-tree changes described here |
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
| `is_agari/positive` | 816 hands | 28.771 us | 28.657–28.903 us | 35.3 ns/hand |
| `is_agari/negative` | 200 hands | 6.452 us | 6.431–6.474 us | 32.3 ns/hand |
| `is_tenpai` | 200 hands | 134.060 us | 132.880–135.390 us | 670 ns/hand |
| `find_divisions` | 816 hands | 92.054 us | 91.858–92.270 us | 113 ns/hand |
| `hand_evaluator/calc_4p` | 816 hands | 482.840 us | 480.610–485.230 us | 0.592 us/hand |
| `hand_evaluator/calc_3p` | 402 hands | 246.440 us | 245.630–247.230 us | 0.613 us/hand |
| `calculate_score` | 28 score cases | 89.800 ns | 89.307–90.337 ns | 3.21 ns/case |

The end-to-end 4P and 3P yaku/scoring paths are both about 0.6 us per winning
hand on this machine. The 3P normalized result is approximately 4% slower,
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

The covered layouts include the frozen feature paths and the opt-in public-
history DREV-v2 paths:

| Variant | Layout | Shape per observation |
|---|---|---:|
| 4P | base | `74 x 34` |
| 4P | extended | `215 x 34` |
| 4P | SP | `178 x 34` |
| 4P | DREV v1 | `9 x 34` |
| 4P | DREV v2 | `81 x 34` |
| 4P | extended + SP + DREV v1 | `402 x 34` |
| 4P | extended + SP + DREV v2 | `474 x 34` |
| 3P | base | `74 x 27` |
| 3P | extended | `215 x 27` |
| 3P | SP | `178 x 27` |
| 3P | DREV v1 | `9 x 27` |
| 3P | DREV v2 | `81 x 27` |
| 3P | extended + SP + DREV v1 | `402 x 27` |
| 3P | extended + SP + DREV v2 | `474 x 27` |

The numerical 16-observation tables immediately below predate the 81-channel
DREV-v2 layout. Their `DREV` and `combined` rows are the frozen DREV-v1 and
402-channel paths respectively; current v2 measurements are reported in the
dedicated section below.

#### 4-player feature batch

| Layout / ownership | Batch estimate | 95% CI | Per observation | `into` time reduction |
|---|---:|---:|---:|---:|
| base / allocate | 12.280 us | 12.123–12.476 us | 0.768 us | — |
| base / into | 9.639 us | 9.598–9.689 us | 0.602 us | 21.50% |
| extended / allocate | 693.610 us | 691.650–695.730 us | 43.351 us | — |
| extended / into | 692.800 us | 682.520–708.730 us | 43.300 us | 0.12% |
| SP / allocate | 7.881 ms | 7.735–8.073 ms | 492.550 us | — |
| SP / into | 7.955 ms | 7.782–8.176 ms | 497.175 us | -0.94% |
| DREV v1 / allocate | 9.646 us | 9.618–9.678 us | 0.603 us | — |
| DREV v1 / into | 9.456 us | 9.421–9.497 us | 0.591 us | 1.97% |
| combined v1 / allocate | 8.390 ms | 8.373–8.408 ms | 524.344 us | — |
| combined v1 / into | 8.405 ms | 8.377–8.441 ms | 525.300 us | -0.18% |

#### 3-player feature batch

| Layout / ownership | Batch estimate | 95% CI | Per observation | `into` time reduction |
|---|---:|---:|---:|---:|
| base / allocate | 9.270 us | 9.219–9.315 us | 0.579 us | — |
| base / into | 6.884 us | 6.857–6.913 us | 0.430 us | 25.75% |
| extended / allocate | 649.070 us | 645.790–653.030 us | 40.567 us | — |
| extended / into | 643.420 us | 639.450–648.030 us | 40.214 us | 0.87% |
| SP / allocate | 9.044 ms | 8.915–9.270 ms | 565.219 us | — |
| SP / into | 8.916 ms | 8.891–8.941 ms | 557.225 us | 1.41% |
| DREV v1 / allocate | 9.353 us | 9.199–9.619 us | 0.585 us | — |
| DREV v1 / into | 9.117 us | 8.972–9.344 us | 0.570 us | 2.52% |
| combined v1 / allocate | 9.576 ms | 9.548–9.608 ms | 598.525 us | — |
| combined v1 / into | 9.549 ms | 9.522–9.578 ms | 596.813 us | 0.29% |

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
| closed, 0-shanten | 17.542 us | 17.469–17.644 us |
| closed, 1-shanten | 20.713 us | 20.541–21.040 us |
| closed, 2-shanten | 130.850 us | 128.540–135.220 us |
| closed, 3-shanten | 6.817 ms | 6.705–7.003 ms |
| closed, 5-shanten light path | 13.948 us | 13.564–14.510 us |
| open 0-shanten with aka and dora | 73.064 us | 72.722–73.479 us |
| 1-shanten, horizon 3 | 17.582 us | 17.521–17.650 us |
| 1-shanten, horizon 10 | 20.556 us | 20.510–20.607 us |
| 1-shanten, horizon 17 | 26.886 us | 26.654–27.288 us |
| calculate discard candidates | 310.020 ns | 304.340–321.670 ns |
| encode result, allocate | 1.082 us | 1.058–1.118 us |
| encode result, caller buffer | 724.960 ns | 721.440–728.970 ns |
| calculate + allocate encode | 18.838 us | 18.678–19.102 us |
| build shared `FeatureContext` | 243.010 ns | 241.960–244.160 ns |
| build SP and DREV inputs independently | 252.440 ns | 250.800–254.620 ns |
| build SP and DREV inputs from one context | 236.560 ns | 235.430–237.800 ns |

The 3-shanten recursive case is about 52 times slower than the 2-shanten case
and 329 times slower than the 1-shanten case. In the progressed corpora, SP
accounts for approximately 94% of both 4P and 3P combined encode time. SP
calculation is therefore the primary feature-generation optimization target;
output allocation and the shared preprocessing context are secondary.

The caller-buffer SP result encoder is 33.01% faster than the allocating result
encoder, but result encoding itself is only about 1 us. Sharing one
`FeatureContext` reduces SP+DREV input-preparation time by 6.29%, also a small
absolute improvement compared with the recursive SP calculation.

### Public-history DREV v2

The current DREV-v2 benchmark uses engine-produced observations with complete
runtime history. Fixture construction is outside the timed loop; encoding
includes public Observation validation, history-sidecar preflight, risk/value
calculation, and output encoding. Both 4P and 3P measure:

- standalone DREV v1 and v2 on the common 16-observation corpus;
- standalone DREV v2 at batch sizes 1, 16, and 256;
- allocating and caller-buffer (`into`) output modes; and
- the frozen 402-channel and new 474-channel combined encoders on batch 16.

The current implementation was measured on 2026-08-15 from revision
`b4eb87418b0c37e5563ab1fec7932e7852e4a58e` plus the DREV-v2 working-tree
changes, on the machine and toolchain listed above. The tables use the full
Criterion run:

```bash
cargo bench -p riichienv-core --bench feature_bench -- drev --noplot
```

The common batch-16 corpus is intentionally retained for a direct v1/v2
comparison. It contains early-to-mid-hand observations, so the separate
scaling table samples evenly across the full 256-observation progressed corpus.

| Variant / layout / ownership | Batch estimate | 95% CI | Per observation | V2/V1 |
|---|---:|---:|---:|---:|
| 4P DREV v1 / allocate | 10.614 us | 10.587–10.642 us | 0.663 us | — |
| 4P DREV v2 / allocate | 51.995 us | 51.739–52.370 us | 3.250 us | 4.90x |
| 4P DREV v1 / into | 10.543 us | 10.470–10.632 us | 0.659 us | — |
| 4P DREV v2 / into | 50.886 us | 49.986–52.691 us | 3.180 us | 4.83x |
| 3P DREV v1 / allocate | 9.348 us | 9.308–9.400 us | 0.584 us | — |
| 3P DREV v2 / allocate | 37.201 us | 37.084–37.333 us | 2.325 us | 3.98x |
| 3P DREV v1 / into | 9.127 us | 9.088–9.185 us | 0.570 us | — |
| 3P DREV v2 / into | 35.283 us | 35.135–35.462 us | 2.205 us | 3.87x |

The v2 scaling groups select observations evenly across the same 256-position
corpus for every batch size. This avoids comparing an opening-only batch of one
against a late-hand batch of 256.

| Variant / batch / ownership | Batch estimate | 95% CI | Per observation |
|---|---:|---:|---:|
| 4P / 1 / allocate | 4.161 us | 4.149–4.175 us | 4.161 us |
| 4P / 1 / into | 4.039 us | 4.022–4.056 us | 4.039 us |
| 4P / 16 / allocate | 75.261 us | 75.085–75.453 us | 4.704 us |
| 4P / 16 / into | 73.655 us | 73.352–73.988 us | 4.603 us |
| 4P / 256 / allocate | 1.200 ms | 1.195–1.206 ms | 4.688 us |
| 4P / 256 / into | 1.189 ms | 1.185–1.194 ms | 4.645 us |
| 3P / 1 / allocate | 2.122 us | 2.116–2.128 us | 2.122 us |
| 3P / 1 / into | 2.040 us | 2.026–2.058 us | 2.040 us |
| 3P / 16 / allocate | 50.141 us | 49.908–50.404 us | 3.134 us |
| 3P / 16 / into | 47.722 us | 47.608–47.842 us | 2.983 us |
| 3P / 256 / allocate | 787.040 us | 785.500–788.770 us | 3.074 us |
| 3P / 256 / into | 774.890 us | 773.490–776.380 us | 3.027 us |

History depth explains the difference between the early common batch and the
full-corpus scaling groups:

| Variant / corpus | Min | Median | P95 | Max public discards |
|---|---:|---:|---:|---:|
| 4P / first 16 | 0 | 6 | 10 | 11 |
| 4P / full 256 | 0 | 37 | 73 | 79 |
| 3P / first 16 | 0 | 8 | 13 | 14 |
| 3P / full 256 | 0 | 29 | 56 | 61 |

The 474-channel combined path remains SP-dominated:

| Variant / ownership | Frozen 402ch | V2 474ch | 474/402 |
|---|---:|---:|---:|
| 4P / allocate | 8.147 ms | 8.053 ms | 0.988x |
| 4P / into | 7.928 ms | 8.030 ms | 1.013x |
| 3P / allocate | 9.495 ms | 9.555 ms | 1.006x |
| 3P / into | 9.591 ms | 9.520 ms | 0.993x |

The extra history and 72 additional planes make standalone v2 materially more
expensive than the frozen nine-plane heuristic, but even the longest measured
4P corpus stays near 5 us per observation. Caller-owned output usually helps,
and does so in every final scaling group. In the combined path, recursive SP
still dominates: all four 474/402 ratios are within about 1.3% and cross 1.0
in both directions, so the total times are effectively benchmark noise and
must not be used to infer DREV's standalone cost.

Timing is a throughput measurement, not a quality claim. The current seeded
hidden-state oracle satisfies the exact-zero gate on 17,920
candidate/opponent cells (3,871 hard-safe, zero false positives). Promotion
from experimental use still requires a representative labelled replay corpus
and reports for Ron calibration, expected-loss error, candidate ranking regret,
and per-yakuman recall.

### Frozen versus prefix-corrected SP series

On 2026-08-14, the prefix-semantics group was measured from revision
`cca35a200a7982fbfae134444ec5184c327d2ff6` plus the working-tree changes in
this section, on the same arm64 macOS machine with Rust 1.92.0:

```bash
cargo bench -p riichienv-core --bench sp_bench -- 'sp/prefix_semantics' --noplot
```

Each fixture asserts its best post-discard shanten before timing. The frozen
path is the model-compatible 4P v0 implementation; corrected v1 independently
solves each prefix from the current draw and returns a version-tagged result.

| SP prefix benchmark | Estimate | 95% CI | Corrected/frozen |
|---|---:|---:|---:|
| 4P s2, horizon 3, frozen v0 | 105.90 us | 105.76–106.06 us | — |
| 4P s2, horizon 3, corrected v1 | 531.51 us | 528.25–536.18 us | 5.02x |
| 4P s3, horizon 4, frozen v0 | 5.473 ms | 5.414–5.546 ms | — |
| 4P s3, horizon 4, corrected v1 | 52.754 ms | 52.298–53.313 ms | 9.64x |
| 3P s2, horizon 3, frozen v1 | 149.24 us | 148.66–149.99 us | — |
| 3P s2, horizon 3, corrected v2 | 672.85 us | 667.18–679.48 us | 4.51x |
| 3P s3, horizon 4, frozen v1 | 696.91 us | 686.34–716.15 us | — |
| 3P s3, horizon 4, corrected v2 | 6.456 ms | 6.360–6.567 ms | 9.26x |

The corrected implementation deliberately clears and rebuilds the probability
DP for every prefix. This is a simple correctness reference, not a production
optimization: cost grows rapidly with both shanten and horizon. Regular
Observation and batch encoders therefore remain on the frozen ABI. Before a
prefix-corrected layout becomes a default feature path, the transition graph
must be shared across deadlines and the resulting implementation must be
remeasured on representative 4P/3P replay positions.

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
3. DREV v1/v2 at batch 16 and DREV-v2 scaling at batches 1/16/256, split by
   4P/3P and allocate/into;
4. frozen 402-channel and v2 474-channel combined encoders, with SP and DREV
   reported separately;
5. true SP s0/s1/s2/s3 fixtures and the open aka/dora fixture;
6. caller-buffer SP encoding and shared `FeatureContext` preprocessing;
7. frozen versus prefix-corrected 4P/3P s2/s3 reference costs, reported as
   informational ratios until the corrected transition graph is shared.

Do not gate on a single absolute number across heterogeneous CI runners. Use a
fixed runner or a named local baseline, and treat a statistically significant
change as actionable only when it also exceeds a practical threshold (for
example 5%).
