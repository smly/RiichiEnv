# Benchmark Tasks

This project includes benchmarks to evaluate the performance and accuracy of the hand calculation logic.

## MJAI Log Parser

Loads compressed MJAI JSONL data, restores game states, and verifies round-end scores through cross-calculation.

## Agari Calculator

Calculates yaku, han, and fu for all winning hands extracted from MJAI logs. This benchmark compares three different implementations.

### Implementations

- **`riichienv` (Rust Core)**: 
  The core implementation in Rust. It utilizes optimized bitsets and efficient backtracking for agari detection and yaku calculation. This method measures the bare performance of the Rust engine with minimal Python overhead, as the data stays within the Rust extension.
- **`riichienv` (Python Wrapper)**:
  Uses the `riichienv.AgariCalculator` Python class. This wrapper provides a high-level API but introduces overhead due to Python object instantiation, attribute access, and data conversion between Python and Rust.
- **`mahjong` (Python Package)**:
  The baseline comparison using the popular `mahjong` Python package (v1.4.0). It is a feature-rich, pure Python implementation.

### Results

The following results were obtained by processing 6,503 agari situations extracted from MJAI logs.

| Implementation | Throughput (agari/sec) | Notes |
| :--- | ---: | :--- |
| **`riichienv` (Rust Core)** | **~762,398.23** | Direct calculate on `AgariContext` (Rust side) |
| **`riichienv` (Python Wrapper)** | 145,592.33 | Using `riichienv.AgariCalculator` (Python side) |
| **`mahjong` (Python Package)** | 5,902.76 | Pure Python implementation |

> [!NOTE]
> `riichienv`'s Rust core is approximately **130x faster** than the pure Python `mahjong` package while maintaining 100% accuracy. The Python wrapper overhead is around 5x, yet it remains significantly faster than the baseline.

### How to Run

To run the benchmark on your machine:

```bash
❯ cd ../benchmark
❯ uv run agari
Uninstalled 1 package in 1ms
Installed 1 package in 3ms
Total Agari situations: 6503
riichienv: 0.0080s (814706.50 agari/sec)
riichienv-py: 0.0416s (156297.97 agari/sec)
mahjong:   1.0810s (6015.68 agari/sec)
```

See details in [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md#benchmarks).