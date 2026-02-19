# riichienv-core

[![crates.io](https://img.shields.io/crates/v/riichienv-core)](https://crates.io/crates/riichienv-core)
[![docs.rs](https://docs.rs/riichienv-core/badge.svg)](https://docs.rs/riichienv-core)
[![License](https://img.shields.io/crates/l/riichienv-core)](https://github.com/smly/RiichiEnv/blob/main/LICENSE)

A high-performance Japanese Mahjong (Riichi) game engine implemented in Rust.

This is the core engine behind [RiichiEnv](https://github.com/smly/RiichiEnv), a Gym-style simulation environment for Riichi Mahjong.

## Features

- **Game state management** -- 4-player Riichi Mahjong simulation with configurable game modes (East-only, Hanchan, single round)
- **Hand evaluation** -- Comprehensive agari (winning hand) detection with all standard yaku, including yakuman
- **Score calculation** -- Han/fu-based scoring with support for honba, tsumo/ron, and oya/ko distinctions
- **Rule presets** -- Built-in presets for Tenhou and MJSoul rule sets, with granular customization options

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
riichienv-core = "0.3"
```

### Hand evaluation

```rust
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::parser::parse_hand;
use riichienv_core::types::Conditions;

// Parse a hand from MPSZ notation
let (tiles_136, melds) = parse_hand("111m33p123s111z");

// Create evaluator and check tenpai / calculate agari
let evaluator = HandEvaluator::new(tiles_136, melds);
assert!(evaluator.is_tenpai());
```

### Score calculation

```rust
use riichienv_core::score::calculate_score;

// Calculate score for a 3 han 30 fu ron by non-dealer, 0 honba
let score = calculate_score(3, 30, false, false, 0);
println!("Total: {} points", score.total);
```

### Game simulation

```rust
use riichienv_core::state::GameState;
use riichienv_core::rule::GameRule;

// Create a new game with Tenhou rules
let rule = GameRule::default_tenhou();
let mut state = GameState::new(rule);

// Get observation for the current player
let obs = state.get_observation(0);
let legal_actions = obs.legal_actions();

// Advance the game state
state.step(legal_actions[0].clone());
```

### Replay parsing

```rust
use riichienv_core::replay::MjaiReplay;

let replay = MjaiReplay::new("path/to/replay.json").unwrap();
for kyoku in replay.kyoku_step_iterators() {
    for (state, obs) in kyoku {
        // Process each step
    }
}
```

## Modules

| Module | Description |
|---|---|
| `action` | Action types (`Discard`, `Chi`, `Pon`, `Kan`, `Riichi`, `Ron`, `Tsumo`, etc.) and game phase tracking |
| `observation` | Player-facing game state views with legal actions and MJAI event history |
| `state` | Full game state management, wall handling, and legal action validation |
| `hand_evaluator` | Agari detection, tenpai checking, wait calculation, and riichi candidate analysis |
| `parser` | MPSZ notation parsing for tiles and hands |
| `types` | Core data types: `Hand`, `Wind`, `Meld`, `MeldType`, `Conditions`, `WinResult` |
| `rule` | Game rule configuration with Tenhou/MJSoul presets |
| `score` | Han/fu-based score calculation |
| `replay` | MJAI and MJSoul replay parsing with step-by-step iteration |
| `win_projection` | Win probability and expected value projection |
| `errors` | Error types (`RiichiError`) and result alias (`RiichiResult`) |

## Tile representation

- **136-format**: Each of 34 tile types x 4 copies (indices 0-135), used for actual game state
- **34-format**: Normalized tile type indices (0-33), used for calculations
- **MPSZ notation**: `1m`-`9m` (man), `1p`-`9p` (pin), `1s`-`9s` (sou), `1z`-`7z` (honors)
- Red fives are represented at indices 16, 52, 88 in 136-format

## License

Apache-2.0
