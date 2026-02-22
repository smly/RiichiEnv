# riichienv-core

[![crates.io](https://img.shields.io/crates/v/riichienv-core)](https://crates.io/crates/riichienv-core)
[![docs.rs](https://docs.rs/riichienv-core/badge.svg)](https://docs.rs/riichienv-core)
[![License](https://img.shields.io/crates/l/riichienv-core)](https://github.com/smly/RiichiEnv/blob/main/LICENSE)

A high-performance Japanese Mahjong (Riichi) game engine implemented in Rust.

This is the core engine behind [RiichiEnv](https://github.com/smly/RiichiEnv), a Gym-style simulation environment for Riichi Mahjong.

## Features

- **Game state management** -- 4-player and 3-player (sanma) Riichi Mahjong simulation with configurable game modes (East-only, Hanchan, single round)
- **Hand evaluation** -- Comprehensive agari (winning hand) detection with all standard yaku, including yakuman
- **Score calculation** -- Han/fu-based scoring with support for honba, tsumo/ron, and oya/ko distinctions
- **Rule presets** -- Built-in presets for Tenhou and MJSoul rule sets, with granular customization options
- **WASM support** -- Compile to WebAssembly via the `wasm` feature flag for browser-side computation (waits, shanten, scoring)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
riichienv-core = "0.3"
```

### Hand evaluation

```rust
use riichienv_core::hand_evaluator::HandEvaluator;

// Parse a hand from MPSZ notation and create evaluator
let evaluator = HandEvaluator::hand_from_text("123m456p789s1122z").unwrap();

// Check tenpai (requires exactly 13 tiles in hand + melds)
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
use std::collections::HashMap;

// Create a new game with Tenhou rules
// GameState::new(game_mode, skip_mjai_logging, seed, round_wind, rule)
let rule = GameRule::default_tenhou();
let mut state = GameState::new(0, true, None, 0, rule);

// Get observation for the current player
let obs = state.get_observation(0);
let legal_actions = obs.legal_actions_method();

// Advance the game state (actions is a HashMap<u8, Action>)
let mut actions = HashMap::new();
actions.insert(0, legal_actions[0].clone());
state.step(&actions);
```

## Modules

| Module | Description |
|---|---|
| `action` | Action types (`Discard`, `Chi`, `Pon`, `Kan`, `Riichi`, `Ron`, `Tsumo`, `Kita`, etc.) and game phase tracking |
| `observation` | Player-facing game state views with legal actions and MJAI event history (4-player) |
| `observation_3p` | Player-facing game state views for 3-player games |
| `state` | Full game state management, wall handling, and legal action validation (4-player) |
| `state_3p` | Game state management for 3-player games with Kita/BaBei support |
| `game_variant` | `GameStateVariant` enum dispatching between 4-player and 3-player game states |
| `hand_evaluator` | Agari detection, tenpai checking, wait calculation, and riichi candidate analysis (4-player) |
| `hand_evaluator_3p` | Hand evaluation for 3-player games |
| `parser` | MPSZ notation parsing for tiles and hands |
| `types` | Core data types: `Hand`, `Wind`, `Meld`, `MeldType`, `Conditions`, `WinResult` |
| `rule` | Game rule configuration with Tenhou/MJSoul presets (4-player and sanma) |
| `score` | Han/fu-based score calculation |
| `replay` | MJAI and MJSoul replay parsing with step-by-step iteration (requires `python` feature) |
| `errors` | Error types (`RiichiError`) and result alias (`RiichiResult`) |

## Tile representation

- **136-format**: Each of 34 tile types x 4 copies (indices 0-135), used for actual game state
- **34-format**: Normalized tile type indices (0-33), used for calculations
- **MPSZ notation**: `1m`-`9m` (man), `1p`-`9p` (pin), `1s`-`9s` (sou), `1z`-`7z` (honors)
- Red fives are represented at indices 16, 52, 88 in 136-format

## License

Apache-2.0
