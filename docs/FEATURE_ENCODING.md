# Feature Encoding Specification

This document describes the observation feature encoding used for `obs.encode()` in RiichiEnv.
The encoding produces a **(C, 34)** tensor, where C is the number of channels and 34 corresponds to the tile types (0-33).

## Channels Definition

**Total Channels: 74**

### Basic Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **0 - 3** | **Hand** | Binary encoding of hand tile counts (1, 2, 3, 4). <br> Ch 0: count >= 1, Ch 1: count >= 2, ... |
| **4** | **Red Tiles** | 1 if the tile in hand is Red. |
| **5 - 8** | **Melds (Self)** | Binary encoding of self melds (up to 4 melds). |
| **9** | **Dora Indicators** | 1 if tile is a Dora indicator. |

### Discard History Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **10 - 13** | **Discards (Self, Recent 4)** | Last 4 discards of self. |
| **14 - 25** | **Discards (Opponents, Recent 4)** | Last 4 discards of each opponent (Player +1, +2, +3). <br> (3 players * 4 channels = 12). |
| **26 - 29** | **Discard Counts (All Players)** | Normalized discard count per player (/ 24.0). <br> Broadcast across all 34 tiles. |
| **64 - 67** | **Extended Discards (Self, 5-8)** | 5th to 8th most recent self discards. |
| **68 - 69** | **Extended Discards (Opponent 1, 5-6)** | 5th to 6th most recent discards of first opponent. |

### Game State Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **30** | **Tiles Left in Wall** | Normalized remaining tiles (/ 70.0). <br> Broadcast across all 34 tiles. |
| **31** | **Riichi (Self)** | Broadcast 1 if self declared Riichi. |
| **32 - 34** | **Riichi (Opponents)** | Broadcast 1 if opponent declared Riichi. |
| **35** | **Round Wind** | Broadcast 1 at the tile index corresponding to the Round Wind (27-30). |
| **36** | **Self Wind** | Broadcast 1 at the tile index corresponding to the Self Wind (27-30). |
| **37** | **Honba** | Normalized honba count (/ 10.0). <br> Broadcast across all 34 tiles. |
| **38** | **Riichi Sticks** | Normalized riichi stick count (/ 5.0). <br> Broadcast across all 34 tiles. |

### Score and Rank Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **39 - 42** | **Scores (0-100000)** | Normalized scores for all 4 players (/ 100000.0). <br> Broadcast across all 34 tiles. |
| **43 - 46** | **Scores (0-30000)** | Normalized scores for all 4 players (/ 30000.0). <br> Broadcast across all 34 tiles. Provides finer granularity for closer scores. |
| **49 - 52** | **Rank** | One-hot encoding of current player rank (0-3, based on scores). |

### Tenpai and Wait Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **47** | **Waits** | 1 at tile indices of waiting tiles (if tenpai). |
| **48** | **Is Tenpai** | Broadcast 1 if tenpai, 0 otherwise. |

### Round Progress Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **53** | **Kyoku Index** | Normalized kyoku index (/ 8.0). <br> Broadcast across all 34 tiles. |
| **54** | **Round Progress** | Combined round progress: (round_wind * 4 + kyoku_index) / 7.0. <br> Provides a single normalized value for overall game progress. <br> Broadcast across all 34 tiles. |

### Dora and Meld Count Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **55 - 58** | **Dora Count (Per Player)** | Normalized dora count visible for each player (/ 12.0). <br> Counts dora in melds, discards, and self hand. <br> Broadcast across all 34 tiles. |
| **59 - 62** | **Melds Count (Per Player)** | Normalized meld count for each player (/ 4.0). <br> Broadcast across all 34 tiles. |

### Tile Visibility Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **63** | **Tiles Seen** | Normalized count of visible tiles per tile type (/ 4.0). <br> Includes: self hand, all melds, all discards, dora indicators. |

### Tsumogiri (Auto-Discard) Features

| Channel Index | Description | Details |
| :--- | :--- | :--- |
| **70 - 73** | **Tsumogiri Flags** | 1 if last discard was tsumogiri (draw-and-discard), 0 otherwise. <br> One channel per player. <br> Broadcast across all 34 tiles. |

**Total Channels: 74**

## Feature Encoding Details

### Tile Encoding
- Tiles are encoded by type (0-33), where each tile ID is divided by 4 to get the tile type
- Tile types: 0-8 (man), 9-17 (pin), 18-26 (sou), 27-30 (winds), 31-33 (dragons)
- Red fives (5mr, 5pr, 5sr) are tracked separately in channel 4

### Normalization
- Scores: Two scales (0-100000 and 0-30000) for different granularities
- Counts: Normalized by maximum expected value (e.g., discards /24, melds /4, dora /12)
- Progress: Normalized by maximum rounds (e.g., kyoku /8, round progress /7)

### Broadcast Channels
Some features are broadcast (repeated) across all 34 tile positions:
- Game state info (honba, riichi sticks, tiles left)
- Scores and ranks
- Round progress indicators
- Count-based features (discard counts, dora counts, meld counts)

This allows the model to access global game state information alongside tile-specific features.

## Alternative Encoding: Exponential Decay Discard History

In addition to the standard `encode()` method, RiichiEnv provides an alternative encoding for discard history using exponential decay weighting, inspired by Mortal v3/v4.

### Method: `encode_discard_history_decay(decay_rate=0.2)`

Returns a **(4, 34)** tensor where:
- **Row 0**: Self discard history
- **Rows 1-3**: Opponent discard history (in order)

Each tile type accumulates weighted values from all discards of that type:

```
weight = exp(-decay_rate × age)
```

Where:
- `age`: Number of turns since the discard (0 = most recent)
- `decay_rate`: Decay rate parameter (default 0.2, matching Mortal)

### Example

If a player discarded tiles in this order: [1m, 2m, 1m, 3m]

With decay_rate = 0.2:
```
1m: exp(-0.2 × 3) + exp(-0.2 × 1) = 0.549 + 0.819 = 1.368
2m: exp(-0.2 × 2) = 0.670
3m: exp(-0.2 × 0) = 1.000
```

### Advantages

1. **Compact representation**: 1 channel per player vs. 4-24 channels for discrete history
2. **Full history**: All discards are included with appropriate weighting
3. **Smooth temporal representation**: Continuous weighting over time
4. **Accumulation**: Multiple discards of the same tile naturally accumulate

### Usage

```python
import riichienv as renv
import struct

env = renv.RiichiEnv()
obs_dict = env.reset()
obs = obs_dict[0]

# Get exponential decay encoding
encoded_bytes = obs.encode_discard_history_decay()
# Or with custom decay rate
encoded_bytes = obs.encode_discard_history_decay(decay_rate=0.3)

# Convert to numpy array
import numpy as np
decay_history = np.frombuffer(encoded_bytes, dtype=np.float32).reshape(4, 34)
# decay_history[0] = self discard history
# decay_history[1:4] = opponent discard histories
```

### Comparison with Standard Encoding

| Feature | Standard `encode()` | `encode_discard_history_decay()` |
|---------|---------------------|----------------------------------|
| **Self history** | Last 4 + Last 4 more (8 channels) | All history (1 channel) |
| **Opponent history** | Last 4 each × 3 + extras (14 channels) | All history × 3 (3 channels) |
| **Total channels** | 22 channels for discards | 4 channels total |
| **Temporal info** | Discrete positions | Continuous weights |
| **Full history** | No (only recent) | Yes (with decay) |

## Alternative Encoding: Yaku Possibility

RiichiEnv also provides a rule-based yaku (winning hand) possibility checker that determines whether certain yaku are definitely possible or impossible based on observable information.

### Method: `encode_yaku_possibility()`

Returns a **(4, 21, 2)** tensor where:
- **Dimension 0 (4)**: Player index (0=self, 1-3=opponents)
- **Dimension 1 (21)**: Yaku type (see list below)
- **Dimension 2 (2)**: [Tsumo possibility, Ron possibility]

Values:
- `1.0`: Yaku is possible or unknown (conservative estimate)
- `0.0`: Yaku is definitely impossible based on visible information

### Supported Yaku (21 types)

| Index | Yaku | Detection Logic |
|-------|------|-----------------|
| 0 | **Tanyao** | Impossible if terminals/honors in melds |
| 1-3 | **Yakuhai (Dragons)** | White, Green, Red - impossible if 3+ visible |
| 4 | **Yakuhai (Round Wind)** | Impossible if 3+ visible |
| 5 | **Yakuhai (Seat Wind)** | Impossible if 3+ visible |
| 6 | **Honitsu (Half Flush)** | Impossible if 2+ suits in melds |
| 7 | **Chinitsu (Full Flush)** | Impossible if 2+ suits or honors in melds |
| 8 | **Toitoi (All Triplets)** | Impossible if any chi (sequence) in melds |
| 9 | **Chiitoitsu (Seven Pairs)** | Impossible if any melds (tsumo only) |
| 10 | **Shousangen** | Impossible if any dragon type has 4 visible |
| 11 | **Daisangen** | Impossible if any dragon has 2+ visible without pon |
| 12 | **Tsuuiisou (All Honors)** | Impossible if number tiles in melds |
| 13 | **Chinroutou (All Terminals)** | Impossible if honors or simples in melds |
| 14 | **Honroutou** | Impossible if simples in melds |
| 15 | **Kokushi (Thirteen Orphans)** | Impossible if any melds or if any required terminal/honor type has 4 visible |
| 16 | **Chanta (Outside Hand)** | Impossible if any meld contains only simples (no terminals/honors) |
| 17 | **Junchan (Pure Outside Hand)** | Impossible if honors in melds or any meld has no terminals |
| 18 | **Sanshoku Doujun (Three Color Straight)** | Conservative estimate (usually Unknown) |
| 19 | **Iipeikou (Pure Double Sequence)** | Impossible if any melds (closed hand required) |
| 20 | **Ittsu (Straight)** | Conservative estimate (usually Unknown) |

### Usage

```python
import riichienv as renv
import numpy as np

env = renv.RiichiEnv()
obs_dict = env.reset()
obs = obs_dict[0]

# Get yaku possibility encoding
encoded_bytes = obs.encode_yaku_possibility()
yaku_poss = np.frombuffer(encoded_bytes, dtype=np.float32).reshape(4, 21, 2)

# Check if player 1 can win with tanyao
player1_tanyao_tsumo = yaku_poss[1, 0, 0]  # 0=tanyao, 0=tsumo
player1_tanyao_ron = yaku_poss[1, 0, 1]    # 0=tanyao, 1=ron

# Check toitoi possibility for player 2
player2_toitoi = yaku_poss[2, 8, :]  # 8=toitoi
```

### Applications

1. **Defense**: If opponent cannot make daisangen (big three dragons), it's safer to discard dragon tiles
2. **Push/Fold**: Estimate opponent hand strength based on possible yaku
3. **Risk Assessment**: Calculate danger level of discarding specific tiles
4. **Feature Engineering**: Provide rule-based features for ML models

### Design Philosophy

- **Rule-based**: No machine learning, purely deterministic
- **Conservative**: When uncertain, assume yaku is possible (avoid false negatives)
- **Observable only**: Based on visible information (melds, discards, dora)
- **Fast**: Simple checks with minimal computation

## Alternative Encoding: Furiten-Aware Ron Possibility

RiichiEnv provides an encoding that considers furiten (見逃し) and tsumogiri patterns to estimate ron possibility.

### Method: `encode_furiten_ron_possibility()`

Returns a **(4, 21)** tensor where:
- **Dimension 0 (4)**: Player index (0=self, 1-3=opponents)
- **Dimension 1 (21)**: Yaku type (same indices as `encode_yaku_possibility`)

Values:
- `1.0`: Ron is likely possible (player's hand appears to have changed)
- `0.0`: Ron is likely impossible (player doing consecutive tsumogiri, suggesting furiten)

### Logic

When a player does **3 or more consecutive tsumogiri** (auto-discards), their hand hasn't changed. According to furiten rules:
- They cannot ron on any tile in their own discard pile
- If they've been doing tsumogiri, they've been seeing and rejecting the same waiting tiles
- Therefore, ron on their recent discards is very unlikely (furiten state)

### Usage

```python
import riichienv as renv
import numpy as np

env = renv.RiichiEnv()
obs_dict = env.reset()
obs = obs_dict[0]

# Get furiten-aware ron possibility
encoded_bytes = obs.encode_furiten_ron_possibility()
furiten_ron = np.frombuffer(encoded_bytes, dtype=np.float32).reshape(4, 21)

# Check if opponent 1 can ron with tanyao
if furiten_ron[1, 0] == 0.0:
    print("Opponent 1 likely in furiten (consecutive tsumogiri)")
```

### Applications

1. **Defense**: If opponent doing consecutive tsumogiri, tiles in their river are safer
2. **Risk Assessment**: Estimate danger level of specific discards
3. **Furiten Detection**: Identify when opponents are likely in furiten state
4. **Feature Engineering**: Provide tsumogiri-based features for ML models

## Alternative Encoding: Shanten and Tile Efficiency

RiichiEnv provides shanten (deficiency number) calculation and tile efficiency features for evaluating hand strength and playability.

### Method: `encode_shanten_efficiency()`

Returns a **(4, 4)** tensor where:
- **Dimension 0 (4)**: Player index (0=self, 1-3=opponents)
- **Dimension 1 (4)**: Feature type [shanten, effective_tiles, best_ukeire, turn_progress]

Features:
- **Shanten (normalized /8)**: Distance to tenpai. -1=tenpai, 0=1-shanten, etc.
- **Effective Tiles (normalized /34)**: Number of tile types that reduce shanten when drawn
- **Best Ukeire (normalized /80)**: Maximum number of tiles that improve hand after optimal discard
- **Turn Progress (normalized /18)**: Current turn number in the round

Values for opponents (players 1-3) are set to 0.5 (unknown) since their hands are hidden.

### Usage

```python
import riichienv as renv
import struct

env = renv.RiichiEnv()
obs_dict = env.reset()
obs = obs_dict[0]

# Get shanten and efficiency features
encoded_bytes = obs.encode_shanten_efficiency()
values = struct.unpack('16f', encoded_bytes)  # 4 players × 4 features

# Player 0 (self) features
shanten_norm = values[0]        # Normalized shanten
effective_norm = values[1]      # Normalized effective tile types
ukeire_norm = values[2]         # Normalized best ukeire
turn_norm = values[3]           # Normalized turn count

# Denormalize
actual_shanten = shanten_norm * 8.0
effective_types = effective_norm * 34.0
best_ukeire = ukeire_norm * 80.0
turn_number = turn_norm * 18.0

print(f"Shanten: {actual_shanten:.1f}")
print(f"Effective tile types: {effective_types:.0f}")
print(f"Best ukeire: {best_ukeire:.0f} tiles")
print(f"Turn: {turn_number:.0f}")
```

### Applications

1. **Hand Evaluation**: Quantify hand strength and progression toward tenpai
2. **Decision Making**: Choose discards that maximize ukeire
3. **Opponent Modeling**: Estimate opponent hand state based on their discards
4. **Feature Engineering**: Provide tile efficiency features for ML models

### Shanten Calculation

The shanten calculation considers three patterns:
1. **Standard Form**: 4 melds + 1 pair (standard winning hand)
2. **Seven Pairs**: 7 pairs of tiles (chitoi)
3. **Thirteen Orphans**: All terminals and honors (kokushi musou)

The final shanten is the minimum across all three patterns.

## Implementation

The encoding is implemented in [native/src/observation.rs](../native/src/observation.rs):
- Standard encoding: `encode()` method (74 channels)
- Exponential decay: `encode_discard_history_decay()` method (4 channels)
- Yaku possibility: `encode_yaku_possibility()` method (4×21×2 = 168 values)
- Furiten-aware ron possibility: `encode_furiten_ron_possibility()` method (4×21 = 84 values)
- Shanten and efficiency: `encode_shanten_efficiency()` method (4×4 = 16 values)

Yaku checking logic: [native/src/yaku_checker.rs](../native/src/yaku_checker.rs)
Shanten calculation: [native/src/shanten.rs](../native/src/shanten.rs)
