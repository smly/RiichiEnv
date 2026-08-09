# Action Encoding Specification

This document describes the mapping between `Action` objects and integer IDs used for model training and inference in RiichiEnv.

## Action Space

### 4-Player Mode (82 actions, IDs 0-81)

| Action ID | Action Type | Description |
| :--- | :--- | :--- |
| **0 - 36** | **Discard** | **Discard a tile.** <br> ID corresponds to the tile type index (0-36). <br> `ID = floor(tile_id / 4)`. <br> Red tiles are mapped to their normal counterparts. |
| **37** | **Riichi** | **Declare Riichi.** |
| **38** | **Chi (Low)** | **Chi (Low).** <br> The target tile is the lowest in the sequence (e.g., target 3, consume 4-5). |
| **39** | **Chi (Mid)** | **Chi (Mid).** <br> The target tile is the middle in the sequence (e.g., target 4, consume 3-5). |
| **40** | **Chi (High)** | **Chi (High).** <br> The target tile is the highest in the sequence (e.g., target 5, consume 3-4). |
| **41** | **Pon** | **Pon.** <br> Declaring a Pon on the discarded tile. |
| **42 - 78** | **Kan** | **Kan a specific tile.** <br> Covers Ankan (Closed Kan), Kakan (Added Kan), and Daiminkan (Open Kan). <br> `ID = 42 + tile_idx`. <br> Example: Ankan 0m -> ID 42, Daiminkan 1m -> ID 43. |
| **79** | **Agari** | **Agari (Win).** <br> Covers both Ron and Tsumo. |
| **80** | **Ryukyoku** | **Kyushukyuhai / Ryukyoku.** |
| **81** | **Pass** | **Pass / No Action.** |

### 3-Player Mode (60 actions, IDs 0-59)

In 3-player (sanma) mode, the action space is compacted to 60 actions. Manzu 2-8 tiles are excluded (only 1m and 9m remain), Chi is not allowed, and Kita (BaBei) is added.

| Action ID | Action Type | Description |
| :--- | :--- | :--- |
| **0 - 26** | **Discard** | **Discard a tile.** <br> 27 valid tile types in sanma (1m, 9m, 1-9p, 1-9s, E/S/W/N, P/F/C). |
| **27** | **Riichi** | **Declare Riichi.** |
| **28** | **Pon** | **Pon.** |
| **29 - 55** | **Kan** | **Kan a specific tile.** <br> `ID = 29 + compact_tile_idx`. |
| **56** | **Agari** | **Agari (Win).** <br> Covers both Ron and Tsumo. |
| **57** | **Ryukyoku** | **Kyushukyuhai / Ryukyoku.** |
| **58** | **Pass** | **Pass / No Action.** |
| **59** | **Kita** | **Kita (BaBei / 北抜き) declaration.** |

## Red-aware v1 action space (#210)

The legacy layouts above remain the default for existing models. New models
can opt into the red-aware v1 layout:

| Variant | v0 size | v1 size |
|---|---:|---:|
| 4P | 82 | 164 |
| 3P | 60 | 120 |

For every v0 ID `i`, v1 reserves `2*i` and `2*i+1`:

- `2*i` means the action does not discard or voluntarily consume a red five;
- `2*i+1` means a discard is the red five, or a Chi/Pon consumes a red five.

Kan, win, riichi, pass, ryukyoku, and Kita actions use the even slot because
there is no model choice between preserving and consuming a red five. Their
odd slots remain masked out. This keeps the mapping mechanically related to
v0 while allowing the hand-changing red-five choice to round-trip exactly.

Rust exposes `ActionEncoderV1` and `ObservationVariant::{action_mask_v1,
select_action_v1}`. Python observations expose `mask_v1()`,
`find_action_v1()`, `action_space_size_v1`, and actions expose `encode_v1()`.
WASM decisions include `legalActionIdsV1`/`actionMaskV1`, with
`actionMaskV1()` and `stepActionIdsV1()` for inference.

## Legacy v0 limitations

### Red 5 Ambiguity in Calls (Chi/Pon)

The v0 encoding does **not** distinguish between using a Red 5 or a Normal 5 when making a call (Chi or Pon) if both are available in the hand. Use v1 for new models.
-   **Example**: If a player holds `[0m (Red 5), 5m (Normal 5)]` and calls Pon on a discarded `5m`, the action ID `41` (Pon) is used regardless of whether the player consumes the Red 5 or the Normal 5.

### Tsumogiri/Tedashi Ambiguity

The current encoding does **not** distinguish between discarding a tile that was just drawn (Tsumogiri) and discarding a tile that was previously drawn (Tedashi).
-   **Example**: If a player draws a `5m` and immediately discards it, the action ID `5` (Discard 5m) is used. However, the encoding does not differentiate this from a situation where the player had previously discarded a `5m` and then drew another `5m` to discard.
