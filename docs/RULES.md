# Game Rules Configuration

Detailed game mechanics can be configured using the `GameRule` struct.

> [!NOTE]
> Configurable rule options are still under development and not yet exhaustive. If you find missing rules or have suggestions for additional configuration options, please report them via [GitHub Issues #43](https://github.com/smly/RiichiEnv/issues/43).

## Pao (Responsibility Payment)

When a winning hand includes a Pao-triggering Yakuman (Daisangen or Daisuushii), the Pao player shares payment responsibility with the deal-in player (Ron) or bears it alone (Tsumo).

For a **single Pao Yakuman** (e.g., Daisangen only), both platforms behave identically:

- **Ron**: The total is split 50/50 between the Pao player and the deal-in player.
- **Tsumo**: The Pao player pays the full amount. Other players pay nothing.

The `yakuman_pao_is_liability_only` flag only matters for **composite hands where Pao and non-Pao Yakumans coexist** (e.g., Daisangen + Tsuuiisou). In this case:

**Ron**: The flag controls the split between the Pao player and the deal-in player (both 4P and 3P):

- **Tenhou** (`false`): The **total** Ron amount is split 50/50 between the Pao player and the deal-in player.
- **Mahjong Soul** (`true`): Only the **Pao-triggering Yakuman portion** is split 50/50. The non-Pao Yakuman portion is paid entirely by the deal-in player.

**Tsumo**: The flag controls how the non-Pao Yakuman portion is settled (both 4P and 3P):

- **Tenhou** (`false`): Pao player pays the full Tsumo amount including non-Pao Yakumans. Other players pay nothing.
- **Mahjong Soul** (`true`): Pao player pays only the Pao-triggering Yakuman portion. The remaining Yakumans are split normally among all non-winning players.

| Flag | Description |
|------|-------------|
| `.yakuman_pao_is_liability_only` | Whether to limit Pao liability to the Pao-triggering Yakuman portion only when combined with non-Pao Yakumans (Mahjong Soul style). Affects both Ron and Tsumo settlement in both 4P and 3P. If false, Pao covers the full amount (Tenhou style). Does not affect single-Yakuman wins (both styles are identical). |

## Kuikae (Swap Calling) Restriction

Controls whether players are forbidden from discarding a tile that completes the same group they just called (including flank tiles for Chi). Both Tenhou and Mahjong Soul enforce this restriction.

| Flag | Description |
|------|-------------|
| `.kuikae_forbidden` | When `True`, kuikae is forbidden: after Chi/Pon, the called tile and flank tiles cannot be discarded. When `False`, no kuikae restriction applies. |

## Kokushi Musou Rules

| Flag | Description |
|------|-------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | Whether to allow Ron on a closed Kan (Ankan) for Kokushi Musou (Chankan). |
| `.is_kokushi_musou_13machi_double` | Whether to treat Kokushi Musou 13-sided wait as a Double Yakuman. |

## Double Yakuman Pattern Rules

Controls whether specific Yakuman pattern variants are treated as Double Yakuman. Tenhou treats all pattern variants as single Yakuman, while Mahjong Soul treats them as Double Yakuman. Note that combinations of independent Yakuman (e.g., Tsuuiisou + Daisangen) always stack regardless of these flags.

| Flag | Description |
|------|-------------|
| `.is_suuankou_tanki_double` | Whether to treat Suuankou Tanki (四暗刻単騎) as a Double Yakuman. |
| `.is_junsei_chuurenpoutou_double` | Whether to treat Junsei Chuurenpoutou (純正九蓮宝燈) as a Double Yakuman. |
| `.is_daisuushii_double` | Whether to treat Daisuushii (大四喜) as a Double Yakuman. |

## Sanchaho (Triple Ron)

| Flag | Description |
|------|-------------|
| `.sanchaho_is_draw` | Whether triple ron (三家和, all non-discarders declaring Ron simultaneously) causes an abortive draw. When enabled (Tenhou), no scoring occurs and the round ends as a draw with renchan. When disabled (Mahjong Soul), all three Ron declarations are processed normally. |

## Platform-Specific Rule Presets

Differences in standard ranked match rules across major platforms and Mortal.

### 4-Player Presets

| Flag | `default_tenhou()` | `default_mjsoul()` | `default_mortal()` |
|------|--------|--------------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` | `False` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` | `False` |
| `.is_suuankou_tanki_double` | `False` | `True` | `False` |
| `.is_junsei_chuurenpoutou_double` | `False` | `True` | `False` |
| `.is_daisuushii_double` | `False` | `True` | `False` |
| `.yakuman_pao_is_liability_only` | `False` | `True` | `False` |
| `.sanchaho_is_draw` | `True` | `False` | `True` |

### 3-Player (Sanma) Presets

| Flag | `default_tenhou_sanma()` | `default_mjsoul_sanma()` | `default_mortal_sanma()` |
|------|--------|--------------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` | `False` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` | `False` |
| `.is_suuankou_tanki_double` | `False` | `True` | `False` |
| `.is_junsei_chuurenpoutou_double` | `False` | `True` | `False` |
| `.is_daisuushii_double` | `False` | `True` | `False` |
| `.yakuman_pao_is_liability_only` | `False` | `True` | `False` |
