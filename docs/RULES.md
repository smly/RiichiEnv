# Game Rules Configuration

Detailed game mechanics can be configured using the `GameRule` struct.

> [!NOTE]
> Configurable rule options are still under development and not yet exhaustive. If you find missing rules or have suggestions for additional configuration options, please report them via [GitHub Issues #43](https://github.com/smly/RiichiEnv/issues/43).

## "Responsibility Payment" behavior for composite Yakumans

- **Tenhou**: Pao player pays the full amount for Tsumo (including non-Pao Yakumans) and splits the full amount for Ron.
- **Mahjong Soul**: Pao player is liable only for the Pao-triggering Yakuman part. The remaining Yakumans are treated as normal settlement (split among all for Tsumo, or paid by deal-in player for Ron).

| Flag | Description |
|------|-------------|
| `.yakuman_pao_is_liability_only` | Whether to limit Pao liability to the specific Pao-triggering Yakuman only (Mahjong Soul style). If false, Pao covers the full amount (Tenhou style). |

## Kuikae (Swap Calling) Mode

Controls whether players can discard a tile that completes the same sequence they just called.

| Mode | Description |
|------|-------------|
| `KuikaeMode.None` | No kuikae restriction. |
| `KuikaeMode.Basic` | Basic kuikae restriction (cannot discard the called tile). |
| `KuikaeMode.StrictFlank` | Strict kuikae restriction including flank tiles (Tenhou/MJSoul standard). |

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

## Open Kan Dora Reveal Timing

Controls when dora indicators are revealed after an open kan (Daiminkan/Kakan) declaration.

Ankan (closed kan) always reveals dora immediately before the rinshan tsumo, regardless of this flag.

| Flag | Description |
|------|-------------|
| `.open_kan_dora_after_discard` | Whether open kan (Daiminkan/Kakan) dora is revealed after the discard. When `True` (Mahjong Soul style), dora is revealed after the discard. When `False` (Tenhou style), dora is revealed before the discard. |

### Event Order

**Ankan (Closed Kan)** - always the same:
```
ankan → dora → tsumo (rinshan) → dahai
```
Note: Rinshan kaihou (winning on rinshan draw) includes the kan dora.

**Kakan/Daiminkan (Open/Added Kan)**:
```
kakan → tsumo (rinshan) → dahai → dora
daiminkan → tsumo (rinshan) → dahai → dora
```

### Usage

```python
from riichienv import RiichiEnv, GameRule

# Tenhou style: open kan dora before discard (default for default_tenhou())
rule = GameRule(open_kan_dora_after_discard=False)
env = RiichiEnv(rule=rule)

# Mahjong Soul style: open kan dora after discard (default for default_mjsoul())
rule = GameRule(open_kan_dora_after_discard=True)
env = RiichiEnv(rule=rule)
```

## Platform-Specific Rule Presets

Differences in standard ranked match rules across major platforms.

### 4-Player Presets

| Flag | `default_tenhou()` | `default_mjsoul()` |
|------|--------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` |
| `.is_suuankou_tanki_double` | `False` | `True` |
| `.is_junsei_chuurenpoutou_double` | `False` | `True` |
| `.is_daisuushii_double` | `False` | `True` |
| `.yakuman_pao_is_liability_only` | `False` | `True` |
| `.sanchaho_is_draw` | `True` | `False` |

| `.kuikae_mode` | `StrictFlank` | `StrictFlank` |
| `.open_kan_dora_after_discard` | `False` | `True` |

### 3-Player (Sanma) Presets

| Flag | `default_tenhou_sanma()` | `default_mjsoul_sanma()` |
|------|--------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` |
| `.is_suuankou_tanki_double` | `False` | `True` |
| `.is_junsei_chuurenpoutou_double` | `False` | `True` |
| `.is_daisuushii_double` | `False` | `True` |
| `.yakuman_pao_is_liability_only` | `False` | `True` |

| `.kuikae_mode` | `StrictFlank` | `StrictFlank` |
| `.open_kan_dora_after_discard` | `False` | `True` |
