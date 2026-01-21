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

## Kokushi Musou Rules

| Flag | Description |
|------|-------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | Whether to allow Ron on a closed Kan (Ankan) for Kokushi Musou (Chankan). |
| `.is_kokushi_musou_13machi_double` | Whether to treat Kokushi Musou 13-sided wait as a Double Yakuman. |

## Platform-Specific Rule Presets

Differences in standard ranked match rules across major platforms.

| Flag | Tenhou | Mahjong Soul |
|------|--------|--------------|
| `.yakuman_pao_is_liability_only` | `False` | `True` |
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` |
