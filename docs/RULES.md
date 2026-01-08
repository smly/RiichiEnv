# Game Rules Configuration

Detailed game mechanics can be configured using the `GameRule` struct.

## Kokushi Musou Rules

| Name | Flag | Description |
|------|-----------|-------------|
| Allows Ron on Ankan for Kokushi Musou | `.allows_ron_on_ankan_for_kokushi_musou` | Whether to allow Ron on a closed Kan (Ankan) for Kokushi Musou (Chankan). |
| Is Kokushi Musou 13-wait Double Yakuman | `.is_kokushi_musou_13machi_double` | Whether to treat Kokushi Musou 13-sided wait as a Double Yakuman. |

## Platform-Specific Rule Presets

Differences in standard ranked match rules across major platforms.

| Flag | Tenhou | Mahjong Soul |
|------|--------|--------------|
| `.allows_ron_on_ankan_for_kokushi_musou` | `False` | `True` |
| `.is_kokushi_musou_13machi_double` | `False` | `True` |

> [!NOTE]
> Configurable rule options are still under development and not yet exhaustive. If you find missing rules or have suggestions for additional configuration options, please report them via [GitHub Issues #43](https://github.com/smly/RiichiEnv/issues/43).