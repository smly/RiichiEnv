from dataclasses import dataclass


@dataclass
class Meld:
    type: str  # "chi", "pon", "kan", "nuki"
    tiles: list[int]  # 136-based tile IDs
    opened: bool = True
    called_tile: int | None = None  # The tile claimed
    who: int = 0
    # Absolute player ID (seat index) 0-3
    # matching mahjong.meld: index in the players list, not relative to the current player

    # Constants to match mahjong.meld.Meld usage or preferred usage
    CHI = "chi"
    PON = "pon"
    KAN = "kan"
    NUKI = "nuki"  # Kita?
