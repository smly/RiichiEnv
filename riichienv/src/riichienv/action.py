import json
from dataclasses import dataclass, field
from enum import IntEnum

from .convert import tid_to_mjai


class ActionType(IntEnum):
    DISCARD = 0
    CHI = 1
    PON = 2
    DAIMINKAN = 3  # Open Kan
    RON = 4  # Claim win
    RIICHI = 5  # Declare Riichi
    TSUMO = 6  # Self-draw win
    PASS = 7  # Pass on claim
    ANKAN = 8  # Closed Kan
    KAKAN = 9  # Add Kan (Chankan)
    KYUSHU_KYUHAI = 10  # Nine Terminal Abortive Draw


@dataclass
class Action:
    type: ActionType
    tile: int | None = None  # For Discard (tile 136 id) or Claim (target tile if needed)
    consume_tiles: list[int] = field(default_factory=list)  # For Meld (tiles from hand to use)

    def to_dict(self):
        return {"type": self.type.value, "tile": self.tile, "consume_tiles": self.consume_tiles}

    @staticmethod
    def from_dict(d):
        return Action(type=ActionType(d["type"]), tile=d.get("tile"), consume_tiles=d.get("consume_tiles", []))

    def to_mjai(self) -> str:
        """
        Convert Action to MJAI protocol JSON string.
        """
        data = {}
        if self.type == ActionType.DISCARD:
            # "tsumogiri" flag is not known here, usually optional for action submission
            data = {"type": "dahai", "pai": tid_to_mjai(self.tile)}
        elif self.type == ActionType.CHI:
            data = {
                "type": "chi",
                "pai": tid_to_mjai(self.tile),
                "consumed": [tid_to_mjai(t) for t in self.consume_tiles],
            }
        elif self.type == ActionType.PON:
            data = {
                "type": "pon",
                "pai": tid_to_mjai(self.tile),
                "consumed": [tid_to_mjai(t) for t in self.consume_tiles],
            }
        elif self.type == ActionType.DAIMINKAN:
            data = {
                "type": "daiminkan",
                "pai": tid_to_mjai(self.tile),
                "consumed": [tid_to_mjai(t) for t in self.consume_tiles],
            }
        elif self.type == ActionType.ANKAN:
            # Ankan: consumed implies which tiles. pai is not needed if consumed is full?
            # MJAI ankan: {"type":"ankan", "consumed": [...]}
            data = {
                "type": "ankan",
                "consumed": [tid_to_mjai(t) for t in self.consume_tiles],
            }
        elif self.type == ActionType.KAKAN:
            data = {
                "type": "kakan",
                "pai": tid_to_mjai(self.tile),
                "consumed": [tid_to_mjai(t) for t in self.consume_tiles],
            }
        elif self.type == ActionType.RIICHI:
            data = {"type": "reach"}
        elif self.type == ActionType.TSUMO:
            data = {"type": "hora"}
        elif self.type == ActionType.RON:
            data = {"type": "hora"}
        elif self.type == ActionType.KYUSHU_KYUHAI:
            data = {"type": "ryukyoku"}
        elif self.type == ActionType.PASS:
            data = {"type": "none"}
        else:
            raise ValueError(f"Unknown ActionType for MJAI conversion: {self.type}")

        return json.dumps(data, separators=(",", ":"))
