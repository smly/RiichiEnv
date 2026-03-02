from ._riichienv import Action, Action3P, ActionType  # type: ignore

__all__ = ["Action", "Action3P", "ActionType"]

# Uppercase aliases for backward compatibility
ActionType.DISCARD = ActionType.Discard
ActionType.CHI = ActionType.Chi
ActionType.PON = ActionType.Pon
ActionType.DAIMINKAN = ActionType.Daiminkan
ActionType.RON = ActionType.Ron
ActionType.RIICHI = ActionType.Riichi
ActionType.TSUMO = ActionType.Tsumo
ActionType.PASS = ActionType.Pass
ActionType.ANKAN = ActionType.Ankan
ActionType.KAKAN = ActionType.Kakan
ActionType.KYUSHU_KYUHAI = ActionType.KyushuKyuhai
