from ._riichienv import Action, Action3P, ActionType

__all__ = ["Action", "Action3P", "ActionType"]

# PascalCase aliases for backward compatibility (deprecated)
ActionType.Discard = ActionType.DISCARD
ActionType.Chi = ActionType.CHI
ActionType.Pon = ActionType.PON
ActionType.Daiminkan = ActionType.DAIMINKAN
ActionType.Ron = ActionType.RON
ActionType.Riichi = ActionType.RIICHI
ActionType.Tsumo = ActionType.TSUMO
ActionType.Pass = ActionType.PASS
ActionType.Ankan = ActionType.ANKAN
ActionType.Kakan = ActionType.KAKAN
ActionType.KyushuKyuhai = ActionType.KYUSHU_KYUHAI
ActionType.Kita = ActionType.KITA
