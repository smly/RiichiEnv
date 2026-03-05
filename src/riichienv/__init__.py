from . import consts, convert
from ._riichienv import (  # type: ignore
    GameRule,
    Kyoku,
    Meld,
    MeldType,
    MjaiReplay,
    MjSoulReplay,
    Observation,
    Phase,
    RiichiEnv,
    Score,
    Wind,
    WinResult,
    WinResultContext,
    Yaku,
    calculate_score,
    calculate_shanten,
    calculate_shanten_3p,
    check_riichi_candidates,
    get_all_yaku,
    get_yaku_by_id,
    parse_hand,
    parse_tile,
)
from .action import Action, Action3P, ActionType
from .game_mode import GameType
from .hand import Conditions, HandEvaluator, HandEvaluator3P

EAST = Wind.East
SOUTH = Wind.South
WEST = Wind.West
NORTH = Wind.North


def _get_viewer(self):  # type: ignore[no-untyped-def]
    # Lazy import to avoid circular dependency: visualizer imports from riichienv
    from riichienv.visualizer import GameViewer  # noqa: PLC0415

    return GameViewer(self.mjai_log)


RiichiEnv.get_viewer = _get_viewer  # type: ignore[attr-defined]


__all__ = [
    "consts",
    "convert",
    "WinResultContext",
    "Kyoku",
    "Meld",
    "MeldType",
    "Observation",
    "MjSoulReplay",
    "MjaiReplay",
    "Score",
    "Wind",
    "calculate_score",
    "calculate_shanten",
    "calculate_shanten_3p",
    "check_riichi_candidates",
    "parse_hand",
    "parse_tile",
    "Action",
    "Action3P",
    "ActionType",
    "RiichiEnv",
    "GameRule",
    "Phase",
    "GameType",
    "WinResult",
    "HandEvaluator",
    "HandEvaluator3P",
    "Conditions",
    "EAST",
    "SOUTH",
    "WEST",
    "NORTH",
    "Yaku",
    "get_yaku_by_id",
    "get_all_yaku",
]
