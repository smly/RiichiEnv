from enum import IntEnum
from typing import Any

class GameRule:
    allows_ron_on_ankan_for_kokushi_musou: bool
    is_kokushi_musou_13machi_double: bool
    def __init__(
        self,
        allows_ron_on_ankan_for_kokushi_musou: bool = False,
        is_kokushi_musou_13machi_double: bool = False,
    ) -> None: ...
    @staticmethod
    def default_tenhou() -> GameRule: ...
    @staticmethod
    def default_mjsoul() -> GameRule: ...

class Wind:
    East: Wind
    South: Wind
    West: Wind
    North: Wind
    def __int__(self) -> int: ...

class MeldType:
    Chi: MeldType
    Peng: MeldType
    Gang: MeldType
    Angang: MeldType
    Addgang: MeldType
    def __int__(self) -> int: ...

class Phase(IntEnum):
    WaitAct = 0
    WaitResponse = 1

class ActionType:
    Discard: ActionType
    Chi: ActionType
    Pon: ActionType
    Daiminkan: ActionType
    Ankan: ActionType
    Kakan: ActionType
    Riichi: ActionType
    Tsumo: ActionType
    Ron: ActionType
    Pass: ActionType
    KyushuKyuhai: ActionType
    # Upper case aliases
    DISCARD: ActionType
    CHI: ActionType
    PON: ActionType
    DAIMINKAN: ActionType
    ANKAN: ActionType
    KAKAN: ActionType
    RIICHI: ActionType
    TSUMO: ActionType
    RON: ActionType
    PASS: ActionType
    KYUSHU_KYUHAI: ActionType
    def __int__(self) -> int: ...

class Action:
    action_type: ActionType
    tile: int
    consume_tiles: list[int]

    def __init__(self, action_type: ActionType, tile: int = 0, consume_tiles: list[int] = []): ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_mjai(self) -> str: ...

class Meld:
    meld_type: MeldType
    tiles: list[int]
    opened: bool
    def __init__(self, meld_type: MeldType, tiles: list[int], opened: bool): ...

class Conditions:
    tsumo: bool
    riichi: bool
    double_riichi: bool
    ippatsu: bool
    haitei: bool
    houtei: bool
    rinshan: bool
    chankan: bool
    tsumo_first_turn: bool
    player_wind: Wind
    round_wind: Wind
    kyoutaku: int
    tsumi: int
    def __init__(
        self,
        tsumo: bool = False,
        riichi: bool = False,
        double_riichi: bool = False,
        ippatsu: bool = False,
        haitei: bool = False,
        houtei: bool = False,
        rinshan: bool = False,
        chankan: bool = False,
        tsumo_first_turn: bool = False,
        player_wind: Wind | int = 0,
        round_wind: Wind | int = 0,
        kyoutaku: int = 0,
        tsumi: int = 0,
    ): ...

class Agari:
    agari: bool
    yakuman: bool
    ron_agari: int
    tsumo_agari_oya: int
    tsumo_agari_ko: int
    yaku: list[int]
    han: int
    fu: int

class AgariContext:
    actual: bool
    agari_tile: int
    conditions: Conditions
    dora_indicators: list[int]
    expected_fu: int
    expected_han: int
    expected_yaku: list[int]
    melds: list[Meld]
    seat: int
    tiles: list[int]
    ura_indicators: list[int]
    def calculate(self) -> Agari: ...
    def create_calculator(self) -> AgariCalculator: ...
    def __iter__(self) -> AgariContextIterator: ...

class AgariContextIterator:
    def __next__(self) -> AgariContext: ...
    def __iter__(self) -> AgariContextIterator: ...

class AgariCalculator:
    def __init__(self, tiles: list[int], melds: list[Meld]): ...
    def calc(
        self, win_tile: int, dora_indicators: list[int], ura_indicators: list[int], conditions: Conditions
    ) -> Agari: ...
    def is_tenpai(self) -> bool: ...
    def get_waits(self) -> list[int]: ...
    @staticmethod
    def hand_from_text(text: str) -> AgariCalculator: ...

class Observation:
    events: list[Any]
    hand: list[int]
    player_id: int
    prev_events_size: int
    def new_events(self) -> list[str]: ...
    def legal_actions(self) -> list[Action]: ...
    def select_action_from_mjai(self, mjai: str | dict[str, Any]) -> Action | None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __init__(self, *args: Any, **kwargs: Any): ...

class Kyoku:
    events: list[dict]
    def take_agari_contexts(self) -> list[list[AgariContext]]: ...
    def __iter__(self) -> KyokuIterator: ...

class KyokuIterator:
    def __next__(self) -> Any: ...
    def __iter__(self) -> KyokuIterator: ...

class ReplayGame:
    num_rounds: int
    @staticmethod
    def from_json(json_str: str) -> ReplayGame: ...
    def take_kyokus(self) -> list[Kyoku]: ...
    def verify(self) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any): ...

class RiichiEnv:
    oya: int
    riichi_sticks: int
    honba: int
    current_player: int
    phase: Phase
    needs_tsumo: bool
    drawn_tile: int | None
    hands: list[list[int]]
    riichi_declared: list[bool]
    wall: list[int]
    discards: list[list[int]]
    mjai_log: list[dict[str, Any]]
    _custom_honba: int
    _custom_round_wind: int

    active_players: list[int]
    agari_results: list[Any]
    current_claims: dict[int, Any]
    dora_indicators: list[int]
    double_riichi_declared: list[bool]
    forbidden_discards: list[list[int]]
    game_type: Any
    ippatsu_cycle: list[bool]
    is_done: bool
    is_first_turn: bool
    is_rinshan_flag: bool
    kyoku_idx: int
    last_agari_results: Any
    last_discard: tuple[int, int] | None
    melds: list[list[Meld]]
    missed_agari_doujun: list[bool]
    missed_agari_riichi: list[bool]
    skip_mjai_logging: bool  # If True, disables MJAI event logging for performance.
    nagashi_eligible: list[bool]
    needs_initialize_next_round: bool
    pending_is_draw: bool
    pending_kan: int | None
    pending_kan_dora_count: int
    pending_oya_won: bool
    player_event_counts: list[int]
    points: list[int]
    riichi_pending_acceptance: int | None
    riichi_stage: list[int]
    rinshan_draw_count: int
    round_end_scores: list[int]
    round_wind: int
    salt: str
    score_deltas: list[int]
    seed: int
    turn_count: int
    wall_digest: str
    pao: list[dict[int, int]]

    def __init__(
        self,
        game_mode: str | int | None = None,
        skip_mjai_logging: bool = False,  # If True, disables MJAI logging (required for visualizer).
        seed: int | None = None,
        round_wind: int | None = None,
        rule: GameRule | None = None,
    ) -> None: ...
    @property
    def game_mode(self) -> int: ...
    def scores(self) -> list[int]: ...
    def points(self) -> list[int]: ...
    def ranks(self) -> list[int]: ...
    def reset(
        self, oya: int | None = None, honba: int | None = None, *args: Any, **kwargs: Any
    ) -> dict[int, Observation]: ...
    def step(
        self, action: Action | int | dict[int, Action] | None = None, *args: Any, **kwargs: Any
    ) -> dict[int, Observation]: ...
    def done(self) -> bool: ...
    def get_observations(self, players: list[int] | None = None) -> dict[int, Observation]: ...
    def get_obs_py(self, player_id: int) -> Observation: ...
    def _check_midway_draws(self) -> Any: ...
    def _get_legal_actions(self, player_id: int) -> list[Action]: ...
    def _get_ura_markers(self) -> list[int]: ...
    def _get_ura_markers_u8(self) -> list[int]: ...
    def _get_waits(self, player_id: int) -> list[int]: ...
    def _is_furiten(self, player_id: int) -> bool: ...
    def _reveal_kan_dora(self) -> None: ...

class Score:
    total: int
    def pay_ron(self, *args: Any, **kwargs: Any) -> None: ...
    def pay_tsumo_ko(self, *args: Any, **kwargs: Any) -> None: ...
    def pay_tsumo_oya(self, *args: Any, **kwargs: Any) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any): ...

def calculate_score(han: int, fu: int, is_oya: bool, is_tsumo: bool) -> tuple[int, int]: ...
def check_riichi_candidates(tiles: list[int]) -> list[int]: ...
def parse_hand(hand_str: str) -> tuple[list[int], list[Meld]]: ...
def parse_tile(tile_str: str) -> int: ...

__all__ = [
    "Action",
    "ActionType",
    "GameRule",
    "Agari",
    "AgariCalculator",
    "AgariContext",
    "AgariContextIterator",
    "Conditions",
    "Kyoku",
    "KyokuIterator",
    "Meld",
    "MeldType",
    "Observation",
    "Phase",
    "ReplayGame",
    "RiichiEnv",
    "Score",
    "Wind",
    "calculate_score",
    "check_riichi_candidates",
    "parse_hand",
    "parse_tile",
]
