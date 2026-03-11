from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from riichienv.visualizer import GameViewer

class GameRule:
    allows_ron_on_ankan_for_kokushi_musou: bool
    is_kokushi_musou_13machi_double: bool
    is_suuankou_tanki_double: bool
    is_junsei_chuurenpoutou_double: bool
    is_daisuushii_double: bool
    yakuman_pao_is_liability_only: bool
    sanchaho_is_draw: bool
    kuikae_forbidden: bool
    def __init__(
        self,
        allows_ron_on_ankan_for_kokushi_musou: bool = False,
        is_kokushi_musou_13machi_double: bool = False,
        is_suuankou_tanki_double: bool = False,
        is_junsei_chuurenpoutou_double: bool = False,
        is_daisuushii_double: bool = False,
        yakuman_pao_is_liability_only: bool = False,
        sanchaho_is_draw: bool = False,
        kuikae_forbidden: bool = True,
    ) -> None: ...
    @staticmethod
    def default_tenhou() -> GameRule: ...
    @staticmethod
    def default_mjsoul() -> GameRule: ...
    @staticmethod
    def default_tenhou_sanma() -> GameRule: ...

class Wind:
    East: Wind
    South: Wind
    West: Wind
    North: Wind
    def __int__(self) -> int: ...

class MeldType:
    Chi: MeldType
    Pon: MeldType
    Daiminkan: MeldType
    Ankan: MeldType
    Kakan: MeldType
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

class Action3P:
    action_type: ActionType
    tile: int | None
    consume_tiles: list[int]
    actor: int | None

    def __init__(
        self,
        type: ActionType = ActionType.Pass,  # noqa: A002
        tile: int | None = None,
        consume_tiles: list[int] = [],
        actor: int | None = None,
    ): ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_mjai(self) -> str: ...
    def encode(self) -> int: ...

class Meld:
    meld_type: MeldType
    tiles: list[int]
    opened: bool
    from_who: int
    def __init__(self, meld_type: MeldType, tiles: list[int], opened: bool, from_who: int = -1): ...

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
    riichi_sticks: int
    honba: int
    kita_count: int
    is_sanma: bool
    num_players: int
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
        riichi_sticks: int = 0,
        honba: int = 0,
        kita_count: int = 0,
        is_sanma: bool = False,
        num_players: int = 4,
    ): ...

class Yaku:
    id: int
    name: str
    name_en: str
    tenhou_id: int
    mjsoul_id: int
    def __repr__(self) -> str: ...

class WinResult:
    is_win: bool
    yakuman: bool
    ron_agari: int
    tsumo_agari_oya: int
    tsumo_agari_ko: int
    yaku: list[int]
    han: int
    fu: int
    pao_payer: int | None
    has_win_shape: bool
    def yaku_list(self) -> list[Yaku]: ...

class WinResultContext:
    actual: WinResult
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
    def calculate(self, calculator: HandEvaluator, conditions: Conditions | None = None) -> WinResult: ...
    def create_calculator(self) -> HandEvaluator: ...

class WinResultContextIterator:
    def __next__(self) -> WinResultContext: ...
    def __iter__(self) -> WinResultContextIterator: ...

class HandEvaluator:
    def __init__(self, tiles: list[int], melds: list[Meld] = []): ...
    def calc(
        self, win_tile: int, dora_indicators: list[int], ura_indicators: list[int], conditions: Conditions
    ) -> WinResult: ...
    def is_tenpai(self) -> bool: ...
    def get_waits(self) -> list[int]: ...
    @staticmethod
    def hand_from_text(text: str) -> HandEvaluator: ...

class HandEvaluator3P:
    def __init__(self, tiles: list[int], melds: list[Meld] = []): ...
    def calc(
        self,
        win_tile: int,
        dora_indicators: list[int] = [],
        ura_indicators: list[int] = [],
        conditions: Conditions | None = None,
    ) -> WinResult: ...
    def is_tenpai(self) -> bool: ...
    def get_waits(self) -> list[int]: ...
    def get_waits_u8(self) -> list[int]: ...
    @staticmethod
    def hand_from_text(text: str) -> HandEvaluator3P: ...

class Observation:
    events: list[Any]
    hand: list[int]
    player_id: int
    prev_events_size: int
    def new_events(self) -> list[str]:
        """Return MJAI JSON events unseen by this player since their previous observation.

        On a player's first observation in a hand, this may include
        ``start_game`` / ``start_kyoku`` and earlier events from the hand.
        """
        ...
    def legal_actions(self) -> list[Action]: ...
    def select_action_from_mjai(self, mjai: str | dict[str, Any]) -> Action | None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def serialize_to_base64(self) -> str: ...
    @staticmethod
    def deserialize_from_base64(s: str) -> Observation: ...
    def encode_discard_history_decay(self, decay_rate: float | None = None) -> bytes: ...
    def encode_yaku_possibility(self) -> bytes: ...
    def encode_furiten_ron_possibility(self) -> bytes: ...
    def __init__(self, *args: Any, **kwargs: Any): ...

class Kyoku:
    events: list[dict]
    rule: GameRule
    def grp_features(self) -> dict[str, Any]: ...
    def take_win_result_contexts(self) -> WinResultContextIterator: ...
    def take_grp_features(self) -> dict[str, Any]: ...
    def steps(
        self, seat: int | None = None, rule: GameRule | None = None, skip_single_action: bool | None = None
    ) -> KyokuStepIterator: ...
    def __iter__(self) -> KyokuIterator: ...

class KyokuIterator:
    def __next__(self) -> Any: ...
    def __iter__(self) -> KyokuIterator: ...

class KyokuStepIterator:
    def __next__(self) -> Any: ...
    def __iter__(self) -> KyokuStepIterator: ...

class MjSoulReplay:
    num_rounds: int
    @staticmethod
    def from_json(json_str: str) -> MjSoulReplay: ...
    @staticmethod
    def from_dict(paifu: dict) -> MjSoulReplay: ...
    def take_kyokus(self) -> list[Kyoku]: ...
    def verify(self) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class MjaiReplay:
    def __init__(self) -> None: ...
    @staticmethod
    def from_jsonl(path: str, rule: str | None = None) -> MjaiReplay: ...
    def num_rounds(self) -> int: ...
    def take_kyokus(self) -> KyokuIterator: ...

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
    skip_mjai_logging: bool
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
        skip_mjai_logging: bool = False,
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
    def step(self, actions: dict[int, Action | Action3P]) -> dict[int, Observation]: ...
    def done(self) -> bool: ...
    def get_observations(self, players: list[int] | None = None) -> dict[int, Observation]: ...
    def get_obs_py(self, player_id: int) -> Observation: ...
    def _check_midway_draws(self) -> Any: ...
    def _get_legal_actions(self, player_id: int) -> list[Action]: ...
    def _get_ura_markers(self) -> list[int]: ...
    def _get_ura_markers_u8(self) -> list[int]: ...
    def _get_waits(self, player_id: int) -> list[int]: ...
    def _is_furiten(self, player_id: int) -> bool: ...
    def get_viewer(self) -> GameViewer: ...
    def _reveal_kan_dora(self) -> None: ...
    def apply_event(self, event: dict[str, Any]) -> None:
        """Apply an MJAI event to advance the game state.

        Use this for replay parsing and training data generation where
        observations are obtained separately via ``get_observation()``.
        For online inference, prefer ``observe_event()``.
        """
        ...
    def observe_event(self, event: dict[str, Any], player_id: int) -> Observation | None:
        """Apply an MJAI event and return an Observation if *player_id* needs to act.

        Returns ``None`` for events that never require decisions
        (start_game, start_kyoku, dora, hora, ryukyoku, etc.) and when
        the tracked player has no legal actions after the event.

        This is the recommended API for online inference: feed events
        one at a time and act whenever a non-None observation is returned.
        """
        ...

class Score:
    total: int
    pay_ron: int
    pay_tsumo_oya: int
    pay_tsumo_ko: int

def calculate_score(han: int, fu: int, is_oya: bool, is_tsumo: bool, honba: int, num_players: int = 4) -> Score: ...
def calculate_shanten(hand_tiles: list[int]) -> int: ...
def calculate_shanten_3p(hand_tiles: list[int]) -> int: ...
def check_riichi_candidates(tiles: list[int]) -> list[int]: ...
def parse_hand(hand_str: str) -> tuple[list[int], list[Meld]]: ...
def parse_tile(tile_str: str) -> int: ...
def get_yaku_by_id(id_: int) -> Yaku | None: ...
def get_all_yaku() -> list[Yaku]: ...

__all__ = [
    "Action",
    "Action3P",
    "ActionType",
    "GameRule",
    "WinResult",
    "HandEvaluator",
    "HandEvaluator3P",
    "WinResultContext",
    "WinResultContextIterator",
    "Conditions",
    "Kyoku",
    "KyokuIterator",
    "Meld",
    "MeldType",
    "Observation",
    "Phase",
    "MjSoulReplay",
    "MjaiReplay",
    "RiichiEnv",
    "Score",
    "Wind",
    "calculate_score",
    "calculate_shanten",
    "calculate_shanten_3p",
    "check_riichi_candidates",
    "parse_hand",
    "parse_tile",
    "Yaku",
    "get_yaku_by_id",
    "get_all_yaku",
]
