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
    DISCARD: ActionType
    CHI: ActionType
    PON: ActionType
    DAIMINKAN: ActionType
    RON: ActionType
    RIICHI: ActionType
    TSUMO: ActionType
    PASS: ActionType
    ANKAN: ActionType
    KAKAN: ActionType
    KYUSHU_KYUHAI: ActionType
    KITA: ActionType
    # PascalCase aliases (deprecated)
    Discard: ActionType
    Chi: ActionType
    Pon: ActionType
    Daiminkan: ActionType
    Ron: ActionType
    Riichi: ActionType
    Tsumo: ActionType
    Pass: ActionType
    Ankan: ActionType
    Kakan: ActionType
    KyushuKyuhai: ActionType
    Kita: ActionType
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
        type: ActionType = ActionType.PASS,  # noqa: A002
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
    """A 4-player (yonma) game observation for a single player.

    Contains the player's hand, visible game events, and the set of
    legal actions available at the current decision point.  Returned by
    :meth:`RiichiEnv.step`, :meth:`RiichiEnv.reset`, and related methods.

    Attributes:
        player_id: Seat index of the observing player (0-3).
        hand: Tile ids currently in the player's hand.
        events: Accumulated MJAI JSON event strings visible to this player.
        prev_events_size: Number of events already consumed; used internally
            by :meth:`new_events` to return only unseen events.
    """

    events: list[Any]
    hand: list[int]
    player_id: int
    prev_events_size: int
    def new_events(self) -> list[str]:
        """Return MJAI JSON events unseen by this player since their previous observation.

        Each call returns only the events accumulated since the last observation
        for this player.  On the first observation of a hand this may include
        ``start_game``, ``start_kyoku``, and earlier events from the hand.

        Returns:
            A list of MJAI JSON strings, one per event.

        Example::

            obs = observations[player_id]
            for ev_json in obs.new_events():
                ev = json.loads(ev_json)
                print(ev["type"])   # e.g. "tsumo", "dahai", "pon", ...
        """
        ...
    def legal_actions(self) -> list[Action]:
        """Return the list of legal actions available to the player.

        The returned actions represent every valid move the player can make
        at the current decision point (e.g. discard a tile, declare riichi,
        call pon/chi/kan, tsumo, ron, or pass).

        Returns:
            A list of :class:`Action` objects.  An empty list means the
            player has no decision to make at this point.

        Example::

            obs = observations[player_id]
            actions = obs.legal_actions()
            # Pick a random legal action
            import random
            chosen = random.choice(actions)
            observations = env.step({player_id: chosen})
        """
        ...
    def select_action_from_mjai(self, mjai: str | dict[str, Any]) -> Action | None:
        """Find the legal action matching an MJAI event, or ``None`` if no match."""
        ...
    def to_dict(self) -> dict[str, Any]:
        """Convert this observation to a plain dictionary."""
        ...
    def serialize_to_base64(self) -> str:
        """Serialize to a base64-encoded JSON string."""
        ...
    @staticmethod
    def deserialize_from_base64(s: str) -> Observation:
        """Deserialize from a base64-encoded JSON string."""
        ...
    def encode(self) -> bytes:
        """Encode the observation into a compact binary feature representation.

        Produces a ``(74, 34)`` float32 feature tensor serialized as raw bytes.
        The 74 channels include hand tile counts, melds, dora indicators,
        discard history, riichi status, wind, scores, waits, and more.

        Returns:
            Raw bytes of length ``74 * 34 * 4`` (float32).

        Example::

            import numpy as np

            obs = observations[player_id]
            buf = obs.encode()
            features = np.frombuffer(buf, dtype=np.float32).reshape(74, 34)
        """
        ...
    def encode_discard_history_decay(self, decay_rate: float | None = None) -> bytes:
        """Encode discard history with exponential decay.

        Shape: ``(4, 34)`` / dtype: ``float32``.

        Each of the 4 players has a 34-tile plane where earlier discards
        are exponentially decayed toward zero.
        """
        ...
    def encode_yaku_possibility(self) -> bytes:
        """Encode yaku possibility features for each player.

        Shape: ``(4, 21, 2)`` / dtype: ``float32``.

        For each of the 4 players, 21 yaku types × 2 (tsumo / ron).
        """
        ...
    def encode_furiten_ron_possibility(self) -> bytes:
        """Encode furiten / ron possibility features.

        Shape: ``(4, 21)`` / dtype: ``float32``.

        For each of the 4 players, ron-possibility flags based on
        tsumogiri patterns across 21 yaku types.
        """
        ...
    def encode_shanten_efficiency(self) -> bytes:
        """Encode shanten and tile efficiency features.

        Shape: ``(4, 4)`` / dtype: ``float32``.

        For each of the 4 players: ``[shanten, effective_tiles,
        best_ukeire, turn_count]``.
        """
        ...
    def encode_kawa_overview(self) -> bytes:
        """Encode river (kawa) overview features for all players.

        Shape: ``(4, 7, 34)`` / dtype: ``float32``.

        For each of the 4 players, 7 channels
        (count1–count4, aka5m, aka5p, aka5s) × 34 tile types.
        """
        ...
    def encode_fuuro_overview(self) -> bytes:
        """Encode open-meld (fuuro) overview features for all players.

        Shape: ``(4, 4, 5, 34)`` / dtype: ``float32``.

        For each of the 4 players, up to 4 melds × 5 channels
        (tile1–tile4, aka) × 34 tile types.
        """
        ...
    def encode_ankan_overview(self) -> bytes:
        """Encode closed-kan (ankan) overview features for all players.

        Shape: ``(4, 34)`` / dtype: ``float32``.

        Binary flags indicating which tile types each player has
        declared as closed kan.
        """
        ...
    def encode_action_availability(self) -> bytes:
        """Encode which action types are currently available.

        Shape: ``(11,)`` / dtype: ``float32``.

        Flags for: ``[riichi, chi_low, chi_mid, chi_high, pon,
        daiminkan, ankan, kakan, agari, kyuushukyuuhai, pass]``.
        """
        ...
    def encode_riichi_sutehais(self) -> bytes:
        """Encode the riichi declaration discard tiles for all players.

        Shape: ``(3, 3)`` / dtype: ``float32``.

        For each of the 3 opponents: ``[tile_type, is_aka, is_dora]``.
        """
        ...
    def encode_last_tedashis(self) -> bytes:
        """Encode the last hand-picked discard for all players.

        Shape: ``(3, 3)`` / dtype: ``float32``.

        For each of the 3 opponents: ``[tile_type, is_aka, is_dora]``.
        """
        ...
    def encode_pass_context(self) -> bytes:
        """Encode contextual features relevant to the pass action.

        Shape: ``(3,)`` / dtype: ``float32``.

        Context about the currently offered tile:
        ``[tile_type, is_aka, is_dora]``.
        """
        ...
    def encode_discard_candidates(self) -> bytes:
        """Encode candidate tiles for discard selection.

        Shape: ``(5,)`` / dtype: ``float32``.

        ``[hand_size, keep_shanten_ratio, increase_shanten_ratio,
        is_tenpai, riichi_declared]``.
        """
        ...
    def encode_extended(self) -> bytes:
        """Encode extended features (combination of multiple feature planes).

        Shape: ``(215, 34)`` / dtype: ``float32``.

        Concatenation of base (74) + discard_decay (4) + shanten (16) +
        ankan (4) + fuuro (80) + action_avail (11) + discard_cand (5) +
        pass_ctx (3) + last_tedashis (9) + riichi_sutehais (9) = 215
        channels, each with 34 tile-type columns.

        Example::

            import numpy as np

            buf = obs.encode_extended()
            features = np.frombuffer(buf, dtype=np.float32).reshape(215, 34)
        """
        ...
    def encode_seq_sparse(self, game_style: int = 1) -> bytes:
        """Encode sequence features as sparse token ids.

        Shape: variable-length 1-D array (up to 25 elements) / dtype: ``uint16``.
        """
        ...
    def encode_seq_numeric(self) -> bytes:
        """Encode sequence features as numeric values.

        Shape: ``(12,)`` / dtype: ``float32``.
        """
        ...
    def encode_seq_progression(self) -> bytes:
        """Encode sequence progression features.

        Shape: ``(N, 5)`` / dtype: ``uint16``.

        Each row represents one action-history step with 5 token values.
        """
        ...
    def encode_seq_candidates(self) -> bytes:
        """Encode sequence candidate features.

        Shape: ``(M, 4)`` / dtype: ``uint16``.

        Each row represents one legal action candidate with 4 token values.
        """
        ...
    def __init__(self, *args: Any, **kwargs: Any): ...

class Observation3P:
    """A 3-player (sanma) game observation for a single player.

    Similar to :class:`Observation` but tailored for 3-player mahjong.
    Arrays indexed by player use length-3 lists ordered as
    ``[self, shimocha, kamicha]`` (relative seat order).

    Attributes:
        player_id: Seat index of the observing player (0-2).
        hands: Tile ids for each player's hand (own hand is fully visible;
            others are empty unless revealed).
        melds: Open / closed melds for each player.
        discards: Discarded tile ids for each player.
        dora_indicators: Currently visible dora indicator tiles.
        scores: Point totals for each player.
        riichi_declared: Whether each player has declared riichi.
        honba: Current honba (repeat) counter.
        riichi_sticks: Number of riichi sticks on the table.
        round_wind: Round wind (0=East, 1=South, 2=West).
        oya: Seat index of the dealer.
        kyoku_index: Current kyoku (hand) number within the round.
        waits: Winning tile ids if the player is tenpai.
        is_tenpai: Whether the player is currently tenpai.
        tsumogiri_flags: Per-player flags indicating tsumogiri for each discard.
        riichi_sutehais: The tile discarded for riichi declaration per player,
            or ``None`` if not yet declared.
        last_tedashis: The last tedashi (hand-picked discard) per player,
            or ``None``.
        last_discard: The most recently discarded tile id, or ``None``.
    """

    player_id: int
    hands: list[list[int]]
    melds: list[list[Meld]]
    discards: list[list[int]]
    dora_indicators: list[int]
    scores: list[int]
    riichi_declared: list[bool]
    honba: int
    riichi_sticks: int
    round_wind: int
    oya: int
    kyoku_index: int
    waits: list[int]
    is_tenpai: bool
    tsumogiri_flags: list[list[bool]]
    riichi_sutehais: list[int | None]
    last_tedashis: list[int | None]
    last_discard: int | None
    @property
    def hand(self) -> list[int]:
        """Shorthand for ``hands[player_id]``."""
        ...
    def events(self) -> list[Any]:
        """Return accumulated MJAI JSON events visible to this player."""
        ...
    def legal_actions(self) -> list[Action3P]:
        """Return the list of legal actions available to the player.

        The returned actions represent every valid move the player can make
        at the current decision point (e.g. discard a tile, declare riichi,
        call pon/kan, tsumo, ron, kita, or pass).

        Returns:
            A list of :class:`Action3P` objects.  An empty list means the
            player has no decision to make at this point.

        Example::

            obs = observations[player_id]
            actions = obs.legal_actions()
            # Pick a random legal action
            import random
            chosen = random.choice(actions)
            observations = env.step({player_id: chosen})
        """
        ...
    def mask(self) -> bytes:
        """Return a boolean action mask as raw bytes."""
        ...
    def action_space_size(self) -> int:
        """Return the total action space size for 3-player mahjong."""
        ...
    def find_action(self, action_id: int) -> Action3P | None:
        """Find the legal action whose encoded id equals *action_id*, or ``None``."""
        ...
    def select_action_from_mjai(self, mjai_data: str | dict[str, Any]) -> Action3P | None:
        """Find the legal action matching an MJAI event, or ``None`` if no match."""
        ...
    def new_events(self) -> list[str]:
        """Return MJAI JSON events unseen by this player since their previous observation.

        Each call returns only the events accumulated since the last observation
        for this player.  On the first observation of a hand this may include
        ``start_game``, ``start_kyoku``, and earlier events from the hand.

        Returns:
            A list of MJAI JSON strings, one per event.

        Example::

            obs = observations[player_id]
            for ev_json in obs.new_events():
                ev = json.loads(ev_json)
                print(ev["type"])   # e.g. "tsumo", "dahai", "pon", ...
        """
        ...
    def to_dict(self) -> dict[str, Any]:
        """Convert this observation to a plain dictionary."""
        ...
    def serialize_to_base64(self) -> str:
        """Serialize to a base64-encoded JSON string."""
        ...
    @staticmethod
    def deserialize_from_base64(s: str) -> Observation3P:
        """Deserialize from a base64-encoded JSON string."""
        ...
    def encode_discard_history_decay(self, decay_rate: float | None = None) -> bytes:
        """Encode discard history with exponential decay.

        Shape: ``(3, 27)`` / dtype: ``float32``.

        Each of the 3 players has a 27-tile plane where earlier discards
        are exponentially decayed toward zero.
        """
        ...
    def encode_furiten_ron_possibility(self) -> bytes:
        """Encode furiten / ron possibility features.

        Shape: ``(3, 21)`` / dtype: ``float32``.

        For each of the 3 players, ron-possibility flags based on
        tsumogiri patterns across 21 yaku types.
        """
        ...
    def encode_yaku_possibility(self) -> bytes:
        """Encode yaku possibility features for each player.

        Shape: ``(3, 21, 2)`` / dtype: ``float32``.

        For each of the 3 players, 21 yaku types × 2 (tsumo / ron).
        """
        ...
    def encode(self) -> bytes:
        """Encode the observation into a compact binary feature representation.

        Shape: ``(74, 27)`` / dtype: ``float32``.

        Produces a ``(74, 27)`` float32 feature tensor serialized as raw
        bytes.  The 74 channels include hand tile counts, melds, dora
        indicators, discard history, riichi status, wind, scores, waits,
        and more.  The tile dimension uses the 27-tile compact encoding
        for 3-player mahjong.

        Returns:
            Raw bytes of length ``74 * 27 * 4`` (float32).

        Example::

            import numpy as np

            obs = observations[player_id]
            buf = obs.encode()
            features = np.frombuffer(buf, dtype=np.float32).reshape(74, 27)
        """
        ...
    def encode_shanten_efficiency(self) -> bytes:
        """Encode shanten and tile efficiency features.

        Shape: ``(3, 4)`` / dtype: ``float32``.

        For each of the 3 players: ``[shanten, effective_tiles,
        best_ukeire, turn_count]``.  ``effective_tiles`` is normalized
        by 27 (compact tile count).
        """
        ...
    def encode_kawa_overview(self) -> bytes:
        """Encode river (kawa) overview features for all players.

        Shape: ``(3, 7, 27)`` / dtype: ``float32``.

        For each of the 3 players, 7 channels
        (count1–count4, aka5m, aka5p, aka5s) × 27 compact tile types.
        """
        ...
    def encode_fuuro_overview(self) -> bytes:
        """Encode open-meld (fuuro) overview features for all players.

        Shape: ``(3, 4, 5, 27)`` / dtype: ``float32``.

        For each of the 3 players, up to 4 melds × 5 channels
        (tile1–tile4, aka) × 27 compact tile types.
        """
        ...
    def encode_ankan_overview(self) -> bytes:
        """Encode closed-kan (ankan) overview features for all players.

        Shape: ``(3, 27)`` / dtype: ``float32``.

        Binary flags indicating which tile types each player has
        declared as closed kan.
        """
        ...
    def encode_action_availability(self) -> bytes:
        """Encode which action types are currently available.

        Shape: ``(11,)`` / dtype: ``float32``.

        Flags for: ``[riichi, chi_low, chi_mid, chi_high, pon,
        daiminkan, ankan, kakan, agari, kyuushukyuuhai, pass]``.
        """
        ...
    def encode_riichi_sutehais(self) -> bytes:
        """Encode the riichi declaration discard tiles for all players.

        Shape: ``(2, 3)`` / dtype: ``float32``.

        For each of the 2 opponents: ``[tile_type, is_aka, is_dora]``.
        """
        ...
    def encode_last_tedashis(self) -> bytes:
        """Encode the last hand-picked discard for all players.

        Shape: ``(2, 3)`` / dtype: ``float32``.

        For each of the 2 opponents: ``[tile_type, is_aka, is_dora]``.
        """
        ...
    def encode_pass_context(self) -> bytes:
        """Encode contextual features relevant to the pass action.

        Shape: ``(3,)`` / dtype: ``float32``.

        Context about the currently offered tile:
        ``[tile_type, is_aka, is_dora]``.
        """
        ...
    def encode_discard_candidates(self) -> bytes:
        """Encode candidate tiles for discard selection.

        Shape: ``(5,)`` / dtype: ``float32``.

        ``[hand_size, keep_shanten_ratio, increase_shanten_ratio,
        is_tenpai, riichi_declared]``.
        """
        ...
    def encode_extended(self) -> bytes:
        """Encode extended features (combination of multiple feature planes).

        Shape: ``(215, 27)`` / dtype: ``float32``.

        Concatenation of base (74) + discard_decay (4) + shanten (16) +
        ankan (4) + fuuro (80) + action_avail (11) + discard_cand (5) +
        pass_ctx (3) + last_tedashis (9) + riichi_sutehais (9) = 215
        channels, each with 27 compact tile-type columns.  Channels that
        correspond to the absent 4th player are zero-padded.

        Example::

            import numpy as np

            buf = obs.encode_extended()
            features = np.frombuffer(buf, dtype=np.float32).reshape(215, 27)
        """
        ...
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
    "Observation3P",
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
