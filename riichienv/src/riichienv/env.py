import hashlib
import json
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from . import _riichienv  # type: ignore
from ._riichienv import Meld, MeldType, Wind  # type: ignore
from .action import Action, ActionType
from .game_mode import GameType
from .hand import AgariCalculator, Conditions
from .rules import get_rule

# Wind lookup
WINDS = [Wind.East, Wind.South, Wind.West, Wind.North]


@dataclass
class Observation:
    player_id: int
    hand: list[int]  # 136-based tile IDs
    events: list[dict[str, Any]]  # MJAI events
    prev_events_size: int = 0
    _legal_actions: list[Action] = field(default_factory=list)  # Internal storage for legal actions

    def legal_actions(self) -> list[Action]:
        """
        Returns the list of legal actions available to the player.
        """
        return self._legal_actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "hand": self.hand,
            "events": self.events,
            "prev_events_size": self.prev_events_size,
            "legal_actions": [a.to_dict() for a in self._legal_actions],
        }

    def new_events(self) -> list[str]:
        """
        Returns only the new events since the last observation.
        """
        return [json.dumps(ev, separators=(",", ":")) for ev in self.events[self.prev_events_size :]]

    def select_action_from_mjai(self, mjai_action: str) -> Action | None:
        """
        Select a legal action corresponding to the given MJAI action string.
        Returns matched Action or None.
        """
        try:
            target_json = json.loads(mjai_action)
        except json.JSONDecodeError:
            return None

        if target_json.get("type") == "none":
            for action in self._legal_actions:
                if action.type == ActionType.PASS:
                    return action

        for action in self._legal_actions:
            if action.type == ActionType.PASS and target_json.get("type") == "dahai":
                if target_json.get("tsumogiri") is True:
                    return action

            if action.type == ActionType.KAKAN and target_json.get("type") == "kakan":
                if target_json.get("pai") == _to_mjai_tile(action.tile):
                    return action

            act_json = json.loads(action.to_mjai())
            if all(target_json.get(k) == v for k, v in act_json.items()):
                return action
        return None


def _to_mjai_tile(tile_136: int) -> str:
    """
    Convert 136-based tile ID to MJAI tile string.
    """
    kind = tile_136 // 36
    if kind < 3:  # Suit
        suit_char = ["m", "p", "s"][kind]
        offset = tile_136 % 36
        num = offset // 4 + 1

        is_red = False
        if num == 5:
            if tile_136 in [16, 52, 88]:
                is_red = True

        return f"{num}{suit_char}{'r' if is_red else ''}"
    else:  # Honor
        offset = tile_136 - 108
        num = offset // 4 + 1
        mjai_honors = ["E", "S", "W", "N", "P", "F", "C"]
        if 1 <= num <= 7:
            return mjai_honors[num - 1]
        return f"{num}z"


class Phase(IntEnum):
    WaitAct = 0  # Current player's turn (Discard/Tsumo/Kan/Riichi)
    WaitResponse = 1  # Other players' turn to claim (Ron/Pon/Chi)


class RiichiEnv:
    def __init__(
        self,
        seed: int | None = None,
        game_type: GameType = GameType.YON_IKKYOKU,
        round_wind: int = 0,
        initial_scores: list[int] | None = None,
        kyotaku: int = 0,
        honba: int = 0,
        mjai_mode: bool = False,
    ):
        self.rule = get_rule(game_type)
        self.mjai_mode = mjai_mode

        self._seed = seed
        self._rng = random.Random(seed)

        self.game_type = game_type
        self._custom_round_wind = round_wind
        self._custom_initial_scores = initial_scores
        self._custom_kyotaku = kyotaku
        self._custom_honba = honba

        # Game State
        self.wall: list[int] = []
        self.hands: dict[int, list[int]] = {}
        self.melds: dict[int, list[Meld]] = {0: [], 1: [], 2: [], 3: []}
        self.discards: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        self.current_player: int = 0
        self.turn_count: int = 0
        self.is_done: bool = False
        self.needs_tsumo: bool = False
        self._scores: list[int] = [25000, 25000, 25000, 25000]
        self.score_deltas: list[int] = [0, 0, 0, 0]
        self.riichi_sticks: int = 0
        self.riichi_declared: list[bool] = [False, False, False, False]
        self.riichi_stage: list[bool] = [False, False, False, False]
        self.double_riichi_declared: list[bool] = [False, False, False, False]
        self.agari_results: dict[int, Any] = {}
        self.last_agari_results: dict[int, Any] = {}
        self.round_end_scores: list[int] | None = None
        self.needs_initialize_round: bool = False
        self.needs_tsumo: bool = False

        # Phases
        self.phase: Phase = Phase.WaitAct
        self.active_players: list[int] = [0]
        self.last_discard: dict[str, Any] | None = None

        self.current_claims: dict[int, list[Action]] = {}
        self.pending_kan: tuple[int, Action] | None = None

        # Round State
        self.oya: int = 0
        self.kyoku_idx: int = 1
        self.dora_indicators: list[int] = []
        self.rinshan_draw_count: int = 0
        self.pending_kan_dora_count: int = 0

        self.kyoku_log_start_idx: int = 0
        self.is_rinshan_flag: bool = False
        self.is_first_turn: bool = True
        self.missed_agari_riichi: list[bool] = [False, False, False, False]
        self.missed_agari_doujun: list[bool] = [False, False, False, False]
        self.riichi_pending_acceptance: int | None = None
        self.nagashi_eligible: list[bool] = [True] * 4
        self.drawn_tile: int | None = None

        # Security
        self.wall_digest: str = ""
        self.salt: str = ""

        # MJAI Logging
        self.mjai_log: list[dict[str, Any]] = []
        self._player_event_counts: list[int] = [0, 0, 0, 0]

        self.kyoku_idx = 0
        self.honba = 0
        self.riichi_sticks = 0
        self.score_deltas = [0, 0, 0, 0]
        self.mjai_log = []
        self.kyoku_log_start_idx = 0
        self.pao: list[dict[int, int]] = [{} for _ in range(4)]

    def _initialize_round(
        self,
        oya: int,
        bakaze: int,
        honba: int,
        kyotaku: int,
        wall: list[int] | None = None,
        scores: list[int] | None = None,
    ):
        self.oya = oya
        self._custom_round_wind = bakaze
        self._custom_honba = honba
        self.riichi_sticks = kyotaku
        self.is_done = False

        self.kyoku_idx = oya + 1
        self.dora_indicators = []
        self.pending_kan_dora_count = 0

        self.ippatsu_eligible = [False] * 4
        self.double_riichi_declared = [False] * 4
        self.is_rinshan_flag = False
        self.rinshan_draw_count = 0
        self.is_first_turn = True
        self.missed_agari_riichi = [False] * 4
        self.missed_agari_doujun = [False] * 4
        self.riichi_pending_acceptance = None
        self.nagashi_eligible = [True] * 4
        if self.agari_results:
            self.last_agari_results = self.agari_results
        self.agari_results = {}
        self.pao = [{} for _ in range(4)]
        self.score_deltas = [0, 0, 0, 0]
        self.kyoku_log_start_idx = len(self.mjai_log)
        self.riichi_declared = [False] * 4
        self.riichi_stage = [False] * 4
        self.turn_count = 0
        self.pending_kan = None

        if wall is not None:
            self.wall = list(reversed(wall))
            assert len(self.wall) > 13, "Wall must have at least 13 tiles."
            self.dora_indicators = [self.wall[4]]
        else:
            self.wall = list(range(136))
            self._rng.shuffle(self.wall)
            self.dora_indicators = [self.wall[4]]

        self.salt = "".join([chr(self._rng.randint(33, 126)) for _ in range(16)])
        wall_str = ",".join(map(str, self.wall))
        self.wall_digest = hashlib.sha256((wall_str + self.salt).encode("utf-8")).hexdigest()

        self.hands = {0: [], 1: [], 2: [], 3: []}
        self.discards = {0: [], 1: [], 2: [], 3: []}
        self.melds = {0: [], 1: [], 2: [], 3: []}

        if scores is not None:
            self._scores = scores[:]
        # Note: If scores is None, we keep current self._scores (useful for multi-round)

        self.riichi_declared = [False, False, False, False]
        self.riichi_stage = [False, False, False, False]
        self.turn_count = 0
        self.current_player = self.oya
        self.phase = Phase.WaitAct
        self.active_players = []  # Wait for tsumo in first step
        self.needs_tsumo = True
        self.current_claims = {}
        self.pending_kan = None
        self.last_discard = None

        # self.mjai_log should be handled by the caller or specialized method

        for _ in range(3):
            for pid in range(4):
                pid_ = (pid + self.oya) % 4
                for _ in range(4):
                    self.hands[pid_].append(self.wall.pop())
        for pid in range(4):
            pid_ = (pid + self.oya) % 4
            self.hands[pid_].append(self.wall.pop())

        for pid in range(4):
            self.hands[pid].sort()

        self.drawn_tile = None
        self.needs_tsumo = True

        tehais = [[_to_mjai_tile(t) for t in self.hands[pid]] for pid in range(4)]
        start_kyoku_event = {
            "type": "start_kyoku",
            "bakaze": ["E", "S", "W", "N"][self._custom_round_wind % 4],
            "kyoku": self.oya + 1,
            "honba": self._custom_honba,
            "kyotaku": self.riichi_sticks,
            "oya": self.oya,
            "dora_marker": _to_mjai_tile(self.dora_indicators[0]),
            "tehais": tehais,
            "scores": self._scores,
        }
        self.mjai_log.append(start_kyoku_event)
        self.score_deltas = [0, 0, 0, 0]

    def reset(
        self,
        *,
        seed: int | None = None,
        oya: int | None = None,
        bakaze: int | None = None,
        honba: int | None = None,
        kyotaku: int | None = None,
        wall: list[int] | None = None,
        scores: list[int] | None = None,
    ) -> dict[int, Observation]:
        self._rng = random.Random(self._seed)

        self.is_done = False
        self.hands = {0: [], 1: [], 2: [], 3: []}
        self.discards = {0: [], 1: [], 2: [], 3: []}
        self.melds = {0: [], 1: [], 2: [], 3: []}
        if not self.mjai_log:
            self.mjai_log.append({"type": "start_game", "names": ["Player0", "Player1", "Player2", "Player3"], "id": 0})
        self._initialize_round(
            oya=oya if oya is not None else 0,
            bakaze=bakaze if bakaze is not None else self._custom_round_wind,
            honba=honba if honba is not None else self._custom_honba,
            kyotaku=kyotaku if kyotaku is not None else self._custom_kyotaku,
            wall=wall,
            scores=scores
            if scores is not None
            else (self._custom_initial_scores[:] if self._custom_initial_scores else [25000, 25000, 25000, 25000]),
        )
        self.needs_initialize_round = False
        self.needs_tsumo = True
        self.active_players = []

        if self.mjai_mode:
            return self.get_observations(self.active_players)

        # Auto-advance to the first actionable state (Oya's draw)
        return self.step({})

    def done(self) -> bool:
        return self.is_done

    def scores(self) -> list[int]:
        return self._scores

    def ranks(self) -> list[int]:
        """
        スコアの降順、同点の場合は起家に近い（インデックスが小さい）順に順位を決定
        """
        sorted_indices = sorted(range(4), key=lambda i: (-self._scores[i], i))
        ranks = [0] * 4
        for rank, player_idx in enumerate(sorted_indices, 1):
            ranks[player_idx] = rank
        return ranks

    def points(self, preset_rule: str = "basic") -> list[int]:
        """
        素点+順位点で持ち点計算
        """
        preset_rules: dict[str, dict[str, Any]] = {
            "basic": {
                "soten_weight": 1,
                "soten_base": 25000,
                "jun_weight": [50, 10, -10, -50],
            },
            "ouza-tyoujyo": {
                "soten_weight": 0,
                "soten_base": 25000,
                "jun_weight": [100, 40, -40, -100],
            },
            "ouza-normal": {
                "soten_weight": 0,
                "soten_base": 25000,
                "jun_weight": [50, 20, -20, -50],
            },
        }

        rule = preset_rules.get(preset_rule)
        if rule is None:
            raise ValueError(f"Unknown preset rule: {preset_rule}")

        soten_weight: int = rule["soten_weight"]
        soten_base: int = rule["soten_base"]
        jun_weight: list[int] = rule["jun_weight"]
        ranks = self.ranks()
        return [
            int((self._scores[i] - soten_base) / 1000.0 * soten_weight + jun_weight[ranks[i] - 1]) for i in range(4)
        ]

    def _perform_discard(self, discard_tile_id: int) -> dict[int, Observation]:
        is_reach_declaration = False
        if self.riichi_stage[self.current_player]:
            self.riichi_stage[self.current_player], self.riichi_declared[self.current_player] = False, True
            is_reach_declaration = True
            self.riichi_pending_acceptance = self.current_player
            self.ippatsu_eligible[self.current_player] = True
            if len(self.discards[self.current_player]) == 0 and not any(len(m) for m in self.melds.values()):
                self.double_riichi_declared[self.current_player] = True
        elif self.riichi_declared[self.current_player]:
            self.ippatsu_eligible[self.current_player] = False

        # Nagashi Mangan check
        if discard_tile_id // 4 not in [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]:
            self.nagashi_eligible[self.current_player] = False

        tsumogiri = self.drawn_tile == discard_tile_id
        if tsumogiri:
            self.drawn_tile = None
        else:
            if self.drawn_tile is not None:
                self.hands[self.current_player].append(self.drawn_tile)
                self.drawn_tile = None
            self.hands[self.current_player].remove(discard_tile_id)
        self.discards[self.current_player].append(discard_tile_id)
        self.hands[self.current_player].sort()
        # Check for pending Kan Dora reveal (Min-kan/Kakan: reveal after discard)
        # Move BEFORE dahai to ensure dahai is the last event in the batch for bots.
        while self.pending_kan_dora_count > 0:
            self._reveal_kan_dora(post_rinshan=False)
            self.pending_kan_dora_count -= 1

        self.mjai_log.append(
            {
                "type": "dahai",
                "actor": self.current_player,
                "pai": _to_mjai_tile(discard_tile_id),
                "tsumogiri": tsumogiri,
                "reach": is_reach_declaration,
            }
        )
        self.last_discard = {"seat": self.current_player, "tile": discard_tile_id}

        self.current_claims = {}
        ronners = self._get_ron_potential(discard_tile_id, is_chankan=False)
        for pid in ronners:
            self.current_claims.setdefault(pid, []).append(Action(ActionType.RON, tile=discard_tile_id))

        # Claims (Pon/Chi/Kan/Daiminkan) are NOT allowed on Houtei (last tile).
        # Houtei occurs when wall length is 14 (dead wall only).
        if len(self.wall) > 14:
            for pid in range(4):
                if pid == self.current_player:
                    continue
                if self.riichi_declared[pid]:
                    continue
                for opt in self._can_pon(self.hands[pid], discard_tile_id):
                    self.current_claims.setdefault(pid, []).append(
                        Action(ActionType.PON, tile=discard_tile_id, consume_tiles=opt)
                    )
                for opt in self._can_kan(self.hands[pid], discard_tile_id):
                    self.current_claims.setdefault(pid, []).append(
                        Action(ActionType.DAIMINKAN, tile=discard_tile_id, consume_tiles=opt)
                    )
            next_p = (self.current_player + 1) % 4
            if not self.riichi_declared[next_p]:
                for opt in self._can_chi(self.hands[next_p], discard_tile_id):
                    self.current_claims.setdefault(next_p, []).append(
                        Action(ActionType.CHI, tile=discard_tile_id, consume_tiles=opt)
                    )

        if not self.current_claims:
            # Transition to next player's WaitAct and set needs_tsumo flag.
            self._accept_riichi()
            self.current_player = (self.current_player + 1) % 4
            self.missed_agari_doujun[self.current_player] = False
            self.needs_tsumo = True
            self.is_rinshan_flag = False
            self.phase = Phase.WaitAct
            self.active_players = []

            if self.mjai_mode:
                return self.get_observations([])
            return self.step({})

        self.phase = Phase.WaitResponse
        self.active_players = [
            pid for pid, actions in self.current_claims.items() if any(a.type != ActionType.PASS for a in actions)
        ]
        return self.get_observations(self.active_players)

    def step(self, actions: dict[int, Action]) -> dict[int, Observation]:
        """Execute one step."""
        if self.agari_results:
            self.last_agari_results = self.agari_results
        self.agari_results = {}

        while not self.is_done:
            if self.needs_tsumo:
                # Handle leading tsumo draw in a separate step to ensure it's terminal
                if not self.is_rinshan_flag and self._check_midway_draws():
                    if self.done():
                        return self.get_observations([])
                    if self.mjai_mode:
                        return self.get_observations(self.active_players)
                    continue

                if len(self.wall) <= 14:
                    self._trigger_ryukyoku("exhaustive_draw")
                    if self.done():
                        return self.get_observations([])
                    if self.mjai_mode:
                        return self.get_observations(self.active_players)
                    continue

                if self.is_rinshan_flag:
                    self.drawn_tile = self.wall.pop(0)
                    self.rinshan_draw_count += 1
                else:
                    self.drawn_tile = self.wall.pop()

                self.mjai_log.append(
                    {"type": "tsumo", "actor": self.current_player, "pai": _to_mjai_tile(self.drawn_tile)}
                )
                self.needs_tsumo = False
                self.missed_agari_doujun[self.current_player] = False
                self.active_players = [self.current_player]
                # Return now if we need the player's reaction.
                # In mjai_mode, we always return after tsumo.
                # In non-mjai_mode, we return if the player is NOT in riichi (so they can act).
                # If they ARE in riichi, we continue to the auto-play check below.
                if self.mjai_mode or not self.riichi_declared[self.current_player]:
                    return self.get_observations([self.current_player])
                continue

            if self.phase == Phase.WaitAct and self.current_player not in actions:
                if not self.mjai_mode and self.riichi_declared[self.current_player]:
                    # Auto-play for Riichi player if no special action (Tsumo/Kan) is provided or available.
                    legal = self._get_legal_actions(self.current_player)
                    if not any(a.type in [ActionType.TSUMO, ActionType.ANKAN, ActionType.KAKAN] for a in legal):
                        actions[self.current_player] = Action(ActionType.PASS)

            if set(actions.keys()) != set(self.active_players):
                # If mjai_mode is False, we allow partial or empty actions by returning current observations.
                # This maintains backward compatibility for tests that expect to inspect state at sync points.
                if not self.mjai_mode:
                    return self.get_observations(self.active_players)
                else:
                    raise ValueError(f"Actions required from {self.active_players}, but got {list(actions.keys())}")

            if self.phase == Phase.WaitAct:
                action = actions[self.current_player]
                if action.type == ActionType.RIICHI:
                    if self._scores[self.current_player] < 1000:
                        raise ValueError("Not enough points for Riichi")
                    self.riichi_stage[self.current_player] = True
                    self.mjai_log.append({"type": "reach", "actor": self.current_player})
                    return self.get_observations(self.active_players)
                elif action.type == ActionType.TSUMO:
                    assert self.drawn_tile is not None
                    ura_in = (
                        self._get_ura_markers_tid()
                        if (
                            self.riichi_declared[self.current_player]
                            or self.double_riichi_declared[self.current_player]
                        )
                        else []
                    )
                    is_first = self.is_first_turn and len(self.discards[self.current_player]) == 0
                    res = AgariCalculator(
                        self.hands[self.current_player], self.melds.get(self.current_player, [])
                    ).calc(
                        self.drawn_tile,
                        dora_indicators=self.dora_indicators,
                        conditions=Conditions(
                            tsumo=True,
                            riichi=self.riichi_declared[self.current_player],
                            double_riichi=self.double_riichi_declared[self.current_player],
                            ippatsu=self.ippatsu_eligible[self.current_player],
                            rinshan=self.is_rinshan_flag,
                            haitei=(len(self.wall) <= 14 and not self.is_rinshan_flag),
                            player_wind=WINDS[(self.current_player - self.oya + 4) % 4],
                            round_wind=WINDS[self._custom_round_wind % 4],
                            tsumo_first_turn=is_first,
                        ),
                        ura_indicators=ura_in,
                    )
                    self.agari_results[self.current_player] = res
                    self.is_done = True
                    deltas = self._calculate_deltas(res, self.current_player, is_tsumo=True)
                    ura_markers = []
                    if self.riichi_declared[self.current_player] or self.double_riichi_declared[self.current_player]:
                        ura_markers = self._get_ura_markers()
                    self.mjai_log.append(
                        {
                            "type": "hora",
                            "actor": self.current_player,
                            "target": self.current_player,
                            "tsumo": True,
                            "pai": _to_mjai_tile(self.drawn_tile),
                            "deltas": deltas,
                            "ura_markers": ura_markers,
                        }
                    )
                    self._end_kyoku(is_renchan=(self.current_player == self.oya), is_draw=False)
                    if self.done():
                        return self.get_observations([])
                    return self.get_observations(self.active_players)
                elif action.type == ActionType.KyushuKyuhai:
                    self._trigger_ryukyoku("kyushu_kyuhai")
                    if self.done():
                        return self.get_observations([])
                    return self.get_observations(self.active_players)
                elif action.type in [ActionType.ANKAN, ActionType.KAKAN]:
                    is_chankan = action.type == ActionType.KAKAN
                    assert action.tile is not None

                    # Execute immediately so event is visible
                    if self.drawn_tile is not None:
                        # Append drawn tile to hand before execution so it can be consumed
                        self.hands[self.current_player].append(self.drawn_tile)
                        self.drawn_tile = None

                    self._execute_claim(self.current_player, action)
                    self.is_first_turn = False

                    # Check for Robbing Kan
                    ronners = self._get_ron_potential(action.tile, is_chankan=is_chankan, is_ankan=(not is_chankan))

                    if not ronners:
                        # No Ron, proceed to Rinshan directly
                        self.is_rinshan_flag = True
                        self.ippatsu_eligible = [False] * 4
                        self.needs_tsumo = True
                        if not self.mjai_mode:
                            return self.step({})
                        # In mjai_mode, we must return to let agent see the ANKAN/KAKAN event
                        # before they receive the Rinshan Tsumo?
                        # No, Tsumo is the next event. The ANKAN event is already in the log.
                        # If we return [], user gets info update.
                        # Then next step processes Tsumo.
                        return self.get_observations([])

                    # Pause for Chankan
                    self.phase, self.active_players, self.pending_kan = (
                        Phase.WaitResponse,
                        ronners,
                        action,  # Store action to resume later
                    )
                    self.last_discard = {"seat": self.current_player, "tile": action.tile}
                    self.current_claims = {pid: [Action(ActionType.RON, tile=action.tile)] for pid in ronners}
                    return self.get_observations(self.active_players)
                elif action.type == ActionType.PASS:
                    assert self.riichi_declared[self.current_player]
                    assert self.drawn_tile is not None
                    obs = self._perform_discard(self.drawn_tile)
                    if self.mjai_mode or self.active_players:
                        return obs
                    actions = {}
                    continue
                elif action.type == ActionType.DISCARD:
                    assert action.tile is not None
                    obs = self._perform_discard(action.tile)
                    if self.mjai_mode or self.active_players:
                        return obs
                    actions = {}
                    continue
                else:
                    raise ValueError(f"Action {action.type} not allowed in Phase.WaitAct")

            elif self.phase == Phase.WaitResponse:
                # Set missed agari flags for those who passed Ron
                for pid, act in actions.items():
                    claims = self.current_claims.get(pid, [])
                    has_ron_option = any(c.type == ActionType.RON for c in claims)
                    chosen_ron = act.type == ActionType.RON

                    if has_ron_option and not chosen_ron:
                        if self.riichi_declared[pid]:
                            self.missed_agari_riichi[pid] = True
                        else:
                            self.missed_agari_doujun[pid] = True

                valid_actions = {pid: act for pid, act in actions.items() if act.type != ActionType.PASS}
                ronners = [pid for pid, a in valid_actions.items() if a.type == ActionType.RON]
                if ronners:
                    if self.riichi_pending_acceptance == self.current_player:
                        self.riichi_declared[self.current_player] = False
                        self.riichi_pending_acceptance = None
                    sorted_ronners = sorted(ronners, key=lambda p: (p - self.current_player + 4) % 4)
                    assert self.last_discard is not None
                    tile = self.last_discard["tile"]
                    for idx, winner in enumerate(sorted_ronners):
                        ura_in = (
                            self._get_ura_markers_tid()
                            if (self.riichi_declared[winner] or self.double_riichi_declared[winner])
                            else []
                        )
                        res = AgariCalculator(self.hands[winner], self.melds.get(winner, [])).calc(
                            tile,
                            dora_indicators=self.dora_indicators,
                            conditions=Conditions(
                                tsumo=False,
                                riichi=self.riichi_declared[winner],
                                double_riichi=self.double_riichi_declared[winner],
                                ippatsu=self.ippatsu_eligible[winner],
                                player_wind=WINDS[(winner - self.oya + 4) % 4],
                                round_wind=WINDS[self._custom_round_wind % 4],
                                chankan=(self.pending_kan is not None),
                                houtei=(len(self.wall) <= 14),
                            ),
                            ura_indicators=ura_in,
                        )
                        self.agari_results[winner] = res
                        deltas = self._calculate_deltas(
                            res, winner, is_tsumo=False, loser=self.current_player, include_bonus=(idx == 0)
                        )
                        ura_markers = []
                        if self.riichi_declared[winner] or self.double_riichi_declared[winner]:
                            ura_markers = self._get_ura_markers()
                        self.mjai_log.append(
                            {
                                "type": "hora",
                                "actor": winner,
                                "target": self.current_player,
                                "pai": _to_mjai_tile(tile),
                                "deltas": deltas,
                                "ura_markers": ura_markers,
                            }
                        )
                    self._end_kyoku(is_renchan=(self.oya in ronners), is_draw=False)
                    if self.done():
                        return self.get_observations([])
                    return self.get_observations(self.active_players)

                ponners = [pid for pid, a in valid_actions.items() if a.type in [ActionType.PON, ActionType.DAIMINKAN]]
                if ponners:
                    self.nagashi_eligible[self.current_player] = False  # Discard was called
                    self._accept_riichi()
                    claimer = ponners[0]
                    self._execute_claim(claimer, valid_actions[claimer], from_pid=self.current_player)
                    (
                        self.current_player,
                        self.phase,
                        self.active_players,
                        self.drawn_tile,
                        self.ippatsu_eligible,
                        self.is_first_turn,
                    ) = claimer, Phase.WaitAct, [claimer], None, [False] * 4, False
                    if valid_actions[claimer].type == ActionType.DAIMINKAN:
                        self.is_rinshan_flag = True
                        self.drawn_tile = None
                        self.needs_tsumo = True
                        self.phase, self.active_players = Phase.WaitAct, []  # Wait for tsumo
                        if self.mjai_mode:
                            return self.get_observations([])
                        actions = {}
                        continue
                    else:
                        self.is_rinshan_flag = False
                    return self.get_observations(self.active_players)

                chiers = [pid for pid, a in valid_actions.items() if a.type == ActionType.CHI]
                if chiers:
                    self.nagashi_eligible[self.current_player] = False  # Discard was called
                    self._accept_riichi()
                    claimer = chiers[0]
                    self._execute_claim(claimer, valid_actions[claimer], from_pid=self.current_player)
                    (
                        self.current_player,
                        self.phase,
                        self.active_players,
                        self.drawn_tile,
                        self.ippatsu_eligible,
                        self.is_first_turn,
                    ) = claimer, Phase.WaitAct, [claimer], None, [False] * 4, False
                    self.is_rinshan_flag = False
                    return self.get_observations(self.active_players)

                if self.pending_kan:
                    # Resuming from Chankan interruption (all passed)
                    if self.pending_kan.type == ActionType.KAKAN:
                        self.is_rinshan_flag = True
                        self.ippatsu_eligible = [False] * 4
                        self.needs_tsumo = True
                        self.phase, self.active_players = Phase.WaitAct, []
                        self.pending_kan = None
                        if self.mjai_mode:
                            return self.get_observations([])
                        return self.step({})

                if self._check_midway_draws():
                    if self.done():
                        return self.get_observations([])
                    if self.mjai_mode:
                        return self.get_observations(self.active_players)
                    return self.step({})

                self._accept_riichi()
                self.current_player = (self.current_player + 1) % 4
                self.missed_agari_doujun[self.current_player] = False
                self.phase, self.active_players = Phase.WaitAct, []
                self.drawn_tile = None
                self.is_rinshan_flag = False
                self.needs_tsumo = True
                if self.mjai_mode:
                    return self.get_observations([])
                return self.step({})

            # This part should only be reached if no actions matched above or loop skipped
            raise ValueError(f"Unhandled phase or action combination. Phase: {self.phase}, Actions: {actions}")

    def get_observations(self, player_ids: list[int]) -> dict[int, Observation]:
        obs_dict = {}
        for pid in player_ids:
            hand = self.hands[pid][:]
            if pid == self.current_player and self.drawn_tile is not None:
                hand.append(self.drawn_tile)
            filtered_events = []
            for ev in self.mjai_log:
                ev_copy = ev.copy()
                if ev["type"] == "start_game":
                    ev_copy["id"] = pid
                elif ev["type"] == "start_kyoku":
                    ev_copy["tehais"] = [t if i == pid else ["?"] * len(t) for i, t in enumerate(ev["tehais"])]
                elif ev["type"] == "tsumo" and ev["actor"] != pid:
                    ev_copy["pai"] = "?"
                filtered_events.append(ev_copy)
            obs_dict[pid] = Observation(
                pid, hand, filtered_events, self._player_event_counts[pid], self._get_legal_actions(pid)
            )
            self._player_event_counts[pid] = len(filtered_events)
        return obs_dict

    def _get_legal_actions(self, pid: int) -> list[Action]:
        actions = []
        hand = self.hands[pid][:]
        h14 = hand + ([self.drawn_tile] if self.drawn_tile is not None else [])
        if self.phase == Phase.WaitAct:
            if pid != self.current_player or self.needs_tsumo:
                return []
            if self.riichi_stage[pid]:
                for t in _riichienv.check_riichi_candidates(h14):
                    actions.append(Action(ActionType.DISCARD, tile=t))
                return actions
            current_kyoku_log = self.mjai_log[self.kyoku_log_start_idx :]
            has_discarded = any(e.get("actor") == pid and e["type"] in ["dahai", "reach"] for e in current_kyoku_log)
            any_call = any(e["type"] in ["chi", "pon", "daiminkan", "kakan", "ankan"] for e in current_kyoku_log)
            if not has_discarded and not any_call:
                if len({t // 4 for t in h14 if (t // 4) in {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}}) >= 9:
                    actions.append(Action(ActionType.KyushuKyuhai))
            if self.riichi_declared[pid]:
                if self.drawn_tile is not None:
                    is_first = self.is_first_turn and len(self.discards[pid]) == 0
                    res = AgariCalculator(hand, self.melds.get(pid, [])).calc(
                        self.drawn_tile,
                        conditions=Conditions(
                            tsumo=True,
                            riichi=True,
                            double_riichi=self.double_riichi_declared[pid],
                            ippatsu=self.ippatsu_eligible[pid],
                            player_wind=WINDS[(pid - self.oya + 4) % 4],
                            round_wind=WINDS[self._custom_round_wind % 4],
                            haitei=(len(self.wall) <= 14),
                            rinshan=self.is_rinshan_flag,
                            tsumo_first_turn=is_first,
                        ),
                    )
                    if res.agari and (res.yakuman or (res.han >= 1 and any(y not in [31, 32, 33] for y in res.yaku))):
                        actions.append(Action(ActionType.TSUMO))
                    # Ankan during Riichi
                    t_type = self.drawn_tile // 4
                    if sum(1 for t in h14 if t // 4 == t_type) == 4:
                        tids = sorted([t for t in h14 if t // 4 == t_type])
                        actions.append(Action(ActionType.ANKAN, tile=self.drawn_tile, consume_tiles=tids))

                    actions.append(Action(ActionType.PASS))
                return actions
            for t in h14:
                actions.append(Action(ActionType.DISCARD, tile=t))
            if self.drawn_tile is not None:
                is_first = self.is_first_turn and len(self.discards[pid]) == 0
                res = AgariCalculator(hand, self.melds.get(pid, [])).calc(
                    self.drawn_tile,
                    conditions=Conditions(
                        tsumo=True,
                        riichi=self.riichi_declared[pid],
                        double_riichi=self.double_riichi_declared[pid],
                        ippatsu=self.ippatsu_eligible[pid],
                        player_wind=WINDS[(pid - self.oya + 4) % 4],
                        round_wind=WINDS[self._custom_round_wind % 4],
                        haitei=(len(self.wall) <= 14),
                        rinshan=self.is_rinshan_flag,
                        tsumo_first_turn=is_first,
                    ),
                )
                if res.agari and (res.yakuman or (res.han >= 1 and any(y not in [31, 32, 33] for y in res.yaku))):
                    actions.append(Action(ActionType.TSUMO))
            if all(not m.opened for m in self.melds.get(pid, [])) and self._scores[pid] >= 1000 and len(self.wall) >= 4:
                if _riichienv.check_riichi_candidates(h14):
                    actions.append(Action(ActionType.RIICHI))
            if len(self.wall) > 14:
                for m in self.melds.get(pid, []):
                    if m.meld_type == MeldType.Peng and any(t // 4 == m.tiles[0] // 4 for t in h14):
                        t = [t for t in h14 if t // 4 == m.tiles[0] // 4][0]
                        actions.append(Action(ActionType.KAKAN, tile=t, consume_tiles=m.tiles))
                counts = {}
                for t in h14:
                    counts[t // 4] = counts.get(t // 4, 0) + 1
                for t_type, count in counts.items():
                    if count == 4:
                        tids = sorted([t for t in h14 if t // 4 == t_type])
                        actions.append(Action(ActionType.ANKAN, tile=tids[0], consume_tiles=tids))
        else:
            actions.append(Action(ActionType.PASS))
            if pid in self.current_claims:
                actions.extend(self.current_claims[pid])
        return actions

    def _execute_claim(self, pid: int, action: Action, from_pid: int | None = None):
        hand = self.hands[pid]
        if action.type == ActionType.KAKAN:
            hand.remove(action.tile)
        else:
            for t in action.consume_tiles:
                hand.remove(t)
        m_type, opened = MeldType.Peng, True
        if action.type == ActionType.CHI:
            m_type = MeldType.Chi
        elif action.type == ActionType.DAIMINKAN:
            m_type = MeldType.Gang
        elif action.type == ActionType.ANKAN:
            m_type, opened = MeldType.Angang, False
        elif action.type == ActionType.KAKAN:
            assert action.tile is not None
            m_type = MeldType.Addgang
            found = [
                i
                for i, m in enumerate(self.melds[pid])
                if m.meld_type == MeldType.Peng and m.tiles[0] // 4 == action.tile // 4
            ]
            if found:
                old = self.melds[pid].pop(found[0])
                tiles = sorted(list(old.tiles) + [action.tile])
                self.melds[pid].append(Meld(m_type, tiles, True))
                self.mjai_log.append(
                    {
                        "type": "kakan",
                        "actor": pid,
                        "pai": _to_mjai_tile(action.tile),
                        "consumed": [_to_mjai_tile(tt) for tt in old.tiles],
                    }
                )
                self.pending_kan_dora_count += 1
                return

            # Fallback: If KAKAN but no Pon found, check if hand has 4 tiles (ANKAN treated as KAKAN in log?)
            # Count tiles in hand matching action.tile
            # action.consume_tiles might differ if coming from MJAI.
            # MJAI Kakan usually implies add. But if we fail, we check local hand.
            match_in_hand = [t for t in self.hands[pid] if t // 4 == action.tile // 4]
            # Hand usually has 1 (drawn) + 3 (in hand). Or if pre-removed, check 3.
            # In KAKAN logic start (Line 891), we removed action.tile.
            # So hand should have 3 remaining if it's meant to be ANKAN.
            if len(match_in_hand) >= 3:
                # Treat as ANKAN
                # Remove the 3 tiles
                for t in match_in_hand[:3]:
                    self.hands[pid].remove(t)
                tiles = sorted(match_in_hand[:3] + [action.tile])
                m_type, opened = MeldType.Angang, False
                self._reveal_kan_dora(False)
                self.mjai_log.append(
                    {
                        "type": "ankan",  # Correct log to Ankan
                        "actor": pid,
                        "pai": _to_mjai_tile(action.tile),
                        "consumed": [_to_mjai_tile(t) for t in match_in_hand[:3]],
                    }
                )
                self.melds[pid].append(Meld(m_type, tiles, opened))
                return

        if action.type == ActionType.ANKAN:
            tiles = sorted(action.consume_tiles)
        else:
            assert action.tile is not None
            tiles = sorted(action.consume_tiles + [action.tile])
        self.melds[pid].append(Meld(m_type, tiles, opened))
        if from_pid is not None and m_type in [MeldType.Peng, MeldType.Gang]:
            if sum(1 for m in self.melds[pid] if m.tiles[0] // 4 in [31, 32, 33]) == 3:
                self.pao[pid][37] = from_pid
            if sum(1 for m in self.melds[pid] if m.tiles[0] // 4 in [27, 28, 29, 30]) == 4:
                self.pao[pid][50] = from_pid
        mj_types = {
            ActionType.CHI: "chi",
            ActionType.PON: "pon",
            ActionType.DAIMINKAN: "daiminkan",
            ActionType.ANKAN: "ankan",
        }
        assert action.tile is not None
        if action.type == ActionType.ANKAN:
            self._reveal_kan_dora(False)

        self.mjai_log.append(
            {
                "type": mj_types[action.type],
                "actor": pid,
                "target": from_pid if from_pid is not None else -1,
                "pai": _to_mjai_tile(action.tile),
                "consumed": [_to_mjai_tile(t) for t in action.consume_tiles],
            }
        )
        if action.type == ActionType.DAIMINKAN:
            self.pending_kan_dora_count += 1

    def _can_pon(self, hand: list[int], tile: int) -> list[list[int]]:
        matches = [t for t in hand if t // 4 == tile // 4]
        if len(matches) < 2:
            return []
        reds = [t for t in matches if t in [16, 52, 88]]
        norms = [t for t in matches if t not in [16, 52, 88]]
        opts = []
        if len(norms) >= 2:
            opts.append([norms[0], norms[1]])
        if len(norms) >= 1 and len(reds) >= 1:
            opts.append(sorted([norms[0], reds[0]]))
        return opts

    def _can_kan(self, hand: list[int], tile: int) -> list[list[int]]:
        matches = [t for t in hand if t // 4 == tile // 4]
        return [matches] if len(matches) == 3 else []

    def _can_chi(self, hand: list[int], tile: int) -> list[list[int]]:
        t_type = tile // 4
        if t_type >= 27:
            return []
        idx = t_type % 9
        opts = []

        def get_cand(r):
            m = [t for t in hand if t // 4 == r]
            if not m:
                return []
            res = []
            if any(t in [16, 52, 88] for t in m):
                res.append([t for t in m if t in [16, 52, 88]][0])
            if any(t not in [16, 52, 88] for t in m):
                res.append([t for t in m if t not in [16, 52, 88]][0])
            return res

        pairs = []
        if idx >= 2:
            pairs.append((t_type - 2, t_type - 1))
        if 1 <= idx <= 7:
            pairs.append((t_type - 1, t_type + 1))
        if idx <= 6:
            pairs.append((t_type + 1, t_type + 2))
        for r1, r2 in pairs:
            for c1 in get_cand(r1):
                for c2 in get_cand(r2):
                    opts.append(sorted([c1, c2]))
        return opts

    def _calculate_deltas(self, agari, winner, is_tsumo, loser=None, include_bonus=True):
        deltas = [0, 0, 0, 0]
        h_val = self._custom_honba if include_bonus else 0
        pao_pid = next((self.pao[winner][y] for y in [37, 50] if y in agari.yaku and y in self.pao[winner]), None)
        if is_tsumo:
            if pao_pid is not None:
                total = (
                    (agari.tsumo_agari_ko * 3)
                    if winner == self.oya
                    else (agari.tsumo_agari_oya + agari.tsumo_agari_ko * 2)
                )
                total += h_val * 300
                deltas[winner], deltas[pao_pid] = total, -total
            else:
                h = h_val * 100
                if winner == self.oya:
                    p = agari.tsumo_agari_ko
                    for i in range(4):
                        deltas[i] = -(p + h) if i != winner else (p + h) * 3
                else:
                    po, pk = agari.tsumo_agari_oya, agari.tsumo_agari_ko
                    for i in range(4):
                        deltas[i] = (
                            (-(po + h) if i == self.oya else -(pk + h)) if i != winner else (po + pk * 2 + h * 3)
                        )
        else:
            s_base = agari.ron_agari
            s_total = s_base + h_val * 300
            if pao_pid is not None and loser != pao_pid:
                # Discarder pays half of base Yakuman.
                # Pao player pays half of base Yakuman + all of Honba points.
                assert loser is not None
                assert pao_pid is not None
                deltas[loser] = -(s_base // 2)
                deltas[pao_pid] = -(s_base // 2) - h_val * 300
                deltas[winner] = s_total
            else:
                assert loser is not None
                deltas[loser], deltas[winner] = -s_total, s_total
        if include_bonus and self.riichi_sticks > 0:
            deltas[winner] += self.riichi_sticks * 1000
            self.riichi_sticks = 0
        for i in range(4):
            self._scores[i] += deltas[i]
            self.score_deltas[i] += deltas[i]
        return deltas

    def _reveal_kan_dora(self, post_rinshan: bool = True):
        count = len(self.dora_indicators)
        if count >= 5:
            return
        # Original indices in reversed wall: 4, 6, 8, 10, 12
        # Adjust for items popped from front (Rinshan)
        # Using simplified logic: 1 Kan = 1 Rinshan (pop(0)).
        # Net shift relative to current wall[0] is just +1 per Kan.
        # Original Kan Dora locations are 4, 6, 8...
        # At Kan K (0-based count of PAST kans):
        # We want the K-th Kan Dora (Total Dora count = K+1).
        # It was at 4 + 2*(K+1). Wait.
        # Initial Dora (K=-1?? No). D0 is at 4.
        # D1 is at 6.
        # If we have count=1 existing dora. We reveal D1.
        # Formula: 4 + count * 2 - self.rinshan_draw_count
        idx = 4 + count * 2 - self.rinshan_draw_count
        if 0 <= idx < len(self.wall):
            t = self.wall[idx]
            self.dora_indicators.append(t)
            self.mjai_log.append({"type": "dora", "dora_marker": _to_mjai_tile(t)})

    def _get_ura_markers(self) -> list[str]:
        return [_to_mjai_tile(tid) for tid in self._get_ura_markers_tid()]

    def _get_ura_markers_tid(self) -> list[int]:
        res = []
        for d in self.dora_indicators:
            try:
                i = self.wall.index(d)
                if i + 1 < len(self.wall):
                    res.append(self.wall[i + 1])
            except ValueError as e:
                raise e

        return res

    def _get_waits(self, pid: int) -> set[int]:
        hand = self.hands[pid]
        melds = self.melds.get(pid, [])
        calc_base = AgariCalculator(hand, melds)
        if not calc_base.is_tenpai():
            return set()

        waits = set()
        for t_type in range(34):
            res = calc_base.calc(
                t_type * 4,
                dora_indicators=self.dora_indicators,
                conditions=Conditions(
                    tsumo=False,
                    riichi=self.riichi_declared[pid],
                    double_riichi=self.double_riichi_declared[pid],
                    player_wind=WINDS[(pid - self.oya + 4) % 4],
                    round_wind=WINDS[self._custom_round_wind % 4],
                ),
            )
            if res.agari:
                waits.add(t_type)
        return waits

    def _is_furiten(self, pid: int) -> bool:
        if self.missed_agari_riichi[pid]:
            # print(f"DEBUG: Player {pid} is Furiten (missed agari riichi)")
            return True
        if self.missed_agari_doujun[pid]:
            # print(f"DEBUG: Player {pid} is Furiten (missed agari doujun)")
            return True

        waits = self._get_waits(pid)
        if not waits:
            return False

        discard_types = {t // 4 for t in self.discards[pid]}
        for w in waits:
            if w in discard_types:
                # print(f"DEBUG: Player {pid} is Furiten (wait {w} in discards)")
                return True
        return False

    def _get_ron_potential(self, tile: int, is_chankan: bool, is_ankan: bool = False) -> list[int]:
        res = []
        for p in range(4):
            if p == self.current_player:
                continue
            if self._is_furiten(p):
                continue
            calc = AgariCalculator(self.hands[p], self.melds.get(p, [])).calc(
                tile,
                dora_indicators=self.dora_indicators,
                conditions=Conditions(
                    tsumo=False,
                    riichi=self.riichi_declared[p],
                    double_riichi=self.double_riichi_declared[p],
                    ippatsu=self.ippatsu_eligible[p],
                    player_wind=WINDS[(p - self.oya + 4) % 4],
                    round_wind=WINDS[self._custom_round_wind % 4],
                    houtei=(len(self.wall) <= 14),
                    chankan=is_chankan,
                ),
            )
            if calc.agari:
                # Enforce 1-han minimum and at least one non-dora yaku
                if not calc.yakuman:
                    non_dora_yaku = [y for y in calc.yaku if y not in [31, 32, 33]]
                    if calc.han < 1 or not non_dora_yaku:
                        continue
                if is_ankan and not (calc.han >= 13 and any(y in [42, 49] for y in calc.yaku)):
                    continue
                res.append(p)
        return res

    def _check_midway_draws(self) -> bool:
        discards = []
        for p in range(4):
            if len(self.discards[p]) != 1:
                break
            discards.append(self.discards[p][0])
        else:
            if len(discards) == 4:
                t = discards[0] // 4
                if (
                    27 <= t <= 30
                    and all(d // 4 == t for d in discards)
                    and not any(len(m) for m in self.melds.values())
                ):
                    self._trigger_ryukyoku("sufuurenta")
                    return True
        tk, kp = 0, set()
        for p, ms in self.melds.items():
            for m in ms:
                if m.meld_type in [MeldType.Gang, MeldType.Angang, MeldType.Addgang]:
                    tk += 1
                    kp.add(p)

        if tk >= 4 and len(kp) > 1:
            self._trigger_ryukyoku("suukansansen")
            return True
        if sum(self.riichi_declared) == 4:
            self._trigger_ryukyoku("suurechi")
            return True
        return False

    def _accept_riichi(self):
        if self.riichi_pending_acceptance is not None:
            p = self.riichi_pending_acceptance
            self._scores[p] -= 1000
            self.score_deltas[p] -= 1000
            self.riichi_sticks += 1
            self.mjai_log.append({"type": "reach_accepted", "actor": p, "deltas": [0, 0, 0, 0]})
            self.mjai_log[-1]["deltas"][p] -= 1000
            self.riichi_pending_acceptance = None

    def _trigger_ryukyoku(self, reason: str):
        self._accept_riichi()
        tenpais = [False] * 4
        if reason == "exhaustive_draw":
            for i in range(4):
                calc = AgariCalculator(self.hands[i], self.melds.get(i, []))
                tenpais[i] = calc.is_tenpai()

            # Nagashi Mangan?
            nagashi_winners = [i for i in range(4) if self.nagashi_eligible[i]]
            if nagashi_winners:
                for winner in nagashi_winners:
                    is_oya = winner == self.oya
                    deltas = [0, 0, 0, 0]
                    if is_oya:
                        for i in range(4):
                            deltas[i] = -4000 if i != winner else 12000
                    else:
                        for i in range(4):
                            if i == winner:
                                deltas[i] = 8000
                            elif i == self.oya:
                                deltas[i] = -4000
                            else:
                                deltas[i] = -2000
                    for i in range(4):
                        self._scores[i] += deltas[i]
                        self.score_deltas[i] += deltas[i]
                self.mjai_log.append({"type": "ryukyoku", "reason": "nagashimangan"})
            else:
                num_tp = sum(tenpais)
                if 0 < num_tp < 4:
                    pk, pn = 3000 // num_tp, 3000 // (4 - num_tp)
                    for i in range(4):
                        delta = pk if tenpais[i] else -pn
                        self._scores[i] += delta
                        self.score_deltas[i] += delta

        self.mjai_log.append({"type": "ryukyoku", "reason": reason})

        is_renchan = False
        if reason == "exhaustive_draw":
            is_renchan = tenpais[self.oya]
            # In case of Nagashi Mangan, some rules renchan if Oya has it.
            # But here we follow standard tenpai-renchan.
        elif reason in ["kyushu_kyuhai", "suurechi", "suukansansen", "sufuurenta"]:
            is_renchan = True

        self._end_kyoku(is_renchan=is_renchan, is_draw=True, is_midway_draw=(reason != "exhaustive_draw"))

    def _end_kyoku(self, is_renchan: bool, is_draw: bool = False, is_midway_draw: bool = False):
        """End the current kyoku and prepare for the next one."""
        self.round_end_scores = self._scores[:]
        self.mjai_log.append({"type": "end_kyoku"})
        if self.rule.is_game_over(self, is_renchan, is_draw=is_draw, is_midway_draw=is_midway_draw):
            self.mjai_log.append({"type": "end_game"})
            self.is_done = True
            return

        params = self.rule.get_next_kyoku_params(self, is_renchan, was_draw=is_draw)
        self._initialize_round(**params)
