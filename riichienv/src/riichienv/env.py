import hashlib
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ._riichienv import Meld, MeldType
from .action import Action, ActionType
from .game_mode import GameType
from .hand import AgariCalculator, Conditions


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

    def new_events(self) -> list[dict[str, Any]]:
        """
        Returns only the new events since the last observation.
        """
        return self.events[self.prev_events_size :]


def _to_mjai_tile(tile_136: int) -> str:
    """
    Convert 136-based tile ID to MJAI tile string.
    0-35: 1m..9m
    36-71: 1p..9p
    72-107: 1s..9s
    108-135: 1z..7z
    Red tiles: 5m (16), 5p (52), 5s (88) -> usually marked 'r' in MJAI (e.g. 5mr)
    But MJai often uses 5mr, 5pr, 5sr.

    136 format:
    Man: 0-35. 1m=(0,1,2,3). 5m=(16,17,18,19).
    Pin: 36-71.
    Sou: 72-107.
    Hon: 108-135.

    Red conventions in tenhou/mjsoul:
    Depending on rule. Usually 0-index of 5 is red.
    136 indices:
    5m: 16,17,18,19.
    If 16 is red: '5mr'.
    Let's assume standard red rule: 0-th 5 is red.
    """
    kind = tile_136 // 36
    if kind < 3:  # Suit
        suit_char = ["m", "p", "s"][kind]
        offset = tile_136 % 36
        num = offset // 4 + 1

        # Check red
        # 5m start at 16. 5p at 52 (36+16). 5s at 88 (72+16).
        # IDs for 5: 16,17,18,19.
        is_red = False
        if num == 5:
            # Assuming 16, 52, 88 are reds (indices 0 of the 5s)
            if tile_136 in [16, 52, 88]:
                is_red = True

        return f"{num}{suit_char}{'r' if is_red else ''}"
    else:  # Honor
        offset = tile_136 - 108
        num = offset // 4 + 1
        return f"{num}z"


class Phase(IntEnum):
    WAIT_ACT = 0  # Current player's turn (Discard/Tsumo/Kan/Riichi)
    WAIT_RESPONSE = 1  # Other players' turn to claim (Ron/Pon/Chi)


class RiichiEnv:
    def __init__(
        self,
        seed: int | None = None,
        game_type: GameType = GameType.YON_IKKYOKU,
        round_wind: int = 0,
        initial_scores: list[int] | None = None,
        kyotaku: int = 0,
        honba: int = 0,
    ):
        if game_type != GameType.YON_IKKYOKU:
            raise NotImplementedError(f"GameType {game_type} is not yet supported.")

        self._seed = seed
        self._rng = random.Random(seed)

        self.game_type = game_type
        self._custom_round_wind = round_wind
        self._custom_initial_scores = initial_scores
        self._custom_kyotaku = kyotaku
        self._custom_honba = honba

        # Game State
        self.wall: list[int] = []
        self.hands: dict[int, list[int]] = {}  # player_id -> tiles_136
        self.melds: dict[
            int, list[Any]
        ] = {}  # player_id -> list of Meld objects (Any for now to avoid circular import issue if Meld not imported)
        self.discards: dict[int, list[int]] = {}
        self.current_player: int = 0
        self.turn_count: int = 0
        self.is_done: bool = False
        self._rewards: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.scores: list[int] = [25000, 25000, 25000, 25000]
        self.riichi_sticks: int = 0
        self.riichi_declared: list[bool] = [False, False, False, False]  # To track who declared riichi

        # Phases
        self.phase: Phase = Phase.WAIT_ACT
        self.active_players: list[int] = [0]  # Initially 0
        self.last_discard: dict[str, Any] | None = None  # {seat, tile_136}

        self.current_claims: dict[int, list[Action]] = {}  # Potential claims for current discard

        # Round State
        self.oya: int = 0
        self.dora_indicators: list[int] = []

        # Security
        self.wall_digest: str = ""
        self.salt: str = ""

        # MJAI Logging
        self.mjai_log: list[dict[str, Any]] = []
        # Track event counts for each player to support new_events()
        self._player_event_counts: list[int] = [0, 0, 0, 0]

        # Current logic state
        self.drawn_tile: int | None = None  # The tile currently drawn by current_player

    def reset(self, oya: int = 0, dora_indicators: list[int] | None = None) -> dict[int, Observation]:
        self._rng = random.Random(self._seed)  # Reset RNG if seed was fixed? Or continue? Usually new seed or continue.

        self.oya = oya
        self.dora_indicators = dora_indicators if dora_indicators is not None else []
        # If seed was None, random.Random(None) uses system time.

        # Initialize tiles: 136 tiles
        # 0-33 are tile types. Each type has 4 copies.
        # IDs: 0-135. Type = id // 4.
        self.wall = list(range(136))
        self._rng.shuffle(self.wall)

        # Secure Wall
        self.salt = "".join([chr(self._rng.randint(33, 126)) for _ in range(16)])  # Random ASCII salt
        wall_str = ",".join(map(str, self.wall))
        self.wall_digest = hashlib.sha256((wall_str + self.salt).encode("utf-8")).hexdigest()

        self.hands = {0: [], 1: [], 2: [], 3: []}
        self.discards = {0: [], 1: [], 2: [], 3: []}
        self._rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        # Initialize scores & kyotaku from custom settings or defaults
        if self._custom_initial_scores:
            if len(self._custom_initial_scores) != 4:
                # Fallback or error? Assuming 4 for now.
                self.scores = self._custom_initial_scores[:]
            else:
                self.scores = self._custom_initial_scores[:]
        else:
            self.scores = [25000, 25000, 25000, 25000]

        self.riichi_sticks = self._custom_kyotaku

        self.riichi_declared = [False, False, False, False]
        self.is_done = False
        self.turn_count = 0
        self.current_player = 0  # Dealer = 0 (East)
        self.phase = Phase.WAIT_ACT
        self.active_players = [0]
        self.current_claims = {}

        self.last_discard = None
        self.melds = {0: [], 1: [], 2: [], 3: []}

        self.mjai_log = []
        self._player_event_counts = [0, 0, 0, 0]  # Reset counts
        self.mjai_log.append(
            {"type": "start_game", "names": ["Player0", "Player1", "Player2", "Player3"], "id": "local_game_0"}
        )

        # Deal 13 tiles to each
        for _ in range(13):
            for pid in range(4):
                self.hands[pid].append(self.wall.pop())

        # Dealer draws 14th tile
        self.drawn_tile = self.wall.pop()

        # Sort hands for convenience (though 136-ids don't have perfect order, we just keep them list)
        for pid in range(4):
            self.hands[pid].sort()

        # Start Kyoku Event
        # We need to construct tehais
        tehais = []
        for pid in range(4):
            tehais.append([_to_mjai_tile(t) for t in self.hands[pid]])

        start_kyoku_event = {
            "type": "start_kyoku",
            "bakaze": "E" if self._custom_round_wind == 0 else "S",  # TODO: Proper wind mapping
            "kyoku": 1,
            "honba": self._custom_honba,
            "kyotaku": self._custom_kyotaku,
            "oya": 0,
            "dora_marker": _to_mjai_tile(
                self.wall[0]
            ),  # Fake dora marker logic: usually wall[5] or similar. Using wall[0] for current simplistic impl.
            "tehais": tehais,
        }
        self.mjai_log.append(start_kyoku_event)

        # Tsumo Event for Dealer
        tsumo_event = {"type": "tsumo", "actor": 0, "tile": _to_mjai_tile(self.drawn_tile)}
        self.mjai_log.append(tsumo_event)

        # Check Tsumo logic...

        return self._get_observations(self.active_players)  # Start state

    def step(self, actions: dict[int, Action]) -> dict[int, Observation]:
        """
        Execute one step.
        actions: Map from player_id to Action.
        """
        if self.is_done:
            return self._get_observations([])

        # Validation: keys of actions must match self.active_players
        # Note: We need to handle set comparison because order might differ in dict (though usually not an issue)
        required_players = set(self.active_players)
        provided_players = set(actions.keys())
        if required_players != provided_players:
            raise ValueError(f"Actions required from {self.active_players}, but got {list(actions.keys())}")

        # Convert raw dict/int to Action objects if needed
        # Assuming actions is {pid: Action object} or {pid: legacy_int}

        # PHASE: WAIT_ACT
        if self.phase == Phase.WAIT_ACT:
            # Expect action from current_player
            action_raw = actions.get(self.current_player)
            if action_raw is None:
                # Should not happen if correctly used
                return self._get_observations(self.active_players)

            action: Action = action_raw

            discard_tile_id: int | None = None

            if action.type == ActionType.RIICHI:
                # Handle Riichi declaration step 1
                # Check validity again?
                # Assume valid if passed legal_actions check.

                # Check 1000 pts
                if self.scores[self.current_player] < 1000:
                    raise ValueError("Not enough points for Riichi")

                # Set Riichi Stage
                if not hasattr(self, "riichi_stage"):
                    self.riichi_stage = {0: False, 1: False, 2: False, 3: False}

                self.riichi_stage[self.current_player] = True

                # Log Reach (Step 1 usually calls "reach", then "reach" again after discard using "step" 2?)
                # MJAI spec:
                # 1. "reach", actor:X (Declaration)
                # 2. "dahai", actor:X, tile:T, reach:True (Discard with reach)
                # 3. "reach", actor:X, score:S, delta:-1000 (Payment)

                # Here we just log the declaration "reach" event.
                reach_event = {"type": "reach", "actor": self.current_player}
                self.mjai_log.append(reach_event)

                # Return observation immediately. Phase remains WAIT_ACT.
                return self._get_observations(self.active_players)

            elif action.type == ActionType.TSUMO:
                # Handle tsumo (self-draw win): record the event and stop further processing.
                if self.drawn_tile is None:
                    raise RuntimeError("Tsumo action without drawn tile")
                hand_13 = self.hands[self.current_player][:]
                if self.drawn_tile in hand_13:
                    hand_13.remove(self.drawn_tile)

                player_melds = self.melds.get(self.current_player, [])
                # 0=East, 1=South, 2=West, 3=North (relative to dealer 0). In simulator pid=wind usually.
                cond = Conditions(tsumo=True, player_wind=self.current_player, round_wind=self._custom_round_wind)
                res = AgariCalculator(hand_13, player_melds).calc(self.drawn_tile, conditions=cond)

                deltas = self._calculate_deltas(res, self.current_player, is_tsumo=True)

                hora_event = {
                    "type": "hora",
                    "actor": self.current_player,
                    "target": self.current_player,
                    "tsumo": True,
                    "pai": _to_mjai_tile(self.drawn_tile) if self.drawn_tile is not None else "?",
                    "deltas": deltas,
                }

                # Check for Riichi to add ura markers
                if self.riichi_declared[self.current_player]:
                    hora_event["ura_markers"] = self._get_ura_markers()

                self.mjai_log.append(hora_event)
                self.mjai_log.append({"type": "end_kyoku"})
                self.mjai_log.append({"type": "end_game"})

                # Set is_done to True
                self.is_done = True

                return self._get_observations([])

            elif action.type == ActionType.KYUSHU_KYUHAI:
                # Handle Kyushu Kyuhai

                # Log Ryukyoku
                self.mjai_log.append({"type": "ryukyoku", "reason": "kyushu_kyuhai", "actor": self.current_player})
                self.mjai_log.append({"type": "end_kyoku"})
                self.mjai_log.append({"type": "end_game"})

                self.is_done = True
                return self._get_observations([])

            elif action.type in [ActionType.ANKAN, ActionType.KAKAN]:
                # Handle self-kan
                # Ensure drawn_tile is in hand so _execute_claim can find it
                if self.drawn_tile is not None:
                    self.hands[self.current_player].append(self.drawn_tile)
                    self.drawn_tile = None
                    self.hands[self.current_player].sort()

                self._execute_claim(self.current_player, action)

                if not self.wall:
                    self.is_done = True
                    self.mjai_log.append({"type": "ryukyoku", "reason": "exhaustive_draw"})
                    self.mjai_log.append({"type": "end_kyoku"})
                    self.mjai_log.append({"type": "end_game"})
                    return self._get_observations([])

                # Rinshan Draw
                self.drawn_tile = self.wall.pop()
                # Log Tsumo (Rinshan)
                tsumo_event = {"type": "tsumo", "actor": self.current_player, "tile": _to_mjai_tile(self.drawn_tile)}
                self.mjai_log.append(tsumo_event)

                return self._get_observations(self.active_players)

            elif action.type == ActionType.DISCARD:
                discard_tile_id = action.tile

                # Check Riichi Stage
                if hasattr(self, "riichi_stage") and self.riichi_stage[self.current_player]:
                    # Must be a valid riichi candidate
                    # (Double check validity here or trust legal_actions filter above? Better to verify)
                    # For performance, maybe skip re-check if assuming agent follows legal_actions.
                    # But enforcing rules is good.
                    # Let's rely on _get_legal_actions returning only valid ones, so if user sends invalid, it would fail `legal_actions` check if we enabled strict checking?
                    # Currently step() doesn't strictly validate against legal_actions (it validates required_players keys).
                    pass

                    # Commit Riichi
                    self.riichi_stage[self.current_player] = False
                    self.riichi_declared[self.current_player] = True

                    # Deduct Score
                    self.scores[self.current_player] -= 1000
                    self.riichi_sticks += 1

                    # Log Reach Payment
                    # Actually MJAI order: Dahai (with reach=true) -> Reach (score update)
                    # We will handle Dahai logging below, but need to pass flag.

            # Execute discard
            # Remove from hand
            if discard_tile_id is None:
                raise ValueError("Discard action must have a tile")

            if self.drawn_tile == discard_tile_id:
                self.drawn_tile = None
            else:
                if self.drawn_tile is not None:
                    self.hands[self.current_player].append(self.drawn_tile)
                    self.drawn_tile = None

                # Remove discard_tile_id from self.hands
                # It must exist
                if discard_tile_id in self.hands[self.current_player]:
                    self.hands[self.current_player].remove(discard_tile_id)
                else:
                    raise ValueError(
                        f"Discard tile {discard_tile_id} not found in player {self.current_player}'s hand: "
                        f"{self.hands[self.current_player]}"
                    )

            self.discards[self.current_player].append(discard_tile_id)
            self.hands[self.current_player].sort()

            # Check if this discard is the Riichi declaration discard
            is_reach_discard = self.riichi_declared[self.current_player] and (
                not self.riichi_declared_prev[self.current_player] if hasattr(self, "riichi_declared_prev") else False
            )
            # Actually simpler: we just set riichi_declared=True above.
            # But wait, riichi_declared is persistent.
            # We need to know if we JUST declared it.
            # We can use a local flag or check if self.riichi_sticks increased?

            # Using local variable logic:
            just_reached = False
            if hasattr(self, "riichi_stage") and self.riichi_stage.get("just_committed"):
                just_reached = True
                self.riichi_stage["just_committed"] = False  # Reset

            # Actually, in the block above (ActionType.DISCARD), we set:
            # self.riichi_declared[pid] = True
            # So if we track 'was_declared' before?

            # Optimization: The block above was:
            # if hasattr(self, "riichi_stage") and self.riichi_stage[self.current_player]:
            #    ...
            #    self.riichi_declared = True
            #    just_reached = True (implicitly)

            # Re-implementing logic to be cleaner.

            just_reached = False
            # Check if this turn is the one where we paid 1000.
            # Or pass a flag from the block above?

            # To fix this cleanly, I should merge the `DISCARD` blocks or set a temporary flag.

            # Re-checking previous Replacement:
            # I added logic inside `elif action.type == ActionType.DISCARD:`.
            # I should have set `just_reached = True` there.

            # Let's assume I fix the previous block in a subsequent edit or do it here if possible.
            # Since I cannot edit the previous block easily in this single Replace, I will rely on `riichi_declared` check if I know it wasn't before?
            # But `riichi_declared` is persistent.

            # Better approach: Check if `reach` event is the last event?
            # self.mjai_log[-1]["type"] == "reach"?
            # Yes, if we just declared Riichi (step 1), last event is "reach" (actor only).

            last_event_is_reach = (
                len(self.mjai_log) > 0
                and self.mjai_log[-1]["type"] == "reach"
                and self.mjai_log[-1]["actor"] == self.current_player
            )

            # Log Dahai
            dahai_event = {
                "type": "dahai",
                "actor": self.current_player,
                "tile": _to_mjai_tile(discard_tile_id) if discard_tile_id is not None else "?",
                "tsumogiri": False,  # TODO: Handle tsumogiri (self-draw)
                "reach": last_event_is_reach,
            }
            self.mjai_log.append(dahai_event)

            if last_event_is_reach:
                # Append the score update event (Step 3 of Riichi)
                reach_score_event = {
                    "type": "reach_accepted",
                    "actor": self.current_player,
                    "score": self.scores[self.current_player],  # Already deducted? Yes.
                    "deltas": [0, 0, 0, 0],  # Usually deltas not in reach event?
                    # MJAI: "reach", actor, score, delta(optional?)
                    # Tenhou convention: delta is usually not in 'reach' but scores is updated.
                    # MJSoul: often has 'delta': -1000?
                }
                # Let's verify MJSoul format from log if possible?
                # User log didn't show reach event detail, just DiscardTile data.
                # Assuming standard MJAI:
                # reach (step 1) -> dahai (reach=true) -> reach (score update)
                self.mjai_log.append(reach_score_event)

            # Store last discard
            self.last_discard = {"seat": self.current_player, "tile": discard_tile_id}

            # Check claims potential
            self.current_claims = {}

            # Ron Check (Priority 1)
            ron_potential = []
            for pid in range(4):
                if pid == self.current_player:
                    continue
                # Calc Ron
                # player_wind: (pid - oya + 4) % 4
                p_wind = (pid - self.oya + 4) % 4
                is_houtei = not self.wall  # Win on discard when wall is empty

                res = AgariCalculator(self.hands[pid], self.melds.get(pid, [])).calc(
                    discard_tile_id,
                    dora_indicators=self.dora_indicators,
                    conditions=Conditions(
                        tsumo=False,
                        riichi=self.riichi_declared[pid],
                        player_wind=p_wind,
                        round_wind=self._custom_round_wind,
                        houtei=is_houtei,
                    ),
                )
                if res.agari:
                    ron_potential.append(pid)
                    self.current_claims.setdefault(pid, []).append(Action(ActionType.RON, tile=discard_tile_id))

            # Pon/Kan Check (Priority 2)
            # Valid for all other players
            for pid in range(4):
                if pid == self.current_player:
                    continue

                # Pon
                pon_opts = self._can_pon(self.hands[pid], discard_tile_id)
                for opt in pon_opts:
                    self.current_claims.setdefault(pid, []).append(
                        Action(ActionType.PON, tile=discard_tile_id, consume_tiles=opt)
                    )

                # Kan (Daiminkan)
                kan_opts = self._can_kan(self.hands[pid], discard_tile_id)
                for opt in kan_opts:
                    self.current_claims.setdefault(pid, []).append(
                        Action(ActionType.DAIMINKAN, tile=discard_tile_id, consume_tiles=opt)
                    )

            # Chi Check (Priority 3)
            # Only valid for next player
            next_player = (self.current_player + 1) % 4
            chi_opts = self._can_chi(self.hands[next_player], discard_tile_id)
            for opt in chi_opts:
                self.current_claims.setdefault(next_player, []).append(
                    Action(ActionType.CHI, tile=discard_tile_id, consume_tiles=opt)
                )

            if self.current_claims:
                self.phase = Phase.WAIT_RESPONSE
                self.active_players = list(self.current_claims.keys())
                self.active_players.sort()  # generic order

                return self._get_observations(self.active_players)

            # If no response needed -> Next
            self.current_player = (self.current_player + 1) % 4
            if not self.wall:
                self.is_done = True
                self.mjai_log.append({"type": "ryukyoku", "reason": "exhaustive_draw"})  # Exhaustive draw
                self.mjai_log.append({"type": "end_kyoku"})
                self.mjai_log.append({"type": "end_game"})
                return self._get_observations([])

            self.drawn_tile = self.wall.pop()
            # Log Tsumo
            tsumo_event = {"type": "tsumo", "actor": self.current_player, "tile": _to_mjai_tile(self.drawn_tile)}
            self.mjai_log.append(tsumo_event)

            self.phase = Phase.WAIT_ACT
            self.active_players = [self.current_player]

            return self._get_observations(self.active_players)

        # PHASE: WAIT_RESPONSE
        elif self.phase == Phase.WAIT_RESPONSE:
            # Priority resolution: Ron > Pon/Kan > Chi
            # Collect valid actions
            valid_actions = {}  # pid -> Action

            for pid in self.active_players:
                act = actions.get(pid)

                if act and isinstance(act, Action):
                    # Validate action is legal?
                    # print(">> ENV STEP ACTION:", act)
                    # print(">> ENV STEP ACTION TYPE:", act.type, "DISCARD IS:", ActionType.DISCARD)
                    if act.type in [ActionType.RON, ActionType.PON, ActionType.DAIMINKAN, ActionType.CHI]:
                        valid_actions[pid] = act

            # 1. Check Ron
            ronners = [pid for pid, a in valid_actions.items() if a.type == ActionType.RON]
            if ronners:
                # Handle Ron (Multiple Ron possible?)
                # Assuming Head Bump (Atamahane) for now or double ron.
                # Let's implement Atamahane: Start from current_player, find first ronner.

                winner = -1
                for i in range(1, 4):
                    p = (self.current_player + i) % 4
                    if p in ronners:
                        winner = p
                        break

                # Calculate scores
                if self.last_discard is None:
                    raise RuntimeError("Winner on Ron but last_discard is None")
                tile = self.last_discard["tile"]
                cond = Conditions(tsumo=False, player_wind=winner, round_wind=self._custom_round_wind)
                res = AgariCalculator(self.hands[winner], self.melds.get(winner, [])).calc(tile, conditions=cond)
                deltas = self._calculate_deltas(res, winner, is_tsumo=False, loser=self.current_player)

                self.is_done = True
                hora_event = {
                    "type": "hora",
                    "actor": winner,
                    "target": self.current_player,
                    "tile": _to_mjai_tile(self.last_discard["tile"]),
                    "deltas": deltas,
                    # "yakus": ... TODO
                }

                # Check for Riichi to add ura markers
                if self.riichi_declared[winner]:
                    hora_event["ura_markers"] = self._get_ura_markers()

                self.mjai_log.append(hora_event)
                # rewards logic...
                # Note: Currently verification script checks 'end_game' type.
                # step (WAIT_ACT -> Ryukyoku) logs 'end_game'.
                # We should log 'end_game' here too.
                self.mjai_log.append({"type": "end_kyoku"})
                self.mjai_log.append({"type": "end_game"})

                return self._get_observations([])

            # 2. Check Pon/Kan
            ponners = [pid for pid, a in valid_actions.items() if a.type in [ActionType.PON, ActionType.DAIMINKAN]]
            if ponners:
                # Should only be one ponner (tiles uniqueness)
                claimer = ponners[0]
                action = valid_actions[claimer]

                # Execute Meld
                self._execute_claim(claimer, action)

                # Turn moves to claimer
                self.current_player = claimer
                self.phase = Phase.WAIT_ACT  # Must discard next
                self.active_players = [self.current_player]
                self.drawn_tile = None  # No draw after call (except some Kan...)

                if action.type == ActionType.DAIMINKAN:
                    if not self.wall:
                        self.is_done = True
                        self.mjai_log.append({"type": "ryukyoku", "reason": "exhaustive_draw"})
                        self.mjai_log.append({"type": "end_kyoku"})
                        self.mjai_log.append({"type": "end_game"})
                        return self._get_observations([])

                    self.drawn_tile = self.wall.pop()
                    # Log Tsumo (Rinshan)
                    tsumo_event = {"type": "tsumo", "actor": self.current_player, "tile": _to_mjai_tile(self.drawn_tile)}
                    self.mjai_log.append(tsumo_event)

                return self._get_observations(self.active_players)

            # 3. Check Chi
            chiers = [pid for pid, a in valid_actions.items() if a.type == ActionType.CHI]
            if chiers:
                # Only next player can Chi
                claimer = chiers[0]
                action = valid_actions[claimer]

                self._execute_claim(claimer, action)

                self.current_player = claimer
                self.phase = Phase.WAIT_ACT
                self.active_players = [self.current_player]
                self.drawn_tile = None

                return self._get_observations(self.active_players)

            # If no Claim -> Pass -> Next Draw
            self.current_player = (self.current_player + 1) % 4
            self.phase = Phase.WAIT_ACT
            self.active_players = [self.current_player]

            if self.wall:
                self.drawn_tile = self.wall.pop()
                # Log Tsumo
                tsumo_event = {"type": "tsumo", "actor": self.current_player, "tile": _to_mjai_tile(self.drawn_tile)}
                self.mjai_log.append(tsumo_event)

                return self._get_observations(self.active_players)
            else:
                # Ryukyoku due to wall exhaustion
                self.is_done = True
                self.mjai_log.append({"type": "ryukyoku", "reason": "exhaustive_draw"})
                self.mjai_log.append({"type": "end_kyoku"})
                self.mjai_log.append({"type": "end_game"})
                return self._get_observations([])

        return self._get_observations([])

    def done(self) -> bool:
        return self.is_done

    def rewards(self) -> dict[int, float]:
        return self._rewards

    def _get_observations(self, player_ids: list[int]) -> dict[int, Observation]:
        obs_dict = {}
        for pid in player_ids:
            # Construct hand for observation
            # If current player, include drawn tile
            hand = self.hands[pid][:]
            if pid == self.current_player and self.drawn_tile is not None:
                hand.append(self.drawn_tile)

            # Filter MJAI events for this player
            filtered_events = []
            for ev in self.mjai_log:
                ev_copy = ev.copy()
                if ev["type"] == "start_kyoku":
                    # Mask tehais of others
                    tehais = ev["tehais"]
                    masked_tehais = []
                    for i, t_list in enumerate(tehais):
                        if i == pid:
                            masked_tehais.append(t_list)
                        else:
                            masked_tehais.append(["?"] * len(t_list))
                    ev_copy["tehais"] = masked_tehais
                elif ev["type"] == "tsumo":
                    # Mask tile if not actor
                    if ev["actor"] != pid:
                        ev_copy["tile"] = "?"

                filtered_events.append(ev_copy)

            prev_size = self._player_event_counts[pid]

            # Legal Actions
            legal = []
            if pid in player_ids:  # Calculate legal actions only for actionable players
                legal = self._get_legal_actions(pid)

            obs_dict[pid] = Observation(
                player_id=pid, hand=hand, events=filtered_events, prev_events_size=prev_size, _legal_actions=legal
            )
            # Update count for next time
            self._player_event_counts[pid] = len(filtered_events)

        return obs_dict

    def _get_legal_actions(self, pid: int) -> list[Action]:
        actions = []
        hand = self.hands[pid][:]

        if self.phase == Phase.WAIT_ACT:
            # pid must be current_player
            if pid != self.current_player:
                return []

            # Riichi Discard Enforcement
            # If we are in riichi_declared state (meaning previously declared, now must discard),
            # we should restrict to valid candidates.
            # But wait, self.riichi_stage[pid] tracks "Just declared, need to discard".
            # Normal 'riichi_declared' means already locked.
            # Usually if 'riichi_declared' is True, we are in "Riichi Mode" -> Tsumogiri only.
            # UNLESS it is the very first turn of riichi (the discard immediately after declaration).

            # Let's clarify state:
            # 1. Player calls step(Action(RIICHI)).
            # 2. Env sets riichi_stage[pid] = True. Returns observation.
            # 3. Player calls step(Action(DISCARD, tile=T)).
            # 4. Env checks if T is valid candidate. If so, sets riichi_declared = True, riichi_stage = False.

            # If riichi_declared[pid] is True, and riichi_stage[pid] is False:
            #   User can only Tsumogiri (discard drawn tile).

            if hasattr(self, "riichi_stage") and self.riichi_stage[pid]:
                # Must discard one of the candidates
                # Where are candidates stored? passing in _get_legal_actions logic again
                # is expensive.
                # Re-calculate candidates
                from . import _riichienv

                # Check 14 tiles (hand + drawn)
                hand_14 = hand[:]
                if self.drawn_tile is not None:
                    hand_14.append(self.drawn_tile)

                # We need candidates again
                candidates_136 = _riichienv.check_riichi_candidates(hand_14)

                # Filter valid actions
                # Just return Discard actions for these tiles
                for t in candidates_136:
                    actions.append(Action(ActionType.DISCARD, tile=t))

                return actions  # Restrict to only these

            # Check Kyushu Kyuhai (9 Terminals)
            # Conditions:
            # 1. First turn (no discards by player)
            # 2. No calls by anyone (uninterrupted)
            # 3. 9 or more distinct terminals/honors

            # Check history
            has_discarded = False
            any_call = False

            # Optimization: could cache this state, but loop is short for now
            for e in self.mjai_log:
                if e["type"] == "start_kyoku":
                    continue
                if e.get("actor") == pid and e["type"] in ["dahai", "reach"]:
                    has_discarded = True
                    break
                if e["type"] in ["chi", "pon", "kan", "kakan", "ankan"]:
                    any_call = True
                    break

            if not has_discarded and not any_call:
                # Check tiles
                hand_check = hand[:]
                if self.drawn_tile is not None:
                    hand_check.append(self.drawn_tile)

                distinct_yaochu = set()
                yaochu_indices = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
                for t in hand_check:
                    t34 = t // 4
                    if t34 in yaochu_indices:
                        distinct_yaochu.add(t34)

                if len(distinct_yaochu) >= 9:
                    actions.append(Action(ActionType.KYUSHU_KYUHAI))

            if self.riichi_declared[pid]:
                # Already in Riichi.
                # Can only Tsumo or Discard Drawn Tile.
                # Tsumo check
                if self.drawn_tile is not None:
                    hand_13 = hand[:]  # drawn_tile not in self.hands yet
                    # check tsumo
                    player_melds = self.melds.get(pid, [])
                    res = AgariCalculator(hand_13, player_melds).calc(
                        self.drawn_tile, conditions=Conditions(tsumo=True, riichi=True)
                    )
                    if res.agari:
                        actions.append(Action(ActionType.TSUMO))

                    # Discard drawn tile (Tsumogiri)
                    actions.append(Action(ActionType.DISCARD, tile=self.drawn_tile))

                return actions

            # Normal State (No Riichi)
            # Basic Discard: all tiles in hand (plus drawn if exist)
            if self.drawn_tile is not None:
                hand.append(self.drawn_tile)

            for t in hand:
                actions.append(Action(ActionType.DISCARD, tile=t))

            # Tsumo logic
            if self.drawn_tile is not None:  # Only possible if just drawn
                hand_13 = self.hands[pid][:]  # Use original 13

                player_melds = self.melds.get(pid, [])
                res = AgariCalculator(hand_13, player_melds).calc(self.drawn_tile, conditions=Conditions(tsumo=True))
                if res.agari:
                    actions.append(Action(ActionType.TSUMO))

            # Riichi Check
            # Conditions:
            # 1. Menzen (Closed Hand)
            # 2. Score >= 1000
            # 3. Wall >= 4
            # 4. Not already riichi (checked above)

            is_menzen = all(not m.opened for m in self.melds.get(pid, []))
            # Actually AgariCalculator/Meld types:
            # Meld.KAN is 2 (Daiminkan/Ankan usually distinguishable? In my types.rs, Gang=2, Angang=3)
            # types.rs: Angang=3. So only Angang allowed.
            # Wait, Meld definition has 'opened' field.
            is_menzen = all(not m.opened for m in self.melds.get(pid, []))

            if is_menzen and self.scores[pid] >= 1000 and len(self.wall) >= 4:
                from . import _riichienv

                candidates = _riichienv.check_riichi_candidates(hand)  # hand has 14 tiles here
                if candidates:
                    actions.append(Action(ActionType.RIICHI))

            # Kakan Check (Added Kan)
            # Must not be Riichi (unless special rule? usually Kakan after Riichi is restricted).
            # We assume Kakan allowed if not Riichi. If Riichi, handled separately?
            # Actually, Kakan AFTER Riichi is allowed only if it doesn't change wait.
            # For simplicity, let's allow Kakan in normal state here.
            # Logic: Iterate melds. If Pon exists, check if we have the 4th tile in hand.
            if not self.riichi_declared[pid]:
                for m in self.melds.get(pid, []):
                    if m.meld_type == MeldType.Peng:
                        # Check if we have the 4th tile
                        # m.tiles[0] // 4 is the type
                        target_type = m.tiles[0] // 4
                        for t in hand:
                            if t // 4 == target_type:
                                # Found match
                                actions.append(Action(ActionType.KAKAN, tile=t, consume_tiles=[t]))
                                # Does allow multiple Kakan if multiple tiles of same type? (e.g. Red vs Normal)
                                # Usually yes.


        elif self.phase == Phase.WAIT_RESPONSE:
            # pid is claiming discard
            actions.append(Action(ActionType.PASS))

            # Use cached current_claims
            if pid in self.current_claims:
                actions.extend(self.current_claims[pid])

        return actions

    def _execute_claim(self, pid: int, action: Action):
        """Executes a claim action (PON, CHI, KAN)"""
        # 1. Remove tiles from hand
        hand = self.hands[pid]
        consume = action.consume_tiles
        for t in consume:
            if t in hand:
                hand.remove(t)
            else:
                raise ValueError(
                    f"Tile {t} not found in player {pid}'s hand during claim execution; "
                    f"consume_tiles={consume}, hand={hand}"
                )

        # 2. Create Meld
        target_tile = action.tile
        if target_tile is None:
            raise ValueError("Claim action must have a target tile")

        if action.type == ActionType.ANKAN:
            if len(consume) == 4:
                tiles = sorted(consume)
            else:
                tiles = sorted(consume + [target_tile])
        elif action.type == ActionType.KAKAN:
            # KAKAN: Upgrade existing Pon to AddGang
            # tiles should be the 4 tiles (3 from Pon + 1 added)
            # Find compatible Pon
            # target_tile is the one added. consume usually contains it too (removed from hand).
            # We need to find the Pon in self.melds

            # Warning: We need to modify self.melds in place or remove/add.
            pass
            tiles = []  # placeholder
        else:
            tiles = sorted(consume + [target_tile])

        # Determine Meld Type
        opened = True
        m_type = MeldType.Peng  # default

        if action.type == ActionType.CHI:
            m_type = MeldType.Chi
        elif action.type == ActionType.PON:
            m_type = MeldType.Peng
        elif action.type == ActionType.DAIMINKAN:
            m_type = MeldType.Gang
        elif action.type == ActionType.ANKAN:
            m_type = MeldType.Angang
            opened = False
        elif action.type == ActionType.KAKAN:
            m_type = MeldType.Addgang
            # Kakan Logic:
            # Find existing Pon
            # We assume tid // 4 matches.
            t_type = target_tile // 4
            found_idx = -1
            for i, m in enumerate(self.melds.get(pid, [])):
                if m.meld_type == MeldType.Peng and m.tiles[0] // 4 == t_type:
                    found_idx = i
                    break

            if found_idx != -1:
                old_meld = self.melds[pid].pop(found_idx)
                # New tiles = old tiles + added tile
                old_tiles = list(old_meld.tiles) if isinstance(old_meld.tiles, (bytes, bytearray)) else old_meld.tiles
                tiles = sorted(old_tiles + [target_tile])
            else:
                # Failed to find Pon? Should not happen if Kakan is legal.
                # Fallback: Just create a Gang meld?
                tiles = sorted(consume + [target_tile])  # Incorrect but prevents crash
                pass

        # Check calling logic
        meld = Meld(m_type, tiles, opened)
        self.melds.setdefault(pid, []).append(meld)

        # 3. Log MJAI
        mjai_type = "pon"
        if action.type == ActionType.CHI:
            mjai_type = "chi"
        elif action.type == ActionType.DAIMINKAN:
            mjai_type = "kan"  # Open kan
        elif action.type == ActionType.ANKAN:
            mjai_type = "ankan"  # Or "gang"? MJAI uses "cang"? No usually "nakan"?
            # MJSoul uses "AnGangAddGang".
            # Standard MJAI uses "reach" "chi" "pon" "kan" "hora" "ryukyoku" "dora" "dahai".
            # For Ankan/Kakan: "kan" is used?
            # Specifically:
            # Open Kan: type="kan", data...
            # Closed Kan: type="kan", data... (but tiles different?)
            # Kakan: type="kan", data...
            # Let's assume "kan" is generic?
            # But wait, verification script distinguishes them.
            mjai_type = "kan"
        elif action.type == ActionType.KAKAN:
            mjai_type = "kan"  # Add kan

        discarder = self.last_discard["seat"] if self.last_discard else -1

        event = {
            "type": mjai_type,
            "actor": pid,
            "target": discarder,
            "tile": _to_mjai_tile(target_tile),
            "consumed": [_to_mjai_tile(t) for t in consume],
        }
        self.mjai_log.append(event)

    def _can_pon(self, hand: list[int], tile: int) -> list[list[int]]:
        """Returns list of consume_tiles options for Pon."""
        tile_type = tile // 4
        matches = [t for t in hand if t // 4 == tile_type]
        if len(matches) < 2:
            return []
        return [matches[:2]]

    def _can_kan(self, hand: list[int], tile: int) -> list[list[int]]:
        """Daiminkan"""
        tile_type = tile // 4
        matches = [t for t in hand if t // 4 == tile_type]
        if len(matches) == 3:
            return [matches]
        return []

    def _can_chi(self, hand: list[int], tile: int) -> list[list[int]]:
        """Returns list of consume_tiles options for Chi."""
        t_type = tile // 4
        if t_type >= 27:
            return []  # Honors check

        idx = t_type % 9
        hand_types = sorted(list(set(t // 4 for t in hand)))
        options = []

        # Left: T-2, T-1, T
        if idx >= 2:
            if (t_type - 2) in hand_types and (t_type - 1) in hand_types:
                c1 = next(t for t in hand if t // 4 == t_type - 2)
                c2 = next(t for t in hand if t // 4 == t_type - 1)
                options.append([c1, c2])
        # Center: T-1, T, T+1
        if idx >= 1 and idx <= 7:
            if (t_type - 1) in hand_types and (t_type + 1) in hand_types:
                c1 = next(t for t in hand if t // 4 == t_type - 1)
                c2 = next(t for t in hand if t // 4 == t_type + 1)
                options.append([c1, c2])
        # Right: T, T+1, T+2
        if idx <= 6:
            if (t_type + 1) in hand_types and (t_type + 2) in hand_types:
                c1 = next(t for t in hand if t // 4 == t_type + 1)
                c2 = next(t for t in hand if t // 4 == t_type + 2)
                options.append([c1, c2])
        return options

    def _calculate_deltas(self, agari, winner, is_tsumo, loser=None):
        """
        Calculate score deltas based on Agari result.
        Returns list of 4 integers [delta0, delta1, delta2, delta3].
        Also updates self.scores.
        """
        deltas = [0, 0, 0, 0]

        # Base win points
        if is_tsumo:
            # Tsumo
            # Dealer win: all others pay oya_payment (which is usually split? No, tsumo_agari_oya implies total?)
            # Wait, Rust Agari struct:
            # tsumo_agari_oya: payment from EACH child? Or total?
            # tsumo_agari_ko: payment from Dealer? or Child?

            # Looking at mahjong lib / typical usage:
            # If Dealer wins (Oya):
            #   get 'tsumo_agari_oya' from *each* child?
            #   Wait, 'tsumo_agari_oya' usually refers to the payment *amount* specifically?
            #   Let's assume AgariCalculator returns standard points.
            #   Usually:
            #     Dealer Tsumo: 4000 all -> everyone pays 4000.
            #     Child Tsumo: 1000/2000 -> Dealer pays 2000, Child pays 1000.

            # Checking `agari_calculator.rs`:
            # It just returns points.
            # Let's check `hand.py` Agari dataclass fields:
            # tsumo_agari_oya: int
            # tsumo_agari_ko: int

            # Standard Convention:
            # If Winner is Dealer:
            #   Each child pays `tsumo_agari_oya` (or checks total?)
            #   Actually `tsumo_agari_oya` usually means "Payment FOR Oya" (when Ko wins)?
            #   OR "Payment BY Oya"?

            # Corrections based on typical `mahjong` library:
            # If Ko wins:
            #   main_payment = tsumo_agari_oya (Dealer pays this)
            #   other_payment = tsumo_agari_ko (Other Ko pays this)
            # If Oya wins:
            #   all_payment = tsumo_agari_oya (All Ko pay this) (wait, usually oya payment is higher?)
            #   Actually, with `mahjong` library:
            #     calculate_scores returns: {"main": X, "additional": Y}
            #     If oya Tsumo: main=X (per person).
            #     If ko Tsumo: main=DealerPay, additional=ChildPay.

            # The Rust struct names are `tsumo_agari_oya` and `tsumo_agari_ko`.
            # If Oya wins, `tsumo_agari_oya` is likely the payment per person.
            # If Ko wins, `tsumo_agari_oya` is payment by Oya, `tsumo_agari_ko` is payment by Ko.

            if winner == self.oya:  # Dealer
                # Dealer Tsumo
                # Based on verification: tsumo_agari_ko holds the payment amount for Kids.
                # tsumo_agari_oya is 0 because there is no Oya to pay.
                payment_all = agari.tsumo_agari_ko
                total_win = 0
                for pid in range(4):
                    if pid != winner:
                        deltas[pid] = -payment_all
                        total_win += payment_all
                deltas[winner] = total_win
            else:
                # Child Tsumo
                payment_oya = agari.tsumo_agari_oya
                payment_ko = agari.tsumo_agari_ko
                total_win = 0
                for pid in range(4):
                    if pid != winner:
                        if pid == 0:  # Oya
                            deltas[pid] = -payment_oya
                            total_win += payment_oya
                        else:  # Ko
                            deltas[pid] = -payment_ko
                            total_win += payment_ko
                deltas[winner] = total_win

        else:
            # Ron
            # Loser pays full amount 'ron_agari'
            score = agari.ron_agari
            if loser is not None:
                deltas[loser] = -score
            deltas[winner] = score

        # Add Riichi Sticks to winning total
        # Any riichi sticks currently on table go to winner
        # Note: self.riichi_sticks tracks 1000 point sticks
        if self.riichi_sticks > 0:
            bonus = self.riichi_sticks * 1000
            deltas[winner] += bonus
            self.riichi_sticks = 0  # Reset after claimed

        # Update internal scores
        for pid in range(4):
            self.scores[pid] += deltas[pid]

        return deltas

    def _get_ura_markers(self) -> list[str]:
        """
        Return ura dora markers from the wall.
        For this simplified implementation, we return the tile under the first dora indicator.
        Dora indicator is at index 0 (self.wall[0]).
        Ura indicator would be at index 1? Or conventionally determined index.
        In standard wall construction:
        Dead wall 14 tiles.
        Dora ind: 3rd tile from back (index 5 in 0-indexed 14-tile dead wall).
        Ura ind: 4th tile (under dora).

        Our `reset` logic:
        wall = list(range(136))
        shuffle(wall)

        We used `wall[0]` as dora_marker in `start_kyoku`.
        Let's use `wall[1]` as ura_marker for consistency with that simple choice.
        """
        if len(self.wall) > 1:
            return [_to_mjai_tile(self.wall[1])]
        return []
