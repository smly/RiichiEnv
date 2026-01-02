"""
Verify the Gym-like API with mjsoul replay data.

Usage:

First, scan the game records to find the game records that are not verified.

    uv run python scripts/verify_gym_api_with_mjsoul.py scan

Then, verify the game records in detail.

    uv run python scripts/verify_gym_api_with_mjsoul.py <path_to_game_record> --skip <skip_kyoku> --verbose

For debugging, use "DEBUG=1" to enable debug logging.

    DEBUG=1 uv run python scripts/verify_gym_api_with_mjsoul.py <path_to_game_record> --skip <skip_kyoku> --verbose

"""
import os
import sys
import json
import traceback
import argparse
from typing import Any
from pathlib import Path

import riichienv.convert as cvt
from riichienv.action import ActionType, Action
from riichienv.env import Phase
from riichienv import ReplayGame, RiichiEnv

import logging


class bcolors:
    BGRED = "\033[41m"
    BGGREEN = "\033[42m"
    BGYELLOW = "\033[43m"
    BGBLUE = "\033[44m"
    BGMAGENDA = "\033[45m"
    BGCYAN = "\033[46m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENDA = "\033[95m"
    CYAN = "\033[96m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class LevelFormatter(logging.Formatter):
    def __init__(self, formatters):
        super().__init__()
        self.formatters = formatters

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.formatters[logging.DEBUG])
        return formatter.format(record)


# print(bcolors.YELLOW + "------- Colored STDOUT Test -------" + bcolors.ENDC)
logger = logging.getLogger(__file__)
if logger.handlers:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatters = {
    logging.DEBUG: logging.Formatter(
        f"{bcolors.GREEN}%(asctime)s{bcolors.ENDC} | {bcolors.CYAN}%(levelname)s{bcolors.ENDC} - {bcolors.CYAN}%(message)s{bcolors.ENDC}",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
    logging.INFO: logging.Formatter(
        f"{bcolors.GREEN}%(asctime)s{bcolors.ENDC} | {bcolors.CYAN}%(levelname)s{bcolors.ENDC} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
    logging.WARNING: logging.Formatter(
        f"{bcolors.GREEN}%(asctime)s{bcolors.ENDC} | {bcolors.YELLOW}WARN{bcolors.ENDC} - {bcolors.YELLOW}%(message)s{bcolors.ENDC}",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
    logging.ERROR: logging.Formatter(
        f"{bcolors.GREEN}%(asctime)s{bcolors.ENDC} | {bcolors.RED}%(levelname)s{bcolors.ENDC} -{bcolors.RED} %(message)s{bcolors.ENDC}",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
    logging.CRITICAL: logging.Formatter(
        f"{bcolors.GREEN}%(asctime)s{bcolors.ENDC} | {bcolors.RED}%(levelname)s{bcolors.ENDC} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ),
}
stream_handler.setFormatter(LevelFormatter(formatters))
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

if os.environ.get("DEBUG") == "1":
    logger.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.DEBUG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="scan", help="Path to the game record JSON file.")
    parser.add_argument("--skip", type=int, default=0, help="Number of kyokus to skip.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


class MjsoulEnvVerifier:
    def __init__(self, verbose: bool = True):
        self.env: RiichiEnv = RiichiEnv()
        self.obs_dict: dict[int, Any] | None = None
        self.dora_indicators: list[int] = []
        self.using_paishan = False
        self._verbose = verbose
        self.kyoku_idx = 0

    def verify_game(self, game: Any, skip: int = 0) -> bool:
        kyokus = list(game.take_kyokus())
        for i, kyoku in enumerate(kyokus[skip:]):
            self.kyoku_idx = skip + i
            if not self.verify_kyoku(kyoku):
                return False
        return True

    def _new_round(self, kyoku: Any, event: Any) -> None:
        data = event["data"]
        assert "paishan" in data, "Paishan not found in NewRound event."

        paishan_wall = None
        if "paishan" in data:
            try:
                paishan_wall = cvt.paishan_to_wall(data["paishan"])
                if self._verbose:
                    print(f">> PAISHAN LOADED: {len(paishan_wall)} tiles")
            except Exception as e:
                logger.error(f"Failed to parse paishan: {e}")
                raise

        self.using_paishan = True
        bakaze_idx = data.get("chang", 0)
        scores = data.get("scores")
        honba = data.get("ben", 0)
        kyotaku = data.get("liqibang", 0)
        self.env.reset(oya=data["ju"] % 4, wall=paishan_wall, bakaze=bakaze_idx, scores=scores, honba=honba, kyotaku=kyotaku)
        self.dora_indicators = self.env.dora_indicators[:]

        # 牌山から配牌を決定するロジックの一致を検証
        assert cvt.tid_to_mjai_list(self.env.hands[0]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles0"][:13]))))
        assert cvt.tid_to_mjai_list(self.env.hands[1]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles1"][:13]))))
        assert cvt.tid_to_mjai_list(self.env.hands[2]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles2"][:13]))))
        assert cvt.tid_to_mjai_list(self.env.hands[3]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles3"][:13]))))

        self.env.mjai_log = [
            {
                "type": "start_game",
                "names": ["Player0", "Player1", "Player2", "Player3"],
            },
            {
                "type": "start_kyoku",
                "bakaze": ["E", "S", "W", "N"][data.get("chang", 0)],
                "kyoku": data["ju"] + 1,
                "honba": data.get("ben", 0),
                "kyotaku": data.get("liqibang", 0),
                "oya": data["ju"],
                "dora_marker": cvt.mpsz_to_mjai(data["doras"][0]),
                "tehais": [
                    cvt.mpsz_to_mjai_list(data["tiles0"][:13]),
                    cvt.mpsz_to_mjai_list(data["tiles1"][:13]),
                    cvt.mpsz_to_mjai_list(data["tiles2"][:13]),
                    cvt.mpsz_to_mjai_list(data["tiles3"][:13]),
                ],
            },
        ]
        # 局数は self.env.oya と一致する
        assert self.env.oya == data["ju"] % 4

        # ju は 0-index なので +1 して 1-index にする（1局, 2局, ... と数える）
        self.env.kyoku_idx = data["ju"] + 1
        oya = data["ju"] % 4

        # 最初の親のツモが RiichiEnv で設定したものとログが一致することを確認
        assert cvt.tid_to_mjai(self.env.drawn_tile) == cvt.mpsz_to_mjai(data["tiles{}".format(oya)][13])

        first_actor = data["ju"]
        self.env.mjai_log.append({
            "type": "tsumo",
            "actor": first_actor,
            "tile": cvt.tid_to_mjai(self.env.drawn_tile),
        })
        assert self.env.current_player == first_actor
        assert self.env.active_players == [first_actor]

        self.obs_dict = self.env._get_observations([first_actor])
        assert self.obs_dict is not None

    def _discard_tile(self, event: Any) -> None:
        while self.env.phase != Phase.WAIT_ACT:
            if self._verbose:
                logger.debug(f">> WAITING loop... obs keys: {list(self.obs_dict.keys())} Phase: {self.env.phase}")

            # Skip action
            self.obs_dict = self.env.step({skip_player_id: Action(ActionType.PASS) for skip_player_id in self.obs_dict.keys()})

        player_id = event["data"]["seat"]
        candidate_tiles = set([cvt.tid_to_mpsz(a.tile) for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.DISCARD])
        assert player_id == event["data"]["seat"]
        assert event["data"]["tile"] in candidate_tiles, f"Tile {event['data']['tile']} not in candidate tiles. Log may be repeating history or wall desync."

        # Riichi Step =====================
        is_liqi = event["data"].get("is_liqi", False)
        if is_liqi:
            if self._verbose:
                logger.debug(f">> TRUST: Executing RIICHI step for {player_id}")

            riichi_actions = [a for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.RIICHI]
            assert len(riichi_actions) > 0, "Riichi flag true but no Riichi action."
            self.obs_dict = self.env.step({player_id: riichi_actions[0]})

        if player_id == 0 and len(self.obs_dict[player_id].hand) < 13:
            if self._verbose:
                logger.debug(f"DEBUG: Player 0 Hand Size Mismatch! Size= {len(self.obs_dict[player_id].hand)}")

        # Discard Step ====================
        # obs.legal_actions() に牌譜をトレースするアクションが存在するか確認して、それを選択してステップを進める
        target_mpsz = event["data"]["tile"]
        actions = [a for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.DISCARD and cvt.tid_to_mpsz(a.tile) == target_mpsz]
        assert len(actions) > 0, "No discard action found."
        self.obs_dict = self.env.step({player_id: actions[0]})

    def _liuju(self, event: Any) -> None:
        # 流局 (Midway Draw)
        # 1: 九種九牌, 2: 四風連打, 3: 四槓散了, 4: 四家立直
        data = event["data"]
        lj_type = data.get("type", 0)
        seat = data.get("seat", 0)
        tiles = data.get("tiles", [])

        if self._verbose:
            logger.info(f"LiuJu event: type={lj_type}, seat={seat}, tiles={tiles}")

        if lj_type == 1:
            # 九種九牌: 宣言席のプレイヤーがアクションを選択してステップを進める
            obs = self.obs_dict[seat]
            kyushu_actions = [a for a in obs.legal_actions() if a.type == ActionType.KYUSHU_KYUHAI]
            assert kyushu_actions, f"No KYUSHU_KYUHAI action found for player {seat}"

            self.obs_dict = self.env.step({seat: kyushu_actions[0]})

        # env 側ですでに is_done が True になっているはず
        assert self.env.is_done, f"Env should be done for LiuJu type {lj_type}"

    def _no_tile(self, event: Any) -> None:
        # 荒牌平局 (Exhaustive Draw)
        data = event["data"]
        liujumanguan = data.get("liujumanguan", False)
        players = data.get("players", [])
        scores = data.get("scores", [])

        if self._verbose:
            logger.info(f"NoTile event: liujumanguan={liujumanguan}, players={players}, scores={scores}")

        # env.is_done が True であることを確認
        assert self.env.is_done, "Env should be done for NoTile (Exhaustive Draw)"

    def _hule(self, event: Any) -> None:
        is_zimo = any(h.get("zimo", False) for h in event["data"]["hules"])

        # If Zimo, we must be in WAIT_ACT. If in WAIT_RESPONSE, auto-pass.
        if is_zimo:
            assert self.env.phase == Phase.WAIT_ACT, "Zimo Hule should be in WAIT_ACT"
        else:
            assert self.env.phase == Phase.WAIT_RESPONSE, "Ron Hule should be in WAIT_RESPONSE"

        active_players = list(self.obs_dict.keys())

        if self._verbose or os.environ.get("DEBUG"):
            logger.debug(f">> DEBUG: START _hule. Kyoku {self.kyoku_idx}. event hules {[h['seat'] for h in event['data']['hules']]}, obs_dict.keys={list(self.obs_dict.keys())}")
            logger.debug(f">> HULE EVENT DATA: {event}")
            logger.debug(f">> ENV PHASE: {self.env.phase}")
            logger.debug(f">> ENV current_player: {self.env.current_player}")
            logger.debug(f">> ENV drawn_tile: {self.env.drawn_tile} ({cvt.tid_to_mpsz(self.env.drawn_tile) if self.env.drawn_tile is not None else 'None'})")
            logger.debug(f">> ENV active_players: {self.env.active_players}")
            logger.debug(f">> ENV wall len: {len(self.env.wall)}")
            for pid in range(4):
                for meld in self.env.melds[pid]:
                    logger.debug(f"Meld: {meld.meld_type} {cvt.tid_to_mpsz_list(meld.tiles)} opened={meld.opened}")
                logger.debug(f">> ENV hands[{pid}] len: {len(self.env.hands[pid])}")
                logger.debug(f">> ENV hands[{pid}] content: {cvt.tid_to_mpsz_list(self.env.hands[pid])}")

        winning_actions = {}
        # Phase 1: Preparation and Hand Repair
        for hule in event["data"]["hules"]:
            player_id = hule["seat"]
            
            # Brute Force Hand Repair if inactive or Agari check fails
            assert player_id in active_players, f"Winner {player_id} inactive."

            # Observations for each active player are assumed to be already present in self.obs_dict.

            obs = self.obs_dict[player_id]
            legal_ron = any(a.type in {ActionType.RON, ActionType.TSUMO} for a in obs.legal_actions())
            assert legal_ron, f"Player {player_id} has no RON/TSUMO."

            # Continue logic
            obs = self.obs_dict[player_id]
            match_actions = [a for a in obs.legal_actions() if a.type in {ActionType.RON, ActionType.TSUMO}]

            assert len(match_actions) == 1
            winning_actions[player_id] = match_actions[0]

        # Phase 2: Execution
        step_actions = winning_actions.copy()

        # If in WAIT_RESPONSE (Ron), others might need to PASS
        for pid in self.obs_dict.keys():
            if pid not in step_actions:
                step_actions[pid] = Action(ActionType.PASS)

        self.obs_dict = self.env.step(step_actions)

        # Phase 3: Verification
        for hule in event["data"]["hules"]:
            player_id = hule["seat"]
            if player_id not in winning_actions:
                continue

            # legal_actions() から取り出した hule に対応する action
            action = winning_actions[player_id]
            winning_tile = action.tile
            # Use environment hand (13 tiles) for calculation, as obs.hand might be 14 for Tsumo
            hand_for_calc = self.env.hands[player_id]
            
            if action.type == ActionType.TSUMO:
                winning_tile = self.env.drawn_tile
                assert self.env.drawn_tile is not None, "Tsumo but drawn_tile is None."

            if self._verbose:
                print(">> HULE", hule)
                print(">> HAND", cvt.tid_to_mpsz_list(hand_for_calc))
                print(">> WIN TILE", cvt.tid_to_mpsz(winning_tile))

            # Retrieve Agari result calculated by the environment
            if player_id not in self.env.agari_results:
                raise KeyError(f"Player {player_id} not found in agari_results. Action type: {action.type}. Step presumably failed to register win.")
            
            calc = self.env.agari_results[player_id]
            assert calc.agari
            assert calc.yakuman == hule["yiman"]
            
            if action.type == ActionType.TSUMO:
                # Tsumo Score Check
                # Check Ko payment
                if "point_zimo_xian" in hule and hule["point_zimo_xian"] > 0:
                    if calc.tsumo_agari_ko != hule["point_zimo_xian"]:
                        logger.debug(f">> TSUMO KO MISMATCH: Mine {calc.tsumo_agari_ko}, Expected {hule['point_zimo_xian']}")
                        logger.debug(f">> Calculated Yaku IDs: {calc.yaku}")
                        logger.debug(f">> Calculated Han: {calc.han}")
                        logger.debug(f">> Calculated Fu: {calc.fu}")
                        logger.debug(f">> Expected Fans: {hule.get('fans')}")
                        logger.debug(f">> Expected Fu: {hule.get('fu')}")
                    assert calc.tsumo_agari_ko == hule["point_zimo_xian"]
                
                # Check Oya Payment (if not Dealer)
                # If winner is Oya, there is no Oya payment (all Ko).
                if player_id != self.env.oya:
                    if "point_zimo_qin" in hule and hule["point_zimo_qin"] > 0:
                        if calc.tsumo_agari_oya != hule["point_zimo_qin"]:
                            logger.debug(f">> TSUMO OYA MISMATCH: Mine {calc.tsumo_agari_oya}, Expected {hule['point_zimo_qin']}")
                            logger.debug(f">> Calculated Yaku IDs: {calc.yaku}")
                            logger.debug(f">> Calculated Han: {calc.han}")
                            logger.debug(f">> Calculated Fu: {calc.fu}")
                            logger.debug(f">> Expected Fans: {hule.get('fans')}")
                            logger.debug(f">> Expected Fu: {hule.get('fu')}")
                        assert calc.tsumo_agari_oya == hule["point_zimo_qin"]
                
            else:
                if calc.ron_agari != hule["point_rong"]:
                    logger.debug(f">> RON POINT MISMATCH: Mine {calc.ron_agari}, Expected {hule['point_rong']}")
                    logger.debug(f">> Calculated Yaku IDs: {calc.yaku}")
                    logger.debug(f">> Calculated Han: {calc.han}")
                    logger.debug(f">> Calculated Fu: {calc.fu}")
                    logger.debug(">> Expected Yaku IDs: {}".format(str([x["id"] for x in hule.get('fans', [])])))
                    logger.debug(f">> Expected Han: {hule.get('count')}")
                    logger.debug(f">> Expected Fu: {hule.get('fu')}")
                    logger.debug(f">> Hand: {cvt.tid_to_mpsz_list(self.env.hands[player_id])}")
                    logger.debug(f">> Win Tile: {hule['hu_tile']}")
                    logger.debug(f">> Is Oya: {player_id == self.env.oya}")
                assert calc.ron_agari == hule["point_rong"]

            try:
                if hule.get("yiman", False):
                    # Yakuman check
                    assert calc.yakuman, "Expected Yakuman but calculator returned False"
                    # For Yakuman, hule["count"] is typically number of yakumans (1, 2, etc.)
                    if self._verbose:
                        logger.debug(f">> YAKUMAN VERIFIED. Count: {hule['count']}, Points: {calc.ron_agari or calc.tsumo_agari_ko + calc.tsumo_agari_oya}")
                else:
                    if calc.han != hule["count"] or calc.fu != hule["fu"]:
                        logger.debug(f">> HAN/FU MISMATCH: Mine {calc.han} Han {calc.fu} Fu, Expected {hule['count']} Han {hule['fu']} Fu")
                        logger.debug(f">> Calculated Yaku IDs: {calc.yaku}")
                    if calc.han != hule["count"]:
                        logger.debug(f">> HAN MISMATCH: Mine {calc.han}, Expected {hule['count']}")
                        logger.debug(f">> Calculated Yaku IDs: {calc.yaku}")
                        logger.debug(f">> Expected Yaku IDs: {[f['id'] for f in hule.get('fans', [])]}")
                    assert calc.han == hule["count"]
                    assert calc.fu == hule["fu"]
            except AssertionError as e:
                if self._verbose:
                    logger.debug(f"Mismatch in Han/Fu/Yakuman: Rust calc han={calc.han} fu={calc.fu} yakuman={calc.yakuman}, Expected count={hule['count']} fu={hule['fu']} yiman={hule.get('yiman', False)}")
                raise e

    def _deal_tile(self, event: Any) -> None:
        # RiichiEnv 内部で処理されるので検証のみ
        if "tile" in event["data"] and event["data"]["tile"] and event["data"]["tile"] != "?":
            t_str = event["data"]["tile"]
            t_tid = cvt.mpsz_to_tid(t_str)

            sim_drawn = self.env.drawn_tile
            assert sim_drawn is not None, "Drawn tile is not set while Env is in WAIT_RESPONSE"
            assert t_tid // 4 == sim_drawn // 4, "Drawn tile mismatch. Sim: {} Log: {}".format(cvt.tid_to_mpsz(sim_drawn), t_str)
            assert t_str == cvt.tid_to_mpsz(sim_drawn), "Drawn tile mismatch. Sim: {} Log: {}".format(cvt.tid_to_mpsz(sim_drawn), t_str)
        else:
            logger.error(f"Draw tile is not set. Sim: {cvt.tid_to_mpsz(sim_drawn)}, Log: {t_str}.")
            assert False, "Draw tile is not set while Env is in WAIT_RESPONSE"

    def verify_kyoku(self, kyoku: Any) -> bool:
        try:
            events = kyoku.events()
            for event in events:
                # If Env is waiting for responses (Ron/Pon/Chi) but the Log event is not one of those,
                # it means all players PASSed. We must synchronize the Env.
                while not self.env.is_done and self.env.phase == Phase.WAIT_RESPONSE and event["name"] not in ["Hule", "ChiPengGang", "AnGangAddGang"]:
                    pids = self.env.active_players
                    assert pids, "Active players is empty while Env is in WAIT_RESPONSE"
                    self.obs_dict = self.env.step({pid: Action(ActionType.PASS) for pid in pids})

                match event["name"]:
                    case "NewRound":
                        self._new_round(kyoku, event)
                        assert not self.env.done()

                    case "DiscardTile":
                        self._discard_tile(event)

                    case "DealTile":
                        self._deal_tile(event)
                        assert not self.env.done()

                    case "Hule":
                        self._hule(event)
                        assert self.env.done()

                    case "LiuJu":
                        # 途中流局 | 1: 九種九牌, 2: 四風連打, 3: 四槓散了, 4: 四家立直
                        self._liuju(event)
                        assert self.env.is_done, "Env should be done after LiuJu"
                        
                    case "NoTile":
                        # 荒牌平局
                        self._no_tile(event)
                        assert self.env.is_done, "Env should be done after NoTile"

                    case "AnGangAddGang":
                        self._angang_addgang(event)

                    case "ChiPengGang":
                        player_id = event["data"]["seat"]
                        assert player_id in self.obs_dict
                        self._handle_chipenggang(event, player_id, self.obs_dict[player_id])

                    case _:
                        logger.error("UNHANDLED Event: {}".format(json.dumps(event)))
                        assert False, f"UNHANDLED Event: {event}"

            assert self.env.is_done
            return True

        except AssertionError as e:
            logger.error(f"Verification Assertion Failed: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            logger.error(f"Verification Error: {e}")
            traceback.print_exc()
            return False

    def _angang_addgang(self, event: Any) -> None:
        # Ensure we are in WAIT_ACT for self-actions (Ankan/Kakan)
        while not self.env.is_done and self.env.phase != Phase.WAIT_ACT:
            # Skip action (Pass on claims)
            self.obs_dict = self.env.step({skip_player_id: Action(ActionType.PASS) for skip_player_id in self.obs_dict.keys()})

        player_id = event["data"]["seat"]
        assert player_id in self.env.active_players
        assert len(self.env.active_players) == 1

        obs = self.obs_dict[player_id]
        if event["data"]["type"] == 2:
            # KAKAN (Added Kan)
            kakan_actions = [a for a in obs.legal_actions() if a.type == ActionType.KAKAN]
            assert len(kakan_actions) > 0, "KAKAN action should be included in obs.legal_actions()"
            t = cvt.mpsz_to_tid(event["data"]["tiles"])
            t_base = t // 4

            target_action = None
            for a in kakan_actions:
                if a.tile // 4 == t_base:
                    target_action = a
                    break

            if target_action:
                action = target_action
            else:
                assert False, "KAKAN action should be included in obs.legal_actions()"

            self.obs_dict = self.env.step({player_id: action})

        elif event["data"]["type"] == 3:
            # ANKAN (Closed Kan)
            target_mpsz = event["data"]["tiles"]
            assert isinstance(target_mpsz, str), "ANKAN tiles should be a string"

            # Smart Scan for Ankan
            # We need to find 4 tiles in hand that match the target tile type.
            base_type = target_mpsz.replace("0", "5").replace("r", "") # 0m -> 5m
            found_tids = []
            hand_copy = list(self.obs_dict[player_id].hand)
            for tid in hand_copy:
                t_mpsz = cvt.tid_to_mpsz(tid)
                t_base = t_mpsz.replace("0", "5").replace("r", "")
                if t_base == base_type:
                    found_tids.append(tid)

            if len(found_tids) < 4:
                if self._verbose:
                    print(f">> WARNING: Missing tiles for ANKAN of {target_mpsz}. Found {len(found_tids)}. Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
                    print(f">> TRUST: Patching hand to include 4x {target_mpsz} for ANKAN.")
                missing_count = 4 - len(found_tids)
                for _ in range(missing_count):
                    new_tid = cvt.mpsz_to_tid(target_mpsz) # Canonical
                    # Remove garbage
                    if self.env.hands[player_id]:
                        removed = self.env.hands[player_id].pop(0)
                        print(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand.")
                    self.env.hands[player_id].append(new_tid)
                self.env.hands[player_id].sort()

            action = None
            actions = self.obs_dict[player_id].legal_actions()
            for a in actions:
                if a.type == ActionType.ANKAN and a.tile // 4 == cvt.mpsz_to_tid(target_mpsz) // 4:
                    action = a
                    break

            assert action is not None, "ANKAN action should be included in obs.legal_actions()"
            if self._verbose:
                print(f">> EXECUTING ANKAN Action: {action}")

            self.obs_dict = self.env.step({player_id: action})
            if self._verbose:
                print(">> OBS (AFTER ANKAN)", self.obs_dict)
        else:
            assert False, "UNHANDLED AnGangAddGang"

    def _handle_chipenggang(self, event: Any, player_id: int, obs: Any) -> None:
        data = event["data"]
        call_tile = None
        consume_tiles_mpsz = []
        for i, t in enumerate(data["tiles"]):
            if data["froms"][i] == player_id:
                consume_tiles_mpsz.append(t)
            else:
                call_tile = t
        consume_tiles_mpsz.sort()

        if data["type"] == 1:
            # PON
            pon_actions = [a for a in obs.legal_actions() if a.type == ActionType.PON]
            assert pon_actions, "ActionType.PON not found"

            target_action = None
            for a in pon_actions:
                if cvt.tid_to_mpsz(a.tile) == call_tile:
                    a_consume_mpsz = sorted(cvt.tid_to_mpsz_list(a.consume_tiles))
                    if a_consume_mpsz == consume_tiles_mpsz:
                        target_action = a
                        break

            assert target_action is not None, f"No matching PON action for consumed {consume_tiles_mpsz}. Avail: {[cvt.tid_to_mpsz_list(a.consume_tiles) for a in pon_actions]}"
            action = target_action
            step_actions = {player_id: action}
            for pid in self.obs_dict.keys():
                if pid != player_id:
                    step_actions[pid] = Action(ActionType.PASS)
            self.obs_dict = self.env.step(step_actions)
            if self._verbose:
                print(">> OBS (AFTER PON)", self.obs_dict)

        elif data["type"] == 0:
            # CHI
            chi_actions = [a for a in obs.legal_actions() if a.type == ActionType.CHI]
            assert chi_actions, "ActionType.CHI not found"

            target_action = None
            for a in chi_actions:
                if cvt.tid_to_mpsz(a.tile) == call_tile:
                    a_consume_mpsz = sorted(cvt.tid_to_mpsz_list(a.consume_tiles))
                    if a_consume_mpsz == consume_tiles_mpsz:
                        target_action = a
                        break

            assert target_action is not None, f"No matching CHI action for consumed {consume_tiles_mpsz}. Avail: {[cvt.tid_to_mpsz_list(a.consume_tiles) for a in chi_actions]}"
            action = target_action
            step_actions = {player_id: action}
            for pid in self.obs_dict.keys():
                if pid != player_id:
                    step_actions[pid] = Action(ActionType.PASS)
            self.obs_dict = self.env.step(step_actions)
            if self._verbose:
                print(">> OBS (AFTER CHI)", self.obs_dict)

        elif data["type"] == 2:
            # DAIMINKAN (Open Kan)
            kan_actions = [a for a in obs.legal_actions() if a.type == ActionType.DAIMINKAN]
            assert kan_actions, "ActionType.DAIMINKAN not found"

            target_action = None
            for a in kan_actions:
                if cvt.tid_to_mpsz(a.tile) == call_tile:
                    a_consume_mpsz = sorted(cvt.tid_to_mpsz_list(a.consume_tiles))
                    if a_consume_mpsz == consume_tiles_mpsz:
                        target_action = a
                        break

            assert target_action is not None, f"No matching DAIMINKAN action for consumed {consume_tiles_mpsz}"
            action = target_action
            step_actions = {player_id: action}
            for pid in self.obs_dict.keys():
                if pid != player_id:
                    step_actions[pid] = Action(ActionType.PASS)
            self.obs_dict = self.env.step(step_actions)
            if self._verbose:
                print(">> OBS (AFTER DAIMINKAN)", self.obs_dict)

        else:
            print(f">> WARNING: Unhandled ChiPengGang type {event['data']['type']}")


def main(path: str, skip: int = 0, verbose: bool = False) -> None:
    game = ReplayGame.from_json(path)
    logger.info(f"Verifying {path}...")
    verifier = MjsoulEnvVerifier(verbose=verbose)
    if not verifier.verify_game(game, skip=skip):
        sys.exit(1)


def scan(verbose: bool = False) -> None:
    valid_kyoku = 0
    for path in sorted(Path("data/game_record_4p_jad_2025-12-14_out/").glob("251214*.json.gz")):
        logger.info(f"Verifying {path}")
        verifier = MjsoulEnvVerifier(verbose=verbose)
        game = ReplayGame.from_json(str(path))
        kyokus = list(game.take_kyokus())

        for i, kyoku in enumerate(kyokus):
            # Start of kyoku scores:
            start_scores = kyoku.events()[0]["data"]["scores"]
            
            if verifier.verify_kyoku(kyoku):
                valid_kyoku += 1
            else:
                logger.info(f"Valid kyoku: {valid_kyoku} kyokus")
                logger.error(f"Invalid kyoku {path}/{i}")
                sys.exit(1)

            # Verify score_deltas internal consistency
            for p in range(4):
                assert start_scores[p] + verifier.env.score_deltas[p] == verifier.env.scores()[p], \
                    f"Score delta mismatch for player {p} in kyoku {i}: {start_scores[p]} + {verifier.env.score_deltas[p]} != {verifier.env.scores()[p]}"

            if i + 1 < len(kyokus):
                expected_scores = kyokus[i+1].events()[0]["data"]["scores"]
                if verifier.env.scores() != expected_scores:
                    logger.error(f"Score mismatch at end of kyoku {i}:")
                    logger.error(f"  Env: {verifier.env.scores()}")
                    logger.error(f"  Log: {expected_scores}")
                    assert verifier.env.scores() == expected_scores


if __name__ == "__main__":
    args = parse_args()
    if args.path == "scan":
        scan(verbose=args.verbose)
    else:
        main(args.path, skip=args.skip, verbose=args.verbose)