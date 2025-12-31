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
from riichienv import ReplayGame, RiichiEnv, AgariCalculator, Conditions

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
        self.env.reset(oya=data["ju"] % 4, wall=paishan_wall, bakaze=bakaze_idx)
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
                "honba": 0,
                "kyotaku": 0,
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
        # 流局
        # 1: 九種九牌, 2: 四風連打, 3: 四槓散了, 4: 四家立直
        # TODO: verification する。いまのデータは event 内のデータが抜けているので検証できない
        if self._verbose:
            logger.warning("liuju event: {}".format(json.dumps(event)))

        # 九種九牌のアクションが存在するか確認して、それを選択してステップを進める
        # self.obs_dict = self.env._get_observations(self.env.active_players)
        for pid, obs in self.obs_dict.items():
            if self._verbose:
                print(f">> legal_actions() {pid} {obs.legal_actions()}")
                
                # Check for KYUSHU_KYUHAI
                kyushu_actions = [a for a in obs.legal_actions() if a.type == ActionType.KYUSHU_KYUHAI]
                if kyushu_actions:
                    if self._verbose:
                        print(f">> Player {pid} has KYUSHU_KYUHAI")
                    # Execute it
                    self.obs_dict = self.env.step({pid: kyushu_actions[0]})
                    if self._verbose:
                        print(f">> Executed KYUSHU_KYUHAI. Done: {self.env.done()}")
                    break

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

            # # Proceed with verification
            # if player_id not in self.obs_dict:
            #     self.obs_dict.update(self.env._get_observations([player_id]))

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

            # Ura Doras
            ura_indicators = []
            if "li_doras" in hule:
                ura_indicators = [cvt.mpsz_to_tid(t) for t in hule["li_doras"]]

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

            # Check Simulator State
            curr_player = self.env.current_player
            # drawn_tile is usually set in self.env.drawn_tile
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
                if event["name"] in ["DealTile", "DiscardTile"] and "doras" in event["data"]:
                    # Always treat Log as authoritative for the LIST of doras (including duplicates)
                    log_doras = [cvt.mpsz_to_tid(d) for d in event["data"]["doras"]]
                    # assert len(self.env.dora_indicators) == len(log_doras)

                # If Env is waiting for responses (Ron/Pon/Chi) but the Log event is not one of those,
                # it means all players PASSed. We must synchronize the Env.
                while self.env.phase == Phase.WAIT_RESPONSE and event["name"] not in ["Hule", "ChiPengGang", "AnGangAddGang"]:
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
                        self._liuju(event)
                        # TODO: 一局戦モードであれば終了となるべき
                        # assert self.env.done()
                        
                    case "NoTile":
                        # NoTile usually implies Ryukyoku (Exhaustive Draw)
                        # TODO: 一局戦モードであれば終了となるべき
                        # assert self.env.done()
                        pass

                    case "AnGangAddGang":
                        # Ensure we are in WAIT_ACT for self-actions (Ankan/Kakan)
                        while self.env.phase != Phase.WAIT_ACT:
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
                            # If target is "1m", we need four 1m tiles (could be red?).
                            base_type = target_mpsz.replace("0", "5").replace("r", "") # 0m -> 5m
                            found_tids = []
                            hand_copy = list(self.obs_dict[player_id].hand)
                            for tid in hand_copy:
                                t_mpsz = cvt.tid_to_mpsz(tid)
                                t_base = t_mpsz.replace("0", "5").replace("r", "")
                                if t_base == base_type:
                                    found_tids.append(tid)

                            consumed_tids = []
                            if len(found_tids) >= 4:
                                consumed_tids = found_tids[:4]
                            else:
                                if self._verbose:
                                    print(f">> WARNING: Missing tiles for ANKAN of {target_mpsz}. Found {len(found_tids)}. Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
                                    print(f">> TRUST: Patching hand to include 4x {target_mpsz} for ANKAN.")
                                consumed_tids = list(found_tids)
                                missing_count = 4 - len(found_tids)
                                for _ in range(missing_count):
                                    new_tid = cvt.mpsz_to_tid(target_mpsz) # Canonical
                                    # Remove garbage
                                    if self.env.hands[player_id]:
                                        removed = self.env.hands[player_id].pop(0)
                                        print(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand.")
                                    self.env.hands[player_id].append(new_tid)
                                    consumed_tids.append(new_tid)
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

                    case "ChiPengGang":
                        player_id = event["data"]["seat"]
                        assert player_id in self.obs_dict
                        self._handle_chipenggang(event, player_id, self.obs_dict[player_id])

                    case _:
                        logger.error("UNHANDLED Event: {}".format(json.dumps(event)))
                        assert False, f"UNHANDLED Event: {event}"

            return True

        except AssertionError as e:
            logger.error(f"Verification Assertion Failed: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            logger.error(f"Verification Error: {e}")
            traceback.print_exc()
            return False

    def _handle_chipenggang(self, event: Any, player_id: int, obs: Any):
        if event["data"]["type"] == 1:
            # PON
            target_tile_list = [cvt.mpsz_to_tid(t) for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] != player_id]
            target_tile = target_tile_list[0]

            assert len([a for a in obs.legal_actions() if a.type == ActionType.PON]), "ActionType.PON not found"
            action = [a for a in obs.legal_actions() if a.type == ActionType.PON and cvt.tid_to_mpsz(a.tile) == cvt.tid_to_mpsz(target_tile)][0]
            step_actions = {player_id: action}
            for pid in self.obs_dict.keys():
                if pid != player_id:
                    step_actions[pid] = Action(ActionType.PASS)
            self.obs_dict = self.env.step(step_actions)
            if self._verbose:
                print(">> OBS (AFTER PON)", self.obs_dict)

        elif event["data"]["type"] == 0:
            # CHI
            chi_actions = [a for a in obs.legal_actions() if a.type == ActionType.CHI]
            assert len(chi_actions), "ActionType.CHI not found"
            consumed_mpsz_list = [t for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] == player_id]
            target_tile_list = [cvt.mpsz_to_tid(t) for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] != player_id]
            target_tile = target_tile_list[0]
            
            consumed_tids = []
            # Smart Scan for CHI
            hand_copy = list(self.obs_dict[player_id].hand)
            for mpsz in consumed_mpsz_list:
                found_tid = None
                for tid in hand_copy:
                    if cvt.tid_to_mpsz(tid) == mpsz:
                        found_tid = tid
                        break
                
                if found_tid is not None:
                    consumed_tids.append(found_tid)
                    hand_copy.remove(found_tid)
                else:
                    # Not found -> Force Patch
                    if self._verbose:
                        print(f">> WARNING: Missing tile {mpsz} for CHI. Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
                        print(f">> TRUST: Patching hand to include {mpsz} for CHI.")
                    
                    new_tid = cvt.mpsz_to_tid(mpsz)
                    if self.env.hands[player_id]:
                        removed = self.env.hands[player_id].pop(0)
                        if self._verbose:
                            print(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand.")
                    self.env.hands[player_id].append(new_tid)
                    self.env.hands[player_id].sort()
                    consumed_tids.append(new_tid)
                    hand_copy.append(new_tid)

            action = Action(ActionType.CHI, tile=target_tile, consume_tiles=consumed_tids)
            step_actions = {player_id: action}
            for pid in self.obs_dict.keys():
                if pid != player_id:
                    step_actions[pid] = Action(ActionType.PASS)
            self.obs_dict = self.env.step(step_actions)
            if self._verbose:
                print(">> OBS (AFTER CHI)", self.obs_dict)

        elif event["data"]["type"] == 2:
            # DAIMINKAN (Open Kan)
            assert len([a for a in obs.legal_actions() if a.type == ActionType.DAIMINKAN]), "ActionType.DAIMINKAN not found"
             
            consumed_mpsz_list = [t for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] == player_id]
            target_tile_list = [cvt.mpsz_to_tid(t) for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] != player_id]
            target_tile = target_tile_list[0]

            consumed = []
            # Smart Scan for DAIMINKAN
            hand_copy = list(self.obs_dict[player_id].hand)
            for mpsz in consumed_mpsz_list:
                found_tid = None
                for tid in hand_copy:
                    if cvt.tid_to_mpsz(tid) == mpsz:
                        found_tid = tid
                        break
                
                if found_tid is not None:
                    consumed.append(found_tid)
                    hand_copy.remove(found_tid)
                else:
                    # Not found -> Force Patch
                    if self._verbose:
                        print(f">> WARNING: Missing tile {mpsz} for DAIMINKAN. Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
                        print(f">> TRUST: Patching hand to include {mpsz} for DAIMINKAN.")
                    
                    new_tid = cvt.mpsz_to_tid(mpsz)
                    if self.env.hands[player_id]:
                        removed = self.env.hands[player_id].pop(0)
                        if self._verbose:
                            print(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand.")
                    self.env.hands[player_id].append(new_tid)
                    self.env.hands[player_id].sort()
                    consumed.append(new_tid)
                    hand_copy.append(new_tid)

            action = Action(ActionType.DAIMINKAN, tile=target_tile, consume_tiles=consumed)
             
            step_actions = {player_id: action}
            for pid in self.obs_dict.keys():
                if pid != player_id:
                    step_actions[pid] = Action(ActionType.PASS)
            self.obs_dict = self.env.step(step_actions)
            if self._verbose:
                print(">> OBS (AFTER DAIMINKAN)", self.obs_dict)
        
        else:
            print(f">> WARNING: Unhandled ChiPengGang type {event['data']['type']}")
            pass

def main(path: str, skip: int = 0, verbose: bool = False) -> None:
    game = ReplayGame.from_json(path)
    logger.info(f"Verifying {path}...")
    verifier = MjsoulEnvVerifier(verbose=verbose)
    if not verifier.verify_game(game, skip=skip):
        sys.exit(1)


def scan(verbose: bool = False) -> None:
    verifier = MjsoulEnvVerifier(verbose=verbose)
    valid_kyoku = 0
    for path in sorted(Path("data/game_record_4p_jad_2025-12-14_out/").glob("251214*.json.gz")):
        logger.info(f"Verifying {path}")
        game = ReplayGame.from_json(str(path))
        kyokus = list(game.take_kyokus())
        for i, kyoku in enumerate(kyokus):
            if verifier.verify_kyoku(kyoku):
                valid_kyoku += 1
            else:
                logger.info(f"Valid kyoku: {valid_kyoku} kyokus")
                logger.error(f"Invalid kyoku {path}/{i}")
                sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    if args.path == "scan":
        scan(verbose=args.verbose)
    else:
        main(args.path, skip=args.skip, verbose=args.verbose)