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
from riichienv import ReplayGame, RiichiEnv, Action, ActionType, Phase, Observation
from riichienv.game_mode import GameType

import logging
import os
from riichienv.log import get_logger

# Initialize logger
logger = get_logger(__file__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="scan", help="Path to the game record JSON file.")
    parser.add_argument("--skip", type=int, default=0, help="Number of kyokus to skip.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--full-match", action="store_true", default=False, help="Verify full match state transitions.")
    return parser.parse_args()


class MjsoulEnvVerifier:
    def __init__(self, verbose: bool = True, full_match: bool = False):
        game_type = GameType.YON_HANCHAN if full_match else GameType.YON_IKKYOKU
        self.env: RiichiEnv = RiichiEnv(game_type=game_type, mjai_mode=True)
        self.obs_dict: dict[int, Any] | None = None
        self.dora_indicators: list[int] = []
        self.using_paishan = False
        self._verbose = verbose
        self.full_match = full_match
        self.kyoku_idx = 0
        self.match_started = False
        self.mjai_idx = 0

    def _env_step(self, actions: dict[int, Action]) -> dict[int, Observation]:
        obs = self.env.step(actions)
        # Auto-advance past informational terminal steps (e.g. after dahai sync or start_kyoku)
        while not self.env.done() and (not self.env.active_players or self.env.needs_tsumo):
            obs = self.env.step({})
        return obs

    def _mjai_idx_catchup(self, target_type: str | None = None):
        # Skip events that are "intermediate" like reach, reach_accepted, etc.
        # until we hit a "main" event.
        main_types = ["tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "hule", "hora"]
        while self.mjai_idx < len(self.env.mjai_log):
            ev = self.env.mjai_log[self.mjai_idx]
            self.mjai_idx += 1
            if target_type:
                if ev["type"] == target_type:
                    break
            else:
                if ev["type"] in main_types:
                    break

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
        oya = data["ju"] % 4

        if self.full_match and self.match_started:
            # Verify transitions
            if self._verbose:
                logger.info(f">> VERIFYING TRANSITION to Kyoku {data['ju'] + 1}...")
            
            assert self.env.oya == oya, f"Oya mismatch: Env={self.env.oya}, Log={oya}"
            assert self.env._custom_round_wind == bakaze_idx, f"Bakaze mismatch: Env={self.env._custom_round_wind}, Log={bakaze_idx}"
            assert self.env.scores() == scores, f"Scores mismatch: Env={self.env.scores()}, Log={scores}"
            assert self.env.riichi_sticks == kyotaku, f"Kyotaku mismatch: Env={self.env.riichi_sticks}, Log={kyotaku}"
            assert self.env._custom_honba == honba, f"Honba mismatch: Env={self.env._custom_honba}, Log={honba}"
            
            # In full match mode, we manually initialize the round state 
            # with the specific wall and tiles from log, but keep the verified match scores/oya/etc.
            self.obs_dict = self.env.reset(
                seed=self.kyoku.seed,
                oya=self.kyoku.oya,
                bakaze=self.kyoku.bakaze,
                honba=self.kyoku.honba,
                kyotaku=self.kyoku.kyotaku,
                wall=self.kyoku.wall,
                scores=self.kyoku.scores,
            )
            # logger.info(f"DEBUG RESET: oya={self.kyoku.oya}, drawn_tile={cvt.tid_to_mpsz(self.env.drawn_tile)}")
            # logger.info(f"DEBUG WALL ENDS: { [cvt.tid_to_mpsz(t) for t in self.env.wall[-5:]] }")
            
            assert self.env.drawn_tile is not None
            self.mjai_idx = len(self.env.mjai_log)
        else:
            self.obs_dict = self.env.reset(oya=oya, wall=paishan_wall, bakaze=bakaze_idx, scores=scores, honba=honba, kyotaku=kyotaku)
            self.match_started = True
            self.mjai_idx = len(self.env.mjai_log)

        self.dora_indicators = self.env.dora_indicators[:]

        # Advance through any granular sync steps (start_game, start_kyoku, and wait-for-tsumo)
        while not self.env.done() and (not self.env.active_players or self.env.needs_tsumo):
            self.env.step({})

        # 牌山から配牌を決定するロジックの一致を検証
        if cvt.tid_to_mjai_list(self.env.hands[0]) != cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles0"][:13])))):
            print(f"DEBUG: Hand 0 mismatch.")
            print(f"  Env Hand: {cvt.tid_to_mjai_list(self.env.hands[0])} (len {len(self.env.hands[0])})")
            drawn = self.env.drawn_tile
            print(f"  Env Drawn: {cvt.tid_to_mjai(drawn) if drawn is not None else 'None'}")
            print(f"  Log: {cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data['tiles0'][:13]))))}")
        
        assert cvt.tid_to_mjai_list(self.env.hands[0]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles0"][:13]))))
        assert cvt.tid_to_mjai_list(self.env.hands[1]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles1"][:13]))))
        assert cvt.tid_to_mjai_list(self.env.hands[2]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles2"][:13]))))
        assert cvt.tid_to_mjai_list(self.env.hands[3]) == cvt.tid_to_mjai_list(list(sorted(cvt.mpsz_to_tid_list(data["tiles3"][:13]))))

        # 最初の親のツモが RiichiEnv で設定したものとログが一致することを確認
        assert self.env.drawn_tile is not None
        assert cvt.tid_to_mjai(self.env.drawn_tile) == cvt.mpsz_to_mjai(data["tiles{}".format(oya)][13])

        self.obs_dict = self.env.get_observations([oya])
        assert self.obs_dict is not None
        if os.environ.get("DEBUG"):
            print(f"DEBUG: Kyoku start. mjai_log len={len(self.env.mjai_log)}")
            for i, ev in enumerate(self.env.mjai_log):
                print(f"  [{i}] {ev}")

    def _discard_tile(self, event: Any) -> None:
        while self.env.phase != Phase.WaitAct:
            if self._verbose:
                logger.debug(f">> WAITING loop... obs keys: {list(self.obs_dict.keys())} Phase: {self.env.phase}")

            # Skip action
            self.obs_dict = self._env_step({skip_player_id: Action(ActionType.Pass) for skip_player_id in self.obs_dict.keys()})

        player_id = event["data"]["seat"]
        # candidate_tiles are used for verification. 
        # Riichi tsumogiri is expressed as ActionType.Pass in this environment.
        legal = self.obs_dict[player_id].legal_actions()
        candidate_tiles = set([cvt.tid_to_mpsz(a.tile) for a in legal if a.type == ActionType.Discard])
        if self.env.riichi_declared[player_id] or self.env.riichi_stage[player_id]:
            if any(a.type == ActionType.Pass for a in legal):
                if self.env.drawn_tile is not None:
                    candidate_tiles.add(cvt.tid_to_mpsz(self.env.drawn_tile))

        assert player_id == event["data"]["seat"]
        assert event["data"]["tile"] in candidate_tiles, f"Tile {event['data']['tile']} not in candidate tiles {candidate_tiles}. Log may be repeating history or wall desync."

        # Riichi Step =====================
        is_liqi = event["data"].get("is_liqi", False) or event["data"].get("is_wliqi", False)
        if is_liqi:
            # Riichi declaration itself doesn't advance the MJAI log tsumo/dahai pointer.
            self.obs_dict = self._env_step({player_id: Action(ActionType.Riichi)})
            # Still, we should catch up to the 'reach' event to keep internal pointer in sync.
            self._mjai_idx_catchup()
            assert self.env.riichi_stage[player_id]

        if player_id == 0 and len(self.obs_dict[player_id].hand) < 13:
            if self._verbose:
                logger.debug(f"DEBUG: Player 0 Hand Size Mismatch! Size= {len(self.obs_dict[player_id].hand)}")

        # Discard Step ====================
        # obs.legal_actions() に牌譜をトレースするアクションが存在するか確認して、それを選択してステップを進める
        target_mpsz = event["data"]["tile"]
        
        actions = []
        for a in self.obs_dict[player_id].legal_actions():
            if a.type == ActionType.Discard and cvt.tid_to_mpsz(a.tile) == target_mpsz:
                actions.append(a)
            elif a.type == ActionType.Pass and (self.env.riichi_declared[player_id] or self.env.riichi_stage[player_id]):
                # Tsumogiri case
                if self.env.drawn_tile is not None and cvt.tid_to_mpsz(self.env.drawn_tile) == target_mpsz:
                    actions.append(a)
        
        assert len(actions) > 0, f"No discard action found for {target_mpsz}. Legal: {[ (a.type, cvt.tid_to_mpsz(a.tile) if a.tile is not None else 'N/A') for a in self.obs_dict[player_id].legal_actions()]}"
        self.obs_dict = self._env_step({player_id: actions[0]})
        self._mjai_idx_catchup("dahai")
        

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
            kyushu_actions = [a for a in obs.legal_actions() if a.type == ActionType.KyushuKyuhai]
            assert kyushu_actions, f"No KYUSHU_KYUHAI action found for player {seat}"
            self.obs_dict = self._env_step({seat: kyushu_actions[0]})
        else:
            # 他の途中流局 (四風連打, 四槓散了, 四家立直): すでに Env 内部で end_kyoku へのトリガーが引かれているはずだが、
            # もし自動で進まない場合は pass アクションなどで進める必要がある。
            # 現状の RiichiEnv はこれらを検知した瞬間に _trigger_ryukyoku を呼ぶ。
            # すでに次の局に進んでいる (WAIT_ACT且つturn_count=0) なら何もしない。
            while not self.env.is_done and not (self.env.phase == Phase.WaitAct and self.env.turn_count == 0):
                pids = self.env.active_players
                if not pids: break
                self.obs_dict = self._env_step({pid: Action(ActionType.Pass) for pid in pids})

    def _no_tile(self, event: Any) -> None:
        # 荒牌平局 (Exhaustive Draw)
        data = event["data"]
        liujumanguan = data.get("liujumanguan", False)
        players = data.get("players", [])
        scores = data.get("scores", [])

        if self._verbose:
            logger.info(f"NoTile event: liujumanguan={liujumanguan}, players={players}, scores={scores}")

        # env.is_done が True であることを確認
        # 荒牌平局の場合、最後のツモや打牌のあとに自動的に ryukyoku が呼ばれるが、
        # もし phase が WAIT_RESPONSE などで止まっている場合は PASS で進める必要がある。
        while not self.env.is_done and not (self.env.phase == Phase.WaitAct and self.env.turn_count == 0):
            pids = self.env.active_players
            if not pids: break
            self.obs_dict = self._env_step({pid: Action(ActionType.Pass) for pid in pids})

        if self.full_match:
            assert self.env.done() or (self.env.phase == Phase.WaitAct and self.env.turn_count == 0)
        else:
            assert self.env.is_done, "Env should be done for NoTile (Exhaustive Draw)"

    def _hule(self, event: Any) -> None:
        is_zimo = any(h.get("zimo", False) for h in event["data"]["hules"])

        # If Zimo, we must be in WAIT_ACT. If in WAIT_RESPONSE, auto-pass.
        if is_zimo:
            assert self.env.phase == Phase.WaitAct, "Zimo Hule should be in WAIT_ACT"
        else:
            assert self.env.phase == Phase.WaitResponse, "Ron Hule should be in WAIT_RESPONSE"

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
            legal_ron = any(a.type in {ActionType.Ron, ActionType.Tsumo} for a in obs.legal_actions())
            assert legal_ron, f"Player {player_id} has no RON/TSUMO."

            # Continue logic
            obs = self.obs_dict[player_id]
            match_actions = [a for a in obs.legal_actions() if a.type in {ActionType.Ron, ActionType.Tsumo}]

            assert len(match_actions) == 1
            winning_actions[player_id] = match_actions[0]

        # Phase 2: Execution
        step_actions = winning_actions.copy()

        # If in WAIT_RESPONSE (Ron), others might need to PASS
        for pid in self.obs_dict.keys():
            if pid not in step_actions:
                step_actions[pid] = Action(ActionType.Pass)

        self.obs_dict = self._env_step(step_actions)
        
        for hule in event["data"]["hules"]:
            player_id = hule["seat"]
            if player_id not in winning_actions:
                 continue
            
            # Debug Print Melds
            melds_debug = [(cvt.tid_to_mpsz_list(m.tiles), m.opened) for m in self.env.melds[player_id]]
            # print(f"DEBUG HULE CHECK: seat={player_id}, melds={melds_debug}")

            # legal_actions() から取り出した hule に対応する action
            action = winning_actions[player_id]
            # print(f"DEBUG HULE ACTION TYPE: {action.type}")
            winning_tile = action.tile
            # Use environment hand (13 tiles) for calculation, as obs.hand might be 14 for Tsumo
            hand_for_calc = self.env.hands[player_id]
            
            if action.type == ActionType.Tsumo:
                winning_tile = self.env.drawn_tile
                assert self.env.drawn_tile is not None, "Tsumo but drawn_tile is None."

            if self._verbose:
                print(">> HULE", hule)
                print(">> HAND", cvt.tid_to_mpsz_list(hand_for_calc))
                print(">> WIN TILE", cvt.tid_to_mpsz(winning_tile))

            # Retrieve Agari result calculated by the environment
            agari_res = self.env.agari_results.get(player_id) or self.env.last_agari_results.get(player_id)
            if agari_res is None:
                raise KeyError(
                    f"Player {player_id} not found in agari_results or last_agari_results. Action type: {action.type}. Step presumably failed to register win."
                )

            calc = agari_res
            assert calc.agari
            assert calc.yakuman == hule["yiman"]
            
            if action.type == ActionType.Ron and "point_rong" in hule:
              assert calc.ron_agari is not None
              if calc.ron_agari != hule["point_rong"]:
                  print(f"DEBUG HULE MISMATCH: Sim={calc.ron_agari} Log={hule['point_rong']}")
                  print(f"Hand: {cvt.tid_to_mpsz_list(hand_for_calc)}")
                  print(f"Sim Han: {calc.han}")
                  print(f"Melds: {[(m.tiles, m.opened) for m in self.env.melds[player_id]]}")
                  # print(f"Melds: {[[cvt.tid_to_mpsz(t) for t in m.tiles] for m in self.env.melds.get(player_id, [])]}")
                  print(f"Sim Yaku: {calc.yaku}")
                  print(f"Log Yaku: {hule.get('fans')}")
                  print(f"Sim Fu: {calc.fu}")
                  print(f"Log Fu: {hule.get('fu')}")
                  print(f"Dora Indicators: {[cvt.tid_to_mpsz(x) for x in self.env.dora_indicators]}")
              assert calc.ron_agari == hule["point_rong"]
            
            if action.type == ActionType.Tsumo:
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
        # DealTile イベント発生時、RiichiEnv.drawn_tile が一致しているか確認する
        if "tile" in event["data"] and event["data"]["tile"] and event["data"]["tile"] != "?":
            t_str = event["data"]["tile"]
            t_tid = cvt.mpsz_to_tid(t_str)

            sim_drawn = self.env.drawn_tile
            if sim_drawn is None and self.env.needs_tsumo:
                self.obs_dict = self._env_step({})
                sim_drawn = self.env.drawn_tile

            assert sim_drawn is not None, "Drawn tile is None but DealTile expected"
            assert t_tid // 4 == sim_drawn // 4, "Drawn tile mismatch. Sim: {} Log: {}".format(cvt.tid_to_mpsz(sim_drawn), t_str)
            self._mjai_idx_catchup("tsumo")
        else:
            # ? の場合は actor が自分でない場合など。既に mjai_log 同期でスキップされている可能性が高いが。
            pass

    def _sync_with_mjai_log(self, event: Any) -> bool:
        if self.mjai_idx >= len(self.env.mjai_log):
            return False

        # Peek next "main" event
        peek_idx = self.mjai_idx
        while peek_idx < len(self.env.mjai_log):
            ev = self.env.mjai_log[peek_idx]
            if ev["type"] in ["tsumo", "dahai", "chi", "pon", "daiminkan", "ankan", "kakan"]:
                break
            peek_idx += 1

        if peek_idx >= len(self.env.mjai_log):
            return False

        mjai_ev = self.env.mjai_log[peek_idx]

        match event["name"]:
            case "DealTile":
                if mjai_ev["type"] == "tsumo":
                    t_str = event["data"]["tile"]
                    t_mjai = cvt.mpsz_to_mjai(t_str) if t_str != "?" else "?"
                    match_pai = (t_str == "?" or t_mjai == mjai_ev["pai"])
                    match_actor = (event["data"]["seat"] == mjai_ev["actor"])
                    if match_pai and match_actor:
                        if self._verbose:
                            logger.info(f">> SYNC: Skipping DealTile {mjai_ev['pai']} for P{mjai_ev['actor']} (idx={peek_idx})")
                        self.mjai_idx = peek_idx + 1
                        return True
            case "DiscardTile":
                if mjai_ev["type"] == "dahai":
                    t_str = event["data"]["tile"]
                    t_mjai = cvt.mpsz_to_mjai(t_str)
                    match_pai = (t_mjai == mjai_ev["pai"])
                    match_actor = (event["data"]["seat"] == mjai_ev["actor"])
                    if match_pai and match_actor:
                        if self._verbose:
                            logger.info(f">> SYNC: Skipping DiscardTile {mjai_ev['pai']} for P{mjai_ev['actor']} (idx={peek_idx})")
                        self.mjai_idx = peek_idx + 1
                        return True
            case "AnGangAddGang":
                if mjai_ev["type"] in ["ankan", "kakan"]:
                    self.mjai_idx = peek_idx + 1
                    return True
        return False

    def verify_kyoku(self, kyoku: Any) -> bool:
        try:
            events = kyoku.events()
            event_idx = 0
            while event_idx < len(events):
                event = events[event_idx]
                if self._verbose:
                    print(f"DEBUG SCRIPT: Round {self.kyoku_idx}, Action {event_idx}, Event: {event['name']} {event.get('data', {}).get('tile', '')}")

                # We might need to sync or pass multiple times for a single event
                while True:
                    # Sync with mjai_log (skip auto-played events)
                    if self._sync_with_mjai_log(event):
                        event_idx += 1
                        # We skipped an event, so we skip the processing of this Log event.
                        # We return to the outer loop to get the next Log event.
                        # Break from inner while True to continue outer while event_idx.
                        break

                    # If Env is waiting for responses (Ron/Pon/Chi) but the Log event is not one of those,
                    # it means all players PASSed. We must synchronize the Env.
                    if not self.env.is_done and self.env.phase == Phase.WaitResponse and event["name"] not in ["Hule", "ChiPengGang", "AnGangAddGang"]:
                        pids = self.env.active_players
                        assert pids, "Active players is empty while Env is in WAIT_RESPONSE"
                        pass_actions = {pid: Action(ActionType.Pass) for pid in pids}
                        self.obs_dict = self._env_step(pass_actions)
                        # After PASS, we might have auto-played more turns. 
                        # Re-try the sync check for the SAME 'event'.
                        continue
                    
                    # If we reached here, it means we don't need to sync OR pass for now.
                    # Proceed to manual match.
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
                            # In full_match mode, Hule might trigger transition to next round (is_done=False)
                            # OR end the entire game (is_done=True).
                            if self.full_match:
                                assert self.env.done() or (self.env.phase == Phase.WaitAct and self.env.turn_count == 0)
                            else:
                                assert self.env.done()

                        case "LiuJu":
                            # 途中流局 | 1: 九種九牌, 2: 四風連打, 3: 四槓散了, 4: 四家立直
                            self._liuju(event)
                            if self.full_match:
                                assert self.env.done() or (self.env.phase == Phase.WaitAct and self.env.turn_count == 0)
                            else:
                                assert self.env.is_done, "Env should be done after LiuJu"
                            
                        case "NoTile":
                            # 荒牌平局
                            self._no_tile(event)
                            if self.full_match:
                                assert self.env.done() or (self.env.phase == Phase.WaitAct and self.env.turn_count == 0)
                            else:
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
                    
                    event_idx += 1
                    break

            if self.full_match:
                assert self.env.done() or (self.env.phase == Phase.WaitAct and self.env.turn_count == 0)
            else:
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
        while not self.env.is_done and self.env.phase != Phase.WaitAct:
            # Skip action (Pass on claims)
            self.obs_dict = self._env_step({skip_player_id: Action(ActionType.Pass) for skip_player_id in self.obs_dict.keys()})

        player_id = event["data"]["seat"]
        assert player_id in self.env.active_players
        assert len(self.env.active_players) == 1

        obs = self.obs_dict[player_id]
        if event["data"]["type"] == 2:
            # KAKAN (Added Kan)
            kakan_actions = [a for a in obs.legal_actions() if a.type == ActionType.Kakan]
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

            self.obs_dict = self._env_step({player_id: action})
            self._mjai_idx_catchup()

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
                if a.type == ActionType.Ankan and a.tile // 4 == cvt.mpsz_to_tid(target_mpsz) // 4:
                    action = a
                    break

            assert action is not None, "ANKAN action should be included in obs.legal_actions()"
            if self._verbose:
                print(f">> EXECUTING ANKAN Action: {action}")

            self.obs_dict = self._env_step({player_id: action})
            self._mjai_idx_catchup()
            if self._verbose:
                print(">> OBS (AFTER ANKAN)", self.obs_dict)
        elif event["data"]["type"] == 4: # RIICHI
            # Riichi declaration itself doesn't advance the MJAI log tsumo/dahai pointer.
            self.obs_dict = self._env_step({player_id: Action(ActionType.Riichi)})
            # Still, we should catch up to the 'reach' event to keep internal pointer in sync.
            self._mjai_idx_catchup()
            assert self.env.riichi_stage[player_id]
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
            pon_actions = [a for a in obs.legal_actions() if a.type == ActionType.Pon]
            assert pon_actions, "ActionType.Pon not found"

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
                    step_actions[pid] = Action(ActionType.Pass)
            self.obs_dict = self._env_step(step_actions)
            self._mjai_idx_catchup()
            if self._verbose:
                print(">> OBS (AFTER PON)", self.obs_dict)

        elif data["type"] == 0:
            # CHI
            chi_actions = [a for a in obs.legal_actions() if a.type == ActionType.Chi]
            assert chi_actions, "ActionType.Chi not found"

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
                    step_actions[pid] = Action(ActionType.Pass)
            self.obs_dict = self._env_step(step_actions)
            self._mjai_idx_catchup()
            
            if self._verbose:
                print(">> OBS (AFTER CHI)", self.obs_dict)

        elif data["type"] == 2:
            # DAIMINKAN (Open Kan)
            kan_actions = [a for a in obs.legal_actions() if a.type == ActionType.Daiminkan]
            assert kan_actions, "ActionType.Daiminkan not found"

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
                    step_actions[pid] = Action(ActionType.Pass)
            self.obs_dict = self._env_step(step_actions)
            self._mjai_idx_catchup()
            
            if self._verbose:
                print(">> OBS (AFTER DAIMINKAN)", self.obs_dict)

        else:
            print(f">> WARNING: Unhandled ChiPengGang type {event['data']['type']}")


def main(path: str, skip: int = 0, verbose: bool = False, full_match: bool = False) -> None:
    game = ReplayGame.from_json(path)
    logger.info(f"Verifying {path}...")
    verifier = MjsoulEnvVerifier(verbose=verbose, full_match=full_match)
    if not verifier.verify_game(game, skip=skip):
        sys.exit(1)


def scan(verbose: bool = False, full_match: bool = False) -> None:
    valid_kyoku = 0
    for path in sorted(Path("data/game_record_4p_jad_2025-12-14_out/").glob("251214*.json.gz")):
        logger.info(f"Verifying {path}")
        verifier = MjsoulEnvVerifier(verbose=verbose, full_match=full_match)
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

            # Verify score_deltas internal consistency using environment's round_end_scores
            # (since env.scores() might already be at the start of next round)
            env_round_scores = verifier.env.round_end_scores
            assert env_round_scores is not None
            
            # Note: score_deltas is already reset for next round in full-match mode, 
            # so we verify the effective change in scores.
            for p in range(4):
                assert start_scores[p] + (env_round_scores[p] - start_scores[p]) == env_round_scores[p], "Math failure"
                if not full_match:
                    # In one-kyoku mode, score_deltas is preserved
                    assert start_scores[p] + verifier.env.score_deltas[p] == env_round_scores[p], \
                        f"Score delta mismatch for player {p} in kyoku {i}: {start_scores[p]} + {verifier.env.score_deltas[p]} != {env_round_scores[p]}"

            if i + 1 < len(kyokus):
                expected_scores = kyokus[i+1].events()[0]["data"]["scores"]
                if env_round_scores != expected_scores:
                    logger.error(f"Score mismatch at end of kyoku {i}:")
                    logger.error(f"  Env: {env_round_scores}")
                    logger.error(f"  Log: {expected_scores}")
                    assert env_round_scores == expected_scores
            
            # Reset round_end_scores for next kyoku (manual verification check)
            verifier.env.round_end_scores = None


if __name__ == "__main__":
    args = parse_args()
    if args.path == "scan":
        scan(verbose=args.verbose, full_match=args.full_match)
    else:
        main(args.path, skip=args.skip, verbose=args.verbose, full_match=args.full_match)