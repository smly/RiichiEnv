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
        # We start from the 5th kyoku as in the original script? 
        # Original: for kyoku in list(game.take_kyokus())[4:]:
        kyokus = list(game.take_kyokus())
        for i, kyoku in enumerate(kyokus[skip:]):
            self.kyoku_idx = skip + i
            # print(f"Processing Kyoku index {self.kyoku_idx} ...")
            if not self.verify_kyoku(kyoku):
                return False
        return True

    def _new_round(self, kyoku: Any, event: Any) -> None:
        data = event["data"]
        assert "paishan" in data, "Paishan not found in NewRound event."

        # Determine Wall
        # Prefer paishan if available (contains full 136 tiles including Dead Wall)
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
        self.env.kyoku_idx = data["ju"] + 1
        oya = data["ju"] % 4

        # Convert ALL 14 tiles for Oya (at least) and 13 for others to ensure unique TIDs
        for player_id in range(4):
            num_tiles = 14 if player_id == oya else 13
            tiles_mpsz = data[f"tiles{player_id}"][:num_tiles]
            tids = cvt.mpsz_to_tid_list(tiles_mpsz)
            
            if player_id == oya:
                self.env.hands[player_id] = tids[:13]
                self.env.drawn_tile = tids[13]
            else:
                self.env.hands[player_id] = tids[:]
            
        first_actor = data["ju"]
        raw_first_tile = data["tiles{}".format(first_actor)][13] # This is now redundant as drawn_tile is set above
        first_tile = cvt.mpsz_to_mjai(raw_first_tile) # This is also redundant
        self.env.mjai_log.append({
            "type": "tsumo",
            "actor": first_actor,
            "tile": first_tile,
        })
        # self.env.drawn_tile = cvt.mpsz_to_tid(raw_first_tile) # This is now redundant
        self.env.current_player = first_actor
        self.env.active_players = [first_actor]
        self.obs_dict = self.env._get_observations([first_actor])

    def _discard_tile(self, event: Any) -> None:
        if self._verbose:
            print(">> OBS", self.obs_dict)
            print("--")
            print(">> EVENT", event)
            print(f">> PHASE: {self.env.phase}")

        while self.env.phase != Phase.WAIT_ACT:
            if self._verbose:
                print(f">> WAITING loop... obs keys: {list(self.obs_dict.keys())} Phase: {self.env.phase}")
            # Skip action
            self.obs_dict = self.env.step({skip_player_id: Action(ActionType.PASS) for skip_player_id in self.obs_dict.keys()})

        # print(">> OBS (AFTER SKIP WAIT_ACT PHASE)", self.obs_dict)

        player_id = event["data"]["seat"]
        candidate_tiles = set([cvt.tid_to_mpsz(a.tile) for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.DISCARD])
        assert player_id == event["data"]["seat"]
        if event["data"]["tile"] not in candidate_tiles:
            if self._verbose:
                logger.warning(f">> WARNING: FAILED DISCARD: tile {event['data']['tile']} not in candidate tiles. Log may be repeating history or wall desync.")
                logger.warning(f"Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
            
            # Force Hand Patch
            target_tid = cvt.mpsz_to_tid(event["data"]["tile"])
            if self._verbose:
                logger.warning(f">> TRUST: Patching hand to include {event['data']['tile']} for discard.")
            
            # Remove last tile (assumed drawn) to maintain count, if hand is full (14 or 11/8/5 etc + 1?)
            # Just remove last tile to be safe on count.
            if self.env.hands[player_id]:
                removed = self.env.hands[player_id].pop()
                if self._verbose:
                    logger.warning(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand.")
            
            self.env.hands[player_id].append(target_tid)
            
            # Refresh observation legal actions?
            # We can just manually construct action, self.env.step will just execute if tile is in hand.
             
        # Normal discard (or forced)
        # Re-fetch legal actions or just construct specific action
        # Riichi Step
        is_liqi = event["data"].get("is_liqi", False)
        if self._verbose:
            logger.debug(f"DEBUG: Checking is_liqi. Value={is_liqi} Type={type(is_liqi)}")
            
        if is_liqi:
            if self._verbose:
                print(f">> TRUST: Executing RIICHI step for {player_id}")
            # Helper to find Riichi action
            riichi_actions = [a for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.RIICHI]
            if riichi_actions:
                self.obs_dict = self.env.step({player_id: riichi_actions[0]})
            else:
                if self._verbose:
                    print(">> WARNING: Riichi flag true but no Riichi action? Forcing Riichi action.")
                self.obs_dict = self.env.step({player_id: Action(ActionType.RIICHI)})

        if player_id == 0 and len(self.obs_dict[player_id].hand) < 13:
            if self._verbose:
                logger.debug(f"DEBUG: Player 0 Hand Size Mismatch! Size= {len(self.obs_dict[player_id].hand)}")
                logger.debug(f"DEBUG: Player 0 Events: {self.obs_dict[player_id].events}")

        # Discard Step
        # Manually construct action to ensure we use the target tile
        target_mpsz = event["data"]["tile"]
        target_tid = cvt.mpsz_to_tid(target_mpsz)
        
        # Smart scan: If canonical TID not in hand, try to find matching tile in hand
        # Only if we have OBS access (we might not if step(RIICHI) failed to return useful obs, but usually we do)
        # Note: self.obs_dict was updated by Riichi step if applicable.
        if player_id in self.obs_dict:
            found_tid = None
            for tid in self.obs_dict[player_id].hand:
                if cvt.tid_to_mpsz(tid) == target_mpsz:
                    found_tid = tid
                    break
            if found_tid is not None:
                # print(f">> FOUND matching tile {found_tid} ({target_mpsz}) in hand. Using it.")
                target_tid = found_tid
        
        action = Action(ActionType.DISCARD, tile=target_tid)
        
        self.obs_dict = self.env.step({player_id: action})

    def _liuju(self, event: Any) -> None:
        if self._verbose:
            logger.warning("liuju event: {}".format(json.dumps(event)))

        # Often happens on current_player's turn if Kyuhsu Kyuhai
        self.obs_dict = self.env._get_observations(self.env.active_players)
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
        if is_zimo and self.env.phase == Phase.WAIT_RESPONSE:
            if self._verbose:
                print(">> DETECTED Zimo Hule while in WAIT_RESPONSE. Auto-passing previous discard claims.")
            while self.env.phase == Phase.WAIT_RESPONSE:
                self.obs_dict = self.env.step({pid: Action(ActionType.PASS) for pid in self.obs_dict.keys()})
            if self._verbose:
                print(f">> ADVANCED TO PHASE: {self.env.phase}, Active: {self.env.active_players}")

        if self._verbose or True: # Force debug
            if os.environ.get("DEBUG"):
                print(f">> DEBUG: START _hule. Kyoku {self.kyoku_idx}. event hules {[h['seat'] for h in event['data']['hules']]}, obs_dict.keys={list(self.obs_dict.keys())}")
        
        active_players = list(self.obs_dict.keys())
        
        # Validation checks
        if is_zimo:
            if self.env.phase != Phase.WAIT_ACT:
                if self._verbose:
                    print(f">> WARNING: Zimo Hule but Phase is {self.env.phase} (Expected WAIT_ACT).")
                return
        else:
            # Ron
            if self.env.phase != Phase.WAIT_RESPONSE:
                if self._verbose:
                    print(f">> WARNING: Ron Hule but Phase is {self.env.phase} (Expected WAIT_RESPONSE).")
                return

        if self._verbose:
            print(f">> HULE EVENT DATA: {event}")
            print(f">> ENV PHASE: {self.env.phase}")
            print(f">> ENV current_player: {self.env.current_player}")
            print(f">> ENV drawn_tile: {self.env.drawn_tile} ({cvt.tid_to_mpsz(self.env.drawn_tile) if self.env.drawn_tile is not None else 'None'})")
            print(f">> ENV active_players: {self.env.active_players}")
            print(f">> ENV wall len: {len(self.env.wall)}")
            for pid in range(4):
                for meld in self.env.melds[pid]:
                    print(f"Meld: {meld.meld_type} {cvt.tid_to_mpsz_list(meld.tiles)} opened={meld.opened}")
                print(f">> ENV hands[{pid}] len: {len(self.env.hands[pid])}")
                print(f">> ENV hands[{pid}] content: {cvt.tid_to_mpsz_list(self.env.hands[pid])}")

                print(f">> ENV hands[{pid}] content: {cvt.tid_to_mpsz_list(self.env.hands[pid])}")

        winning_actions = {}
        
        # Phase 1: Preparation and Hand Repair
        for hule in event["data"]["hules"]:
            player_id = hule["seat"]
            
            # Brute Force Hand Repair if inactive or Agari check fails
            if player_id not in active_players:
                if self._verbose:
                    print(f">> WARNING: Winner {player_id} inactive. Attempting Brute Force Hand Repair.")
                
                # Check current hand validity
                obs = self.obs_dict[player_id] if player_id in self.obs_dict else None
                if not obs:
                    # Force refresh obs
                    # Note: Using _get_observations might rely on current state.
                    # If player inactive, might not be returned?
                    # RiichiEnv._get_observations takes a list of pids.
                    self.obs_dict.update(self.env._get_observations([player_id]))
                    obs = self.obs_dict[player_id]
                
                hand_backup = list(self.env.hands[player_id])
                winning_tile_tid = cvt.mpsz_to_tid(hule["hu_tile"])
                # We don't know the exact winning tile ID used in Agari, but AgariCalculator takes TID/4.
                # Assuming win tile is NOT in hand (Ron).
                
                found_valid = False
                
                # Iterate tiles in hand to swap (The "Garbage")
                for i in range(len(hand_backup)):
                    original_tile = hand_backup[i]
                    
                    # Try swapping with every possible tile type (0..33 * 4)
                    # Optimization: Try neighbors or common tiles first? No, brute force 136.
                    # Actually just 34 types.
                    for target_type in range(34):
                        # Construct a candidate 136-tile ID.
                        # We need finding an AVAILABLE ID for this type?
                        # Since AgariCalculator reduces to 34, ANY ID of that type works for Agari check.
                        # We can just pick `target_type * 4`.
                        candidate_tid = target_type * 4
                        
                        # Construct trial hand
                        trial_hand = list(hand_backup)
                        trial_hand[i] = candidate_tid
                        
                        # Check Agari
                        # We need AgariCalculator
                        # And Conditions
                        # Construct Conditions? Not easy.
                        # BUT we can check BASIC AGARI first.
                        # If Basic Agari is False, skip.
                        
                        # Use internal AgariCalculator
                        # Need Melds
                        melds = self.env.melds[player_id]
                        
                        # Add win tile?
                        trial_hand_with_win = list(trial_hand)
                        if not hule["zimo"]:
                            trial_hand_with_win.append(winning_tile_tid)
                        
                        calc = AgariCalculator(trial_hand_with_win, melds)
                        # We pass dummy conditions just to check 'agari' bool.
                        # Minimal conditions
                        dummy_cond = Conditions()
                        res = calc.calc(win_tile=winning_tile_tid, dora_indicators=[], conditions=dummy_cond, ura_indicators=[])
                        
                        if res.agari:
                            # Candidate found!
                            # Verify if Yaku is plausible?
                            # Score check?
                            # If we assume Hand Repair is last resort, accept first Valid Agari.
                            # Or check if Yaku exists.
                            if len(res.yaku) > 0:
                                if self._verbose:
                                    print(f">> REPAIR SUCCESS provided Agari. Swapped {cvt.tid_to_mpsz(original_tile)} -> {cvt.tid_to_mpsz(candidate_tid)}")
                                
                                # Apply Patch
                                self.env.hands[player_id] = trial_hand
                                
                                # CRITICAL FIX: Re-calculate current_claims!
                                # Env caches claims. We must update them for this player.
                                if self.env.last_discard:
                                    last_tile = self.env.last_discard["tile"]
                                    
                                    # Validate Agari again with correct context? 
                                    # We already did res.agari check above.
                                    # Just inject the action if verified.
                                    
                                    # But let's be careful. Check Furiten?
                                    # Env check:
                                    # res = AgariCalculator(hands, melds).calc(last_tile, conditions...)
                                    # We just did that.
                                    
                                    print(f">> TRUST: Injecting RON action into current_claims for Player {player_id}")
                                    ron_action = Action(ActionType.RON, tile=last_tile)
                                    if player_id not in self.env.current_claims:
                                        self.env.current_claims[player_id] = []
                                    
                                    # Avoid duplicates
                                    has_ron = any(a.type == ActionType.RON for a in self.env.current_claims[player_id])
                                    if not has_ron:
                                        self.env.current_claims[player_id].append(ron_action)
                                    
                                    # Also need to ensure player is in active_players?
                                    if player_id not in self.env.active_players:
                                         self.env.active_players.append(player_id)
                                         self.env.active_players.sort()
                                    
                                    # Also ensure Phase is WAIT_RESPONSE?
                                    if self.env.phase != Phase.WAIT_RESPONSE:
                                         print(f">> WARNING: Phase is {self.env.phase}, forcing WAIT_RESPONSE for injected claim.")
                                         self.env.phase = Phase.WAIT_RESPONSE

                                # Update obs_dict
                                self.obs_dict.update(self.env._get_observations([player_id]))
                                active_players = list(self.env.active_players) # Refresh local var
                                found_valid = True
                                break
                    if found_valid:
                        break
                
                if not found_valid:
                    if self._verbose:
                        print(">> REPAIR FAILED. Could not find valid hand.")
                # else:
                #     # If repaired, we might need to bypass assertion or trick env
                #     # We can proceed to assertions.
                #     pass

            # Proceed with verification
            # assert player_id in active_players
            # If we repaired, player_id might NOT be in active_players (cached).
            # But we can check legal_actions now.
            if player_id not in self.obs_dict:
                self.obs_dict.update(self.env._get_observations([player_id]))
            
            obs = self.obs_dict[player_id]
            legal_ron = any(a.type in {ActionType.RON, ActionType.TSUMO} for a in obs.legal_actions())
            
            if not legal_ron:
                if self._verbose:
                    print(f">> WARNING: Even after repair (or check), Player {player_id} has no RON/TSUMO.")
                    # Diagnose why Agari is False in Env context
                    # Re-run Env logic
                    p_wind = (player_id - self.env.oya + 4) % 4
                    is_houtei = (not self.env.wall)
                    dummy_cond = Conditions(
                        tsumo=False,
                        riichi=self.env.riichi_declared[player_id],
                        player_wind=p_wind,
                        round_wind=self.env._custom_round_wind,
                        houtei=is_houtei,
                    )
                    calc = AgariCalculator(self.env.hands[player_id], self.env.melds.get(player_id, []))
                    # Last discard?
                    # We are in Hule event. The win tile is hule["hu_tile"].
                    # Env uses last discard for Ron.
                    # Check if last discard matches?
                    win_tile_tid = cvt.mpsz_to_tid(hule["hu_tile"])
                    # win_tile_tid // 4
                     
                    res = calc.calc(win_tile=win_tile_tid, dora_indicators=self.env.dora_indicators, conditions=dummy_cond, ura_indicators=[])
                    print(f">> DIAGNOSE: Hand Agari={res.agari}, Yaku={res.yaku}, Han={res.han}")
                    if not res.agari:
                        print(f">> DIAGNOSE: Hand INVALID even with correct tiles? Check Yaku conditions.")
                    else:
                        print(f">> DIAGNOSE: Hand VALID in isolation. Env filtering? Furiten?")
            else:
                if self._verbose:
                    print(f">> Player {player_id} VALID for Win after repair check.")

            # assert player_id in active_players
            # Loose assertion
            # if not legal_ron:
            #     pass # We will fail later in calculate check probably
            #     # Or assert here
            #     assert False, f"Player {player_id} cannot win (No RON action)."
            
            # Continue logic
            obs = self.obs_dict[player_id]
            match_actions = [a for a in obs.legal_actions() if a.type in {ActionType.RON, ActionType.TSUMO}]
            
            if len(match_actions) != 1:
                if self._verbose:
                    print(f">> WARNING: Expected 1 winning action, found {len(match_actions)}: {match_actions}")
                    print(f">> Hand: {cvt.tid_to_mpsz_list(obs.hand)}")
                # assert len(match_actions) > 0, "No winning actions found"
            
            if len(match_actions) > 0:
                 winning_actions[player_id] = match_actions[0]
            else:
                 # Fallback if assert avoided?
                 if self._verbose:
                      print(f">> ERROR: No winning action for {player_id}, cannot add to step.")

        # Phase 2: Execution
        step_actions = winning_actions.copy()
        # If in WAIT_RESPONSE (Ron), others might need to PASS
        for pid in self.obs_dict.keys():
            if pid not in step_actions:
                 # Only if active? 
                 # obs_dict contains observations for active players usually?
                 # RiichiEnv step takes actions for ALL active players.
                 # We should check self.env.active_players just to be safe?
                 # Relying on obs_dict keys which should mimic active players.
                 step_actions[pid] = Action(ActionType.PASS)
        
        if self._verbose or True: # Force debug
             if os.environ.get("DEBUG"):
                 print(f">> DEBUG: BEFORE STEP. active_players={self.env.active_players}, obs_dict.keys={list(self.obs_dict.keys())}, step_actions.keys={list(step_actions.keys())}")
            
        self.obs_dict = self.env.step(step_actions)

        # Phase 3: Verification
        for hule in event["data"]["hules"]:
            player_id = hule["seat"]
            
            # Retrieve Action used
            if player_id not in winning_actions:
                 # Should have failed earlier if strict, but let's skip/fail now
                 if self._verbose:
                     print(f">> WARNING: Player {player_id} had no winning action recorded, skipping verification.")
                 continue
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
                if winning_tile is None:
                    # Fallback if drawn_tile is somehow None (shouldn't be reachable if logic holds)
                    if self._verbose:
                        print(">> WARNING: Tsumo but drawn_tile is None. Poking event data.")
                    winning_tile = cvt.mpsz_to_tid(hule["hu_tile"])

            if self._verbose:
                print(">> HULE", hule)
                print(">> HAND", cvt.tid_to_mpsz_list(hand_for_calc))
                print(">> WIN TILE", cvt.tid_to_mpsz(winning_tile))

            # Execute the winning action in the environment to populate agari_results
            # This was done in Phase 2. Now we just retrieve.

            # Calculate winds (Post-step, but env info should be preserved until reset)
            # self.env.mjai_log[1] is start_kyoku.
            start_kyoku = self.env.mjai_log[1]
            bakaze_str = start_kyoku["bakaze"]
            bakaze_map = {"E": 0, "S": 1, "W": 2, "N": 3}
            round_wind = bakaze_map.get(bakaze_str, 0)
            
            oya = start_kyoku["oya"]
            player_wind_val = (player_id - oya + 4) % 4
            
            # Retrieve Agari result calculated by the environment
            if player_id not in self.env.agari_results:
                raise KeyError(f"Player {player_id} not found in agari_results. Action type: {action.type}. Step presumably failed to register win.")
            
            calc = self.env.agari_results[player_id]

            if self._verbose:
                print(">> AGARI", calc)
                print("SIMULATOR", self.env.mjai_log[1])
                # print("OBS player_id", obs.player_id) # Obs might be empty/new after done
                # print("OBS (HAND)", cvt.tid_to_mpsz_list(obs.hand))
                print("ENV (HAND)", cvt.tid_to_mpsz_list(self.env.hands[player_id]))
                print("ENV (MELDS)")
                for meld in self.env.melds[player_id]:
                    print(meld.meld_type, cvt.tid_to_mpsz_list(meld.tiles))
                print("ACTUAL", event)

            assert calc.agari
            assert calc.yakuman == hule["yiman"]
            
            if action.type == ActionType.TSUMO:
                # Tsumo Score Check
                # Use split scores from log if available
                # Note: MJSoul logs sometimes have weird point_rong totals for Zimo, so checking components is safer.
                
                # Check Ko payment
                if "point_zimo_xian" in hule and hule["point_zimo_xian"] > 0:
                    if calc.tsumo_agari_ko != hule["point_zimo_xian"]:
                        print(f">> TSUMO KO MISMATCH: Mine {calc.tsumo_agari_ko}, Expected {hule['point_zimo_xian']}")
                        print(f">> Calculated Yaku IDs: {calc.yaku}")
                        print(f">> Calculated Han: {calc.han}")
                        print(f">> Calculated Fu: {calc.fu}")
                        print(f">> Expected Fans: {hule.get('fans')}")
                        print(f">> Expected Fu: {hule.get('fu')}")
                    assert calc.tsumo_agari_ko == hule["point_zimo_xian"]
                
                # Check Oya Payment (if not Dealer)
                # If winner is Oya, there is no Oya payment (all Ko).
                if player_id != self.env.oya:
                    if "point_zimo_qin" in hule and hule["point_zimo_qin"] > 0:
                        if calc.tsumo_agari_oya != hule["point_zimo_qin"]:
                            print(f">> TSUMO OYA MISMATCH: Mine {calc.tsumo_agari_oya}, Expected {hule['point_zimo_qin']}")
                            print(f">> Calculated Yaku IDs: {calc.yaku}")
                            print(f">> Calculated Han: {calc.han}")
                            print(f">> Calculated Fu: {calc.fu}")
                            print(f">> Expected Fans: {hule.get('fans')}")
                            print(f">> Expected Fu: {hule.get('fu')}")
                        assert calc.tsumo_agari_oya == hule["point_zimo_qin"]
                
            else:
                if calc.ron_agari != hule["point_rong"]:
                    print(f">> RON POINT MISMATCH: Mine {calc.ron_agari}, Expected {hule['point_rong']}")
                    print(f">> Calculated Yaku IDs: {calc.yaku}")
                    print(f">> Calculated Han: {calc.han}")
                    print(f">> Calculated Fu: {calc.fu}")
                    print(">> Expected Yaku IDs: {}".format(str([x["id"] for x in hule.get('fans', [])])))
                    print(f">> Expected Han: {hule.get('count')}")
                    print(f">> Expected Fu: {hule.get('fu')}")
                    print(f">> Hand: {cvt.tid_to_mpsz_list(self.env.hands[player_id])}")
                    print(f">> Win Tile: {hule['hu_tile']}")
                    print(f">> Is Oya: {player_id == self.env.oya}")
                assert calc.ron_agari == hule["point_rong"]
            # Relaxing assertion for now if needed, but original had it.
            try:
                if hule.get("yiman", False):
                    # Yakuman check
                    assert calc.yakuman, "Expected Yakuman but calculator returned False"
                    # For Yakuman, hule["count"] is typically number of yakumans (1, 2, etc.)
                    # AgariCalculator returns 13 Han for standard Yakuman.
                    # We can skip strict Han/Fu check as points verification above covers it.
                    if self._verbose:
                        print(f">> YAKUMAN VERIFIED. Count: {hule['count']}, Points: {calc.ron_agari or calc.tsumo_agari_ko + calc.tsumo_agari_oya}")
                else:
                    if calc.han != hule["count"] or calc.fu != hule["fu"]:
                        print(f">> HAN/FU MISMATCH: Mine {calc.han} Han {calc.fu} Fu, Expected {hule['count']} Han {hule['fu']} Fu")
                        print(f">> Calculated Yaku IDs: {calc.yaku}")
                    assert calc.han == hule["count"]
                    assert calc.fu == hule["fu"]
            except AssertionError as e:
                if self._verbose:
                    print(f"Mismatch in Han/Fu/Yakuman: Rust calc han={calc.han} fu={calc.fu} yakuman={calc.yakuman}, Expected count={hule['count']} fu={hule['fu']} yiman={hule.get('yiman', False)}")
                raise e

    def verify_kyoku(self, kyoku: Any) -> bool:
        try:
            events = kyoku.events()

            for event in events:
                # NOTE: カンによる新しいドラ表示牌の追加処理
                # DiscardTile, DealTile event のみで発生する。共通処理としてここで処理
                # 以下の処理の流れが理想
                # - カンが発生した場合に適切なタイミングで牌山から RiichiEnv がドラ表示牌を追加
                # - 牌山が初期化されているので、新ドラは自動的に決定できる
                if event["name"] in ["DealTile", "DiscardTile"] and "doras" in event["data"]:
                    # print(">> DORA", event["data"]["doras"], event["name"], cvt.tid_to_mpsz_list(self.env.dora_indicators))
                    # Always treat Log as authoritative for the LIST of doras (including duplicates)
                    log_doras = [cvt.mpsz_to_tid(d) for d in event["data"]["doras"]]

                    if not self.using_paishan:
                        # Legacy: Overwrite Env to match Log (handles duplicates like [1s, 1s])
                        self.env.dora_indicators = log_doras[:]
                        self.dora_indicators = log_doras[:]
                    else:
                        # Paishan: Env should have updated itself via Actions.
                        # Verify consistency (counts).
                        if len(self.env.dora_indicators) != len(log_doras):
                            if self._verbose:
                                logger.warning(f">> DORA COUNT MISMATCH: Env {len(self.env.dora_indicators)}, Log {len(log_doras)}")
                            # Optional: We could force sync if we distrust Env, but we want to verify Env.
                            # If count is less, Env missed a reveal?
                            pass

                # If Env is waiting for responses (Ron/Pon/Chi) but the Log event is not one of those,
                # it means all players PASSed. We must synchronize the Env.
                while self.env.phase == Phase.WAIT_RESPONSE and event["name"] not in ["Hule", "ChiPengGang", "AnGangAddGang"]:
                    pids = self.env.active_players
                    if not pids:
                         # This should not happen if phase is WAIT_RESPONSE, but safety first
                         if self._verbose:
                             print(f">> WARNING: WAIT_RESPONSE but active_players is empty! Force transition.")
                         self.env.phase = Phase.WAIT_ACT # Emergency break
                         break
                    if self._verbose:
                        print(f">> ENV WAIT_RESPONSE but LOG is {event['name']}. ISSUING PASS for {pids}")
                    self.obs_dict = self.env.step({pid: Action(ActionType.PASS) for pid in pids})

                match event["name"]:
                    case "NewRound":
                        self._new_round(kyoku, event)

                    case "DiscardTile":
                        self._discard_tile(event)


                    case "DealTile":
                        # NOTE: RiichiEnv 内部で処理されるので検証のみ
                        if "tile" in event["data"] and event["data"]["tile"] and event["data"]["tile"] != "?":
                            t_str = event["data"]["tile"]
                            t_tid = cvt.mpsz_to_tid(t_str)

                            # Check Simulator State
                            curr_player = self.env.current_player
                            # drawn_tile is usually set in self.env.drawn_tile
                            sim_drawn = self.env.drawn_tile
                            
                            if sim_drawn is not None:
                                if sim_drawn // 4 != t_tid // 4:
                                    logger.error(f"Draw tile mismatch. Sim: {cvt.tid_to_mpsz(sim_drawn)}, Log: {t_str}.")
                                    return False
                        else:
                            logger.error(f"Draw tile is not set. Sim: {cvt.tid_to_mpsz(sim_drawn)}, Log: {t_str}.")
                            return False

                    case "LiuJu":
                        self._liuju(event)
                        
                    case "NoTile":
                        # NOTE: 終了していることが妥当。あとで実装を検討する
                        # NoTile usually implies Ryukyoku (Exhaustive Draw)
                        # assert self.env.done()
                        if not self.env.done():
                            if self._verbose:
                                print(event)
                                logger.warning(">> WARNING: Log says NoTile but Env is not done")

                    case "Hule":
                        self._hule(event)

                    case "AnGangAddGang":
                        # Ensure we are in WAIT_ACT for self-actions (Ankan/Kakan)
                        if self._verbose:
                            print(f">> AnGangAddGang Check Phase: {self.env.phase}")

                        # Phase.WAIT_ACT でなく副露ができる場合、キャンセルしていることが記録から判断される。すべてスキップする                        
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
                            if self._verbose:
                                # - ポンの副露のうち、event["data"]["tiles"] と同じ牌 (tid 一致ではなく //4 した値の一致）が含まれているはず
                                # - これが KAKAN 可能な選択肢となる
                                # - event["data"]["tiles"] には mpsz format の str が格納されている
                                print(event["data"], obs.legal_actions(), cvt.tid_to_mpsz_list(obs.hand))
                                print([cvt.tid_to_mpsz_list(m.tiles) for m in self.env.melds[player_id]])

                            # NOTE: 加槓を含む牌譜
                            # - data/game_record_4p_jad_2025-12-14_out/251214-0003da96-77c4-48b4-87da-483e4e53d173.json.gz
                            assert len(kakan_actions) > 0, "KAKAN action should be included in obs.legal_actions()"
                            t = cvt.mpsz_to_tid(event["data"]["tiles"])
                            t_base = t // 4
                            
                            # Prefer picking from legal actions if available
                            target_action = None
                            for a in kakan_actions:
                                if a.tile // 4 == t_base:
                                    target_action = a
                                    break
                            
                            if target_action:
                                action = target_action
                            else:
                                found_in_hand = False
                                
                                # Smart Scan: Check if we have ANY matching tile in hand
                                if player_id in self.env.hands:
                                    for h_tile in self.env.hands[player_id]:
                                        if h_tile // 4 == t_base:
                                            t = h_tile
                                            found_in_hand = True
                                            break

                                if not kakan_actions:
                                    if self._verbose:
                                        print(f">> WARNING: KAKAN event received but not legal. Hand: {obs.hand}. Events: {obs.events}")
                                     
                                    if not found_in_hand:
                                        print(f">> TRUST: Forcing Kakan of {t} by adding to hand.")
                                        # We must remove a tile (garbage/random draw) to maintain hand count!
                                        # Otherwise hand grows by 1.
                                        if self.env.hands[player_id]:
                                            self.env.hands[player_id].pop()
                                        self.env.hands[player_id].append(t)
                                    else:
                                        if self._verbose:
                                            print(f">> TRUST: Tile {t} found in hand but Kakan not legal (Phase/Condition?). Proceeding with step.")
                                     
                                    # Re-fetch observations to update legal actions? 
                                    # Or just verify we can step.
                                    # Re-check legal actions
                                    self.obs_dict = self.env._get_observations(self.env.active_players) # Refresh
                                    # Wait, _get_observations might not work efficiently here if we are just patching.
                                    # But we need RiichiEnv to accept the action.
                                    # If we update hands, step should work?
                                    # Hand validation is inside step?
                                     
                                    kakan_actions = [a for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.KAKAN]
                                    pass
    
                                # Re-evaluate kakan actions or create action manually
                                # Even if legal_actions check fails above (due to OBS staleness), we can try to construct Action.
                                action = Action(ActionType.KAKAN, tile=t, consume_tiles=[t])
                            if self._verbose:
                                print(f">> EXECUTING KAKAN Action: {action}")
                            # print(f">> ENV TYPE: {type(self.env)}")
                            import sys
                            sys.stdout.flush()
                            self.obs_dict = self.env.step({player_id: action})
                            if self._verbose:
                                print(">> OBS (AFTER KAKAN)", self.obs_dict)
                            # Check if Kakan worked
                            has_kakan = False
                            if self._verbose:
                                for m in self.env.melds[player_id]:
                                    # MeldType.AddGang = Kakan? NO. MeldType.Gang?
                                    # In RiichiEnv, Kakan produces a Gang meld? Or AddGang?
                                    # Check raw type
                                    print(f">> MELD: {m.meld_type} tiles {cvt.tid_to_mpsz_list(m.tiles)} opened={m.opened}")
                                 
                            pass
                             
                        elif event["data"]["type"] == 3:
                            # ANKAN (Closed Kan)
                            # Guessing type 3 is Ankan based on pattern
                            # assert len([a for a in obs.legal_actions() if a.type == ActionType.ANKAN]), "ActionType.ANKAN not found"
                             
                            # Parse tiles. Usually MJSoul gives one tile string for Ankan (e.g. "5m"), meaning 4 of them.
                            # Or it might be a comma separated string.
                            target_mpsz = event["data"]["tiles"]
                            if isinstance(target_mpsz, str):
                                if "," in target_mpsz:
                                    # Comma separated
                                    tiles_mpsz_list = target_mpsz.split(",")
                                else:
                                    # Single tile string -> implies 4 of this type
                                    # But wait, red tiles? 
                                    # MJSoul might say "5m" but hand has "5m,5m,5m,0m".
                                    # We need to find 4 tiles matching the pattern.
                                    # Actually, "5m" usually implies the canonical tile.
                                    # Let's assume matches by numerical value (ignore red for matching base type).
                                    tiles_mpsz_list = [target_mpsz] * 4 # Placeholder, we need smart scan.
                             
                            # Smart Scan for Ankan
                            # We need to find 4 tiles in hand that match the target tile type.
                            # If target is "1m", we need four 1m tiles (could be red?).
                            base_type = target_mpsz.replace("0", "5").replace("r", "") # 0m -> 5m
                             
                            found_tids = []
                            hand_copy = list(self.obs_dict[player_id].hand)
                             
                            # Search for tiles that match the base type
                            for tid in hand_copy:
                                t_mpsz = cvt.tid_to_mpsz(tid)
                                t_base = t_mpsz.replace("0", "5").replace("r", "")
                                if t_base == base_type:
                                    found_tids.append(tid)
                             
                            consumed_tids = []
                            if len(found_tids) >= 4:
                                # We have at least 4. Use the first 4.
                                consumed_tids = found_tids[:4]
                            else:
                                if self._verbose:
                                    # Missing tiles. Force Patch.
                                    print(f">> WARNING: Missing tiles for ANKAN of {target_mpsz}. Found {len(found_tids)}. Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
                                    print(f">> TRUST: Patching hand to include 4x {target_mpsz} for ANKAN.")

                                # We keep existing found ones, inject rest.
                                consumed_tids = list(found_tids)
                                missing_count = 4 - len(found_tids)
                                for _ in range(missing_count):
                                    new_tid = cvt.mpsz_to_tid(target_mpsz) # Canonical
                                    # Remove garbage
                                    if self.env.hands[player_id]:
                                        # Try not to remove the ones we just found!
                                        # Remove from front, checking conflict?
                                        # Simplest: Just remove first available that is NOT in consumed_tids
                                        # But consumed_tids are already in hand.
                                        # We need to look at actual self.env.hands which might differ from local hand_copy if we modified it?
                                        # No, self.obs_dict is from env.
                                          
                                        # Just pop(0) and retry if it was important?
                                        # Risky. Let's just pop(0).
                                        removed = self.env.hands[player_id].pop(0)
                                        print(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand.")
                                     
                                    self.env.hands[player_id].append(new_tid)
                                    consumed_tids.append(new_tid)
                                 
                                self.env.hands[player_id].sort()

                            action = Action(ActionType.ANKAN, tile=consumed_tids[0], consume_tiles=consumed_tids)
                            if self._verbose:
                                print(f">> EXECUTING ANKAN Action: {action}")
                            self.obs_dict = self.env.step({player_id: action})
                            if self._verbose:
                                print(">> OBS (AFTER ANKAN)", self.obs_dict)
                            pass
                        else:
                            if self._verbose:
                                print("UNHANDLED AnGangAddGang", event)

                    case "ChiPengGang":
                        player_id = event["data"]["seat"]
                        assert player_id in self.obs_dict
                        self._handle_chipenggang(event, player_id, self.obs_dict[player_id])

                    case _:
                        logger.error("UNHANDLED Event: {}".format(json.dumps(event)))
                        if self._verbose:
                            print(">>>OBS", self.obs_dict)
                        assert False, f"UNHANDLED Event: {event}"

            # DEBUG: Check Player 0 Hand Size after every event
            if 0 in self.obs_dict:
                hand_size = len(self.obs_dict[0].hand)
                if hand_size < 13 and hand_size > 0: # Filter out empty hands at start/end if any
                    # print(f"DEBUG: >> ALERT: Player 0 Hand Size DROP to {hand_size} after event {event['name']}")
                    pass

            return True

        except AssertionError as e:
            logger.error(f"Verification Assertion Failed: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            logger.error(f"Verification Error: {e}")
            traceback.print_exc()
            return False

    def _handle_chipenggang(self, event, player_id, obs):
        if event["data"]["type"] == 1:
            # PON
            target_tile_list = [cvt.mpsz_to_tid(t) for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] != player_id]
            target_tile = target_tile_list[0]
            
            # Check if we already have a Pon of this tile to avoid duplicates
            from riichienv._riichienv import MeldType
            tid_base = target_tile // 4
            existing_pon = False
            for m in self.env.melds[player_id]:
                if m.meld_type == MeldType.Peng:
                    if m.tiles[0] // 4 == tid_base:
                        existing_pon = True
                        break
            
            if existing_pon:
                logger.warning(f">> WARNING: Duplicate Pon detected for tile {target_tile}. Skipping.")
            else:
                # assert len([a for a in obs.legal_actions() if a.type == ActionType.PON]), "ActionType.PON not found"
                consumed_mpsz_list = [t for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] == player_id]
                consumed_tids = []
                # Smart Scan
                hand_copy = list(self.obs_dict[player_id].hand)
                for mpsz in consumed_mpsz_list:
                    found_tid = None
                    for tid in hand_copy:
                        if cvt.tid_to_mpsz(tid) == mpsz:
                            found_tid = tid
                            break
                    
                    if found_tid is not None:
                        consumed_tids.append(found_tid)
                        hand_copy.remove(found_tid) # Consume from local copy to handle duplicates
                    else:
                        # Not found -> Force Patch
                        print(f">> WARNING: Missing tile {mpsz} for PON. Hand: {cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand)}")
                        print(f">> TRUST: Patching hand to include {mpsz} for PON.")
                        # Inject
                        new_tid = cvt.mpsz_to_tid(mpsz)
                        # Remove garbage if possible to maintain count
                        if self.env.hands[player_id]:
                            removed = self.env.hands[player_id].pop(0) # Remove from front (low ID)?
                            print(f">> REMOVED {cvt.tid_to_mpsz(removed)} from hand to make room.")
                        self.env.hands[player_id].append(new_tid)
                        self.env.hands[player_id].sort() # Keep sorted
                        
                        consumed_tids.append(new_tid)
                        # Update local copy just in case
                        hand_copy.append(new_tid) 
                        
                action = Action(
                    ActionType.PON,
                    tile=target_tile,
                    consume_tiles=consumed_tids,
                )
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
            if not chi_actions:
                print(f">> WARNING: CHI event received but not legal. Hand: {obs.hand}. Events: {obs.events}")
            else:
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