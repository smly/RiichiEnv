import sys
import traceback
import argparse
from typing import Any
from pathlib import Path

import riichienv.convert as cvt
from riichienv.action import ActionType, Action
from riichienv.env import Phase
from riichienv import ReplayGame, RiichiEnv, AgariCalculator, Conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the game record JSON file.")
    parser.add_argument("--skip", type=int, default=0, help="Number of kyokus to skip.")
    return parser.parse_args()


class MjsoulEnvVerifier:
    def __init__(self):
        self.env: RiichiEnv = RiichiEnv()
        self.obs_dict: dict[int, Any] | None = None
        self.dora_indicators: list[int] = []

    def verify_game(self, game: Any, skip: int = 0) -> bool:
        # We start from the 5th kyoku as in the original script? 
        # Original: for kyoku in list(game.take_kyokus())[4:]:
        kyokus = list(game.take_kyokus())
        for i, kyoku in enumerate(kyokus[skip:]):
            print(f"Verifying kyoku {skip + i}...")
            if not self.verify_kyoku(kyoku):
                print(f"Verification failed at kyoku {skip + i}.")
                return False
        print("All kyokus verified successfully.")
        return True

    def _new_round(self, kyoku: Any, event: Any) -> None:
        events = kyoku.events()
        env_wall = []
        tid_count = {}
        for event_ in events:
            if event_["name"] == "DealTile":
                tid = cvt.mpsz_to_tid(event_["data"]["tile"])
                cnt = 0
                if tid in tid_count:
                    cnt = tid_count[tid]
                    tid_count[tid] += 1
                else:
                    tid_count[tid] = 1
                tid = tid + cnt
                env_wall.append(tid)
        env_wall = list(reversed(env_wall))

        data = event["data"]
        self.dora_indicators = [cvt.mpsz_to_tid(t) for t in data["doras"]]
        self.env = RiichiEnv()
        self.env.reset()
        self.env.mjai_log = [
            {
                "type": "start_game",
                "names": ["Player0", "Player1", "Player2", "Player3"],
            },
            {
                "type": "start_kyoku",
                "bakaze": "E",
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
        for player_id in range(4):
            self.env.hands[player_id] = cvt.mpsz_to_tid_list(data[f"tiles{player_id}"][:13])
        
        first_actor = data["ju"]
        raw_first_tile = data["tiles{}".format(first_actor)][13]
        first_tile = cvt.mpsz_to_mjai(raw_first_tile)
        self.env.mjai_log.append({
            "type": "tsumo",
            "actor": first_actor,
            "tile": first_tile,
        })
        self.env.drawn_tile = cvt.mpsz_to_tid(raw_first_tile)
        self.env.current_player = first_actor
        self.env.active_players = [first_actor]
        self.env.wall = env_wall
        self.obs_dict = self.env._get_observations([first_actor])

    def _discard_tile(self, event: Any) -> None:
        # print(">> OBS", self.obs_dict)
        # print("--")
        print(">> EVENT", event)
        while self.env.phase != Phase.WAIT_ACT:
            # Skip action
            self.obs_dict = self.env.step({skip_player_id: Action(ActionType.PASS) for skip_player_id in self.obs_dict.keys()})

        # print(">> OBS (AFTER SKIP WAIT_ACT PHASE)", self.obs_dict)

        player_id = event["data"]["seat"]
        candidate_tiles = set([cvt.tid_to_mpsz(a.tile) for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.DISCARD])
        assert player_id == event["data"]["seat"]
        assert event["data"]["tile"] in candidate_tiles
        if event["data"]["is_liqi"]:
            # Riichi declaration
            print(cvt.tid_to_mpsz_list(self.obs_dict[player_id].hand))
            matched_actions = [a for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.RIICHI]
            assert len(matched_actions) == 1, "ActionType.RIICHI not found"
            action = matched_actions[0]
            self.obs_dict = self.env.step({player_id: action})

        # Normal discard
        action = [a for a in self.obs_dict[player_id].legal_actions() if a.type == ActionType.DISCARD and cvt.tid_to_mpsz(a.tile) == event["data"]["tile"]][0]
        self.obs_dict = self.env.step({player_id: action})

    def _liuju(self, event: Any) -> None:
        print(">> LIUJU", event)
        # Often happens on current_player's turn if Kyuhsu Kyuhai
        self.obs_dict = self.env._get_observations(self.env.active_players)
        for pid, obs in self.obs_dict.items():
                print(f">> legal_actions() {pid} {obs.legal_actions()}")
                
                # Check for KYUSHU_KYUHAI
                kyushu_actions = [a for a in obs.legal_actions() if a.type == ActionType.KYUSHU_KYUHAI]
                if kyushu_actions:
                    print(f">> Player {pid} has KYUSHU_KYUHAI")
                    # Execute it
                    self.obs_dict = self.env.step({pid: kyushu_actions[0]})
                    print(f">> Executed KYUSHU_KYUHAI. Done: {self.env.done()}")
                    break

    def _hule(self, event: Any) -> None:
        active_players = self.obs_dict.keys()
        assert self.env.phase == Phase.WAIT_RESPONSE

        for hule in event["data"]["hules"]:
            player_id = hule["seat"]
            assert player_id in active_players
            assert self.obs_dict[player_id]
            obs = self.obs_dict[player_id]
            match_actions = [a for a in obs.legal_actions() if a.type in {ActionType.RON, ActionType.TSUMO}]
            assert len(match_actions) == 1
            action = match_actions[0]

            # Ura Doras
            ura_indicators = []
            if "li_doras" in hule:
                ura_indicators = [cvt.mpsz_to_tid(t) for t in hule["li_doras"]]

            print(">> HULE", hule)
            print(">>", cvt.tid_to_mpsz_list(obs.hand))
            print(">>", cvt.tid_to_mpsz(action.tile))

            # Calculate winds
            # self.env.mjai_log[1] is start_kyoku.
            # We can extract bakaze/oya from there if needed, or from NewRound data.
            # data["doras"] ...
            # But self.env.mjai_log[1] has "bakaze": "E", "oya": 0
            start_kyoku = self.env.mjai_log[1]
            
            # bakaze: E=0, S=1, W=2, N=3
            bakaze_str = start_kyoku["bakaze"]
            bakaze_map = {"E": 0, "S": 1, "W": 2, "N": 3}
            round_wind = bakaze_map.get(bakaze_str, 0)
            
            oya = start_kyoku["oya"]
            # player_wind: (seat - oya + 4) % 4
            player_wind_val = (player_id - oya + 4) % 4
            
            calc = AgariCalculator(obs.hand).calc(
                action.tile, 
                dora_indicators=self.dora_indicators,
                ura_indicators=ura_indicators,
                conditions=Conditions(
                    tsumo=False,
                    riichi=self.env.riichi_declared[player_id],
                    double_riichi=False,
                    ippatsu=False,
                    haitei=False,
                    houtei=False,
                    rinshan=False,
                    chankan=False,
                    tsumo_first_turn=False,
                    player_wind=player_wind_val,
                    round_wind=round_wind,
            ))

            print(">> AGARI", calc)
            print("SIMULATOR", self.env.mjai_log[1])
            print("OBS player_id", obs.player_id)
            print("OBS (HAND)", cvt.tid_to_mpsz_list(obs.hand))
            print("ENV (HAND)", cvt.tid_to_mpsz_list(self.env.hands[player_id]))
            print("ENV (MELDS)")
            for meld in self.env.melds[player_id]:
                print(meld.meld_type, cvt.tid_to_mpsz_list(meld.tiles))
            print("ACTUAL", event)

            assert calc.agari
            assert calc.yakuman == hule["yiman"]
            assert calc.ron_agari == hule["point_rong"]
            
            # Relaxing assertion for now if needed, but original had it.
            try:
                assert calc.han == hule["count"]
                assert calc.fu == hule["fu"]
            except AssertionError as e:
                print(f"Mismatch in Han/Fu: Rust calc han={calc.han} fu={calc.fu}, Expected han={hule['count']} fu={hule['fu']}")
                raise e

    def verify_kyoku(self, kyoku: Any) -> bool:
        try:
            events = kyoku.events()

            for event in events:
                match event["name"]:
                    case "NewRound":
                        self._new_round(kyoku, event)

                    case "DiscardTile":
                        self._discard_tile(event)

                    case "DealTile":
                        # TODO: verify deal tile event with RiichiEnv internal state
                        pass

                    case "LiuJu":
                        self._liuju(event)
                        
                    case "NoTile":
                        player_id = event["data"]["seat"]
                        print(event)

                    case "Hule":
                        self._hule(event)

                    case "ChiPengGang":
                        print(">> OBS", self.obs_dict)
                        print("--")
                        print(">> EVENT", event)
                        player_id = event["data"]["seat"]
                        assert player_id in self.obs_dict
                        obs = self.obs_dict[player_id]
                        if event["data"]["type"] == 1:
                            # PON
                            assert len([a for a in obs.legal_actions() if a.type == ActionType.PON])
                            action = Action(
                                ActionType.PON,
                                tile=[cvt.mpsz_to_tid(t) for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] != player_id][0],
                                consume_tiles=[cvt.mpsz_to_tid(t) for i, t in enumerate(event["data"]["tiles"]) if event["data"]["froms"][i] == player_id],
                            )
                            self.obs_dict = self.env.step({player_id: action})
                            print(">> OBS (AFTER PON)", self.obs_dict)
                        else:
                            print("BREAK", event)
                            print(">>>OBS", self.obs_dict)
                            # Original had break? No, just print.
                            # Actually looking at line 227-230 in original:
                            # case _: break
                            # case ChiPengGang else: print BREAK...
                            # It doesn't break in the else block of ChiPengGang, but maybe it should?
                            pass

                    case _:
                        print("BREAK", event)
                        print(">>>OBS", self.obs_dict)
                        # break # Original had break here
            return True
        except AssertionError as e:
            print(f"Verification Assertion Failed: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Verification Error: {e}")
            traceback.print_exc()
            return False


def main(path: str, skip: int = 0):
    game = ReplayGame.from_json(path)
    print(f"Verifying {path}...")
    verifier = MjsoulEnvVerifier()
    if not verifier.verify_game(game, skip=skip):
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args.path, args.skip)
