"""
Validate 4P RiichiEnv implementation by tracing MjSoul replays using env.step().

Loads MjSoul 4P replay data, reconstructs the wall from paishan, and
traces each kyoku through RiichiEnv("4p-red-half") using env.step(acts).
Verifies that the env accepts all replay actions and end scores match.
"""

import lzma
import glob
import sys
from typing import Optional, Tuple, List, Dict, Any

import tqdm

from riichienv import (
    RiichiEnv,
    Action,
    ActionType,
    Phase,
    GameRule,
    MjSoulReplay,
)
from mjsoul_parser import MjsoulPaifuParser, Paifu

TARGET_FILE_PATTERN = "/data/mjsoul/mahjong_game_record_4p_*/*.bin.xz"
NP = 4


# ---------------------------------------------------------------------------
# Tile parsing utilities
# ---------------------------------------------------------------------------

def parse_tile(tile_str: str) -> Tuple[int, bool]:
    """Parse tile string (e.g. '5m', '0p') to (tile_34, is_red)."""
    num = int(tile_str[0])
    suit = tile_str[1]
    is_red = num == 0
    if is_red:
        num = 5
    suit_offset = {"m": 0, "p": 9, "s": 18, "z": 27}[suit]
    tile_34 = suit_offset + num - 1
    return tile_34, is_red


def parse_paishan(paishan_str: str) -> List[int]:
    """Parse MjSoul paishan string into wall of 136-format tile IDs.

    Assigns unique sub-indices to each copy of the same tile type.
    Red fives (0m/0p/0s) always get sub-index 0.
    """
    used: Dict[int, set] = {}
    wall = []

    for i in range(0, len(paishan_str), 2):
        tile_str = paishan_str[i : i + 2]
        tile_34, is_red = parse_tile(tile_str)

        if tile_34 not in used:
            used[tile_34] = set()

        if is_red:
            sub_idx = 0  # Red five is always sub-index 0
        else:
            is_five = tile_34 in (4, 13, 22)
            if is_five:
                # For normal 5s, try 1,2,3 first (0 is red)
                candidates = [1, 2, 3, 0]
            else:
                candidates = [0, 1, 2, 3]
            sub_idx = next(si for si in candidates if si not in used[tile_34])

        used[tile_34].add(sub_idx)
        wall.append(tile_34 * 4 + sub_idx)

    return wall




def find_tile_in_hand(hand: List[int], tile_str: str) -> Optional[int]:
    """Find a tile in hand matching the tile string. Returns 136-format ID."""
    tile_34, is_red = parse_tile(tile_str)

    # Try exact match (correct red status)
    for t in hand:
        if t // 4 == tile_34:
            t_is_red = t in (16, 52, 88)
            if is_red == t_is_red:
                return t

    # Fallback: any matching tile_34
    for t in hand:
        if t // 4 == tile_34:
            return t

    return None


def find_legal_action(
    legal_actions: list,
    action_type: ActionType,
    tile_str: Optional[str] = None,
) -> Optional[Action]:
    """Find a legal action matching the given type and optionally tile."""
    if tile_str:
        tile_34, is_red = parse_tile(tile_str)
    else:
        tile_34, is_red = None, None

    # Exact match (type + tile_34 + red status)
    for la in legal_actions:
        if la.action_type != action_type:
            continue
        if tile_34 is not None and la.tile is not None:
            la_34 = la.tile // 4
            la_red = la.tile in (16, 52, 88)
            if la_34 == tile_34 and la_red == is_red:
                return la
        elif tile_34 is None:
            return la

    # Fallback: match tile_34 only (ignore red status)
    for la in legal_actions:
        if la.action_type != action_type:
            continue
        if tile_34 is not None and la.tile is not None:
            if la.tile // 4 == tile_34:
                return la
        elif tile_34 is None:
            return la

    return None


# ---------------------------------------------------------------------------
# Event processing helpers
# ---------------------------------------------------------------------------

def submit_all_pass(env: RiichiEnv) -> dict:
    """Submit Pass for all active players in WaitResponse."""
    acts = {}
    for p in env.active_players:
        acts[p] = Action(type=ActionType.Pass)
    return env.step(acts)


def process_discard(env: RiichiEnv, data: dict) -> dict:
    """Process a DiscardTile event."""
    seat = data["seat"]
    tile_str = data["tile"]
    is_liqi = data.get("is_liqi", False)
    is_wliqi = data.get("is_wliqi", False)

    legals = env._get_legal_actions(seat)

    if is_liqi or is_wliqi:
        # Riichi: legal actions have tile=None (just indicating riichi is available).
        # Two-step process: 1) declare riichi, 2) discard the tile.
        has_riichi = any(la.action_type == ActionType.Riichi for la in legals)
        if has_riichi:
            # Step 1: Declare riichi (tile=None matches the legal action)
            env.step({seat: Action(type=ActionType.Riichi)})
            # Step 2: Discard the specific tile
            legals2 = env._get_legal_actions(seat)
            matched = find_legal_action(legals2, ActionType.Discard, tile_str)
            if matched is not None:
                return env.step({seat: matched})

        # TODO: env の内部状態でリーチ宣言開始のタイミングであることを管理するフラグを確認
        pass

        # Fallback: try as Discard (shouldn't normally happen)
        matched = find_legal_action(legals, ActionType.Discard, tile_str)
    else:
        matched = find_legal_action(legals, ActionType.Discard, tile_str)

    if matched is None:
        hand = env.hands[seat]
        action_type = ActionType.Riichi if (is_liqi or is_wliqi) else ActionType.Discard
        raise RuntimeError(
            f"Tile {tile_str} (type={action_type}) not found in legal actions "
            f"for player {seat}: "
            f"phase={env.phase}, active={env.active_players}, done={env.is_done}, "
            f"legals={[(a.action_type, a.tile) for a in legals]}, "
            f"hand={sorted(hand)}"
        )

    return env.step({seat: matched})


def find_best_chi_action(legals: list, data: dict) -> Optional[Action]:
    """Find the best matching Chi action using froms to identify consume tiles."""
    seat = data["seat"]
    tiles_str = data["tiles"]
    froms = data["froms"]

    # Extract consume tile info from replay (tiles from own hand)
    consume_parsed = []
    for ts, f in zip(tiles_str, froms):
        if f == seat:
            t34, is_red = parse_tile(ts)
            consume_parsed.append((t34, is_red))
    consume_parsed.sort()

    chi_actions = [la for la in legals if la.action_type == ActionType.Chi]

    # Exact match: match consume tiles by type and red status
    for la in chi_actions:
        la_parsed = sorted([(t // 4, t in (16, 52, 88)) for t in la.consume_tiles])
        if la_parsed == consume_parsed:
            return la

    # Fallback: match by tile_34 only (ignore red)
    consume_34 = sorted([t34 for t34, _ in consume_parsed])
    for la in chi_actions:
        la_34 = sorted([t // 4 for t in la.consume_tiles])
        if la_34 == consume_34:
            return la

    # Last resort: any chi action
    return chi_actions[0] if chi_actions else None


def find_best_pon_action(legals: list, data: dict) -> Optional[Action]:
    """Find the best matching Pon action using froms to identify consume tiles."""
    seat = data["seat"]
    tiles_str = data["tiles"]
    froms = data["froms"]

    # Extract consume tile info from replay (tiles from own hand)
    consume_parsed = []
    for ts, f in zip(tiles_str, froms):
        if f == seat:
            t34, is_red = parse_tile(ts)
            consume_parsed.append((t34, is_red))
    consume_parsed.sort()

    pon_actions = [la for la in legals if la.action_type == ActionType.Pon]

    # Exact match: match consume tiles by type and red status
    for la in pon_actions:
        la_parsed = sorted([(t // 4, t in (16, 52, 88)) for t in la.consume_tiles])
        if la_parsed == consume_parsed:
            return la

    # Fallback: any pon action
    return pon_actions[0] if pon_actions else None


def process_chi_peng_gang(env: RiichiEnv, data: dict) -> dict:
    """Process a ChiPengGang event (Chi, Pon, or Daiminkan from discard)."""
    seat = data["seat"]
    meld_type = data["type"]  # 0=Chi, 1=Pon, 2=Daiminkan
    tiles_str = data["tiles"]  # List of tile strings in the meld

    if meld_type not in (0, 1, 2):
        raise RuntimeError(f"Unexpected ChiPengGang type {meld_type}")

    # Find the matching legal action for the claiming player
    legals = env._get_legal_actions(seat)
    matched = None

    if meld_type == 0:
        matched = find_best_chi_action(legals, data)
    elif meld_type == 1:
        matched = find_best_pon_action(legals, data)
    elif meld_type == 2:
        called_tile_str = tiles_str[0]
        matched = find_legal_action(legals, ActionType.Daiminkan, called_tile_str)

    if matched is None:
        raise RuntimeError(
            f"No matching legal action for ChiPengGang "
            f"seat={seat} type={meld_type} tiles={tiles_str}, "
            f"legals={[(a.action_type, a.tile, a.consume_tiles) for a in legals]}"
        )

    # Build actions: claim for claimer, Pass for others
    acts = {}
    for p in env.active_players:
        if p == seat:
            acts[p] = matched
        else:
            acts[p] = Action(type=ActionType.Pass)

    return env.step(acts)


def process_angang_addgang(env: RiichiEnv, data: dict) -> dict:
    """Process an AnGangAddGang event (Ankan or Kakan from own hand)."""
    seat = data["seat"]
    gang_type = data["type"]  # 3=Ankan, 2=Kakan
    tile_str = data["tiles"]  # Single tile string

    if gang_type == 3:
        action_type = ActionType.Ankan
    else:
        action_type = ActionType.Kakan

    legals = env._get_legal_actions(seat)
    matched = find_legal_action(legals, action_type, tile_str)

    if matched is None:
        raise RuntimeError(
            f"No matching legal action for AnGangAddGang "
            f"seat={seat} type={gang_type} tile={tile_str}, "
            f"legals={[(a.action_type, a.tile) for a in legals]}"
        )

    return env.step({seat: matched})


def process_tsumo(env: RiichiEnv, data: dict) -> dict:
    """Process a Hule event with zimo=True (Tsumo win)."""
    hule = data["hules"][0]
    seat = hule["seat"]

    legals = env._get_legal_actions(seat)
    matched = find_legal_action(legals, ActionType.Tsumo)

    if matched is None:
        raise RuntimeError(
            f"No matching Tsumo legal action for seat={seat}, "
            f"legals={[(a.action_type, a.tile) for a in legals]}"
        )

    return env.step({seat: matched})


def process_ron(env: RiichiEnv, data: dict) -> dict:
    """Process a Hule event with zimo=False (Ron win, possibly double/triple)."""
    hules = data["hules"]
    ron_seats = {h["seat"] for h in hules}

    acts = {}
    for p in env.active_players:
        if p in ron_seats:
            legals = env._get_legal_actions(p)
            matched = find_legal_action(legals, ActionType.Ron)
            if matched is None:
                raise RuntimeError(
                    f"No matching Ron legal action for seat={p}, "
                    f"legals={[(a.action_type, a.tile) for a in legals]}"
                )
            acts[p] = matched
        else:
            acts[p] = Action(type=ActionType.Pass)

    return env.step(acts)


def process_liuju(env: RiichiEnv, data: dict) -> dict:
    """Process a LiuJu (abortive draw) event."""
    seat = data.get("seat", 0)

    # KyushuKyuhai: player has 9+ terminal/honor tiles on first draw
    legals = env._get_legal_actions(seat)
    matched = find_legal_action(legals, ActionType.KyushuKyuhai)

    if matched is not None:
        return env.step({seat: matched})

    # Other abortive draws should be handled internally by the env
    return {}


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_kyoku(
    env: RiichiEnv,
    kyoku,
    game_uuid: str,
    kyoku_idx: int,
    is_last_kyoku: bool = False,
) -> Tuple[bool, str]:
    """Validate a single kyoku by tracing it through env.step().

    Returns (success, message).
    """
    events = kyoku.events()
    if not events:
        return False, "No events"

    new_round = events[0]
    if new_round["name"] != "NewRound":
        return False, f"First event is not NewRound: {new_round['name']}"

    nr = new_round["data"]

    # Check for paishan
    paishan = nr.get("paishan")
    assert paishan is not None, "Paishan is required for validation"

    # Parse wall
    try:
        wall = parse_paishan(paishan)
    except Exception as e:
        return False, f"Failed to parse paishan: {e}"

    if len(wall) != 136:
        return False, f"Wall has {len(wall)} tiles, expected 136"

    # Setup round
    scores = nr["scores"]
    round_wind = nr["chang"]
    honba = nr["ben"]
    kyotaku = nr["liqibang"]

    # Determine oya from hands (player with 14 tiles after draw)
    # In MjSoul, ju indicates the round number within the wind
    oya = nr["ju"] % 4

    env.reset(
        oya=oya,
        wall=wall,
        round_wind=round_wind,
        scores=scores,
        honba=honba,
        kyotaku=kyotaku,
    )

    # Verify initial hands match
    env_hands = env.hands
    for i in range(4):
        key = f"tiles{i}"
        if key not in nr:
            continue
        replay_hand_strs = nr[key]
        replay_hand_34 = sorted([parse_tile(t)[0] for t in replay_hand_strs])
        env_hand_34 = sorted([t // 4 for t in env_hands[i]])

        # Oya has 14 tiles (including drawn), replay shows 13
        if i == oya:
            if len(env_hand_34) != 14:
                return False, (
                    f"Oya (player {i}) has {len(env_hand_34)} tiles, expected 14"
                )
            if len(replay_hand_34) != 13:
                continue  # Can't verify exactly
        else:
            if env_hand_34 != replay_hand_34:
                return False, (
                    f"Hand mismatch for player {i}: "
                    f"env={env_hand_34} replay={replay_hand_34}"
                )

    # Process events
    last_action_event = "none"
    last_hule_data = None
    last_discard_was_riichi = False
    for ev_idx, event in enumerate(events[1:], start=1):
        name = event["name"]
        data = event["data"]

        # Skip passive events
        if name in ("DealTile", "Dora"):
            # Before skipping, resolve pending WaitResponse (nobody claimed)
            if env.phase == Phase.WaitResponse:
                submit_all_pass(env)
            continue

        if name == "NoTile":
            # Exhaustive draw (荒牌平局)
            if env.phase == Phase.WaitResponse:
                submit_all_pass(env)
            # The env should handle exhaustive draw internally
            # (triggered by _deal_next when wall is empty)
            continue

        # If env is done, the remaining events are post-game
        if env.is_done:
            continue

        try:
            if name == "DiscardTile":
                # Resolve pending WaitResponse if needed
                if env.phase == Phase.WaitResponse:
                    submit_all_pass(env)
                if env.is_done:
                    continue
                process_discard(env, data)
                last_discard_was_riichi = data.get("is_liqi", False) or data.get("is_wliqi", False)
                last_action_event = "discard"

            elif name == "ChiPengGang":
                # Should be in WaitResponse
                if env.phase != Phase.WaitResponse:
                    return False, (
                        f"Event {ev_idx}: ChiPengGang but env not in WaitResponse "
                        f"(phase={env.phase})"
                    )
                process_chi_peng_gang(env, data)
                last_action_event = f"chi_peng_gang_{data['type']}"

            elif name == "AnGangAddGang":
                # Resolve pending WaitResponse if needed
                if env.phase == Phase.WaitResponse:
                    submit_all_pass(env)
                if env.is_done:
                    continue
                process_angang_addgang(env, data)
                last_action_event = f"angang_{data['type']}"

            elif name == "Hule":
                hules = data["hules"]
                is_tsumo = hules[0].get("zimo", False)

                if is_tsumo:
                    # Tsumo in WaitAct
                    if env.phase == Phase.WaitResponse:
                        submit_all_pass(env)
                    if env.is_done:
                        continue
                    process_tsumo(env, data)
                    last_action_event = "tsumo"
                    last_hule_data = data
                else:
                    # Ron in WaitResponse
                    if env.phase != Phase.WaitResponse:
                        return False, (
                            f"Event {ev_idx}: Hule(ron) but env not in WaitResponse "
                            f"(phase={env.phase})"
                        )
                    pre_ron_sticks = env.riichi_sticks
                    pre_ron_scores = env.scores()
                    process_ron(env, data)
                    last_action_event = f"ron_{len(data['hules'])}:sticks={pre_ron_sticks}:pre={pre_ron_scores}:riichi_discard={last_discard_was_riichi}"
                    last_hule_data = data

            elif name == "LiuJu":
                if env.phase == Phase.WaitResponse:
                    submit_all_pass(env)
                if env.is_done:
                    continue
                process_liuju(env, data)
                last_action_event = f"liuju_{data.get('type', '?')}"

        except RuntimeError as e:
            return False, (
                f"Event {ev_idx} ({name}): {e}"
            )

    # Verify end scores
    end_scores = kyoku.end_scores
    env_scores = env.scores()

    if env_scores != end_scores:
        deltas = [env_scores[i] - end_scores[i] for i in range(NP)]
        actual_deltas = [env_scores[i] - scores[i] for i in range(NP)]
        expected_deltas = [end_scores[i] - scores[i] for i in range(NP)]
        hule_info = ""
        if last_hule_data:
            win_results = env.win_results
            for h in last_hule_data["hules"]:
                s = h["seat"]
                replay_pts = f"han={h['count']} fu={h['fu']} ron={h['point_rong']} tsumo_oya={h['point_zimo_qin']} tsumo_ko={h['point_zimo_xian']}"
                wr = win_results.get(s)
                if wr:
                    env_pts = f"han={wr.han} fu={wr.fu} ron={wr.ron_agari} tsumo_oya={wr.tsumo_agari_oya} tsumo_ko={wr.tsumo_agari_ko} yaku={wr.yaku}"
                else:
                    env_pts = "no_win_result"
                hule_info += f" | seat{s} replay[{replay_pts}] env[{env_pts}]"
        return False, (
            f"Score mismatch: init={scores} env={env_scores} exp={end_scores} "
            f"actual_d={actual_deltas} exp_d={expected_deltas} "
            f"last={last_action_event} honba={honba} kyotaku={kyotaku} "
            f"riichi_sticks={env.riichi_sticks} is_last={is_last_kyoku}"
            f"{hule_info}"
        )

    return True, "ok"


def main():
    target_files = sorted(glob.glob(TARGET_FILE_PATTERN, recursive=True))
    if not target_files:
        print(f"No files found matching {TARGET_FILE_PATTERN}")
        sys.exit(1)

    target_files = list(target_files)[:5000]

    total_kyoku = 0
    total_success = 0
    total_fail = 0
    total_skip = 0
    fail_details = []
    fail_categories: Dict[str, int] = {}

    env = RiichiEnv("4p-red-half", rule=GameRule.default_mjsoul())

    for path in tqdm.tqdm(target_files, desc="Processing files", ncols=100):
        with lzma.open(path, "rb") as f:
            data = f.read()
            paifu: Paifu = MjsoulPaifuParser.to_dict(data)

        game_uuid = paifu.header.get("uuid", "unknown")
        game = MjSoulReplay.from_dict(paifu.data)

        num_rounds = game.num_rounds()
        for k, kyoku in enumerate(game.take_kyokus()):
            total_kyoku += 1
            is_last_kyoku = (k == num_rounds - 1)

            success, msg = validate_kyoku(env, kyoku, game_uuid, k, is_last_kyoku=is_last_kyoku)

            if msg.startswith("skip:"):
                total_skip += 1
            elif success:
                total_success += 1
            else:
                total_fail += 1
                # Categorize
                if "legals=[]" in msg or "legals=[], " in msg:
                    cat = "discard_empty_legals"
                elif "not found in legal actions" in msg:
                    cat = "discard_tile_mismatch"
                elif "Score mismatch" in msg:
                    # Sub-categorize by last event
                    if "last_event=tsumo" in msg:
                        cat = "score_mismatch:tsumo"
                    elif "last_event=ron" in msg:
                        cat = "score_mismatch:ron"
                    elif "last_event=discard" in msg:
                        cat = "score_mismatch:draw"
                    elif "last_event=liuju" in msg:
                        cat = "score_mismatch:liuju"
                    else:
                        cat = "score_mismatch:other"
                elif "No matching Ron" in msg:
                    cat = "no_ron"
                elif "No matching Chi" in msg:
                    cat = "no_chi"
                elif "No matching Tsumo" in msg:
                    cat = "no_tsumo"
                elif "not in WaitResponse" in msg:
                    cat = "phase_error"
                else:
                    cat = "other"
                fail_categories[cat] = fail_categories.get(cat, 0) + 1

                if len(fail_details) < 30:
                    fail_details.append(
                        f"  FAIL: {game_uuid} kyoku={k}: {msg}"
                    )

    print()
    print(f"Total kyoku: {total_kyoku}")
    print(f"  Success: {total_success}")
    print(f"  Failed:  {total_fail}")
    print(f"  Skipped: {total_skip}")

    if fail_categories:
        print()
        print("Failure categories:")
        for cat, count in sorted(fail_categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    if fail_details:
        print()
        print("Failure details (first 30):")
        for d in fail_details:
            print(d)


if __name__ == "__main__":
    main()
