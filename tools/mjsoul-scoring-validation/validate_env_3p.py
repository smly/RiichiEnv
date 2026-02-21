"""
Validate 3P RiichiEnv implementation by tracing MjSoul replays using env.step().

Loads MjSoul 3P replay data, reconstructs the wall from paishan, and
traces each kyoku through RiichiEnv("3p-red-half") using env.step(acts).
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

TARGET_FILE_PATTERN = "/data/mjsoul/game_record_3p_thr/2024/01/02/*.bin.xz"
NP = 3


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


def rearrange_dead_wall_for_env(wall: List[int]) -> List[int]:
    """Rearrange dead wall tiles to match env's expected layout.

    MjSoul 3P dead wall (positions 94-107) has 7 stacks (bottom,top pairs):
      Stack 1 [94,95]: D3 indicators (leftmost dora)
      Stack 2 [96,97]: D2 indicators
      Stack 3 [98,99]: D1 indicators (initial dora at pos 99)
      Stack 4 [100,101]: D4 indicators
      Stack 5 [102,103]: D5 indicators
      Stack 6 [104,105]: Rinshan 3,4
      Stack 7 [106,107]: Rinshan 1,2

    Env expects after load_wall reversal (tiles[i] = input[107-i]):
      tiles[0..3] = rinshan (positions 107..104, already correct)
      tiles[4,5]  = D1,U1 → need input[103,102] = paishan[99,98]
      tiles[6,7]  = D2,U2 → need input[101,100] = paishan[97,96]
      tiles[8,9]  = D3,U3 → need input[99,98]   = paishan[95,94]
      tiles[10,11]= D4,U4 → need input[97,96]   = paishan[101,100]
      tiles[12,13]= D5,U5 → need input[95,94]   = paishan[103,102]
    """
    w = list(wall)
    # Save original dora area (positions 94-103)
    orig = [wall[i] for i in range(94, 104)]
    # Rearrange to env's expected order
    w[94] = orig[8]   # old pos 102 → D5/U5 area
    w[95] = orig[9]   # old pos 103
    w[96] = orig[6]   # old pos 100 → D4/U4 area
    w[97] = orig[7]   # old pos 101
    w[98] = orig[0]   # old pos 94  → D3/U3 area
    w[99] = orig[1]   # old pos 95
    w[100] = orig[2]  # old pos 96  → D2/U2 area
    w[101] = orig[3]  # old pos 97
    w[102] = orig[4]  # old pos 98  → D1/U1 area
    w[103] = orig[5]  # old pos 99
    return w


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


def process_chi_peng_gang(env: RiichiEnv, data: dict) -> dict:
    """Process a ChiPengGang event (Pon or Daiminkan from discard)."""
    seat = data["seat"]
    meld_type = data["type"]  # 0=Chi, 1=Pon, 2=Daiminkan
    tiles_str = data["tiles"]  # List of tile strings in the meld

    if meld_type == 1:
        action_type = ActionType.Pon
    elif meld_type == 2:
        action_type = ActionType.Daiminkan
    else:
        raise RuntimeError(f"Unexpected ChiPengGang type {meld_type} in 3P game")

    # Find the matching legal action for the claiming player
    legals = env._get_legal_actions(seat)
    matched = None

    if action_type == ActionType.Pon:
        # Match by the called tile type
        called_tile_str = tiles_str[0]  # Any tile in the meld should match
        matched = find_legal_action(legals, ActionType.Pon, called_tile_str)
        if matched is None:
            # Try other tiles in the meld
            for ts in tiles_str[1:]:
                matched = find_legal_action(legals, ActionType.Pon, ts)
                if matched:
                    break

    elif action_type == ActionType.Daiminkan:
        called_tile_str = tiles_str[0]
        matched = find_legal_action(legals, ActionType.Daiminkan, called_tile_str)

    if matched is None:
        raise RuntimeError(
            f"No matching legal action for ChiPengGang "
            f"seat={seat} type={meld_type} tiles={tiles_str}, "
            f"legals={[a.action_type for a in legals]}"
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


def process_babei(env: RiichiEnv, data: dict) -> dict:
    """Process a BaBei event (Kita / North tile declaration)."""
    seat = data["seat"]

    legals = env._get_legal_actions(seat)
    matched = find_legal_action(legals, ActionType.Kita)

    if matched is None:
        raise RuntimeError(
            f"No matching Kita legal action for seat={seat}, "
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
    """Process a Hule event with zimo=False (Ron win, possibly double)."""
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
    if not paishan:
        return True, "skip:no_paishan"

    # Parse wall
    try:
        wall = parse_paishan(paishan)
    except Exception as e:
        return False, f"Failed to parse paishan: {e}"

    if len(wall) != 108:
        return False, f"Wall has {len(wall)} tiles, expected 108"

    # Rearrange dead wall to match env's expected layout
    wall = rearrange_dead_wall_for_env(wall)

    # Setup round
    scores = nr["scores"]
    round_wind = nr["chang"]
    honba = nr["ben"]
    kyotaku = nr["liqibang"]

    # Determine oya from hands (player with 14 tiles after draw)
    # In MjSoul, ju indicates the round number within the wind
    oya = nr["ju"] % NP

    rule = GameRule.default_mjsoul()
    rule.is_sanma = True
    rule.allow_kita = True
    rule.sanma_tsumo_zon = True

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
    for i in range(NP):
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
            # Exhaustive draw
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

            elif name == "ChiPengGang":
                # Should be in WaitResponse
                if env.phase != Phase.WaitResponse:
                    return False, (
                        f"Event {ev_idx}: ChiPengGang but env not in WaitResponse "
                        f"(phase={env.phase})"
                    )
                process_chi_peng_gang(env, data)

            elif name == "AnGangAddGang":
                # Resolve pending WaitResponse if needed
                if env.phase == Phase.WaitResponse:
                    submit_all_pass(env)
                if env.is_done:
                    continue
                process_angang_addgang(env, data)

            elif name == "BaBei":
                # Kita - resolve pending WaitResponse if needed
                if env.phase == Phase.WaitResponse:
                    submit_all_pass(env)
                if env.is_done:
                    continue
                process_babei(env, data)

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
                else:
                    # Ron in WaitResponse
                    if env.phase != Phase.WaitResponse:
                        return False, (
                            f"Event {ev_idx}: Hule(ron) but env not in WaitResponse "
                            f"(phase={env.phase})"
                        )
                    process_ron(env, data)

            elif name == "LiuJu":
                if env.phase == Phase.WaitResponse:
                    submit_all_pass(env)
                if env.is_done:
                    continue
                process_liuju(env, data)

        except RuntimeError as e:
            return False, (
                f"Event {ev_idx} ({name}): {e}"
            )

    # Verify end scores
    end_scores = kyoku.end_scores
    env_scores = env.scores()

    if env_scores != end_scores:
        deltas = [env_scores[i] - end_scores[i] for i in range(NP)]
        return False, (
            f"Score mismatch: env={env_scores} expected={end_scores} "
            f"delta={deltas}"
        )

    return True, "ok"


def main():
    target_files = sorted(glob.glob(TARGET_FILE_PATTERN, recursive=True))
    if not target_files:
        print(f"No files found matching {TARGET_FILE_PATTERN}")
        sys.exit(1)

    total_kyoku = 0
    total_success = 0
    total_fail = 0
    total_skip = 0
    fail_details = []
    fail_categories: Dict[str, int] = {}

    env = RiichiEnv("3p-red-half")

    for path in tqdm.tqdm(target_files, desc="Processing files", ncols=100):
        with lzma.open(path, "rb") as f:
            data = f.read()
            paifu: Paifu = MjsoulPaifuParser.to_dict(data)

        game_uuid = paifu.header.get("uuid", "unknown")
        game = MjSoulReplay.from_dict(paifu.data)

        for k, kyoku in enumerate(game.take_kyokus()):
            total_kyoku += 1

            success, msg = validate_kyoku(env, kyoku, game_uuid, k)

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
                    cat = "score_mismatch"
                elif "No matching Ron" in msg:
                    cat = "no_ron"
                elif "No matching Kita" in msg:
                    cat = "no_kita"
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
