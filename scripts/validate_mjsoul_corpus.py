"""Strict differential checks of raw Mahjong Soul records against env.step().

Requires mjsoul-parser (the local protobuf decoder) and riichienv. Raw records
remain the oracle: this does not use MjSoulReplay's reconstructed end scores or
apply_event/apply_log_action to advance the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from riichienv import Action, ActionType, GameRule, Phase, RiichiEnv

RED_IDS = {16, 52, 88}
DRAW_REASONS = {1: "kyushu_kyuhai", 2: "sufuurenta", 3: "suukansansen", 4: "suucha_riichi"}


class ReplayMismatchError(AssertionError):
    def __init__(self, category, actual, expected):
        self.category = category
        self.actual = actual
        self.expected = expected
        super().__init__(f"{category}: actual={actual!r}, expected={expected!r}")


def equal(category, actual, expected):
    if actual != expected:
        raise ReplayMismatchError(category, actual, expected)


def tile_key(tile):
    if isinstance(tile, int):
        return tile // 4, tile in RED_IDS
    number, suit = int(tile[0]), "mpsz".index(tile[1])
    return suit * 9 + (4 if number == 0 else number - 1), number == 0


def tile_keys(tiles):
    return sorted(tile_key(t) for t in tiles)


def parse_wall(paishan, np):
    used = Counter()
    wall = []
    for i in range(0, len(paishan), 2):
        kind, red = tile_key(paishan[i : i + 2])
        if red:
            copy = 0
        else:
            copy = used[kind] + int(kind in (4, 13, 22))
            used[kind] += 1
        wall.append(kind * 4 + copy)
    expected = set(range(136)) - (set(range(4, 32)) if np == 3 else set())
    equal("source_wall_size", len(wall), len(expected))
    equal("source_wall_tiles", sorted(wall), sorted(expected))
    return wall


def select_action(env, seat, kind, tile=None, consumed=None, moqie=None):
    candidates = []
    for action in env._get_legal_actions(seat):
        if action.action_type != kind:
            continue
        if tile is not None:
            # A kan event names the tile kind, not necessarily the added copy.
            if kind in (ActionType.ANKAN, ActionType.KAKAN):
                if action.tile // 4 != tile_key(tile)[0]:
                    continue
            elif action.tile is None or tile_key(action.tile) != tile_key(tile):
                continue
        if consumed is not None and tile_keys(action.consume_tiles) != tile_keys(consumed):
            continue
        candidates.append(action)
    if moqie is not None:
        preferred = [a for a in candidates if (a.tile == env.drawn_tile) == moqie]
        if preferred:
            candidates = preferred
    if not candidates:
        raise ReplayMismatchError(
            "legal_action",
            {"seat": seat, "hand": env.hands[seat], "legals": [a.to_dict() for a in env._get_legal_actions(seat)]},
            {"kind": str(kind), "tile": tile, "consumed": consumed},
        )
    return candidates[0]


def check_invariants(env, total, np):
    equal("point_conservation", sum(env.scores()) + 1000 * env.riichi_sticks, total)
    if env.is_done:
        return
    hands, melds = env.hands, env.melds
    meld_tiles = [t for ms in melds for m in ms for t in m.tiles]
    called = set(meld_tiles)
    zones = [t for hand in hands for t in hand] + meld_tiles + env.wall
    zones += [t for hand in env.kita_tiles for t in hand]
    zones += [t for discards in env.discards for t in discards if t not in called]
    equal("tile_conservation_count", len(zones), 108 if np == 3 else 136)
    equal("tile_conservation_unique", len(set(zones)), len(zones))
    for seat, hand in enumerate(hands):
        base = 13 - 3 * len(melds[seat])
        if len(hand) not in (base, base + 1):
            raise ReplayMismatchError("hand_size", {"seat": seat, "size": len(hand)}, [base, base + 1])


def expected_end_scores(events):
    last = events[-1]
    data = last["data"]
    if last["name"] == "Hule":
        return data["scores"]
    if last["name"] == "NoTile" and data["scores"]:
        scores = list(data["scores"][0]["old_scores"])
        for payment in data["scores"]:
            # Protobuf omits the entire delta vector when no points move.
            deltas = payment["delta_scores"] or [0] * len(scores)
            scores = [s + d for s, d in zip(scores, deltas, strict=True)]
        return scores
    scores = list(events[0]["data"]["scores"])
    for event in events:
        d = event["data"]
        if event["name"] == "DiscardTile" and (d.get("is_liqi") or d.get("is_wliqi")):
            scores[d["seat"]] -= 1000
    return scores


def validate_round(events, next_start, mode, counters):  # noqa: PLR0915 - Keep the event trace in order.
    nr = events[0]["data"]
    equal("source_start", events[0]["name"], "NewRound")
    np = len(nr["scores"])
    wall = parse_wall(nr.get("paishan", ""), np)
    rule = GameRule.default_mjsoul()
    single = RiichiEnv(f"{np}p-red-single", rule=rule, seed=42)
    match = RiichiEnv(f"{np}p-red-{'half' if mode in (2, 12) else 'east'}", rule=rule, seed=42)
    kwargs = dict(
        oya=nr["ju"], wall=wall, round_wind=nr["chang"], scores=nr["scores"], honba=nr["ben"], kyotaku=nr["liqibang"]
    )
    for env in (single, match):
        env.reset(**kwargs)
    total = sum(nr["scores"]) + 1000 * nr["liqibang"]
    for seat in range(np):
        equal("initial_hand", tile_keys(single.hands[seat]), tile_keys(nr[f"tiles{seat}"]))
    equal("initial_dora", tile_keys(single.dora_indicators), tile_keys(nr["doras"]))
    equal("initial_wall_count", len(single.wall) - 14, nr["left_tile_count"])
    check_invariants(single, total, np)

    def step(actions):
        single.step(actions)
        match.step(actions)
        counters["steps"] += 1
        check_invariants(single, total, np)

    def pass_all():
        if single.phase == Phase.WaitResponse and not single.is_done:
            step({p: Action(ActionType.PASS) for p in single.active_players})

    def claim(seats, kind, tile=None, consumed=None):
        equal("claimants_active", set(seats) <= set(single.active_players), True)
        actions = {
            p: select_action(single, p, kind, tile, consumed) if p in seats else Action(ActionType.PASS)
            for p in single.active_players
        }
        step(actions)

    index = 0
    try:
        for index, event in enumerate(events[1:], 1):  # noqa: B007 - Used to report the failing event.
            name, data = event["name"], event["data"]
            counters["events"] += 1
            counters[f"event:{name}"] += 1
            if name in ("DealTile", "DiscardTile", "AnGangAddGang", "BaBei"):
                pass_all()
                equal("premature_round_end", single.is_done, False)
                equal("actor", single.current_player, data["seat"])
            if name == "DealTile":
                equal("drawn_tile", tile_key(single.drawn_tile), tile_key(data["tile"]))
                equal("wall_count", len(single.wall) - 14, data["left_tile_count"])
                if data.get("doras"):
                    equal("draw_dora", tile_keys(single.dora_indicators), tile_keys(data["doras"]))
            elif name == "DiscardTile":
                seat = data["seat"]
                if data.get("is_liqi") or data.get("is_wliqi"):
                    step({seat: select_action(single, seat, ActionType.RIICHI)})
                    counters["riichi"] += 1
                action = select_action(single, seat, ActionType.DISCARD, data["tile"], moqie=data.get("moqie"))
                step({seat: action})
                if data.get("doras") and not single.is_done:
                    equal("discard_dora", tile_keys(single.dora_indicators), tile_keys(data["doras"]))
            elif name == "ChiPengGang":
                equal("claim_phase", single.phase, Phase.WaitResponse)
                seat = data["seat"]
                own = [t for t, p in zip(data["tiles"], data["froms"], strict=True) if p == seat]
                called = [t for t, p in zip(data["tiles"], data["froms"], strict=True) if p != seat]
                equal("source_call", len(called), 1)
                kind = [ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN][data["type"]]
                claim({seat}, kind, called[0], own)
                counters[f"call:{kind}"] += 1
            elif name == "AnGangAddGang":
                kind = ActionType.ANKAN if data["type"] == 3 else ActionType.KAKAN
                step({data["seat"]: select_action(single, data["seat"], kind, data["tiles"])})
                counters[f"call:{kind}"] += 1
            elif name == "BaBei":
                step({data["seat"]: select_action(single, data["seat"], ActionType.KITA)})
                counters["kita"] += 1
            elif name == "Hule":
                hules = data["hules"]
                for h in hules:
                    expected_hand = h["hand"] + ([h["hu_tile"]] if h["zimo"] else [])
                    equal("winning_hand", tile_keys(single.hands[h["seat"]]), tile_keys(expected_hand))
                    equal("win_dora", tile_keys(single.dora_indicators), tile_keys(h["doras"]))
                if hules[0]["zimo"]:
                    pass_all()
                    seat = hules[0]["seat"]
                    heavenly_hand = (
                        index == 1
                        and seat == nr["ju"]
                        and any(f["id"] == 35 and f["val"] > 0 for f in hules[0]["fans"])
                    )
                    # Tenhou has no separate draw event: the provider can name
                    # a different tile from the same complete 14-tile hand.
                    win_tile = None if heavenly_hand else hules[0]["hu_tile"]
                    step({seat: select_action(single, seat, ActionType.TSUMO, win_tile)})
                    counters["heavenly_hand_normalizations"] += heavenly_hand
                else:
                    equal("ron_phase", single.phase, Phase.WaitResponse)
                    claim({h["seat"] for h in hules}, ActionType.RON, hules[0]["hu_tile"])
                counters[f"win:{'tsumo' if hules[0]['zimo'] else str(len(hules)) + 'ron'}"] += 1
                equal("winners", sorted(single.win_results), sorted(h["seat"] for h in hules))
                for h in hules:
                    result = single.win_results[h["seat"]]
                    equal("win_han", result.han, h["count"] * (13 if h["yiman"] else 1))
                    # Fu is irrelevant to yakuman payments; providers display
                    # different placeholder values (e.g. kokushi: 25 versus 0).
                    if not h["yiman"]:
                        equal("win_fu", result.fu, h["fu"])
                    equal("win_yaku", sorted(set(result.yaku)), sorted({f["id"] for f in h["fans"] if f["val"] > 0}))
                    counters["wins"] += 1
            elif name == "NoTile":
                pass_all()
                counters["draw:nagashi" if data["liujumanguan"] else "draw:exhaustive"] += 1
            elif name == "LiuJu":
                if data["type"] == 1:
                    pass_all()
                    step({data["seat"]: select_action(single, data["seat"], ActionType.KYUSHU_KYUHAI)})
                else:
                    pass_all()
                counters[f"draw:{DRAW_REASONS.get(data['type'], data['type'])}"] += 1
            else:
                raise ReplayMismatchError("unsupported_event", name, "known raw record")

        equal("round_finished", single.is_done, True)
        log = single.mjai_log
        if events[-1]["name"] != "Hule":
            draw = [e for e in log if e["type"] == "ryukyoku"]
            expected = (
                DRAW_REASONS[events[-1]["data"]["type"]]
                if events[-1]["name"] == "LiuJu"
                else "nagashimangan"
                if events[-1]["data"]["liujumanguan"]
                else "exhaustive_draw"
            )
            equal("draw_reason", [e["reason"] for e in draw], [expected])
        expected_scores = expected_end_scores(events)
        equal("end_scores", single.scores(), expected_scores)
        equal("match_end_scores", match.scores(), expected_scores)
        equal("match_finished", match.is_done, next_start is None)
        if next_start is not None:
            expected_meta = [next_start[k] for k in ("chang", "ju", "ben", "liqibang")]
            actual_meta = [match.round_wind, match.oya, match.honba, match.riichi_sticks]
            equal("next_round", actual_meta, expected_meta)
            equal("score_continuity", match.scores(), next_start["scores"])
        counters["rounds_passed"] += 1
        return match.scores()
    except Exception as error:
        if isinstance(error, ReplayMismatchError):
            error.event_index = index
            error.event_name = events[index]["name"]
        raise


def validate_file(path):
    from mjsoul_parser import MjsoulPaifuParser  # noqa: PLC0415 - Optional raw-record decoder.

    counters = Counter()
    failures = []
    digest = None
    try:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        paifu = MjsoulPaifuParser.to_dict(lzma.decompress(raw))
        mode = paifu.header["config"]["mode"]["mode"]
        equal("source_mode", mode in (1, 2, 11, 12), True)
        equal("source_ranked", paifu.header["config"]["category"], 2)
        last_scores = None
        for index, events in enumerate(paifu.data):
            counters["rounds"] += 1
            counters["3p_rounds" if len(events[0]["data"]["scores"]) == 3 else "4p_rounds"] += 1
            following = paifu.data[index + 1][0]["data"] if index + 1 < len(paifu.data) else None
            try:
                scores = validate_round(events, following, mode, counters)
                if following is None:
                    last_scores = scores
            except Exception as error:
                failures.append(
                    dict(
                        round=index,
                        category=getattr(error, "category", type(error).__name__),
                        event_index=getattr(error, "event_index", None),
                        event_name=getattr(error, "event_name", None),
                        message=str(error),
                    )
                )
        if last_scores is not None:
            final_players = sorted(paifu.header["result"]["players"], key=lambda p: p["seat"])
            final_scores = [p["part_point_1"] for p in final_players]
            counters["final_score_checks"] += 1
            if last_scores != final_scores:
                failures.append(
                    dict(
                        round=len(paifu.data) - 1,
                        category="final_game_scores",
                        message=f"actual={last_scores}, expected={final_scores}",
                    )
                )
        counters["files"] += 1
        counters["files_passed"] += not failures
    except Exception as error:
        failures.append(dict(round=None, category="decode_or_source", message=str(error)))
        counters["file_errors"] += 1
    return dict(path=path, sha256=digest, counts=dict(counters), failures=failures)


def sample_files(root, count, seed):
    """Equal samples per mode/table, spread across dates; UUIDs are deduplicated."""
    rng = random.Random(seed)
    paths, seen = [], set()
    for family in ("3p_thr", "3p_jad", "4p_thr", "4p_jad"):
        days = sorted(p for p in (root / f"game_record_{family}").glob("*/*/*") if p.is_dir())
        rng.shuffle(days)
        quota = count // 4
        collected = 0
        for index, day in enumerate(days):
            files = sorted(day.glob("*.bin.xz"))
            rng.shuffle(files)
            take = max(1, (quota - collected + len(days) - index - 1) // (len(days) - index))
            for path in files:
                if path.name in seen:
                    continue
                paths.append(str(path))
                seen.add(path.name)
                collected += 1
                take -= 1
                if take == 0 or collected == quota:
                    break
            if collected == quota:
                break
        equal("sample_size", collected, quota)
    rng.shuffle(paths)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/mjsoul"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    if args.workers < 1 or (not args.files and not args.manifest and (args.sample <= 0 or args.sample % 4)):
        parser.error("workers must be positive; sample must be a positive multiple of four")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.files:
        paths = args.files
    elif args.manifest:
        paths = json.loads(args.manifest.read_text())
    else:
        paths = sample_files(args.root, args.sample, args.seed)
    (args.output / "manifest.json").write_text(json.dumps(paths, indent=2) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary = dict(
        commit=commit,
        seed=args.seed,
        files_selected=len(paths),
        counts={},
        failure_categories={},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    counts, categories = Counter(), Counter()
    print(json.dumps(dict(event="start", files=len(paths), workers=args.workers)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "results.jsonl").open("w") as output:
        for index, result in enumerate(pool.map(validate_file, paths, chunksize=4), 1):
            output.write(json.dumps(result) + "\n")
            counts.update(result["counts"])
            categories.update(f["category"] for f in result["failures"])
            if index % 100 == 0 or index == len(paths):
                output.flush()
                summary.update(
                    counts=dict(counts), failure_categories=dict(categories), elapsed_seconds=time.monotonic() - started
                )
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(
                    json.dumps(
                        dict(
                            event="progress",
                            files=index,
                            rounds=counts["rounds"],
                            passed=counts["rounds_passed"],
                            failures=sum(categories.values()),
                        )
                    ),
                    flush=True,
                )
    return bool(categories)


if __name__ == "__main__":
    raise SystemExit(main())
