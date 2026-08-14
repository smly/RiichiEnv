"""Differential validation against attributed external replay golden data.

The committed manifest is intentionally small enough for CI.  Larger corpora
remain opt-in: pass one or more additional manifest paths on the command line.
Each manifest freezes both source and derived-fixture bytes (SHA-256), the
normalized action sequence, provider result, score transition, and next-round
metadata.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from riichienv import MjaiReplay, get_all_yaku

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "data" / "external_replay" / "manifest.json"

HONORS = {"E": "1z", "S": "2z", "W": "3z", "N": "4z", "P": "5z", "F": "6z", "C": "7z"}
CALL_TYPES = {0: "chi", 1: "pon", 2: "daiminkan"}


class ExternalReplayError(AssertionError):
    """Raised when replay output diverges from its external oracle."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalReplayError(message)


def _require_sha256(value: Any, context: str) -> None:
    is_lower_hex = (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )
    _require(is_lower_hex, f"{context}: expected 64-character lowercase SHA-256")


def _tile(tile: str | None) -> str | None:
    if tile is None:
        return None
    if tile.endswith("r"):
        return f"0{tile[1]}"
    return HONORS.get(tile, tile)


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    events = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ExternalReplayError(f"{path}:{line_no}: invalid JSON: {error}") from error
    return events


def _rounds(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in events:
        if event.get("type") == "start_kyoku":
            if current is not None:
                rounds.append(current)
            current = {"start": event, "events": [], "complete": False}
        elif current is not None:
            current["events"].append(event)
            if event.get("type") == "end_kyoku":
                current["complete"] = True
                rounds.append(current)
                current = None
        elif event.get("type") not in {"start_game", "end_game"}:
            raise ExternalReplayError(f"gameplay event outside a round: {event.get('type')!r}")
    if current is not None:
        rounds.append(current)
    return rounds


def _start(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "bakaze": event["bakaze"],
        "kyoku": event["kyoku"],
        "honba": event["honba"],
        "kyotaku": event.get("kyotaku", event.get("kyoutaku", 0)),
        "oya": event["oya"],
        "scores": event["scores"],
    }


def _yaku_name_map() -> dict[str, int]:
    result: dict[str, int] = {}
    for yaku in get_all_yaku():
        result[yaku.name] = yaku.id
        result[yaku.name_en.casefold()] = yaku.id
    return result


def _yaku_id(name: str) -> int:
    if name in {"自風 東", "自風 南", "自風 西", "自風 北"} or name.casefold() in {
        "seat wind east",
        "seat wind south",
        "seat wind west",
        "seat wind north",
    }:
        return 10
    if name in {"場風 東", "場風 南", "場風 西", "場風 北"} or name.casefold() in {
        "round wind east",
        "round wind south",
        "round wind west",
        "round wind north",
    }:
        return 11
    mapping = _yaku_name_map()
    yaku_id = mapping.get(name, mapping.get(name.casefold()))
    if yaku_id is None:
        raise ExternalReplayError(f"unknown external yaku label: {name!r}")
    return yaku_id


def _digest(actions: list[dict[str, Any]]) -> str:
    payload = json.dumps(actions, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_raw_actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    last_actor: int | None = None
    last_tile: str | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "tsumo":
            last_actor, last_tile = event["actor"], _tile(event["pai"])
            actions.append({"type": "tsumo", "actor": last_actor, "tile": last_tile})
        elif event_type == "dahai":
            last_actor, last_tile = event["actor"], _tile(event["pai"])
            actions.append({"type": "dahai", "actor": last_actor, "tile": last_tile})
        elif event_type == "reach":
            actions.append({"type": "reach", "actor": event["actor"]})
        elif event_type in {"reach_accepted", "end_kyoku", "end_game"}:
            continue
        elif event_type in {"chi", "pon", "daiminkan", "kan"}:
            actions.append(
                {
                    "type": "daiminkan" if event_type == "kan" else event_type,
                    "actor": event["actor"],
                    "target": event["target"],
                    "tile": _tile(event["pai"]),
                    "consumed": sorted(_tile(tile) for tile in event["consumed"]),
                }
            )
        elif event_type in {"ankan", "kakan"}:
            tile = event.get("pai") or event["consumed"][0]
            last_actor, last_tile = event["actor"], _tile(tile)
            actions.append({"type": event_type, "actor": last_actor, "tile": last_tile})
        elif event_type == "kita":
            last_actor, last_tile = event["actor"], "4z"
            actions.append({"type": "kita", "actor": last_actor})
        elif event_type == "dora":
            actions.append({"type": "dora", "tile": _tile(event["dora_marker"])})
        elif event_type == "hora":
            win_tile = _tile(event.get("pai")) or last_tile
            actions.append(
                {
                    "type": "hora",
                    "actor": event["actor"],
                    "target": event["target"],
                    "tile": win_tile,
                }
            )
        elif event_type == "ryukyoku":
            actions.append({"type": "ryukyoku"})
        else:
            raise ExternalReplayError(f"unsupported raw replay event: {event_type!r}")
    return actions


def _normalize_replay_actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    last_actor: int | None = None
    last_tile: str | None = None
    for event in events:
        name, data = event["name"], event["data"]
        if name == "NewRound":
            continue
        if name == "DealTile":
            last_actor, last_tile = data["seat"], _tile(data["tile"])
            actions.append({"type": "tsumo", "actor": last_actor, "tile": last_tile})
        elif name == "DiscardTile":
            if data["is_liqi"] or data["is_wliqi"]:
                actions.append({"type": "reach", "actor": data["seat"]})
            last_actor, last_tile = data["seat"], _tile(data["tile"])
            actions.append({"type": "dahai", "actor": last_actor, "tile": last_tile})
        elif name == "ChiPengGang":
            actor = data["seat"]
            target_index = next(index for index, seat in enumerate(data["froms"]) if seat != actor)
            actions.append(
                {
                    "type": CALL_TYPES[data["type"]],
                    "actor": actor,
                    "target": data["froms"][target_index],
                    "tile": _tile(data["tiles"][target_index]),
                    "consumed": sorted(
                        _tile(tile) for index, tile in enumerate(data["tiles"]) if index != target_index
                    ),
                }
            )
        elif name == "AnGangAddGang":
            action_type = "ankan" if data["type"] == 3 else "kakan"
            last_actor, last_tile = data["seat"], _tile(data["tiles"])
            actions.append({"type": action_type, "actor": last_actor, "tile": last_tile})
        elif name == "BaBei":
            last_actor, last_tile = data["seat"], "4z"
            actions.append({"type": "kita", "actor": last_actor})
        elif name == "Dora":
            actions.append({"type": "dora", "tile": _tile(data["dora_marker"])})
        elif name == "Hule":
            for hule in data["hules"]:
                actions.append(
                    {
                        "type": "hora",
                        "actor": hule["seat"],
                        "target": hule["seat"] if hule["zimo"] else last_actor,
                        "tile": _tile(hule["hu_tile"]) or last_tile,
                    }
                )
        elif name in {"NoTile", "LiuJu"}:
            actions.append({"type": "ryukyoku"})
        else:
            raise ExternalReplayError(f"unsupported parsed replay event: {name!r}")
    return actions


def _raw_wins(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wins = []
    last_tile = None
    for event in events:
        event_type = event.get("type")
        if event_type in {"tsumo", "dahai", "kakan"}:
            last_tile = _tile(event.get("pai"))
        elif event_type == "ankan":
            last_tile = _tile(event["consumed"][0])
        elif event_type == "kita":
            last_tile = "4z"
        elif event_type == "hora":
            raw_yaku = event.get("yaku", event.get("yakus"))
            yaku_values = None
            if raw_yaku is not None:
                yaku_values = sorted([_yaku_id(name), value] for name, value in raw_yaku)
            payment = None
            if any(event.get(field) is not None for field in ("hora_points", "point_zimo_qin", "point_zimo_xian")):
                payment = {
                    "ron": event.get("hora_points", 0) if event["actor"] != event["target"] else 0,
                    "tsumo_oya": event.get("point_zimo_qin", 0),
                    "tsumo_ko": event.get("point_zimo_xian", 0),
                }
            wins.append(
                {
                    "winner": event["actor"],
                    "target": event["target"],
                    "win_tile": _tile(event.get("pai")) or last_tile,
                    "han": event.get("han", event.get("fan")),
                    "fu": event.get("fu"),
                    "yaku_values": yaku_values,
                    "payment": payment,
                }
            )
    return wins


def _replay_wins(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wins = []
    last_actor = None
    for event in events:
        name, data = event["name"], event["data"]
        if name in {"DealTile", "DiscardTile", "AnGangAddGang", "BaBei"}:
            last_actor = data["seat"]
        elif name == "Hule":
            for hule in data["hules"]:
                wins.append(
                    {
                        "winner": hule["seat"],
                        "target": hule["seat"] if hule["zimo"] else last_actor,
                        "win_tile": _tile(hule["hu_tile"]),
                        "han": hule["count"] or None,
                        "fu": hule["fu"] or None,
                        "yaku_values": sorted([fan["id"], fan.get("val", 1)] for fan in hule["fans"]) or None,
                        "payment": {
                            "ron": hule["point_rong"],
                            "tsumo_oya": hule["point_zimo_qin"],
                            "tsumo_ko": hule["point_zimo_xian"],
                        }
                        if any((hule["point_rong"], hule["point_zimo_qin"], hule["point_zimo_xian"]))
                        else None,
                    }
                )
    return wins


def _accepted_riichi_actors(events: list[dict[str, Any]], players: int) -> set[int]:
    actors: set[int] = set()
    for event in events:
        if event.get("type") != "reach_accepted":
            continue
        actor = event.get("actor")
        if not (isinstance(actor, int) and not isinstance(actor, bool) and 0 <= actor < players):
            raise ExternalReplayError(f"reach_accepted actor {actor!r} is invalid for {players} players")
        _require(actor not in actors, f"player {actor} has duplicate reach_accepted events")
        actors.add(actor)
    return actors


def _sum_deltas(events: list[dict[str, Any]], players: int, accepted_riichi: set[int]) -> list[int] | None:
    deltas = [0] * players
    for actor in accepted_riichi:
        deltas[actor] -= 1000
    found = bool(accepted_riichi)
    for event in events:
        delta = event.get("deltas", event.get("delta"))
        if event.get("type") in {"hora", "ryukyoku"} and delta is not None:
            _require(len(delta) == players, f"result delta has {len(delta)} entries for {players} players")
            deltas = [total + value for total, value in zip(deltas, delta, strict=True)]
            found = True
    return deltas if found else None


def _compare_calculator(
    kyoku: Any,
    expected_wins: list[dict[str, Any]],
    expected_riichi_sticks: int,
    context: str,
) -> None:
    contexts = list(kyoku.take_win_result_contexts())
    _require(len(contexts) == len(expected_wins), f"{context}: calculator win count mismatch")
    for index, (calc_context, expected) in enumerate(zip(contexts, expected_wins, strict=True)):
        prefix = f"{context} win={index}"
        _require(calc_context.seat == expected["winner"], f"{prefix}: calculator winner mismatch")
        _require(calc_context.agari_tile // 4 == _tile_type(expected["win_tile"]), f"{prefix}: win tile mismatch")
        _require(calc_context.conditions.num_players == len(kyoku.scores), f"{prefix}: wrong evaluator player count")
        _require(calc_context.conditions.is_sanma == (len(kyoku.scores) == 3), f"{prefix}: wrong sanma condition")
        _require(calc_context.conditions.honba == kyoku.ben, f"{prefix}: honba condition mismatch")
        _require(
            calc_context.conditions.riichi_sticks == expected_riichi_sticks,
            f"{prefix}: kyotaku condition mismatch",
        )
        actual = calc_context.actual
        _require(actual.is_win, f"{prefix}: calculator rejected external win shape")
        if expected["han"] is not None:
            _require(actual.han == expected["han"], f"{prefix}: han {actual.han} != {expected['han']}")
        if expected["fu"] is not None:
            _require(actual.fu == expected["fu"], f"{prefix}: fu {actual.fu} != {expected['fu']}")
        if expected["yaku_values"] is not None:
            expected_ids = sorted(item[0] for item in expected["yaku_values"] if item[1] > 0)
            _require(sorted(actual.yaku) == expected_ids, f"{prefix}: yaku {actual.yaku} != {expected_ids}")
        payment = expected["payment"]
        if payment is not None:
            if expected["target"] != expected["winner"]:
                _require(actual.ron_agari == payment["ron"], f"{prefix}: ron payment mismatch")
            else:
                _require(
                    actual.tsumo_agari_oya == payment["tsumo_oya"],
                    f"{prefix}: dealer tsumo payment mismatch",
                )
                _require(
                    actual.tsumo_agari_ko == payment["tsumo_ko"],
                    f"{prefix}: non-dealer tsumo payment mismatch",
                )


def _tile_type(tile: str) -> int:
    number, suit = int(tile[0]), tile[1]
    return {"m": 0, "p": 9, "s": 18, "z": 27}[suit] + number - 1


def observe_fixture(fixture: Path, rule: str, validated_rounds: list[int]) -> list[dict[str, Any]]:
    raw_rounds = _rounds(_read_events(fixture))
    replay_rounds = list(MjaiReplay.from_jsonl(str(fixture), rule=rule).take_kyokus())
    observed = []
    for round_index in validated_rounds:
        _require(round_index < len(raw_rounds), f"{fixture}: missing raw round {round_index}")
        _require(round_index < len(replay_rounds), f"{fixture}: missing replay round {round_index}")
        raw_round, replay_round = raw_rounds[round_index], replay_rounds[round_index]
        _require(raw_round["complete"], f"{fixture}: selected raw round {round_index} is incomplete")
        raw_actions = _normalize_raw_actions(raw_round["events"])
        replay_actions = _normalize_replay_actions(list(replay_round.events()))
        _require(raw_actions == replay_actions, _action_diff(fixture, round_index, raw_actions, replay_actions))

        start = _start(raw_round["start"])
        accepted_riichi = _accepted_riichi_actors(raw_round["events"], len(start["scores"]))
        raw_wins = _raw_wins(raw_round["events"])
        replay_wins = _replay_wins(list(replay_round.events()))
        _require(raw_wins == replay_wins, f"{fixture} round={round_index}: replay win metadata mismatch")

        grp = replay_round.grp_features()
        next_round = _start(raw_rounds[round_index + 1]["start"]) if round_index + 1 < len(raw_rounds) else None
        delta = _sum_deltas(raw_round["events"], len(start["scores"]), accepted_riichi)
        end_scores = next_round["scores"] if next_round is not None else grp["end_scores"]
        if delta is not None:
            _require(
                [score + change for score, change in zip(start["scores"], delta, strict=True)] == end_scores,
                f"{fixture} round={round_index}: external delta/end score mismatch",
            )
        _require(grp["end_scores"] == end_scores, f"{fixture} round={round_index}: Replay end score mismatch")
        _require(
            grp["delta_scores"] == [e - s for s, e in zip(start["scores"], end_scores, strict=True)],
            f"{fixture} round={round_index}: Replay delta mismatch",
        )

        replay_start = {
            "bakaze": "ESWN"[replay_round.chang],
            "kyoku": replay_round.ju + 1,
            "honba": replay_round.ben,
            "kyotaku": replay_round.liqibang,
            "oya": replay_round.ju % len(replay_round.scores),
            "scores": replay_round.scores,
        }
        _require(replay_start == start, f"{fixture} round={round_index}: Replay start metadata mismatch")
        if next_round is not None:
            next_replay = replay_rounds[round_index + 1]
            replay_next = {
                "bakaze": "ESWN"[next_replay.chang],
                "kyoku": next_replay.ju + 1,
                "honba": next_replay.ben,
                "kyotaku": next_replay.liqibang,
                "oya": next_replay.ju % len(next_replay.scores),
                "scores": next_replay.scores,
            }
            _require(replay_next == next_round, f"{fixture} round={round_index}: next-round metadata mismatch")

        _compare_calculator(
            replay_round,
            raw_wins,
            start["kyotaku"] + len(accepted_riichi),
            f"{fixture} round={round_index}",
        )
        observed.append(
            {
                "index": round_index,
                "start": start,
                "action_count": len(raw_actions),
                "action_sha256": _digest(raw_actions),
                "wins": raw_wins,
                "delta": delta,
                "end_scores": end_scores,
                "next_round": next_round,
            }
        )
    return observed


def _action_diff(path: Path, round_index: int, expected: list[Any], actual: list[Any]) -> str:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return f"{path} round={round_index} action={index}: raw={left!r} replay={right!r}"
    return f"{path} round={round_index}: action length raw={len(expected)} replay={len(actual)}"


def validate_manifest(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, f"{manifest_path}: unsupported schema")
    count = 0
    for entry in manifest["fixtures"]:
        _require(entry.get("provider") in {"tenhou", "mahjong_soul"}, f"{manifest_path}: invalid provider")
        _require(entry.get("rule") in {"tenhou", "mjsoul"}, f"{manifest_path}: invalid rule preset")
        source = entry.get("source", {})
        for field in ("url", "commit", "sha256", "license", "license_url", "transform"):
            _require(source.get(field), f"{manifest_path}: fixture {entry.get('id')} lacks source.{field}")
        _require_sha256(source["sha256"], f"{manifest_path}: source SHA-256")
        _require_sha256(entry.get("fixture_sha256"), f"{manifest_path}: fixture SHA-256")
        for expected_round in entry["rounds"]:
            _require_sha256(
                expected_round.get("action_sha256"),
                f"{manifest_path}: action SHA-256 for round {expected_round.get('index')}",
            )
        fixture = manifest_path.parent / entry["fixture"]
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        _require(digest == entry["fixture_sha256"], f"{fixture}: fixture SHA-256 mismatch")
        expected_rounds = entry["rounds"]
        observed = observe_fixture(fixture, entry["rule"], [item["index"] for item in expected_rounds])
        _require(observed == expected_rounds, f"{fixture}: observed golden metadata differs from manifest")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate attributed external replay golden manifests")
    parser.add_argument("manifests", nargs="*", type=Path, help="Additional opt-in corpus manifests")
    parser.add_argument("--no-committed", action="store_true", help="Skip the committed CI manifest")
    args = parser.parse_args()
    manifests = ([] if args.no_committed else [DEFAULT_MANIFEST]) + args.manifests
    if not manifests:
        parser.error("no manifests selected")
    try:
        total = sum(validate_manifest(path.resolve()) for path in manifests)
    except Exception as error:
        print(f"FAILED: {error}")
        return 1
    print(f"Validated {total} external replay fixtures from {len(manifests)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
