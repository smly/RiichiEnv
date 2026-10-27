import json
import os

import pytest

from riichienv.visualizer.viewer import MetadataInjector


# Load the log once for all tests
@pytest.fixture(scope="module")
def enriched_log():
    # Use the .bak file as requested (now renamed)
    path = os.path.join(os.path.dirname(__file__), "../riichienv-ui/example_before_injection.jsonl")
    if not os.path.exists(path):
        # Fallback if running from root relative path issues
        path = "riichienv-ui/example_before_injection.jsonl"

    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    injector = MetadataInjector(events)
    return injector.process()


def find_kyoku_events(events, bakaze_char, kyoku_num, honba_num):
    """Finds all events for a specific kyoku."""
    kyoku_events = []
    in_kyoku = False

    for ev in events:
        if ev["type"] == "start_kyoku":
            if ev.get("bakaze") == bakaze_char and ev.get("kyoku") == kyoku_num and ev.get("honba") == honba_num:
                in_kyoku = True
                kyoku_events = [ev]
            else:
                in_kyoku = False
        elif in_kyoku:
            kyoku_events.append(ev)
            if ev["type"] == "end_kyoku":
                in_kyoku = False
                yield kyoku_events


def get_hora_score(kyoku_events):
    """Extracts score from the first hora event in the kyoku."""
    for ev in kyoku_events:
        if ev["type"] == "hora":
            return ev.get("meta", {}).get("score")
    return None


def test_e2_1_fu_correctness(enriched_log):
    """
    E2-1 (East 2, 1 Honba):
    Expectation: 3 Han 40 Fu.
    Issue Fixed: Was calculated as 30 Fu due to ID overflow and sorting bugs.
    """
    kyokus = list(find_kyoku_events(enriched_log, "E", 2, 1))
    assert len(kyokus) > 0, "E2-1 not found in log"

    events = kyokus[0]
    score = get_hora_score(events)

    assert score is not None, "No Hora event in E2-1"
    assert score["fu"] == 40, f"Expected 40 Fu for E2-1, got {score['fu']}"
    assert score["han"] == 3


def test_e4_0_ippatsu_correctness(enriched_log):
    """
    E4-0 (East 4, 0 Honba):
    Expectation: 4 Han 30 Fu (No Ippatsu).
    Issue Fixed: Ippatsu was wrongly awarded (5 Han) because it wasn't cleared on subsequent discard.
    """
    kyokus = list(find_kyoku_events(enriched_log, "E", 4, 0))
    assert len(kyokus) > 0, "E4-0 not found in log"

    events = kyokus[0]
    score = get_hora_score(events)

    assert score is not None, "No Hora event in E4-0"

    # Check Ippatsu (ID 30) is NOT present
    yaku_ids = score["yaku"]
    assert 30 not in yaku_ids, f"Ippatsu (30) should not be present in E4-0. Yaku: {yaku_ids}"

    # Check Han/Fu
    assert score["han"] == 4, f"Expected 4 Han for E4-0, got {score['han']}"
    assert score["fu"] == 30


def test_s4_0_fu_correctness(enriched_log):
    """
    S4-0 (South 4, 0 Honba):
    Expectation: 40 Fu.
    Issue Fixed: Previous fix for S4-0 regarding 136-tile offsets.
    """
    kyokus = list(find_kyoku_events(enriched_log, "S", 4, 0))
    if not kyokus:
        pytest.skip("S4-0 not found in log")
        return

    events = kyokus[0]
    score = get_hora_score(events)

    assert score is not None, "No Hora event in S4-0"
    assert score["fu"] == 40, f"Expected 40 Fu for S4-0, got {score['fu']}"


def test_waits_calculation(enriched_log):
    """
    Verify that waits are calculated and injected into Dahai events.
    """
    found_waits = False
    for ev in enriched_log:
        if ev["type"] == "dahai":
            if "waits" in ev.get("meta", {}):
                found_waits = True
                waits = ev["meta"]["waits"]
                assert isinstance(waits, list)
                break

    assert found_waits, "No waits calculated in any Dahai event"


def make_riichi_round(player_count, actor, *, inline_reach=False, late_reach=False, call=None, call_before=False):
    # Only the winner's hand matters for scoring; other players draw and discard.
    tehais = [[] for _ in range(player_count)]
    tehais[actor] = ["1p", "2p", "3p", "4p", "5pr", "6p", "7s", "8s", "9s", "5s", "5s", "4s", "6s"]
    events = [{"type": "start_kyoku", "tehais": tehais, "dora_marker": "C", "oya": 0, "honba": 1}]

    def discard(pid, tile, reach=False):
        events.append({"type": "tsumo", "actor": pid, "pai": tile})
        if reach and not inline_reach:
            events.append({"type": "reach", "actor": pid})
        events.append({"type": "dahai", "actor": pid, "pai": tile, "reach": reach and inline_reach})
        if reach:
            events.append({"type": "reach_accepted", "actor": pid})

    if call and call_before:
        events.append(call)
    if late_reach:
        for pid in range(player_count):
            discard(pid, "E")
    for pid in range(player_count):
        discard(pid, "S", reach=pid == actor)
    if call and not call_before:
        events.append(call)
    # End the winner's ippatsu window before a later ron.
    discard(actor, "W")
    target = (actor + 1) % player_count
    discard(target, "5s")
    events.extend(
        [
            {"type": "hora", "actor": actor, "target": target, "pai": "5s", "ura_markers": ["4p"]},
            {"type": "end_kyoku"},
        ]
    )
    return events


@pytest.mark.parametrize("player_count,actor", [(4, 0), (4, 3), (3, 0), (3, 2)])
@pytest.mark.parametrize("inline_reach", [False, True])
@pytest.mark.parametrize("late_reach", [False, True])
def test_riichi_scoring_after_first_round(player_count, actor, inline_reach, late_reach):
    events = make_riichi_round(player_count, actor, inline_reach=inline_reach, late_reach=late_reach)
    enriched = MetadataInjector(events).process()
    score = get_hora_score(enriched)
    assert score is not None
    assert set(score["yaku"]) == {2 if late_reach else 18, 32, 33}
    assert score["han"] == (3 if late_reach else 4)
    assert score["fu"] == 40
    # Metadata points exclude honba; the browser adds it when scoring with WASM.
    points = (7700 if actor == 0 else 5200) if late_reach else (12000 if actor == 0 else 8000)
    assert score["points"] == str(points)
    assert enriched[-1]["meta"]["results"][0]["score"] == score


@pytest.mark.parametrize("call_before", [False, True])
@pytest.mark.parametrize(
    "call",
    [
        {"type": "pon", "actor": 1, "target": 0, "pai": "N", "consumed": ["N", "N"]},
        {"type": "ankan", "actor": 1, "consumed": ["N"] * 4},
    ],
)
def test_double_riichi_depends_on_calls_before_declaration(call, call_before):
    events = make_riichi_round(4, 3, call=call, call_before=call_before)
    score = get_hora_score(MetadataInjector(events).process())
    assert score is not None
    assert set(score["yaku"]) == {2 if call_before else 18, 32, 33}


def test_double_riichi_resets_between_rounds():
    events = make_riichi_round(4, 3) + make_riichi_round(4, 3, late_reach=True)
    enriched = MetadataInjector(events).process()
    scores = [ev["meta"]["score"] for ev in enriched if ev["type"] == "hora"]
    assert set(scores[0]["yaku"]) == {18, 32, 33}
    assert set(scores[1]["yaku"]) == {2, 32, 33}
