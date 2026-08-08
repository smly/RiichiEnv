import gzip
from pathlib import Path

import pytest

from riichienv import EventJournal

START_GAME = '{"type":"start_game","names":["A","B","C","D"]}'
START_KYOKU_1 = '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0}'
START_KYOKU_2 = '{"type":"start_kyoku","bakaze":"E","kyoku":2,"honba":0}'
TSUMO = '{"type":"tsumo","actor":0,"pai":"1m"}'
END_KYOKU = '{"type":"end_kyoku"}'
END_GAME = '{"type":"end_game"}'


def test_live_spectator_prefix_and_cursor_delta():
    journal = EventJournal()
    assert journal.push_json(START_GAME) == 1
    assert journal.push_json(START_KYOKU_1) == 2
    assert journal.push_json(TSUMO) == 3
    assert journal.push_json(END_KYOKU) == 4

    assert journal.revision == 4
    assert journal.completed_kyokus == [(1, 4, "E", 1, 0)]
    assert journal.events_for_kyoku(0) == [START_KYOKU_1, TSUMO, END_KYOKU]
    assert journal.prefix_through_completed_kyoku(0) == [START_GAME, START_KYOKU_1, TSUMO, END_KYOKU]

    # Between hands, delay=1 still withholds the just-completed hand.
    cursor = journal.spectator_end()
    assert cursor == 1
    assert journal.spectator_prefix() == [START_GAME]

    journal.push_json(START_KYOKU_2)
    new_cursor, delta = journal.spectator_delta(cursor)
    assert new_cursor == 4
    assert delta == [START_KYOKU_1, TSUMO, END_KYOKU]
    assert journal.spectator_events_since(cursor) == delta
    assert journal.spectator_prefix() == [START_GAME, *delta]
    assert START_KYOKU_2 not in journal.spectator_prefix()
    assert journal.spectator_batch_since(cursor) == {
        "schemaVersion": 1,
        "complete": False,
        "from": cursor,
        "to": new_cursor,
        "events": delta,
    }

    with pytest.raises(ValueError, match="invalid event cursor range"):
        journal.spectator_delta(journal.revision)


@pytest.mark.parametrize("gzip_compressed", [False, True])
def test_from_jsonl_path_supports_plain_and_gzip_magic_bytes(tmp_path, gzip_compressed):
    events = [START_GAME, START_KYOKU_1, END_KYOKU, END_GAME]
    jsonl = "\n".join(events) + "\n"
    # The gzip case deliberately has no .gz suffix: detection is by magic bytes.
    path = tmp_path / "journal.jsonl"
    if gzip_compressed:
        with gzip.open(path, "wt", encoding="utf-8") as output:
            output.write(jsonl)
    else:
        path.write_text(jsonl, encoding="utf-8")

    journal = EventJournal.from_jsonl(path)

    assert journal.events == events
    assert journal.is_complete
    assert not journal.has_in_progress_kyoku
    assert journal.spectator_end() == len(events)
    assert journal.spectator_prefix() == events
    assert journal.to_jsonl() == jsonl


def test_from_jsonl_reads_all_concatenated_gzip_members(tmp_path):
    path = tmp_path / "concatenated.jsonl.gz"
    path.write_bytes(
        gzip.compress(f"{START_GAME}\n".encode())
        + gzip.compress(f"{START_KYOKU_1}\n{END_KYOKU}\n{END_GAME}\n".encode())
    )

    journal = EventJournal.from_jsonl(path)

    assert journal.events == [START_GAME, START_KYOKU_1, END_KYOKU, END_GAME]
    assert journal.is_complete


def test_builders_preserve_raw_events_and_invalid_append_is_non_mutating():
    raw = '{"type":"custom","future":{"x":1},"opaque":" value "}'
    journal = EventJournal.from_jsonl_text(f"{START_GAME}\n{raw}")

    assert journal.events == [START_GAME, raw]
    assert journal.events_since(1) == [raw]
    assert len(journal) == 2

    from_events = EventJournal.from_events([START_GAME, raw])
    assert from_events.events == journal.events

    revision = journal.revision
    with pytest.raises(ValueError, match="without a matching start_kyoku"):
        journal.push_json(END_KYOKU)
    assert journal.revision == revision
    assert journal.events == [START_GAME, raw]


def test_committed_full_game_fixture_has_stable_kyoku_boundaries():
    path = Path(__file__).parent / "data" / "126_204_0_mjai.jsonl"
    journal = EventJournal.from_jsonl(path)

    assert journal.revision == 1383
    assert len(journal.completed_kyokus) == 12
    assert journal.completed_kyokus[0] == (1, 103, "E", 1, 0)
    assert journal.completed_kyokus[-1] == (1292, 1382, "S", 4, 1)
    assert journal.is_complete
    assert journal.spectator_end() == journal.revision


def test_spectator_view_fails_closed_for_unframed_private_event():
    journal = EventJournal.from_events([START_GAME, TSUMO, END_GAME])

    assert journal.is_complete
    assert journal.spectator_end() == 1
    assert journal.spectator_prefix() == [START_GAME]


@pytest.mark.parametrize(
    "unsafe_start",
    [
        '{"type":"start_game","tehais":[["1m"]]}',
        '{"type":"start_game","future":{"opaque":true}}',
        '{"type":"start_game","names":[{"tehai":["1m"]}]}',
    ],
)
def test_spectator_view_never_trusts_unknown_start_game_fields(unsafe_start):
    journal = EventJournal.from_events([unsafe_start, END_GAME])

    assert journal.events[0] == unsafe_start
    assert journal.spectator_end() == 0
    assert journal.spectator_prefix() == []


def test_spectator_view_withholds_standard_seed_until_game_completion():
    seeded_start = '{"type":"start_game","names":["A","B","C","D"],"seed":[1,2]}'
    journal = EventJournal.from_events([seeded_start])
    assert journal.spectator_end() == 0

    for event in (START_KYOKU_1, END_KYOKU, END_GAME):
        journal.push_json(event)

    assert journal.is_complete
    assert journal.spectator_end() == journal.revision
    assert journal.spectator_prefix()[0] == seeded_start


def test_end_game_without_a_framed_kyoku_does_not_release_seed():
    seeded_start = '{"type":"start_game","names":["A","B","C","D"],"seed":[1,2]}'
    journal = EventJournal.from_events([seeded_start, END_GAME])

    assert journal.is_complete
    assert journal.spectator_end() == 0
    assert journal.spectator_prefix() == []

    public_start = EventJournal.from_events([START_GAME, '{"type":"end_game","wall":["PRIVATE"]}'])
    assert public_start.spectator_end() == 1
    assert public_start.spectator_prefix() == [START_GAME]

    framed_private = EventJournal.from_events(
        [
            START_GAME,
            START_KYOKU_1,
            END_KYOKU,
            '{"type":"end_game","wall":["PRIVATE"]}',
        ]
    )
    assert framed_private.spectator_end() == 3
    assert "PRIVATE" not in "\n".join(framed_private.spectator_prefix())

    public_scores = EventJournal.from_events(
        [
            START_GAME,
            START_KYOKU_1,
            END_KYOKU,
            '{"type":"end_game","scores":[30000,25000,25000,20000]}',
        ]
    )
    assert public_scores.spectator_end() == public_scores.revision


def test_missing_or_duplicate_start_game_never_declassifies_ambiguous_events():
    missing = EventJournal.from_events([START_KYOKU_1, END_KYOKU, END_GAME])
    assert missing.is_complete
    assert missing.spectator_end() == 0

    duplicate = EventJournal.from_events([START_GAME, START_GAME, START_KYOKU_1, END_KYOKU, END_GAME])
    assert duplicate.is_complete
    assert duplicate.spectator_end() == 1
    assert duplicate.spectator_prefix() == [START_GAME]


def test_json_number_semantics_match_javascript_safe_integers():
    safe_start = '{"type":"start_game","names":["A","B","C","D"],"id":1.0,"seed":[2e0]}'
    numeric_kyoku = '{"type":"start_kyoku","bakaze":"E","kyoku":1.0,"honba":2e0}'
    safe = EventJournal.from_events([safe_start, numeric_kyoku, END_KYOKU, END_GAME])
    assert safe.completed_kyokus == [(1, 3, "E", 1, 2)]
    assert safe.spectator_end() == safe.revision

    unsafe = EventJournal.from_events(
        [
            '{"type":"start_game","id":9007199254740992}',
            START_KYOKU_1,
            END_KYOKU,
            END_GAME,
        ]
    )
    assert unsafe.spectator_end() == 0


@pytest.mark.parametrize(
    "duplicate",
    [
        '{"type":"private_wall=1m2m3m","type":"start_game"}',
        '{"type":"start_game","names":["PRIVATE"],"names":["A","B","C","D"]}',
        '{"type":"start_game","future":{"secret":1,"secret":2}}',
    ],
)
def test_duplicate_json_keys_are_rejected_without_mutation(duplicate):
    journal = EventJournal()

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        journal.push_json(duplicate)

    assert journal.revision == 0
    assert journal.events == []


def test_duplicate_key_error_does_not_echo_private_key_or_payload():
    private = '{"type":"start_game","TOKEN_SECRET":"PRIVATE","TOKEN_SECRET":"SECRET"}'

    with pytest.raises(ValueError, match="duplicate JSON object key") as caught:
        EventJournal().push_json(private)

    assert "TOKEN_SECRET" not in str(caught.value)
    assert "PRIVATE" not in str(caught.value)
    assert "SECRET" not in str(caught.value)
