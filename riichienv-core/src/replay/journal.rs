//! Append-only MJAI event journal for static replay and live spectators.
//!
//! The journal intentionally does not interpret game rules. It preserves the
//! original JSON text, indexes completed kyoku boundaries, and exposes cursor-
//! based slices. This keeps replay delivery and spectator-delay policy out of
//! the mutable game engine.

use std::fmt;
use std::io::BufRead;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};

use crate::errors::{RiichiError, RiichiResult};

/// Version of the cursor/delta response envelope. Raw MJAI events retain
/// their own existing wire format inside the envelope.
pub const EVENT_JOURNAL_SCHEMA_VERSION: u16 = 1;

/// Largest integer that has the same exact meaning in Rust JSON and
/// JavaScript's `Number` representation.
const JSON_MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

/// Monotonic event position. A cursor points between events.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct EventCursor(pub usize);

impl EventCursor {
    pub const ZERO: Self = Self(0);

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Metadata available on a `start_kyoku` event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KyokuKey {
    pub bakaze: Option<String>,
    pub kyoku: Option<u8>,
    pub honba: Option<u8>,
}

/// Half-open event range for one completed kyoku.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KyokuSpan {
    pub start: EventCursor,
    pub end: EventCursor,
    pub key: KyokuKey,
}

/// Borrowed cursor response suitable for transport adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EventBatch<'a> {
    pub schema_version: u16,
    pub from: EventCursor,
    pub to: EventCursor,
    pub complete: bool,
    pub events: &'a [String],
}

impl KyokuSpan {
    pub fn len(&self) -> usize {
        self.end.0.saturating_sub(self.start.0)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

#[derive(Debug, Clone)]
struct PendingKyoku {
    start: EventCursor,
    key: KyokuKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    StartGame,
    StartKyoku,
    EndKyoku,
    EndGame,
    Other,
}

#[derive(Debug)]
struct EventHeader {
    kind: EventKind,
    key: KyokuKey,
    start_game_safety: StartGameSafety,
    end_game_safe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartGameSafety {
    Public,
    WithholdUntilComplete,
    Unsafe,
}

/// Canonical append-only event source for replay consumers.
#[derive(Debug, Clone, Default)]
pub struct EventJournal {
    events: Vec<String>,
    completed_kyokus: Vec<KyokuSpan>,
    current_kyoku: Option<PendingKyoku>,
    game_complete: bool,
    /// A valid replay/live stream starts with exactly one `start_game`.
    saw_initial_start_game: bool,
    /// `end_game` may release live-only metadata only after a structurally
    /// complete framed game, never merely because an untrusted stream emitted
    /// an event named `end_game`.
    completion_trusted: bool,
    /// First event observed outside an explicit kyoku boundary. Spectator
    /// views never cross this cursor, even if a malformed stream later emits
    /// `end_game`.
    spectator_ceiling: Option<EventCursor>,
    /// Standard but live-sensitive prelude (currently a validated shuffle
    /// seed). It may be released only after a structurally valid end_game.
    live_spectator_ceiling: Option<EventCursor>,
}

impl EventJournal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a journal from MJAI JSONL text. A final newline is optional.
    /// An unfinished final kyoku remains in-progress and is never promoted to
    /// `completed_kyokus` merely because EOF was reached.
    pub fn from_jsonl(jsonl: &str) -> RiichiResult<Self> {
        let mut journal = Self::new();
        for (line_index, line) in jsonl.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            journal
                .push_json(line)
                .map_err(|error| RiichiError::Parse {
                    input: format!("line {}", line_index + 1),
                    message: error.to_string(),
                })?;
        }
        Ok(journal)
    }

    /// Stream JSONL from any buffered reader without requiring a filesystem
    /// path. Compression and transport remain adapter concerns.
    pub fn from_jsonl_reader(reader: impl BufRead) -> RiichiResult<Self> {
        let mut journal = Self::new();
        for (line_index, line) in reader.lines().enumerate() {
            let line = line.map_err(|error| RiichiError::Serialization {
                message: format!("failed to read JSONL line {}: {error}", line_index + 1),
            })?;
            if line.trim().is_empty() {
                continue;
            }
            journal
                .push_json(&line)
                .map_err(|error| RiichiError::Parse {
                    input: format!("line {}", line_index + 1),
                    message: error.to_string(),
                })?;
        }
        Ok(journal)
    }

    pub fn from_events(events: impl IntoIterator<Item = String>) -> RiichiResult<Self> {
        let mut journal = Self::new();
        for event in events {
            journal.push_json(event)?;
        }
        Ok(journal)
    }

    /// Append one raw MJAI event while preserving its exact JSON text.
    pub fn push_json(&mut self, event: impl AsRef<str>) -> RiichiResult<EventCursor> {
        let event = event.as_ref();
        let header = parse_header(event)?;

        if self.game_complete {
            return Err(RiichiError::InvalidState {
                message: "cannot append an event after end_game".to_string(),
            });
        }
        match header.kind {
            EventKind::StartKyoku if self.current_kyoku.is_some() => {
                return Err(RiichiError::InvalidState {
                    message: "start_kyoku encountered before the current kyoku ended".to_string(),
                });
            }
            EventKind::EndKyoku if self.current_kyoku.is_none() => {
                return Err(RiichiError::InvalidState {
                    message: "end_kyoku encountered without a matching start_kyoku".to_string(),
                });
            }
            EventKind::EndGame if self.current_kyoku.is_some() => {
                return Err(RiichiError::InvalidState {
                    message: "end_game encountered before the current kyoku ended".to_string(),
                });
            }
            _ => {}
        }

        let event_cursor = EventCursor(self.events.len());
        self.events.push(event.to_string());
        match header.kind {
            EventKind::StartGame => {
                if event_cursor == EventCursor::ZERO && !self.saw_initial_start_game {
                    self.saw_initial_start_game = true;
                } else if self.spectator_ceiling.is_none() {
                    // A duplicate or mid-stream start marker makes the feed
                    // ambiguous. Preserve it for replay/debugging, but never
                    // let a later end_game turn it into spectator output.
                    self.spectator_ceiling = Some(event_cursor);
                }

                match header.start_game_safety {
                    StartGameSafety::Public => {}
                    StartGameSafety::WithholdUntilComplete => {
                        if self.live_spectator_ceiling.is_none() {
                            self.live_spectator_ceiling = Some(event_cursor);
                        }
                    }
                    StartGameSafety::Unsafe => {
                        if self.spectator_ceiling.is_none() {
                            self.spectator_ceiling = Some(event_cursor);
                        }
                    }
                }
            }
            EventKind::StartKyoku => {
                if !self.saw_initial_start_game && self.spectator_ceiling.is_none() {
                    // A partial or forged stream has no trustworthy point at
                    // which an end_game marker may declassify hand data.
                    self.spectator_ceiling = Some(event_cursor);
                }
                self.current_kyoku = Some(PendingKyoku {
                    start: event_cursor,
                    key: header.key,
                });
            }
            EventKind::EndKyoku => {
                let pending = self.current_kyoku.take().expect("validated above");
                self.completed_kyokus.push(KyokuSpan {
                    start: pending.start,
                    end: EventCursor(self.events.len()),
                    key: pending.key,
                });
            }
            EventKind::EndGame => {
                if !header.end_game_safe && self.spectator_ceiling.is_none() {
                    // Unknown end_game extensions are retained in the full
                    // journal, but could carry concealed data and are never
                    // copied into the spectator stream.
                    self.spectator_ceiling = Some(event_cursor);
                }
                self.game_complete = true;
                self.completion_trusted = self.saw_initial_start_game
                    && !self.completed_kyokus.is_empty()
                    && self.spectator_ceiling.is_none();
                if !self.completion_trusted && self.spectator_ceiling.is_none() {
                    // Preserve malformed/partial input, but do not expose the
                    // end marker (or any unknown fields carried by it) merely
                    // because its `type` claims the game is complete.
                    self.spectator_ceiling = Some(event_cursor);
                }
            }
            EventKind::Other => {
                if self.current_kyoku.is_none() && self.spectator_ceiling.is_none() {
                    self.spectator_ceiling = Some(event_cursor);
                }
            }
        }
        Ok(EventCursor(self.events.len()))
    }

    /// Append a parsed JSON value using serde's canonical compact rendering.
    pub fn push_value(&mut self, event: &Value) -> RiichiResult<EventCursor> {
        self.push_json(event.to_string())
    }

    pub fn revision(&self) -> EventCursor {
        EventCursor(self.events.len())
    }

    pub fn is_complete(&self) -> bool {
        self.game_complete
    }

    pub fn has_in_progress_kyoku(&self) -> bool {
        self.current_kyoku.is_some()
    }

    pub fn current_kyoku_start(&self) -> Option<EventCursor> {
        self.current_kyoku.as_ref().map(|pending| pending.start)
    }

    pub fn events(&self) -> &[String] {
        &self.events
    }

    pub fn completed_kyokus(&self) -> &[KyokuSpan] {
        &self.completed_kyokus
    }

    pub fn events_since(&self, cursor: EventCursor) -> RiichiResult<&[String]> {
        self.slice(cursor, self.revision())
    }

    pub fn event_batch_since(&self, cursor: EventCursor) -> RiichiResult<EventBatch<'_>> {
        let to = self.revision();
        Ok(EventBatch {
            schema_version: EVENT_JOURNAL_SCHEMA_VERSION,
            from: cursor,
            to,
            complete: self.game_complete,
            events: self.slice(cursor, to)?,
        })
    }

    pub fn events_for_kyoku(&self, index: usize) -> RiichiResult<&[String]> {
        let span = self
            .completed_kyokus
            .get(index)
            .ok_or_else(|| RiichiError::InvalidState {
                message: format!(
                    "completed kyoku index {index} is out of range (count={})",
                    self.completed_kyokus.len()
                ),
            })?;
        self.slice(span.start, span.end)
    }

    pub fn prefix_through_completed_kyoku(&self, index: usize) -> RiichiResult<&[String]> {
        let span = self
            .completed_kyokus
            .get(index)
            .ok_or_else(|| RiichiError::InvalidState {
                message: format!(
                    "completed kyoku index {index} is out of range (count={})",
                    self.completed_kyokus.len()
                ),
            })?;
        self.slice(EventCursor::ZERO, span.end)
    }

    /// End cursor for a spectator feed delayed by whole kyokus.
    ///
    /// With `delay_kyokus = 1`, an in-progress kyoku N is withheld and the
    /// prefix ends exactly at kyoku N-1's `end_kyoku`. Between kyokus, the
    /// newly completed kyoku remains withheld until the next `start_kyoku`.
    /// Once a structurally valid, spectator-safe `end_game` arrives, the full
    /// log is available.
    pub fn spectator_end(&self, delay_kyokus: usize) -> EventCursor {
        let unconstrained = if delay_kyokus == 0 || self.game_complete {
            self.revision()
        } else {
            let current_consumes_delay = usize::from(self.current_kyoku.is_some());
            let completed_to_withhold = delay_kyokus.saturating_sub(current_consumes_delay);
            let visible_completed = self
                .completed_kyokus
                .len()
                .saturating_sub(completed_to_withhold);

            if visible_completed > 0 {
                self.completed_kyokus[visible_completed - 1].end
            } else {
                // Prelude events such as start_game are safe; never include
                // the first withheld start_kyoku.
                self.completed_kyokus
                    .first()
                    .map(|span| span.start)
                    .or_else(|| self.current_kyoku.as_ref().map(|pending| pending.start))
                    .unwrap_or_else(|| self.revision())
            }
        };

        let permanent = self
            .spectator_ceiling
            .map(|ceiling| ceiling.0)
            .unwrap_or(usize::MAX);
        let while_live = if self.completion_trusted {
            usize::MAX
        } else {
            self.live_spectator_ceiling
                .map(|ceiling| ceiling.0)
                .unwrap_or(usize::MAX)
        };
        EventCursor(unconstrained.0.min(permanent).min(while_live))
    }

    pub fn spectator_prefix(&self, delay_kyokus: usize) -> &[String] {
        &self.events[..self.spectator_end(delay_kyokus).0]
    }

    /// Return only newly visible spectator events after `cursor`.
    pub fn spectator_events_since(
        &self,
        cursor: EventCursor,
        delay_kyokus: usize,
    ) -> RiichiResult<&[String]> {
        self.slice(cursor, self.spectator_end(delay_kyokus))
    }

    pub fn spectator_batch_since(
        &self,
        cursor: EventCursor,
        delay_kyokus: usize,
    ) -> RiichiResult<EventBatch<'_>> {
        let to = self.spectator_end(delay_kyokus);
        Ok(EventBatch {
            schema_version: EVENT_JOURNAL_SCHEMA_VERSION,
            from: cursor,
            to,
            complete: self.game_complete,
            events: self.slice(cursor, to)?,
        })
    }

    pub fn to_jsonl(&self) -> String {
        if self.events.is_empty() {
            return String::new();
        }
        let mut jsonl = self.events.join("\n");
        jsonl.push('\n');
        jsonl
    }

    fn slice(&self, start: EventCursor, end: EventCursor) -> RiichiResult<&[String]> {
        if start.0 > end.0 || end.0 > self.events.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "invalid event cursor range {}..{} for revision {}",
                    start.0,
                    end.0,
                    self.events.len()
                ),
            });
        }
        Ok(&self.events[start.0..end.0])
    }
}

fn parse_header(event: &str) -> RiichiResult<EventHeader> {
    let UniqueJsonValue(value) =
        serde_json::from_str::<UniqueJsonValue>(event).map_err(|error| RiichiError::Parse {
            input: "<redacted MJAI event>".to_string(),
            message: format!("invalid MJAI JSON event: {error}"),
        })?;
    let object = value.as_object().ok_or_else(|| RiichiError::Parse {
        input: "<redacted MJAI event>".to_string(),
        message: "MJAI event must be a JSON object".to_string(),
    })?;
    let event_type =
        object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| RiichiError::Parse {
                input: "<redacted MJAI event>".to_string(),
                message: "MJAI event must contain a string 'type' field".to_string(),
            })?;

    let kind = match event_type {
        "start_game" => EventKind::StartGame,
        "start_kyoku" => EventKind::StartKyoku,
        "end_kyoku" => EventKind::EndKyoku,
        "end_game" => EventKind::EndGame,
        _ => EventKind::Other,
    };
    let key = if kind == EventKind::StartKyoku {
        KyokuKey {
            bakaze: object
                .get("bakaze")
                .and_then(Value::as_str)
                .map(str::to_owned),
            kyoku: object.get("kyoku").and_then(json_u8),
            honba: object.get("honba").and_then(json_u8),
        }
    } else {
        KyokuKey::default()
    };
    let start_game_safety = if kind == EventKind::StartGame {
        classify_start_game(object)
    } else {
        StartGameSafety::Public
    };
    let end_game_safe = kind != EventKind::EndGame || classify_end_game(object);
    Ok(EventHeader {
        kind,
        key,
        start_game_safety,
        end_game_safe,
    })
}

/// `serde_json::Value` intentionally accepts duplicate object keys using
/// last-key-wins semantics. That is unsafe when the accepted raw JSON is later
/// returned verbatim to spectators, because validation could inspect a
/// different logical value than a downstream parser. Build a temporary value
/// with duplicate rejection at every object level while retaining the original
/// raw text in the journal itself.
struct UniqueJsonValue(Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(UniqueJsonValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON object key"));
            }
            let UniqueJsonValue(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueJsonValue(Value::Object(values)))
    }
}

/// Only explicitly typed, non-secret start_game metadata can enter a live
/// spectator prefix. Unknown fields remain preserved in the full journal but
/// fail closed because they could carry a wall seed or concealed tiles.
fn classify_start_game(object: &serde_json::Map<String, Value>) -> StartGameSafety {
    let mut has_seed = false;
    for (key, value) in object {
        let valid = match key.as_str() {
            "type" => value.as_str() == Some("start_game"),
            "names" => value
                .as_array()
                .is_some_and(|names| names.iter().all(Value::is_string)),
            "id" => value.is_string() || is_json_safe_integer(value),
            "seed" => {
                has_seed = true;
                value
                    .as_array()
                    .is_some_and(|seed| seed.iter().all(is_json_safe_integer))
            }
            _ => false,
        };
        if !valid {
            return StartGameSafety::Unsafe;
        }
    }
    if has_seed {
        StartGameSafety::WithholdUntilComplete
    } else {
        StartGameSafety::Public
    }
}

/// `end_game` is itself part of the spectator payload, so retain unknown
/// extensions in the canonical journal but expose only established public
/// result metadata. Canonical RiichiEnv logs currently contain only `type`.
fn classify_end_game(object: &serde_json::Map<String, Value>) -> bool {
    object.iter().all(|(key, value)| match key.as_str() {
        "type" => value.as_str() == Some("end_game"),
        "scores" | "ranks" => value.as_array().is_some_and(|values| {
            matches!(values.len(), 3 | 4) && values.iter().all(is_json_safe_integer)
        }),
        _ => false,
    })
}

fn is_json_safe_integer(value: &Value) -> bool {
    let Some(number) = value.as_number() else {
        return false;
    };
    if let Some(value) = number.as_i64() {
        return (-JSON_MAX_SAFE_INTEGER..=JSON_MAX_SAFE_INTEGER).contains(&value);
    }
    if let Some(value) = number.as_u64() {
        return value <= JSON_MAX_SAFE_INTEGER as u64;
    }
    number.as_f64().is_some_and(|value| {
        value.is_finite() && value.fract() == 0.0 && value.abs() <= JSON_MAX_SAFE_INTEGER as f64
    })
}

fn json_u8(value: &Value) -> Option<u8> {
    if !is_json_safe_integer(value) {
        return None;
    }
    let value = value.as_f64()?;
    (0.0..=u8::MAX as f64)
        .contains(&value)
        .then_some(value as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(event_type: &str) -> String {
        format!(r#"{{"type":"{event_type}"}}"#)
    }

    fn start(kyoku: u8) -> String {
        format!(r#"{{"type":"start_kyoku","bakaze":"E","kyoku":{kyoku},"honba":0}}"#)
    }

    #[test]
    fn unfinished_eof_is_not_a_completed_kyoku() {
        let jsonl = [event("start_game"), start(1), event("tsumo")].join("\n");
        let journal = EventJournal::from_jsonl(&jsonl).unwrap();
        assert!(journal.has_in_progress_kyoku());
        assert!(journal.completed_kyokus().is_empty());
        assert_eq!(journal.spectator_prefix(1), &[event("start_game")]);
    }

    #[test]
    fn one_kyoku_delay_advances_only_when_next_kyoku_starts() {
        let mut journal = EventJournal::new();
        journal.push_json(event("start_game")).unwrap();
        journal.push_json(start(1)).unwrap();
        journal.push_json(event("tsumo")).unwrap();
        journal.push_json(event("end_kyoku")).unwrap();

        // Between hands, the just-completed hand remains withheld.
        assert_eq!(journal.spectator_end(1), EventCursor(1));

        journal.push_json(start(2)).unwrap();
        assert_eq!(journal.spectator_end(1), EventCursor(4));
        assert_eq!(
            journal.spectator_prefix(1).last().map(String::as_str),
            Some(r#"{"type":"end_kyoku"}"#)
        );
        assert!(
            !journal
                .spectator_prefix(1)
                .iter()
                .any(|line| line == &start(2))
        );
    }

    #[test]
    fn completed_game_exposes_full_log() {
        let events = vec![
            event("start_game"),
            start(1),
            event("end_kyoku"),
            event("end_game"),
        ];
        let journal = EventJournal::from_events(events.clone()).unwrap();
        assert!(journal.is_complete());
        assert_eq!(journal.spectator_prefix(1), events.as_slice());
    }

    #[test]
    fn cursor_delta_concatenation_matches_visible_prefix() {
        let mut journal = EventJournal::new();
        journal.push_json(event("start_game")).unwrap();
        let cursor = journal.spectator_end(1);
        journal.push_json(start(1)).unwrap();
        journal.push_json(event("end_kyoku")).unwrap();
        journal.push_json(start(2)).unwrap();

        let delta = journal.spectator_events_since(cursor, 1).unwrap();
        assert_eq!(delta, &[start(1), event("end_kyoku")]);

        let batch = journal.spectator_batch_since(cursor, 1).unwrap();
        assert_eq!(batch.schema_version, 1);
        assert_eq!(batch.from, cursor);
        assert_eq!(batch.to, EventCursor(3));
        assert_eq!(batch.events, delta);
        assert!(!batch.complete);
    }

    #[test]
    fn invalid_boundaries_do_not_mutate_the_journal() {
        let mut journal = EventJournal::new();
        journal.push_json(event("start_game")).unwrap();
        let revision = journal.revision();
        assert!(journal.push_json(event("end_kyoku")).is_err());
        assert_eq!(journal.revision(), revision);

        journal.push_json(start(1)).unwrap();
        let revision = journal.revision();
        assert!(journal.push_json(start(2)).is_err());
        assert_eq!(journal.revision(), revision);
    }

    #[test]
    fn raw_unknown_fields_are_preserved() {
        let raw = r#"{"type":"custom","future":{"x":1},"opaque":" value "}"#;
        let mut journal = EventJournal::new();
        journal.push_json(raw).unwrap();
        assert_eq!(journal.events(), &[raw.to_string()]);
        assert!(journal.spectator_prefix(1).is_empty());
    }

    #[test]
    fn unframed_gameplay_event_is_never_exposed_by_spectator_views() {
        let mut journal = EventJournal::new();
        journal.push_json(event("start_game")).unwrap();
        journal
            .push_json(r#"{"type":"tsumo","actor":0,"pai":"1m"}"#)
            .unwrap();
        journal.push_json(event("end_game")).unwrap();

        assert!(journal.is_complete());
        assert_eq!(journal.spectator_end(1), EventCursor(1));
        assert_eq!(journal.spectator_prefix(1), &[event("start_game")]);
    }

    #[test]
    fn start_game_unknown_or_private_fields_fail_closed() {
        for raw in [
            r#"{"type":"start_game","tehais":[["1m"]]}"#,
            r#"{"type":"start_game","future":{"opaque":true}}"#,
            r#"{"type":"start_game","names":[{"tehai":["1m"]}]}"#,
        ] {
            let journal =
                EventJournal::from_events(vec![raw.to_string(), event("end_game")]).unwrap();
            assert_eq!(journal.spectator_end(1), EventCursor::ZERO);
            assert!(journal.spectator_prefix(1).is_empty());
            assert_eq!(journal.events()[0], raw);
        }
    }

    #[test]
    fn standard_start_game_seed_is_withheld_live_but_released_on_completion() {
        let seeded = r#"{"type":"start_game","names":["A","B","C","D"],"seed":[1,2]}"#;
        let mut journal = EventJournal::from_events(vec![seeded.to_string()]).unwrap();
        assert_eq!(journal.spectator_end(1), EventCursor::ZERO);

        journal.push_json(start(1)).unwrap();
        journal.push_json(event("end_kyoku")).unwrap();
        journal.push_json(event("end_game")).unwrap();
        assert_eq!(journal.spectator_end(1), journal.revision());
    }

    #[test]
    fn end_game_without_a_framed_kyoku_does_not_release_live_metadata() {
        let seeded = r#"{"type":"start_game","names":["A","B","C","D"],"seed":[1,2]}"#;
        let journal =
            EventJournal::from_events(vec![seeded.to_string(), event("end_game")]).unwrap();

        assert!(journal.is_complete());
        assert_eq!(journal.spectator_end(1), EventCursor::ZERO);
        assert!(journal.spectator_prefix(1).is_empty());

        let public_start = EventJournal::from_events(vec![
            event("start_game"),
            r#"{"type":"end_game","wall":["PRIVATE"]}"#.to_string(),
        ])
        .unwrap();
        assert_eq!(public_start.spectator_end(1), EventCursor(1));
        assert_eq!(public_start.spectator_prefix(1), &[event("start_game")]);

        let framed_private = EventJournal::from_events(vec![
            event("start_game"),
            start(1),
            event("end_kyoku"),
            r#"{"type":"end_game","wall":["PRIVATE"]}"#.to_string(),
        ])
        .unwrap();
        assert_eq!(framed_private.spectator_end(1), EventCursor(3));
        assert!(!framed_private.spectator_prefix(1)[2].contains("PRIVATE"));

        let public_scores = EventJournal::from_events(vec![
            event("start_game"),
            start(1),
            event("end_kyoku"),
            r#"{"type":"end_game","scores":[30000,25000,25000,20000]}"#.to_string(),
        ])
        .unwrap();
        assert_eq!(public_scores.spectator_end(1), public_scores.revision());
    }

    #[test]
    fn missing_or_duplicate_start_game_never_declassifies_the_ambiguous_suffix() {
        let missing =
            EventJournal::from_events(vec![start(1), event("end_kyoku"), event("end_game")])
                .unwrap();
        assert!(missing.is_complete());
        assert_eq!(missing.spectator_end(1), EventCursor::ZERO);

        let duplicate = EventJournal::from_events(vec![
            event("start_game"),
            event("start_game"),
            start(1),
            event("end_kyoku"),
            event("end_game"),
        ])
        .unwrap();
        assert!(duplicate.is_complete());
        assert_eq!(duplicate.spectator_end(1), EventCursor(1));
        assert_eq!(duplicate.spectator_prefix(1), &[event("start_game")]);
    }

    #[test]
    fn json_number_semantics_match_javascript_safe_integers() {
        let safe_start = r#"{"type":"start_game","names":["A","B","C","D"],"id":1.0,"seed":[2e0]}"#;
        let numeric_kyoku = r#"{"type":"start_kyoku","bakaze":"E","kyoku":1.0,"honba":2e0}"#;
        let journal = EventJournal::from_events(vec![
            safe_start.to_string(),
            numeric_kyoku.to_string(),
            event("end_kyoku"),
            event("end_game"),
        ])
        .unwrap();
        assert_eq!(
            journal.completed_kyokus()[0].key,
            KyokuKey {
                bakaze: Some("E".to_string()),
                kyoku: Some(1),
                honba: Some(2),
            }
        );
        assert_eq!(journal.spectator_end(1), journal.revision());

        let negative_zero = EventJournal::from_events(vec![
            event("start_game"),
            r#"{"type":"start_kyoku","kyoku":-0,"honba":-0.0}"#.to_string(),
            event("end_kyoku"),
        ])
        .unwrap();
        assert_eq!(negative_zero.completed_kyokus()[0].key.kyoku, Some(0));
        assert_eq!(negative_zero.completed_kyokus()[0].key.honba, Some(0));

        let unsafe_integer = r#"{"type":"start_game","id":9007199254740992}"#;
        let journal = EventJournal::from_events(vec![
            unsafe_integer.to_string(),
            start(1),
            event("end_kyoku"),
            event("end_game"),
        ])
        .unwrap();
        assert_eq!(journal.spectator_end(1), EventCursor::ZERO);
    }

    #[test]
    fn duplicate_json_keys_are_rejected_without_mutation() {
        for raw in [
            r#"{"type":"private_wall=1m2m3m","type":"start_game"}"#,
            r#"{"type":"start_game","names":["PRIVATE"],"names":["A","B","C","D"]}"#,
            r#"{"type":"start_game","future":{"secret":1,"secret":2}}"#,
        ] {
            let mut journal = EventJournal::new();
            let error = journal.push_json(raw).unwrap_err();
            assert!(error.to_string().contains("duplicate JSON object key"));
            assert_eq!(journal.revision(), EventCursor::ZERO);
            assert!(journal.events().is_empty());
        }
    }

    #[test]
    fn malformed_input_errors_do_not_echo_private_event_text() {
        let private = r#"{"type":"start_game","TOKEN_SECRET":"PRIVATE","TOKEN_SECRET":"SECRET"}"#;
        let error = EventJournal::from_jsonl(private).unwrap_err().to_string();
        assert!(error.contains("duplicate JSON object key"));
        assert!(!error.contains("TOKEN_SECRET"));
        assert!(!error.contains("PRIVATE"));
        assert!(!error.contains("SECRET"));
    }

    #[test]
    fn externally_constructed_reversed_span_has_safe_length() {
        let span = KyokuSpan {
            start: EventCursor(2),
            end: EventCursor(1),
            key: KyokuKey::default(),
        };
        assert_eq!(span.len(), 0);
        assert!(span.is_empty());
    }
}
