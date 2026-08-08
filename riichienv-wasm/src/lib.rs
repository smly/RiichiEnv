use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use wasm_bindgen::prelude::*;

use riichienv_core::action::Phase;
use riichienv_core::drev::{self, DrevInput, encode_drev};
use riichienv_core::engine::{
    Decision, EngineConfig, EngineStepError, EventLogPolicy, GameEngine as CoreGameEngine,
    GameMode, GameSnapshot,
};
use riichienv_core::hand_evaluator::HandEvaluator;
use riichienv_core::hand_evaluator_3p::HandEvaluator3P;
use riichienv_core::parser::{mjai_to_tid, tid_to_mjai};
use riichienv_core::replay::{
    EVENT_JOURNAL_SCHEMA_VERSION, EventCursor, EventJournal as CoreEventJournal,
};
use riichienv_core::sp::{self, SpInput, encode_sp};
use riichienv_core::types::{Conditions, Meld, MeldType, TILE_MAX, Wind};
use riichienv_core::{score, yaku};

const SPECTATOR_DELAY_KYOKUS: usize = 1;

#[wasm_bindgen(typescript_custom_section)]
const ENGINE_TYPES: &str = r#"
export interface JournalDeltaWire {
    readonly schemaVersion: 1;
    readonly complete: boolean;
    readonly from: number;
    readonly to: number;
    readonly events: readonly string[];
}

export interface CompletedKyokuSpan {
    readonly start: number;
    readonly end: number;
    readonly key: {
        readonly bakaze: string | null;
        readonly kyoku: number | null;
        readonly honba: number | null;
    };
}

export interface EngineDecision {
    readonly playerId: number;
    readonly legalActionIds: readonly number[];
    readonly actionMask: readonly number[];
    readonly observationBase64: string;
    readonly baseShape: readonly [number, number];
    readonly extendedShape: readonly [number, number];
}

export interface GameSnapshotWire {
    readonly mode: string;
    readonly numPlayers: number;
    readonly phase: 'WaitAct' | 'WaitResponse';
    readonly currentPlayer: number;
    readonly activePlayers: readonly number[];
    readonly scores: readonly number[];
    readonly roundWind: number;
    readonly kyoku: number;
    readonly honba: number;
    readonly riichiSticks: number;
    readonly wallTilesRemaining: number;
    readonly done: boolean;
}

export interface EngineErrorWire {
    readonly kind: 'invalidActionBatch' | 'illegalAction' | 'state';
    readonly playerId?: number;
    readonly message: string;
}

export interface ActionIdInput {
    readonly playerId: number;
    readonly actionId: number;
}

export interface StepActionIdsResult {
    readonly snapshot: GameSnapshotWire;
    readonly events: readonly string[];
    readonly error: EngineErrorWire | null;
    readonly nextDecisions: readonly EngineDecision[];
}
"#;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct RawEventDelta<'a> {
    schema_version: u16,
    complete: bool,
    from: usize,
    to: usize,
    events: &'a [String],
}

fn to_js_value<T: serde::Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true);
    value
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("Serialization error: {error}")))
}

fn journal_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Append-only MJAI event journal exposed to JavaScript.
///
/// Events cross the WASM boundary as raw JSON strings so unknown fields and
/// their original representation remain intact. Spectator methods always use
/// a one-kyoku delay: an in-progress kyoku is withheld from `start_kyoku`, and
/// a structurally framed `end_game` releases the complete log.
#[wasm_bindgen(js_name = EventJournal)]
pub struct WasmEventJournal {
    inner: CoreEventJournal,
}

#[wasm_bindgen]
impl WasmEventJournal {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: CoreEventJournal::new(),
        }
    }

    /// Build a journal from newline-delimited raw MJAI JSON events.
    #[wasm_bindgen(js_name = fromJsonl)]
    pub fn from_jsonl(jsonl: &str) -> Result<WasmEventJournal, JsValue> {
        CoreEventJournal::from_jsonl(jsonl)
            .map(|inner| Self { inner })
            .map_err(journal_error)
    }

    /// Append one raw MJAI JSON object and return the new event cursor.
    #[wasm_bindgen(js_name = pushRawJson)]
    pub fn push_raw_json(&mut self, event_json: &str) -> Result<usize, JsValue> {
        self.inner
            .push_json(event_json)
            .map(EventCursor::index)
            .map_err(journal_error)
    }

    /// Current append revision. Cursors point between events.
    #[wasm_bindgen(getter)]
    pub fn revision(&self) -> usize {
        self.inner.revision().index()
    }

    #[wasm_bindgen(getter, js_name = complete)]
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    #[wasm_bindgen(getter, js_name = hasInProgressKyoku)]
    pub fn has_in_progress_kyoku(&self) -> bool {
        self.inner.has_in_progress_kyoku()
    }

    /// All raw events in append order.
    #[wasm_bindgen(js_name = rawEvents, unchecked_return_type = "readonly string[]")]
    pub fn raw_events(&self) -> Result<JsValue, JsValue> {
        to_js_value(self.inner.events())
    }

    /// Full-log cursor delta `{ from, to, events }`.
    #[wasm_bindgen(js_name = eventsSince, unchecked_return_type = "JournalDeltaWire")]
    pub fn events_since(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] cursor: JsValue,
    ) -> Result<JsValue, JsValue> {
        let cursor = parse_js_usize(&cursor, "cursor").map_err(journal_error)?;
        let from = EventCursor(cursor);
        let events = self.inner.events_since(from).map_err(journal_error)?;
        to_js_value(&RawEventDelta {
            schema_version: EVENT_JOURNAL_SCHEMA_VERSION,
            complete: self.inner.is_complete(),
            from: cursor,
            to: self.inner.revision().index(),
            events,
        })
    }

    /// Completed kyoku ranges as half-open `{ start, end, key }` spans.
    #[wasm_bindgen(
        js_name = completedKyokuSpans,
        unchecked_return_type = "readonly CompletedKyokuSpan[]"
    )]
    pub fn completed_kyoku_spans(&self) -> Result<JsValue, JsValue> {
        to_js_value(self.inner.completed_kyokus())
    }

    /// Safe end cursor for a one-kyoku-delayed spectator.
    #[wasm_bindgen(getter, js_name = spectatorCursor)]
    pub fn spectator_cursor(&self) -> usize {
        self.inner.spectator_end(SPECTATOR_DELAY_KYOKUS).index()
    }

    /// Schema-v1 raw safe prefix for a one-kyoku-delayed spectator.
    #[wasm_bindgen(
        js_name = spectatorSafePrefix,
        unchecked_return_type = "JournalDeltaWire"
    )]
    pub fn spectator_safe_prefix(&self) -> Result<JsValue, JsValue> {
        let to = self.spectator_cursor();
        to_js_value(&RawEventDelta {
            schema_version: EVENT_JOURNAL_SCHEMA_VERSION,
            complete: self.inner.is_complete(),
            from: 0,
            to,
            events: self.inner.spectator_prefix(SPECTATOR_DELAY_KYOKUS),
        })
    }

    /// Newly visible one-kyoku-delayed spectator events.
    #[wasm_bindgen(
        js_name = spectatorEventsSince,
        unchecked_return_type = "JournalDeltaWire"
    )]
    pub fn spectator_events_since(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] cursor: JsValue,
    ) -> Result<JsValue, JsValue> {
        let cursor = parse_js_usize(&cursor, "cursor").map_err(journal_error)?;
        let events = self
            .inner
            .spectator_events_since(EventCursor(cursor), SPECTATOR_DELAY_KYOKUS)
            .map_err(journal_error)?;
        to_js_value(&RawEventDelta {
            schema_version: EVENT_JOURNAL_SCHEMA_VERSION,
            complete: self.inner.is_complete(),
            from: cursor,
            to: self.spectator_cursor(),
            events,
        })
    }

    #[wasm_bindgen(js_name = toJsonl)]
    pub fn to_jsonl(&self) -> String {
        self.inner.to_jsonl()
    }
}

impl Default for WasmEventJournal {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ActionIdInput {
    player_id: u8,
    action_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct DecisionJs {
    player_id: u8,
    legal_action_ids: Vec<usize>,
    action_mask: Vec<u8>,
    observation_base64: String,
    base_shape: [usize; 2],
    extended_shape: [usize; 2],
}

impl DecisionJs {
    fn from_decision(decision: &Decision) -> Result<Self, String> {
        let observation = &decision.observation;
        let base_shape = observation.base_feature_shape();
        let extended_shape = observation.extended_feature_shape();
        Ok(Self {
            player_id: decision.player_id,
            legal_action_ids: observation
                .legal_action_ids()
                .map_err(|error| error.to_string())?,
            action_mask: observation
                .action_mask()
                .map_err(|error| error.to_string())?,
            observation_base64: observation
                .serialize_to_base64()
                .map_err(|error| error.to_string())?,
            base_shape: [base_shape.0, base_shape.1],
            extended_shape: [extended_shape.0, extended_shape.1],
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineErrorJs {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    player_id: Option<u8>,
    message: String,
}

impl EngineErrorJs {
    fn invalid_action_batch(message: impl Into<String>, player_id: Option<u8>) -> Self {
        Self {
            kind: "invalidActionBatch".to_string(),
            player_id,
            message: message.into(),
        }
    }
}

impl From<EngineStepError> for EngineErrorJs {
    fn from(error: EngineStepError) -> Self {
        match error {
            EngineStepError::IllegalAction { player_id, message } => Self {
                kind: "illegalAction".to_string(),
                player_id: Some(player_id),
                message,
            },
            EngineStepError::State { message } => Self {
                kind: "state".to_string(),
                player_id: None,
                message,
            },
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct StepActionIdsResultJs {
    snapshot: GameSnapshotJs,
    events: Vec<String>,
    error: Option<EngineErrorJs>,
    next_decisions: Vec<DecisionJs>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GameSnapshotJs {
    mode: &'static str,
    num_players: u8,
    phase: &'static str,
    current_player: u8,
    active_players: Vec<u8>,
    scores: Vec<i32>,
    round_wind: u8,
    kyoku: u8,
    honba: u8,
    riichi_sticks: u32,
    wall_tiles_remaining: usize,
    done: bool,
}

impl From<&GameSnapshot> for GameSnapshotJs {
    fn from(snapshot: &GameSnapshot) -> Self {
        Self {
            mode: snapshot.mode.as_str(),
            num_players: snapshot.num_players,
            phase: match snapshot.phase {
                Phase::WaitAct => "WaitAct",
                Phase::WaitResponse => "WaitResponse",
            },
            current_player: snapshot.current_player,
            active_players: snapshot.active_players.clone(),
            scores: snapshot.scores.clone(),
            round_wind: snapshot.round_wind,
            kyoku: snapshot.kyoku,
            honba: snapshot.honba,
            riichi_sticks: snapshot.riichi_sticks,
            wall_tiles_remaining: snapshot.wall_tiles_remaining,
            done: snapshot.done,
        }
    }
}

/// Variant-independent game engine for browser and JavaScript runtimes.
///
/// JavaScript submits model action IDs only. Each ID is resolved strictly
/// against the cached pending observation that produced the corresponding
/// decision; callers never construct raw Rust `Action` values.
#[wasm_bindgen(js_name = GameEngine)]
pub struct WasmGameEngine {
    inner: CoreGameEngine,
    pending: Vec<Decision>,
}

#[wasm_bindgen]
impl WasmGameEngine {
    /// Construct an engine from a canonical mode string such as
    /// `4p-red-single` or `3p-red-half`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        mode: &str,
        #[wasm_bindgen(unchecked_optional_param_type = "number | null")] seed: Option<JsValue>,
        #[wasm_bindgen(unchecked_optional_param_type = "boolean | null")] event_log: Option<
            JsValue,
        >,
    ) -> Result<WasmGameEngine, JsValue> {
        let seed = parse_optional_js_number(seed, "seed").map_err(journal_error)?;
        let event_log = parse_optional_js_bool(event_log, "eventLog").map_err(journal_error)?;
        Self::build(mode, seed, event_log).map_err(journal_error)
    }

    /// Pending player decisions. Repeated calls do not consume observation
    /// event cursors or replace the observations used by `stepActionIds`.
    #[wasm_bindgen(unchecked_return_type = "readonly EngineDecision[]")]
    pub fn decisions(&self) -> Result<JsValue, JsValue> {
        to_js_value(&decision_views(&self.pending).map_err(journal_error)?)
    }

    /// Base-v0 features for one pending player as a `Float32Array`.
    #[wasm_bindgen(js_name = baseFeatures)]
    pub fn base_features(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] player_id: JsValue,
    ) -> Result<Vec<f32>, JsValue> {
        let player_id = parse_js_u8(&player_id, "playerId").map_err(journal_error)?;
        self.pending_decision(player_id)
            .and_then(|decision| {
                decision
                    .observation
                    .encode_base_features()
                    .map_err(|error| error.to_string())
            })
            .map_err(journal_error)
    }

    /// Extended-v0 features for one pending player as a `Float32Array`.
    #[wasm_bindgen(js_name = extendedFeatures)]
    pub fn extended_features(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] player_id: JsValue,
    ) -> Result<Vec<f32>, JsValue> {
        let player_id = parse_js_u8(&player_id, "playerId").map_err(journal_error)?;
        self.pending_decision(player_id)
            .and_then(|decision| {
                decision
                    .observation
                    .encode_extended_features()
                    .map_err(|error| error.to_string())
            })
            .map_err(journal_error)
    }

    /// Dense action mask for one pending player as a `Uint8Array`.
    #[wasm_bindgen(js_name = actionMask)]
    pub fn action_mask(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] player_id: JsValue,
    ) -> Result<Vec<u8>, JsValue> {
        let player_id = parse_js_u8(&player_id, "playerId").map_err(journal_error)?;
        self.pending_decision(player_id)
            .and_then(|decision| {
                decision
                    .observation
                    .action_mask()
                    .map_err(|error| error.to_string())
            })
            .map_err(journal_error)
    }

    /// Step all pending players by action ID.
    #[wasm_bindgen(js_name = stepActionIds, unchecked_return_type = "StepActionIdsResult")]
    pub fn step_action_ids(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "readonly ActionIdInput[]")] actions: JsValue,
    ) -> Result<JsValue, JsValue> {
        let inputs: Vec<ActionIdInput> = serde_wasm_bindgen::from_value(actions)
            .map_err(|error| JsValue::from_str(&format!("Invalid action ID batch: {error}")))?;
        let outcome = self.step_inputs(inputs).map_err(journal_error)?;
        to_js_value(&outcome)
    }

    #[wasm_bindgen(unchecked_return_type = "GameSnapshotWire")]
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        to_js_value(&GameSnapshotJs::from(&self.inner.snapshot()))
    }

    /// Complete retained MJAI log as raw JSON event strings.
    #[wasm_bindgen(js_name = mjaiLog, unchecked_return_type = "readonly string[]")]
    pub fn mjai_log(&self) -> Result<JsValue, JsValue> {
        to_js_value(self.inner.mjai_log())
    }

    /// Current engine log indexed as an EventJournal.
    #[wasm_bindgen(js_name = eventJournal)]
    pub fn event_journal(&self) -> Result<WasmEventJournal, JsValue> {
        self.inner
            .event_journal()
            .map(|inner| WasmEventJournal { inner })
            .map_err(journal_error)
    }
}

impl WasmGameEngine {
    fn build(mode: &str, seed: Option<f64>, event_log: Option<bool>) -> Result<Self, String> {
        let mode = GameMode::from_str(mode).map_err(|error| error.to_string())?;
        let seed = parse_js_seed(seed)?;
        let mut config = EngineConfig::new(mode).with_event_log(if event_log.unwrap_or(true) {
            EventLogPolicy::Full
        } else {
            EventLogPolicy::Off
        });
        if let Some(seed) = seed {
            config = config.with_seed(seed);
        }

        let mut inner = CoreGameEngine::new(config).map_err(|error| error.to_string())?;
        let pending = inner.decisions();
        Ok(Self { inner, pending })
    }

    fn step_inputs(&mut self, inputs: Vec<ActionIdInput>) -> Result<StepActionIdsResultJs, String> {
        let mut seen = HashSet::with_capacity(inputs.len());
        let pending_by_player: HashMap<u8, &Decision> = self
            .pending
            .iter()
            .map(|decision| (decision.player_id, decision))
            .collect();

        let validation_error = if self.pending.is_empty() {
            Some(EngineErrorJs::invalid_action_batch(
                "the engine has no pending decisions",
                None,
            ))
        } else if inputs.len() != self.pending.len() {
            Some(EngineErrorJs::invalid_action_batch(
                format!(
                    "action batch contains {} entries, expected {}",
                    inputs.len(),
                    self.pending.len()
                ),
                None,
            ))
        } else {
            None
        };

        if let Some(error) = validation_error {
            return self.validation_outcome(error);
        }

        let mut actions = HashMap::with_capacity(inputs.len());
        for input in inputs {
            if !seen.insert(input.player_id) {
                return self.validation_outcome(EngineErrorJs::invalid_action_batch(
                    format!("duplicate action for player {}", input.player_id),
                    Some(input.player_id),
                ));
            }
            let Some(decision) = pending_by_player.get(&input.player_id) else {
                return self.validation_outcome(EngineErrorJs::invalid_action_batch(
                    format!("player {} has no pending decision", input.player_id),
                    Some(input.player_id),
                ));
            };
            let action = match decision.observation.select_action(input.action_id) {
                Ok(action) => action,
                Err(error) => {
                    return self.validation_outcome(EngineErrorJs::invalid_action_batch(
                        error.to_string(),
                        Some(input.player_id),
                    ));
                }
            };
            actions.insert(input.player_id, action);
        }

        if let Some(missing) = self
            .pending
            .iter()
            .find(|decision| !seen.contains(&decision.player_id))
        {
            return self.validation_outcome(EngineErrorJs::invalid_action_batch(
                format!("missing action for player {}", missing.player_id),
                Some(missing.player_id),
            ));
        }

        let outcome = self.inner.step(&actions);
        self.pending = outcome.decisions;
        Ok(StepActionIdsResultJs {
            snapshot: GameSnapshotJs::from(&outcome.snapshot),
            events: outcome.events,
            error: outcome.error.map(EngineErrorJs::from),
            next_decisions: decision_views(&self.pending)?,
        })
    }

    fn pending_decision(&self, player_id: u8) -> Result<&Decision, String> {
        self.pending
            .iter()
            .find(|decision| decision.player_id == player_id)
            .ok_or_else(|| format!("player {player_id} has no pending decision"))
    }

    fn validation_outcome(&self, error: EngineErrorJs) -> Result<StepActionIdsResultJs, String> {
        Ok(StepActionIdsResultJs {
            snapshot: GameSnapshotJs::from(&self.inner.snapshot()),
            events: Vec::new(),
            error: Some(error),
            next_decisions: decision_views(&self.pending)?,
        })
    }
}

fn decision_views(decisions: &[Decision]) -> Result<Vec<DecisionJs>, String> {
    decisions.iter().map(DecisionJs::from_decision).collect()
}

fn parse_js_seed(seed: Option<f64>) -> Result<Option<u64>, String> {
    const JS_MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    let Some(seed) = seed else {
        return Ok(None);
    };
    if !seed.is_finite() || !(0.0..=JS_MAX_SAFE_INTEGER).contains(&seed) || seed.fract() != 0.0 {
        return Err("seed must be a non-negative JavaScript safe integer".to_string());
    }
    Ok(Some(seed as u64))
}

fn parse_optional_js_number(value: Option<JsValue>, name: &str) -> Result<Option<f64>, String> {
    value
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| format!("{name} must be a JavaScript number"))
        })
        .transpose()
}

fn parse_optional_js_bool(value: Option<JsValue>, name: &str) -> Result<Option<bool>, String> {
    value
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| format!("{name} must be a JavaScript boolean"))
        })
        .transpose()
}

fn parse_js_usize(value: &JsValue, name: &str) -> Result<usize, String> {
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{name} must be a JavaScript number"))?;
    parse_usize_number(value, name)
}

fn parse_usize_number(value: f64, name: &str) -> Result<usize, String> {
    const JS_MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    let max = JS_MAX_SAFE_INTEGER.min(usize::MAX as f64);
    if !value.is_finite() || !(0.0..=max).contains(&value) || value.fract() != 0.0 {
        return Err(format!(
            "{name} must be a non-negative integer no greater than {max:.0}"
        ));
    }
    Ok(value as usize)
}

fn parse_js_u8(value: &JsValue, name: &str) -> Result<u8, String> {
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{name} must be a JavaScript number"))?;
    parse_u8_number(value, name)
}

fn parse_u8_number(value: f64, name: &str) -> Result<u8, String> {
    if !value.is_finite() || !(0.0..=u8::MAX as f64).contains(&value) || value.fract() != 0.0 {
        return Err(format!(
            "{name} must be an integer between 0 and {}",
            u8::MAX
        ));
    }
    Ok(value as u8)
}

/// Input format for melds passed from JavaScript.
#[derive(serde::Deserialize)]
struct MeldInput {
    meld_type: String,
    tiles: Vec<u8>,
}

impl MeldInput {
    fn to_meld(&self) -> Meld {
        let meld_type = match self.meld_type.as_str() {
            "chi" => MeldType::Chi,
            "pon" => MeldType::Pon,
            "daiminkan" => MeldType::Daiminkan,
            "ankan" => MeldType::Ankan,
            "kakan" => MeldType::Kakan,
            _ => MeldType::Chi,
        };
        Meld::new(
            meld_type,
            self.tiles.clone(),
            meld_type != MeldType::Ankan,
            -1,
            None,
        )
    }
}

/// Input format for scoring conditions passed from JavaScript.
#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct ConditionsInput {
    tsumo: bool,
    riichi: bool,
    double_riichi: bool,
    ippatsu: bool,
    haitei: bool,
    houtei: bool,
    rinshan: bool,
    chankan: bool,
    tsumo_first_turn: bool,
    player_wind: u8,
    round_wind: u8,
    honba: u32,
    kita_count: u8,
    is_sanma: bool,
    is_kokushi_musou_13machi_double: bool,
    is_suuankou_tanki_double: bool,
    is_junsei_chuurenpoutou_double: bool,
    is_daisuushii_double: bool,
}

impl ConditionsInput {
    fn to_conditions(&self) -> Conditions {
        Conditions {
            tsumo: self.tsumo,
            riichi: self.riichi,
            double_riichi: self.double_riichi,
            ippatsu: self.ippatsu,
            haitei: self.haitei,
            houtei: self.houtei,
            rinshan: self.rinshan,
            chankan: self.chankan,
            tsumo_first_turn: self.tsumo_first_turn,
            player_wind: Wind::from(self.player_wind),
            round_wind: Wind::from(self.round_wind),
            riichi_sticks: 0,
            honba: self.honba,
            kita_count: self.kita_count,
            is_sanma: self.is_sanma,
            num_players: if self.is_sanma { 3 } else { 4 },
        }
    }
}

/// Output format for scoring results returned to JavaScript.
#[derive(serde::Serialize)]
struct ScoreResult {
    is_win: bool,
    yakuman: bool,
    han: u32,
    fu: u32,
    ron_agari: u32,
    tsumo_agari_oya: u32,
    tsumo_agari_ko: u32,
    yaku: Vec<u32>,
}

fn apply_double_yakuman_rules(score: &mut ScoreResult, conditions: &ConditionsInput) {
    if !score.yakuman || score.han <= 13 {
        return;
    }

    let mut cap = 0u32;
    for &y in &score.yaku {
        match y {
            yaku::ID_JUNSEI_CHUUREN if !conditions.is_junsei_chuurenpoutou_double => cap += 13,
            yaku::ID_SUANKO_TANKI if !conditions.is_suuankou_tanki_double => cap += 13,
            yaku::ID_KOKUSHI_13 if !conditions.is_kokushi_musou_13machi_double => cap += 13,
            yaku::ID_DAISUUSHI if !conditions.is_daisuushii_double => cap += 13,
            _ => {}
        }
    }

    if cap == 0 {
        return;
    }

    score.han = score.han.saturating_sub(cap).max(13);
    let score_res = score::calculate_score(
        score.han as u8,
        0,
        conditions.player_wind % 4 == Wind::East as u8,
        conditions.tsumo,
        conditions.honba,
        if conditions.is_sanma { 3 } else { 4 },
    );
    score.ron_agari = score_res.pay_ron;
    score.tsumo_agari_oya = score_res.pay_tsumo_oya;
    score.tsumo_agari_ko = score_res.pay_tsumo_ko;
}

/// Calculate wait tiles (machi) for a given hand.
///
/// Input: JSON array of tile IDs (136-encoding) for hand tiles.
/// Melds: JSON array of meld objects (optional).
/// Returns: JSON array of wait tile types (34-encoding).
#[wasm_bindgen]
pub fn calc_waits(tiles_json: &str, melds_json: &str) -> Result<JsValue, JsValue> {
    let tiles: Vec<u8> = serde_json::from_str(tiles_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse tiles: {}", e)))?;

    let meld_inputs: Vec<MeldInput> = serde_json::from_str(melds_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse melds: {}", e)))?;

    let melds: Vec<Meld> = meld_inputs.iter().map(|m| m.to_meld()).collect();

    let evaluator = HandEvaluator::new(tiles, melds);
    let waits = evaluator.get_waits_u8();

    serde_wasm_bindgen::to_value(&waits)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Calculate score for a winning hand.
///
/// Input: hand tiles (136-encoding), win tile, dora indicators, and conditions.
/// Returns: score result as JSON.
#[wasm_bindgen]
pub fn calc_score(
    tiles_json: &str,
    melds_json: &str,
    win_tile: u8,
    dora_json: &str,
    ura_json: &str,
    conditions_json: &str,
) -> Result<JsValue, JsValue> {
    let tiles: Vec<u8> = serde_json::from_str(tiles_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse tiles: {}", e)))?;

    let meld_inputs: Vec<MeldInput> = serde_json::from_str(melds_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse melds: {}", e)))?;

    let dora_indicators: Vec<u8> = serde_json::from_str(dora_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse dora: {}", e)))?;

    let ura_indicators: Vec<u8> = serde_json::from_str(ura_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse ura: {}", e)))?;

    let cond_input: ConditionsInput = serde_json::from_str(conditions_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse conditions: {}", e)))?;

    let melds: Vec<Meld> = meld_inputs.iter().map(|m| m.to_meld()).collect();
    let conditions = cond_input.to_conditions();

    let result = if cond_input.is_sanma {
        let evaluator = HandEvaluator3P::new(tiles, melds);
        evaluator.calc(win_tile, dora_indicators, ura_indicators, Some(conditions))
    } else {
        let evaluator = HandEvaluator::new(tiles, melds);
        evaluator.calc(win_tile, dora_indicators, ura_indicators, Some(conditions))
    };

    let mut score = ScoreResult {
        is_win: result.is_win,
        yakuman: result.yakuman,
        han: result.han,
        fu: result.fu,
        ron_agari: result.ron_agari,
        tsumo_agari_oya: result.tsumo_agari_oya,
        tsumo_agari_ko: result.tsumo_agari_ko,
        yaku: result.yaku,
    };
    apply_double_yakuman_rules(&mut score, &cond_input);

    serde_wasm_bindgen::to_value(&score)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Convert MJAI tile string to 136-encoding tile ID.
#[wasm_bindgen]
pub fn mjai_to_tile_id(mjai: &str) -> Option<u8> {
    mjai_to_tid(mjai)
}

#[derive(serde::Serialize)]
struct SpCandidateJs {
    tile: u8,
    exp_value: f32,
    tenpai_prob: f32,
    win_prob: f32,
    num_required_tiles: f32,
    num_yaku_progress_tiles: f32,
    min_point: f32,
    mean_point: f32,
    max_point: f32,
    yaku_mask: u32,
    required_tiles: Vec<f32>,
    yaku_progress_tiles: Vec<f32>,
    point_achievement_probs: Vec<f32>,
    tenpai_series: Vec<f32>,
    win_series: Vec<f32>,
    ev_series: Vec<f32>,
}

#[derive(serde::Serialize)]
struct SpFeaturesJs {
    sp_channels: usize,
    tile_max: usize,
    encoded: Vec<f32>,
    candidates: Vec<SpCandidateJs>,
}

#[derive(serde::Serialize)]
struct DrevFeaturesJs {
    drev_channels: usize,
    tile_max: usize,
    encoded: Vec<f32>,
    anpai_norm: Vec<f32>,
    suji_norm: Vec<f32>,
    kabe: Vec<f32>,
    reach_genbutsu_norm: Vec<f32>,
    n_reach_norm: f32,
    opp_tenpai_prob: Vec<f32>,
    threat: Vec<f32>,
}

/// Compute SP features for a hand state.
///
/// Input JSON matches `SpInput`. Returns `{ sp_channels, tile_max, encoded,
/// candidates }`. `encoded` is the row-major `[SP_CHANNELS][TILE_MAX]`
/// channel buffer (length 178 * 34); `candidates` is a per-discard summary
/// sorted by descending immediate EV (matches the in-engine ordering).
#[wasm_bindgen]
pub fn calc_sp_features(input_json: &str) -> Result<JsValue, JsValue> {
    let input: SpInput = serde_json::from_str(input_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse SpInput: {}", e)))?;

    let result = sp::calculate_sp(&input);
    let encoded = encode_sp(&result);

    let candidates: Vec<SpCandidateJs> = result
        .candidates
        .iter()
        .map(|c| SpCandidateJs {
            tile: c.tile,
            exp_value: c.exp_values[0],
            tenpai_prob: c.tenpai_probs[0],
            win_prob: c.win_probs[0],
            num_required_tiles: c.num_required_tiles,
            num_yaku_progress_tiles: c.num_yaku_progress_tiles,
            min_point: c.min_point,
            mean_point: c.mean_point,
            max_point: c.max_point,
            yaku_mask: c.yaku_mask,
            required_tiles: c.required_tiles.to_vec(),
            yaku_progress_tiles: c.yaku_progress_tiles.to_vec(),
            point_achievement_probs: c.point_achievement_probs.to_vec(),
            tenpai_series: c.tenpai_probs.to_vec(),
            win_series: c.win_probs.to_vec(),
            ev_series: c.exp_values.to_vec(),
        })
        .collect();

    let out = SpFeaturesJs {
        sp_channels: sp::SP_CHANNELS,
        tile_max: TILE_MAX,
        encoded,
        candidates,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Compute Deal-in Risk EV (DREV) features for a hand state.
///
/// Input JSON matches `DrevInput`. Returns `{ drev_channels, tile_max,
/// encoded, anpai_norm, suji_norm, kabe, reach_genbutsu_norm, n_reach_norm,
/// opp_tenpai_prob, threat }`. `encoded` is the row-major
/// `[DREV_CHANNELS][TILE_MAX]` channel buffer (length 9 * 34).
#[wasm_bindgen]
pub fn calc_drev_features(input_json: &str) -> Result<JsValue, JsValue> {
    let input: DrevInput = serde_json::from_str(input_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse DrevInput: {}", e)))?;

    let result = drev::calculate_drev(&input);
    let encoded = encode_drev(&result);

    let out = DrevFeaturesJs {
        drev_channels: drev::DREV_CHANNELS,
        tile_max: TILE_MAX,
        encoded,
        anpai_norm: result.anpai_norm.to_vec(),
        suji_norm: result.suji_norm.to_vec(),
        kabe: result.kabe.to_vec(),
        reach_genbutsu_norm: result.reach_genbutsu_norm.to_vec(),
        n_reach_norm: result.n_reach_norm,
        opp_tenpai_prob: result.opp_tenpai_prob.to_vec(),
        threat: result.threat.to_vec(),
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Convert 136-encoding tile ID to MJAI tile string.
#[wasm_bindgen]
pub fn tile_id_to_mjai(tid: u8) -> String {
    tid_to_mjai(tid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_yakuman_score(yaku_id: u32) -> ScoreResult {
        ScoreResult {
            is_win: true,
            yakuman: true,
            han: 26,
            fu: 0,
            ron_agari: 64000,
            tsumo_agari_oya: 32000,
            tsumo_agari_ko: 16000,
            yaku: vec![yaku_id],
        }
    }

    #[test]
    fn tenhou_default_caps_suuankou_tanki_to_single_yakuman() {
        let conditions = ConditionsInput {
            player_wind: Wind::South as u8,
            ..Default::default()
        };
        let mut score = double_yakuman_score(yaku::ID_SUANKO_TANKI);

        apply_double_yakuman_rules(&mut score, &conditions);

        assert_eq!(score.han, 13);
        assert_eq!(score.ron_agari, 32000);
    }

    #[test]
    fn enabled_suuankou_tanki_rule_keeps_double_yakuman() {
        let conditions = ConditionsInput {
            player_wind: Wind::South as u8,
            is_suuankou_tanki_double: true,
            ..Default::default()
        };
        let mut score = double_yakuman_score(yaku::ID_SUANKO_TANKI);

        apply_double_yakuman_rules(&mut score, &conditions);

        assert_eq!(score.han, 26);
        assert_eq!(score.ron_agari, 64000);
    }

    #[test]
    fn event_journal_wrapper_uses_one_kyoku_spectator_delay() {
        let mut journal = WasmEventJournal::new();
        journal.push_raw_json(r#"{"type":"start_game"}"#).unwrap();
        journal
            .push_raw_json(r#"{"type":"start_kyoku","kyoku":1}"#)
            .unwrap();
        journal.push_raw_json(r#"{"type":"end_kyoku"}"#).unwrap();

        assert_eq!(journal.spectator_cursor(), 1);
        assert_eq!(journal.inner.completed_kyokus()[0].start.index(), 1);
        assert_eq!(journal.inner.completed_kyokus()[0].end.index(), 3);

        journal
            .push_raw_json(r#"{"type":"start_kyoku","kyoku":2}"#)
            .unwrap();
        assert_eq!(journal.spectator_cursor(), 3);
        assert_eq!(journal.revision(), 4);
    }

    #[test]
    fn event_journal_wrapper_releases_full_log_after_end_game() {
        let mut journal = WasmEventJournal::new();
        journal.push_raw_json(r#"{"type":"start_game"}"#).unwrap();
        journal
            .push_raw_json(r#"{"type":"start_kyoku","kyoku":1}"#)
            .unwrap();
        journal.push_raw_json(r#"{"type":"end_kyoku"}"#).unwrap();
        journal.push_raw_json(r#"{"type":"end_game"}"#).unwrap();

        assert!(journal.is_complete());
        assert_eq!(journal.spectator_cursor(), journal.revision());
    }

    #[test]
    fn event_journal_wrapper_does_not_trust_an_unframed_end_game() {
        let mut journal = WasmEventJournal::new();
        journal
            .push_raw_json(r#"{"type":"start_game","seed":[1.0]}"#)
            .unwrap();
        journal
            .push_raw_json(r#"{"type":"end_game","wall":["PRIVATE"]}"#)
            .unwrap();

        assert!(journal.is_complete());
        assert_eq!(journal.spectator_cursor(), 0);
    }

    #[test]
    fn event_journal_delta_envelope_has_stable_schema_and_completion() {
        let events = vec![r#"{"type":"start_game"}"#.to_string()];
        let value = serde_json::to_value(RawEventDelta {
            schema_version: EVENT_JOURNAL_SCHEMA_VERSION,
            complete: false,
            from: 0,
            to: 1,
            events: &events,
        })
        .unwrap();

        assert_eq!(value["schemaVersion"], 1);
        assert_eq!(value["complete"], false);
        assert_eq!(value["from"], 0);
        assert_eq!(value["to"], 1);
    }

    #[test]
    fn game_engine_binding_exposes_four_player_action_id_contract() {
        let mut engine = WasmGameEngine::build("4p-red-single", Some(42.0), Some(true)).unwrap();
        let decisions = decision_views(&engine.pending).unwrap();

        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].base_shape, [74, 34]);
        assert_eq!(decisions[0].extended_shape, [215, 34]);
        assert_eq!(decisions[0].action_mask.len(), 82);
        assert!(!decisions[0].observation_base64.is_empty());

        let player_id = decisions[0].player_id;
        assert_eq!(
            engine
                .pending_decision(player_id)
                .unwrap()
                .observation
                .encode_base_features()
                .unwrap()
                .len(),
            74 * 34
        );
        assert_eq!(
            engine
                .pending_decision(player_id)
                .unwrap()
                .observation
                .encode_extended_features()
                .unwrap()
                .len(),
            215 * 34
        );

        let outcome = engine
            .step_inputs(vec![ActionIdInput {
                player_id,
                action_id: decisions[0].legal_action_ids[0],
            }])
            .unwrap();

        assert!(outcome.error.is_none());
        assert_eq!(outcome.snapshot.num_players, 4);
        assert!(!outcome.events.is_empty());
        assert!(!outcome.next_decisions.is_empty());
    }

    #[test]
    fn game_engine_binding_exposes_three_player_shapes() {
        let engine = WasmGameEngine::build("3p-red-single", Some(7.0), Some(true)).unwrap();
        let decisions = decision_views(&engine.pending).unwrap();

        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].base_shape, [74, 27]);
        assert_eq!(decisions[0].extended_shape, [215, 27]);
        assert_eq!(decisions[0].action_mask.len(), 60);

        let player_id = decisions[0].player_id;
        assert_eq!(
            engine
                .pending_decision(player_id)
                .unwrap()
                .observation
                .encode_base_features()
                .unwrap()
                .len(),
            74 * 27
        );
        assert_eq!(
            engine
                .pending_decision(player_id)
                .unwrap()
                .observation
                .encode_extended_features()
                .unwrap()
                .len(),
            215 * 27
        );
    }

    #[test]
    fn game_engine_binding_rejects_ids_outside_pending_observation() {
        let mut engine = WasmGameEngine::build("4p-red-single", Some(42.0), Some(true)).unwrap();
        let player_id = engine.pending[0].player_id;
        let before = GameSnapshotJs::from(&engine.inner.snapshot());

        let outcome = engine
            .step_inputs(vec![ActionIdInput {
                player_id,
                action_id: usize::MAX,
            }])
            .unwrap();

        assert_eq!(outcome.error.as_ref().unwrap().kind, "invalidActionBatch");
        assert_eq!(outcome.snapshot, before);
        assert!(outcome.events.is_empty());
        assert_eq!(outcome.next_decisions.len(), engine.pending.len());
    }

    #[test]
    fn game_engine_binding_validates_javascript_seed_range() {
        assert_eq!(parse_js_seed(None).unwrap(), None);
        assert_eq!(parse_js_seed(Some(42.0)).unwrap(), Some(42));
        assert!(parse_js_seed(Some(-1.0)).is_err());
        assert!(parse_js_seed(Some(1.5)).is_err());
        assert!(parse_js_seed(Some(f64::INFINITY)).is_err());
    }

    #[test]
    fn binding_integer_parsers_reject_javascript_coercions() {
        assert_eq!(parse_u8_number(0.0, "playerId").unwrap(), 0);
        for invalid in [256.0, -256.0, 0.5, f64::NAN, f64::INFINITY] {
            assert!(parse_u8_number(invalid, "playerId").is_err());
        }

        assert_eq!(parse_usize_number(0.0, "cursor").unwrap(), 0);
        for invalid in [-1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert!(parse_usize_number(invalid, "cursor").is_err());
        }
    }
}
