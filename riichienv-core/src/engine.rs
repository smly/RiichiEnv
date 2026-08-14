//! Stable, variant-independent game-engine facade.
//!
//! `state::GameState` and `state_3p::GameState3P` remain available for
//! compatibility and advanced testing. New Rust consumers should prefer this
//! module: it keeps 4-player/3-player dispatch, reset semantics, observations,
//! and batched orchestration behind one typed API.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::action::{
    ACTION_SPACE_3P, ACTION_SPACE_3P_V1, ACTION_SPACE_4P, ACTION_SPACE_4P_V1, Action,
    ActionEncoder, ActionEncoderV1, Phase,
};
use crate::drev::DREV_CHANNELS;
use crate::drev_v2::DREV_V2_CHANNELS;
use crate::errors::{RiichiError, RiichiResult};
use crate::game_variant::GameStateVariant;
use crate::observation::{OBS_BASE_CHANNELS, OBS_EXTENDED_CHANNELS, OBS_TILE_TYPES, Observation};
use crate::observation_3p::{
    OBS_3P_BASE_CHANNELS, OBS_3P_EXTENDED_CHANNELS, OBS_3P_TILE_TYPES, Observation3P,
};
use crate::replay::EventJournal;
use crate::rule::GameRule;
use crate::sp::SP_CHANNELS;
use crate::types::{TILES_3P, TILES_4P, is_sanma_excluded_tile};

const MAX_SAFE_ENGINE_SCORE_ABS: i32 = 100_000_000;
const MAX_SAFE_RIICHI_STICKS: u32 = 100_000;

/// Supported game modes with an explicit, validated representation.
/// Numeric values intentionally match the legacy `game_mode: u8` API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GameMode {
    #[default]
    #[serde(rename = "4p-red-single")]
    FourPlayerSingle = 0,
    #[serde(rename = "4p-red-east")]
    FourPlayerEast = 1,
    #[serde(rename = "4p-red-half")]
    FourPlayerHalf = 2,
    #[serde(rename = "3p-red-single")]
    ThreePlayerSingle = 3,
    #[serde(rename = "3p-red-east")]
    ThreePlayerEast = 4,
    #[serde(rename = "3p-red-half")]
    ThreePlayerHalf = 5,
}

impl GameMode {
    pub const ALL: [Self; 6] = [
        Self::FourPlayerSingle,
        Self::FourPlayerEast,
        Self::FourPlayerHalf,
        Self::ThreePlayerSingle,
        Self::ThreePlayerEast,
        Self::ThreePlayerHalf,
    ];

    pub const fn id(self) -> u8 {
        self as u8
    }

    pub const fn num_players(self) -> u8 {
        if self.id() >= 3 { 3 } else { 4 }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FourPlayerSingle => "4p-red-single",
            Self::FourPlayerEast => "4p-red-east",
            Self::FourPlayerHalf => "4p-red-half",
            Self::ThreePlayerSingle => "3p-red-single",
            Self::ThreePlayerEast => "3p-red-east",
            Self::ThreePlayerHalf => "3p-red-half",
        }
    }
}

impl TryFrom<u8> for GameMode {
    type Error = RiichiError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.id() == value)
            .ok_or_else(|| RiichiError::InvalidState {
                message: format!("unknown game mode id {value}; expected 0..=5"),
            })
    }
}

impl FromStr for GameMode {
    type Err = RiichiError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.as_str() == value)
            .ok_or_else(|| RiichiError::InvalidState {
                message: format!("unknown game mode '{value}'"),
            })
    }
}

/// Event logging policy for simulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventLogPolicy {
    /// Retain the full MJAI log for servers, debugging, and replay.
    #[default]
    Full,
    /// Skip MJAI serialization in the engine hot path.
    Off,
}

/// Construction options for [`GameEngine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub mode: GameMode,
    pub rule: GameRule,
    pub seed: Option<u64>,
    pub round_wind: u8,
    pub event_log: EventLogPolicy,
}

impl EngineConfig {
    pub fn new(mode: GameMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_rule(mut self, rule: GameRule) -> Self {
        self.rule = rule;
        self
    }

    pub fn with_event_log(mut self, event_log: EventLogPolicy) -> Self {
        self.event_log = event_log;
        self
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            mode: GameMode::default(),
            rule: GameRule::default(),
            seed: None,
            round_wind: 0,
            event_log: EventLogPolicy::Full,
        }
    }
}

/// Explicit reset options shared by Rust and language bindings.
#[derive(Debug, Clone, Default)]
pub struct ResetOptions {
    pub oya: u8,
    pub wall: Option<Vec<u8>>,
    pub round_wind: u8,
    pub scores: Option<Vec<i32>>,
    pub honba: u8,
    pub riichi_sticks: u32,
    pub seed: Option<u64>,
}

/// A player-facing observation independent of the game variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObservationVariant {
    FourPlayer(Observation),
    ThreePlayer(Observation3P),
}

impl ObservationVariant {
    pub fn player_id(&self) -> u8 {
        match self {
            Self::FourPlayer(obs) => obs.player_id,
            Self::ThreePlayer(obs) => obs.player_id,
        }
    }

    pub fn num_players(&self) -> u8 {
        match self {
            Self::FourPlayer(_) => 4,
            Self::ThreePlayer(_) => 3,
        }
    }

    pub fn tile_types(&self) -> usize {
        match self {
            Self::FourPlayer(_) => OBS_TILE_TYPES,
            Self::ThreePlayer(_) => OBS_3P_TILE_TYPES,
        }
    }

    pub fn action_space_size(&self) -> usize {
        match self {
            Self::FourPlayer(_) => ACTION_SPACE_4P,
            Self::ThreePlayer(_) => ACTION_SPACE_3P,
        }
    }

    /// Size of the red-aware v1 model action space.
    pub fn action_space_size_v1(&self) -> usize {
        match self {
            Self::FourPlayer(_) => ACTION_SPACE_4P_V1,
            Self::ThreePlayer(_) => ACTION_SPACE_3P_V1,
        }
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        match self {
            Self::FourPlayer(obs) => obs.legal_actions_method(),
            Self::ThreePlayer(obs) => obs
                .legal_actions_method()
                .into_iter()
                .map(|action| action.0)
                .collect(),
        }
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        select_encoded_action(&self.legal_actions(), self.num_players(), action_id)
    }

    /// Encode all legal actions using the model ABI for this variant.
    pub fn legal_action_ids(&self) -> RiichiResult<Vec<usize>> {
        let encoder = ActionEncoder::from_num_players(self.num_players());
        let mut seen = vec![false; self.action_space_size()];
        let mut action_ids = Vec::new();
        for action in self.legal_actions() {
            let encoded = encoder.encode(&action)?;
            let action_id = usize::try_from(encoded).map_err(|_| RiichiError::InvalidAction {
                message: format!("action encoded to a negative id: {encoded}"),
            })?;
            let Some(already_seen) = seen.get_mut(action_id) else {
                return Err(RiichiError::InvalidAction {
                    message: format!(
                        "action id {action_id} exceeds action space {}",
                        self.action_space_size()
                    ),
                });
            };
            if !*already_seen {
                *already_seen = true;
                action_ids.push(action_id);
            }
        }
        Ok(action_ids)
    }

    /// Encode all legal actions using the red-aware v1 model ABI.
    pub fn legal_action_ids_v1(&self) -> RiichiResult<Vec<usize>> {
        legal_action_ids_with(
            &self.legal_actions(),
            ActionEncoderV1::from_num_players(self.num_players()),
            self.action_space_size_v1(),
        )
    }

    /// Dense `u8` action mask with the stable 82-id (4P) or 60-id (3P)
    /// layout used by existing models.
    pub fn action_mask(&self) -> RiichiResult<Vec<u8>> {
        let mut mask = vec![0; self.action_space_size()];
        for action_id in self.legal_action_ids()? {
            let Some(value) = mask.get_mut(action_id) else {
                return Err(RiichiError::InvalidAction {
                    message: format!("action id {action_id} exceeds action space {}", mask.len()),
                });
            };
            *value = 1;
        }
        Ok(mask)
    }

    /// Dense red-aware v1 mask with 164 slots (4P) or 120 slots (3P).
    pub fn action_mask_v1(&self) -> RiichiResult<Vec<u8>> {
        let mut mask = vec![0; self.action_space_size_v1()];
        for action_id in self.legal_action_ids_v1()? {
            mask[action_id] = 1;
        }
        Ok(mask)
    }

    pub fn select_action(&self, action_id: usize) -> RiichiResult<Action> {
        select_encoded_action(&self.legal_actions(), self.num_players(), action_id).ok_or_else(
            || RiichiError::InvalidAction {
                message: format!(
                    "action id {action_id} is not legal for player {}",
                    self.player_id()
                ),
            },
        )
    }

    /// Resolve a red-aware v1 ID back to its exact legal action.
    pub fn select_action_v1(&self, action_id: usize) -> RiichiResult<Action> {
        select_encoded_action_v1(&self.legal_actions(), self.num_players(), action_id).ok_or_else(
            || RiichiError::InvalidAction {
                message: format!(
                    "v1 action id {action_id} is not legal for player {}",
                    self.player_id()
                ),
            },
        )
    }

    pub fn new_events(&self) -> Vec<String> {
        match self {
            Self::FourPlayer(obs) => obs.new_events(),
            Self::ThreePlayer(obs) => obs.new_events(),
        }
    }

    pub fn serialize_to_base64(&self) -> RiichiResult<String> {
        match self {
            Self::FourPlayer(obs) => obs.serialize_to_base64(),
            Self::ThreePlayer(obs) => obs.serialize_to_base64(),
        }
    }

    pub fn encode_base_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_base_features(),
            Self::ThreePlayer(obs) => obs.encode_base_features(),
        }
    }

    pub fn encode_extended_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_extended_features(),
            Self::ThreePlayer(obs) => obs.encode_extended_features(),
        }
    }

    pub fn encode_sp_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_sp_features(),
            Self::ThreePlayer(obs) => obs.encode_sp_features(),
        }
    }

    pub fn encode_drev_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_drev_features(),
            Self::ThreePlayer(obs) => obs.encode_drev_features(),
        }
    }

    pub fn encode_drev_v2_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_drev_v2_features(),
            Self::ThreePlayer(obs) => obs.encode_drev_v2_features(),
        }
    }

    pub fn encode_extended_with_sp_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_extended_with_sp_features(),
            Self::ThreePlayer(obs) => obs.encode_extended_with_sp_features(),
        }
    }

    pub fn encode_extended_with_sp_drev_v2_features(&self) -> RiichiResult<Vec<f32>> {
        match self {
            Self::FourPlayer(obs) => obs.encode_extended_with_sp_drev_v2_features(),
            Self::ThreePlayer(obs) => obs.encode_extended_with_sp_drev_v2_features(),
        }
    }

    pub fn base_feature_shape(&self) -> (usize, usize) {
        match self {
            Self::FourPlayer(_) => (OBS_BASE_CHANNELS, OBS_TILE_TYPES),
            Self::ThreePlayer(_) => (OBS_3P_BASE_CHANNELS, OBS_3P_TILE_TYPES),
        }
    }

    pub fn extended_feature_shape(&self) -> (usize, usize) {
        match self {
            Self::FourPlayer(_) => (OBS_EXTENDED_CHANNELS, OBS_TILE_TYPES),
            Self::ThreePlayer(_) => (OBS_3P_EXTENDED_CHANNELS, OBS_3P_TILE_TYPES),
        }
    }

    pub fn sp_feature_shape(&self) -> (usize, usize) {
        (SP_CHANNELS, self.tile_types())
    }

    pub fn drev_feature_shape(&self) -> (usize, usize) {
        (DREV_CHANNELS, self.tile_types())
    }

    pub fn drev_v2_feature_shape(&self) -> (usize, usize) {
        (DREV_V2_CHANNELS, self.tile_types())
    }

    pub fn extended_with_sp_feature_shape(&self) -> (usize, usize) {
        let extended_channels = match self {
            Self::FourPlayer(_) => OBS_EXTENDED_CHANNELS,
            Self::ThreePlayer(_) => OBS_3P_EXTENDED_CHANNELS,
        };
        (
            extended_channels + SP_CHANNELS + DREV_CHANNELS,
            self.tile_types(),
        )
    }

    pub fn extended_with_sp_drev_v2_feature_shape(&self) -> (usize, usize) {
        let extended_channels = match self {
            Self::FourPlayer(_) => OBS_EXTENDED_CHANNELS,
            Self::ThreePlayer(_) => OBS_3P_EXTENDED_CHANNELS,
        };
        (
            extended_channels + SP_CHANNELS + DREV_V2_CHANNELS,
            self.tile_types(),
        )
    }
}

fn select_encoded_action(actions: &[Action], num_players: u8, action_id: usize) -> Option<Action> {
    let encoder = ActionEncoder::from_num_players(num_players);
    actions
        .iter()
        .filter(|action| {
            encoder
                .encode(action)
                .is_ok_and(|encoded| encoded >= 0 && encoded as usize == action_id)
        })
        // The v0 action space collapses physically distinct red/non-red
        // choices. New action-ID facades preserve akadora whenever the same
        // encoded action is possible without consuming one.
        .min_by_key(|action| red_five_cost(action))
        .cloned()
}

fn select_encoded_action_v1(
    actions: &[Action],
    num_players: u8,
    action_id: usize,
) -> Option<Action> {
    let encoder = ActionEncoderV1::from_num_players(num_players);
    actions
        .iter()
        .filter(|action| {
            encoder
                .encode(action)
                .is_ok_and(|encoded| encoded >= 0 && encoded as usize == action_id)
        })
        .min_by_key(|action| red_five_cost(action))
        .cloned()
}

fn legal_action_ids_with(
    actions: &[Action],
    encoder: ActionEncoderV1,
    action_space_size: usize,
) -> RiichiResult<Vec<usize>> {
    let mut seen = vec![false; action_space_size];
    let mut action_ids = Vec::new();
    for action in actions {
        let encoded = encoder.encode(action)?;
        let action_id = usize::try_from(encoded).map_err(|_| RiichiError::InvalidAction {
            message: format!("action encoded to a negative id: {encoded}"),
        })?;
        let Some(already_seen) = seen.get_mut(action_id) else {
            return Err(RiichiError::InvalidAction {
                message: format!("action id {action_id} exceeds action space {action_space_size}"),
            });
        };
        if !*already_seen {
            *already_seen = true;
            action_ids.push(action_id);
        }
    }
    Ok(action_ids)
}

fn red_five_cost(action: &Action) -> usize {
    let is_red = |tile: u8| matches!(tile, 16 | 52 | 88);
    usize::from(action.tile.is_some_and(is_red))
        + action
            .consume_tiles
            .iter()
            .filter(|&&tile| is_red(tile))
            .count()
}

/// One pending player decision.
#[derive(Debug, Clone)]
pub struct Decision {
    pub player_id: u8,
    pub observation: ObservationVariant,
}

/// Read-only state intended for servers, UIs, and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GameSnapshot {
    pub mode: GameMode,
    pub num_players: u8,
    pub phase: Phase,
    pub current_player: u8,
    pub active_players: Vec<u8>,
    pub scores: Vec<i32>,
    pub round_wind: u8,
    pub kyoku: u8,
    pub honba: u8,
    pub riichi_sticks: u32,
    pub wall_tiles_remaining: usize,
    pub done: bool,
}

/// Result of one state transition.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    pub decisions: Vec<Decision>,
    pub events: Vec<String>,
    pub snapshot: GameSnapshot,
    pub error: Option<EngineStepError>,
}

/// Typed view of errors reported by the legacy state machines.
///
/// The original message is retained so compatibility adapters can continue
/// emitting the exact legacy chombo reason while new callers avoid parsing it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum EngineStepError {
    IllegalAction { player_id: u8, message: String },
    State { message: String },
}

impl EngineStepError {
    pub fn message(&self) -> &str {
        match self {
            Self::IllegalAction { message, .. } | Self::State { message } => message,
        }
    }

    fn from_legacy(message: String) -> Self {
        const PREFIX: &str = "Error: Illegal Action by Player ";
        if let Some(player_id) = message
            .strip_prefix(PREFIX)
            .and_then(|value| value.parse::<u8>().ok())
        {
            Self::IllegalAction { player_id, message }
        } else {
            Self::State { message }
        }
    }
}

/// Pure Rust facade over the 4-player and 3-player engines.
#[derive(Debug)]
pub struct GameEngine {
    state: GameStateVariant,
    mode: GameMode,
    journal_cache: RwLock<Option<JournalCache>>,
}

#[derive(Debug, Clone)]
struct JournalCache {
    source_len: usize,
    source_tail: Option<String>,
    journal: EventJournal,
}

impl Clone for GameEngine {
    fn clone(&self) -> Self {
        let journal_cache = self
            .journal_cache
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        Self {
            state: self.state.clone(),
            mode: self.mode,
            journal_cache: RwLock::new(journal_cache),
        }
    }
}

impl GameEngine {
    pub fn new(config: EngineConfig) -> RiichiResult<Self> {
        if config.round_wind >= 4 {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "round_wind {} is out of range; expected 0..=3",
                    config.round_wind
                ),
            });
        }
        let state = GameStateVariant::new(
            config.mode.id(),
            config.event_log == EventLogPolicy::Off,
            config.seed,
            config.round_wind,
            config.rule,
        );
        Ok(Self {
            state,
            mode: config.mode,
            journal_cache: RwLock::new(None),
        })
    }

    /// Strict adapter for callers that still receive the legacy numeric mode.
    pub fn from_legacy_mode(
        game_mode: u8,
        skip_mjai_logging: bool,
        seed: Option<u64>,
        round_wind: u8,
        rule: GameRule,
    ) -> RiichiResult<Self> {
        let mode = GameMode::try_from(game_mode)?;
        Self::new(EngineConfig {
            mode,
            rule,
            seed,
            round_wind,
            event_log: if skip_mjai_logging {
                EventLogPolicy::Off
            } else {
                EventLogPolicy::Full
            },
        })
    }

    pub fn mode(&self) -> GameMode {
        self.mode
    }

    pub fn num_players(&self) -> u8 {
        self.mode.num_players()
    }

    pub fn is_done(&self) -> bool {
        match &self.state {
            GameStateVariant::FourPlayer(state) => state.is_done,
            GameStateVariant::ThreePlayer(state) => state.is_done,
        }
    }

    pub fn phase(&self) -> Phase {
        match &self.state {
            GameStateVariant::FourPlayer(state) => state.phase,
            GameStateVariant::ThreePlayer(state) => state.phase,
        }
    }

    pub fn active_players(&self) -> &[u8] {
        match &self.state {
            GameStateVariant::FourPlayer(state) => &state.active_players,
            GameStateVariant::ThreePlayer(state) => &state.active_players,
        }
    }

    pub fn observe(&mut self, player_id: u8) -> RiichiResult<ObservationVariant> {
        if player_id >= self.num_players() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "player id {player_id} is out of range for {} players",
                    self.num_players()
                ),
            });
        }
        Ok(match &mut self.state {
            GameStateVariant::FourPlayer(state) => {
                ObservationVariant::FourPlayer(state.get_observation(player_id))
            }
            GameStateVariant::ThreePlayer(state) => {
                ObservationVariant::ThreePlayer(state.get_observation(player_id))
            }
        })
    }

    /// Consume active players' event cursors and return a flat inference batch.
    pub fn decisions(&mut self) -> Vec<Decision> {
        if self.is_done() {
            return Vec::new();
        }
        let players = self.active_players().to_vec();
        players
            .into_iter()
            .filter_map(|player_id| {
                self.observe(player_id).ok().map(|observation| Decision {
                    player_id,
                    observation,
                })
            })
            .collect()
    }

    pub fn reset(&mut self, options: ResetOptions) -> RiichiResult<Vec<Decision>> {
        validate_reset_options(self.mode, &options)?;
        *self
            .journal_cache
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;

        match &mut self.state {
            GameStateVariant::FourPlayer(state) => {
                state.last_error = None;
                if let Some(seed) = options.seed {
                    state.seed = Some(seed);
                    state.wall.seed = Some(seed);
                    state.wall.hand_index = 0;
                }
                let scores = options
                    .scores
                    .unwrap_or_else(|| vec![state.mode.starting_score(); 4]);
                state.reset();
                state._initialize_round(
                    options.oya,
                    options.round_wind,
                    options.honba,
                    options.riichi_sticks,
                    options.wall,
                    Some(scores),
                );
            }
            GameStateVariant::ThreePlayer(state) => {
                state.last_error = None;
                if let Some(seed) = options.seed {
                    state.seed = Some(seed);
                    state.wall.seed = Some(seed);
                    state.wall.hand_index = 0;
                }
                let scores = options
                    .scores
                    .unwrap_or_else(|| vec![crate::state_3p::game_mode::starting_score(); 3]);
                state.reset();
                state._initialize_round(
                    options.oya,
                    options.round_wind,
                    options.honba,
                    options.riichi_sticks,
                    options.wall,
                    Some(scores),
                );
            }
        }
        Ok(self.decisions())
    }

    pub fn step(&mut self, actions: &HashMap<u8, Action>) -> StepOutcome {
        if self.is_done() && actions.is_empty() {
            return StepOutcome {
                decisions: Vec::new(),
                events: Vec::new(),
                snapshot: self.snapshot(),
                error: None,
            };
        }

        let pending_players = if self.is_done() {
            Vec::new()
        } else {
            self.active_players().to_vec()
        };
        let invalid_player = actions
            .keys()
            .find(|&&player_id| !pending_players.contains(&player_id))
            .copied();
        let missing_player = pending_players
            .iter()
            .find(|&&player_id| !actions.contains_key(&player_id))
            .copied();
        if let Some(player_id) = invalid_player {
            return StepOutcome {
                decisions: Vec::new(),
                events: Vec::new(),
                snapshot: self.snapshot(),
                error: Some(EngineStepError::State {
                    message: format!("player {player_id} has no pending decision"),
                }),
            };
        }
        if let Some(player_id) = missing_player {
            return StepOutcome {
                decisions: Vec::new(),
                events: Vec::new(),
                snapshot: self.snapshot(),
                error: Some(EngineStepError::State {
                    message: format!("missing action for pending player {player_id}"),
                }),
            };
        }
        let previous_log_len = self.mjai_log().len();
        match &mut self.state {
            GameStateVariant::FourPlayer(state) => {
                state.last_error = None;
                state.step(actions);
            }
            GameStateVariant::ThreePlayer(state) => {
                state.last_error = None;
                state.step(actions);
            }
        }
        let events = self
            .mjai_log()
            .get(previous_log_len..)
            .unwrap_or_default()
            .to_vec();
        let error = match &self.state {
            GameStateVariant::FourPlayer(state) => state.last_error.clone(),
            GameStateVariant::ThreePlayer(state) => state.last_error.clone(),
        }
        .map(EngineStepError::from_legacy);
        let decisions = if error.is_none() {
            self.decisions()
        } else {
            Vec::new()
        };
        StepOutcome {
            decisions,
            events,
            snapshot: self.snapshot(),
            error,
        }
    }

    pub fn snapshot(&self) -> GameSnapshot {
        match &self.state {
            GameStateVariant::FourPlayer(state) => GameSnapshot {
                mode: self.mode,
                num_players: 4,
                phase: state.phase,
                current_player: state.current_player,
                active_players: state.active_players.clone(),
                scores: state.players.iter().map(|player| player.score).collect(),
                round_wind: state.round_wind,
                kyoku: state.kyoku_idx,
                honba: state.honba,
                riichi_sticks: state.riichi_sticks,
                wall_tiles_remaining: state.wall.drawable_count as usize,
                done: state.is_done,
            },
            GameStateVariant::ThreePlayer(state) => GameSnapshot {
                mode: self.mode,
                num_players: 3,
                phase: state.phase,
                current_player: state.current_player,
                active_players: state.active_players.clone(),
                scores: state.players.iter().map(|player| player.score).collect(),
                round_wind: state.round_wind,
                kyoku: state.kyoku_idx,
                honba: state.honba,
                riichi_sticks: state.riichi_sticks,
                wall_tiles_remaining: state.wall.drawable_count as usize,
                done: state.is_done,
            },
        }
    }

    pub fn mjai_log(&self) -> &[String] {
        match &self.state {
            GameStateVariant::FourPlayer(state) => &state.mjai_log,
            GameStateVariant::ThreePlayer(state) => &state.mjai_log,
        }
    }

    /// Index the retained log into the replay/spectator boundary.
    ///
    /// The cache is built lazily and only parses newly appended events on
    /// subsequent calls. Simulations with logging disabled therefore pay no
    /// journal maintenance cost in the step hot path.
    pub fn event_journal(&self) -> RiichiResult<EventJournal> {
        let events = self.mjai_log();
        let mut cache_slot = self
            .journal_cache
            .write()
            .map_err(|_| RiichiError::InvalidState {
                message: "event journal cache lock was poisoned".to_string(),
            })?;
        let can_append = cache_slot.as_ref().is_some_and(|cache| {
            cache.source_len <= events.len()
                && (cache.source_len == 0
                    || events.get(cache.source_len - 1) == cache.source_tail.as_ref())
        });

        if !can_append {
            let journal = EventJournal::from_events(events.iter().cloned())?;
            *cache_slot = Some(JournalCache {
                source_len: events.len(),
                source_tail: events.last().cloned(),
                journal,
            });
        } else if let Some(cache) = cache_slot.as_mut() {
            for event in &events[cache.source_len..] {
                cache.journal.push_json(event)?;
                cache.source_len += 1;
                cache.source_tail = Some(event.clone());
            }
        }

        Ok(cache_slot
            .as_ref()
            .expect("journal cache is initialized above")
            .journal
            .clone())
    }

    /// Compatibility escape hatch for incremental downstream migrations.
    pub fn state(&self) -> &GameStateVariant {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut GameStateVariant {
        *self
            .journal_cache
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        &mut self.state
    }

    pub fn into_state(self) -> GameStateVariant {
        self.state
    }
}

fn validate_reset_options(mode: GameMode, options: &ResetOptions) -> RiichiResult<()> {
    let expected_players = mode.num_players() as usize;
    if options.oya as usize >= expected_players {
        return Err(RiichiError::InvalidState {
            message: format!(
                "oya {} is out of range for {expected_players} players",
                options.oya
            ),
        });
    }
    if options.round_wind >= 4 {
        return Err(RiichiError::InvalidState {
            message: format!(
                "round_wind {} is out of range; expected 0..=3",
                options.round_wind
            ),
        });
    }
    if let Some(scores) = &options.scores
        && scores.len() != expected_players
    {
        return Err(RiichiError::InvalidState {
            message: format!(
                "scores length {} does not match number of players {expected_players}",
                scores.len()
            ),
        });
    }
    if let Some(score) = options.scores.as_ref().and_then(|scores| {
        scores.iter().find(|&&score| {
            !(-MAX_SAFE_ENGINE_SCORE_ABS..=MAX_SAFE_ENGINE_SCORE_ABS).contains(&score)
        })
    }) {
        return Err(RiichiError::InvalidState {
            message: format!(
                "score {score} is outside the safe engine range -{MAX_SAFE_ENGINE_SCORE_ABS}..={MAX_SAFE_ENGINE_SCORE_ABS}"
            ),
        });
    }
    if options.riichi_sticks > MAX_SAFE_RIICHI_STICKS {
        return Err(RiichiError::InvalidState {
            message: format!(
                "riichi_sticks {} exceeds the safe engine limit {MAX_SAFE_RIICHI_STICKS}",
                options.riichi_sticks
            ),
        });
    }
    if let Some(wall) = &options.wall {
        validate_wall(mode, wall)?;
    }
    Ok(())
}

fn validate_wall(mode: GameMode, wall: &[u8]) -> RiichiResult<()> {
    let expected_tiles = if mode.num_players() == 3 {
        TILES_3P
    } else {
        TILES_4P
    };
    if wall.len() != expected_tiles {
        return Err(RiichiError::InvalidState {
            message: format!(
                "wall length {} does not match {expected_tiles} tiles for {}",
                wall.len(),
                mode.as_str()
            ),
        });
    }

    let mut seen = [false; TILES_4P];
    for &tile in wall {
        let tile_index = tile as usize;
        if tile_index >= TILES_4P {
            return Err(RiichiError::InvalidState {
                message: format!("wall contains invalid physical tile id {tile}"),
            });
        }
        if mode.num_players() == 3 && is_sanma_excluded_tile(tile) {
            return Err(RiichiError::InvalidState {
                message: format!("sanma wall contains excluded physical tile id {tile}"),
            });
        }
        if std::mem::replace(&mut seen[tile_index], true) {
            return Err(RiichiError::InvalidState {
                message: format!("wall contains duplicate physical tile id {tile}"),
            });
        }
    }
    Ok(())
}

/// A decision tagged with its environment index for batched inference.
#[derive(Debug, Clone)]
pub struct BatchDecision {
    pub env_index: usize,
    pub player_id: u8,
    pub observation: ObservationVariant,
}

/// Outcome of stepping a vector of environments once.
#[derive(Debug, Clone)]
pub struct BatchStepOutcome {
    pub decisions: Vec<BatchDecision>,
    pub snapshots: Vec<GameSnapshot>,
    pub errors: Vec<Option<EngineStepError>>,
}

/// Vectorized orchestration for self-play.
///
/// State transitions remain deterministic and sequential. Observations are
/// flattened so callers can run one model batch and scatter actions back by
/// `(env_index, player_id)`.
#[derive(Debug, Clone)]
pub struct BatchGameEngine {
    engines: Vec<GameEngine>,
}

impl BatchGameEngine {
    pub fn new(configs: impl IntoIterator<Item = EngineConfig>) -> RiichiResult<Self> {
        let engines = configs
            .into_iter()
            .map(GameEngine::new)
            .collect::<RiichiResult<Vec<_>>>()?;
        Ok(Self { engines })
    }

    pub fn homogeneous(config: EngineConfig, count: usize) -> RiichiResult<Self> {
        Self::try_homogeneous(config, count)
    }

    /// Construct a homogeneous batch without panicking on an impossible
    /// capacity request. Language bindings use this for untrusted counts.
    pub fn try_homogeneous(config: EngineConfig, count: usize) -> RiichiResult<Self> {
        let mut engines = Vec::new();
        engines
            .try_reserve_exact(count)
            .map_err(|error| RiichiError::InvalidState {
                message: format!("cannot allocate batch of {count} engines: {error}"),
            })?;
        for index in 0..count {
            let mut item = config.clone();
            if let Some(seed) = item.seed {
                item.seed = Some(seed.wrapping_add(index as u64));
            }
            engines.push(GameEngine::new(item)?);
        }
        Ok(Self { engines })
    }

    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    pub fn engines(&self) -> &[GameEngine] {
        &self.engines
    }

    pub fn engines_mut(&mut self) -> &mut [GameEngine] {
        &mut self.engines
    }

    pub fn decisions(&mut self) -> Vec<BatchDecision> {
        self.engines
            .iter_mut()
            .enumerate()
            .flat_map(|(env_index, engine)| {
                engine
                    .decisions()
                    .into_iter()
                    .map(move |decision| BatchDecision {
                        env_index,
                        player_id: decision.player_id,
                        observation: decision.observation,
                    })
            })
            .collect()
    }

    pub fn step(&mut self, actions: &[HashMap<u8, Action>]) -> RiichiResult<BatchStepOutcome> {
        if actions.len() != self.engines.len() {
            return Err(RiichiError::InvalidState {
                message: format!(
                    "received {} action maps for {} environments",
                    actions.len(),
                    self.engines.len()
                ),
            });
        }

        let mut decisions = Vec::new();
        let mut snapshots = Vec::with_capacity(self.engines.len());
        let mut errors = Vec::with_capacity(self.engines.len());
        for (env_index, (engine, env_actions)) in self.engines.iter_mut().zip(actions).enumerate() {
            let outcome = engine.step(env_actions);
            decisions.extend(outcome.decisions.into_iter().map(|decision| BatchDecision {
                env_index,
                player_id: decision.player_id,
                observation: decision.observation,
            }));
            snapshots.push(outcome.snapshot);
            errors.push(outcome.error);
        }
        Ok(BatchStepOutcome {
            decisions,
            snapshots,
            errors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionType;

    #[test]
    fn typed_game_modes_round_trip() {
        for mode in GameMode::ALL {
            assert_eq!(GameMode::try_from(mode.id()).unwrap(), mode);
            assert_eq!(mode.as_str().parse::<GameMode>().unwrap(), mode);
        }
        assert!(GameMode::try_from(6).is_err());
        assert!("default".parse::<GameMode>().is_err());
    }

    #[test]
    fn engine_config_rejects_invalid_round_wind() {
        let config = EngineConfig {
            round_wind: 4,
            ..EngineConfig::default()
        };
        assert!(GameEngine::new(config).is_err());
    }

    #[test]
    fn core_engine_wire_uses_the_canonical_camel_case_schema() {
        let engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let snapshot = serde_json::to_value(engine.snapshot()).unwrap();
        assert_eq!(snapshot["mode"], "4p-red-single");
        assert_eq!(snapshot["phase"], "WaitAct");
        assert_eq!(snapshot["numPlayers"], 4);
        assert!(snapshot.get("num_players").is_none());

        let error = serde_json::to_value(EngineStepError::IllegalAction {
            player_id: 2,
            message: "bad action".to_string(),
        })
        .unwrap();
        assert_eq!(error["kind"], "illegalAction");
        assert_eq!(error["playerId"], 2);
    }

    #[test]
    fn four_player_engine_returns_unified_decisions() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let decisions = engine.decisions();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].observation.base_feature_shape(), (74, 34));
        let mask = decisions[0].observation.action_mask().unwrap();
        assert_eq!(mask.len(), ACTION_SPACE_4P);
        assert_eq!(
            mask.iter().filter(|&&value| value == 1).count(),
            decisions[0]
                .observation
                .legal_action_ids()
                .unwrap()
                .into_iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        );

        let action = decisions[0].observation.legal_actions()[0].clone();
        let actions = HashMap::from([(decisions[0].player_id, action)]);
        let outcome = engine.step(&actions);
        assert!(outcome.error.is_none());
        assert_eq!(outcome.snapshot.num_players, 4);
        assert!(!outcome.events.is_empty());
    }

    #[test]
    fn three_player_engine_hides_action_wrapper() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::ThreePlayerSingle).with_seed(42)).unwrap();
        let decisions = engine.decisions();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].observation.base_feature_shape(), (74, 27));
        assert_eq!(
            decisions[0].observation.action_space_size(),
            ACTION_SPACE_3P
        );
        assert!(!decisions[0].observation.legal_actions().is_empty());
    }

    #[test]
    fn observations_report_the_discarded_tile_not_the_actor() {
        for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
            let mut engine = GameEngine::new(EngineConfig::new(mode).with_seed(42)).unwrap();
            let decision = engine.decisions().remove(0);
            let discard = decision
                .observation
                .legal_actions()
                .into_iter()
                .find(|action| {
                    action.action_type == ActionType::Discard
                        && action.tile.is_some_and(|tile| tile > 20)
                })
                .expect("opening hand must have a non-1m discard");
            let discarded_tile = discard.tile.unwrap() as u32;
            let outcome = engine.step(&HashMap::from([(decision.player_id, discard)]));
            assert!(outcome.error.is_none());
            for next in outcome.decisions {
                let observed = match next.observation {
                    ObservationVariant::FourPlayer(observation) => observation.last_discard,
                    ObservationVariant::ThreePlayer(observation) => observation.last_discard,
                };
                assert_eq!(
                    observed,
                    Some(discarded_tile),
                    "wrong last discard in {mode:?}"
                );
            }
        }
    }

    #[test]
    fn four_player_drev_normalises_genbutsu_across_all_three_opponents() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let decision = engine.decisions().remove(0);
        let discard = decision
            .observation
            .legal_actions()
            .into_iter()
            .find(|action| action.action_type == ActionType::Discard && action.tile.is_some())
            .unwrap();
        let tile_type = (discard.tile.unwrap() / 4) as usize;
        let outcome = engine.step(&HashMap::from([(decision.player_id, discard)]));
        for next in outcome.decisions {
            let encoded = next.observation.encode_drev_features().unwrap();
            assert!((encoded[tile_type] - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn batch_engine_flattens_environment_and_player_ids() {
        let mut batch = BatchGameEngine::homogeneous(
            EngineConfig::new(GameMode::FourPlayerSingle).with_seed(7),
            3,
        )
        .unwrap();
        let decisions = batch.decisions();
        assert_eq!(decisions.len(), 3);
        assert_eq!(
            decisions
                .iter()
                .map(|decision| decision.env_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn fallible_batch_constructor_rejects_capacity_overflow() {
        let result = BatchGameEngine::try_homogeneous(EngineConfig::default(), usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn event_journal_hides_the_current_kyoku() {
        let engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let journal = engine.event_journal().unwrap();
        assert!(journal.has_in_progress_kyoku());
        assert_eq!(journal.spectator_prefix(1).len(), 1);
        assert!(journal.spectator_prefix(1)[0].contains(r#""type":"start_game""#));
    }

    #[test]
    fn event_journal_cache_tracks_appended_engine_events() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let first_revision = engine.event_journal().unwrap().revision();
        let decision = engine.decisions().remove(0);
        let discard = decision
            .observation
            .legal_actions()
            .into_iter()
            .find(|action| action.action_type == ActionType::Discard)
            .unwrap();
        let outcome = engine.step(&HashMap::from([(decision.player_id, discard)]));
        assert!(outcome.error.is_none());

        let second = engine.event_journal().unwrap();
        assert!(second.revision() > first_revision);
        assert_eq!(second.events(), engine.mjai_log());
        assert_eq!(
            engine.event_journal().unwrap().revision(),
            second.revision()
        );
    }

    #[test]
    fn illegal_actions_have_a_typed_outcome_and_do_not_poison_later_steps() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let player_id = engine.active_players()[0];
        let outcome = engine.step(&HashMap::from([(
            player_id,
            Action::new(
                crate::action::ActionType::Pass,
                None,
                vec![],
                Some(player_id),
            ),
        )]));
        assert!(matches!(
            outcome.error,
            Some(EngineStepError::IllegalAction {
                player_id: actual,
                ..
            }) if actual == player_id
        ));

        let next = engine.step(&HashMap::new());
        assert!(next.error.is_none());
    }

    #[test]
    fn logging_can_be_disabled_for_rollout_hot_paths() {
        let mut engine = GameEngine::new(
            EngineConfig::new(GameMode::FourPlayerSingle)
                .with_seed(42)
                .with_event_log(EventLogPolicy::Off),
        )
        .unwrap();
        assert!(engine.mjai_log().is_empty());
        let decision = engine.decisions().remove(0);
        let action = decision.observation.legal_actions().remove(0);
        let outcome = engine.step(&HashMap::from([(decision.player_id, action)]));
        assert!(outcome.events.is_empty());
    }

    #[test]
    fn missing_pending_actions_are_rejected_without_becoming_implicit_passes() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let before = engine.snapshot();
        let outcome = engine.step(&HashMap::new());

        assert!(matches!(outcome.error, Some(EngineStepError::State { .. })));
        assert_eq!(engine.snapshot(), before);
        assert!(outcome.events.is_empty());
    }

    #[test]
    fn reset_validates_inputs_before_mutating_state() {
        for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
            let mut engine = GameEngine::new(EngineConfig::new(mode).with_seed(42)).unwrap();
            let before = engine.snapshot();

            let invalid_oya = ResetOptions {
                oya: mode.num_players(),
                ..ResetOptions::default()
            };
            assert!(engine.reset(invalid_oya).is_err());
            assert_eq!(engine.snapshot(), before);

            let invalid_wall = ResetOptions {
                wall: Some(Vec::new()),
                ..ResetOptions::default()
            };
            assert!(engine.reset(invalid_wall).is_err());
            assert_eq!(engine.snapshot(), before);

            let unsafe_scores = ResetOptions {
                scores: Some(vec![i32::MAX; mode.num_players() as usize]),
                ..ResetOptions::default()
            };
            assert!(engine.reset(unsafe_scores).is_err());
            assert_eq!(engine.snapshot(), before);

            let unsafe_sticks = ResetOptions {
                riichi_sticks: u32::MAX,
                ..ResetOptions::default()
            };
            assert!(engine.reset(unsafe_sticks).is_err());
            assert_eq!(engine.snapshot(), before);
        }
    }

    #[test]
    fn reset_seed_restarts_the_deterministic_wall_sequence() {
        for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
            let mut fresh = GameEngine::new(EngineConfig::new(mode).with_seed(42)).unwrap();
            let expected = fresh
                .decisions()
                .remove(0)
                .observation
                .serialize_to_base64()
                .unwrap();

            let mut reset = GameEngine::new(EngineConfig::new(mode).with_seed(999)).unwrap();
            let actual = reset
                .reset(ResetOptions {
                    seed: Some(42),
                    ..ResetOptions::default()
                })
                .unwrap()
                .remove(0)
                .observation
                .serialize_to_base64()
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn observing_a_non_current_player_never_exposes_the_drawn_tile() {
        for mode in [GameMode::FourPlayerSingle, GameMode::ThreePlayerSingle] {
            let mut engine = GameEngine::new(EngineConfig::new(mode).with_seed(42)).unwrap();
            let hidden = engine.observe(1).unwrap();
            match hidden {
                ObservationVariant::FourPlayer(observation) => {
                    assert_eq!(observation.drawn_tile, None)
                }
                ObservationVariant::ThreePlayer(observation) => {
                    assert_eq!(observation.drawn_tile, None)
                }
            }
        }
    }

    #[test]
    fn terminal_engines_never_emit_empty_action_decisions() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        match engine.state_mut() {
            GameStateVariant::FourPlayer(state) => state.is_done = true,
            GameStateVariant::ThreePlayer(_) => unreachable!(),
        }
        assert!(engine.decisions().is_empty());
        assert!(engine.step(&HashMap::new()).decisions.is_empty());
    }

    #[test]
    fn action_keys_without_pending_decisions_are_rejected_without_mutation() {
        let mut engine =
            GameEngine::new(EngineConfig::new(GameMode::ThreePlayerSingle).with_seed(42)).unwrap();
        let before = engine.snapshot();
        let outcome = engine.step(&HashMap::from([(
            3,
            Action::new(crate::action::ActionType::Pass, None, Vec::new(), Some(3)),
        )]));
        assert!(matches!(outcome.error, Some(EngineStepError::State { .. })));
        assert_eq!(engine.snapshot(), before);
        assert!(outcome.events.is_empty());
    }

    #[test]
    fn snapshot_wall_count_is_the_drawable_live_wall() {
        let four =
            GameEngine::new(EngineConfig::new(GameMode::FourPlayerSingle).with_seed(42)).unwrap();
        let three =
            GameEngine::new(EngineConfig::new(GameMode::ThreePlayerSingle).with_seed(42)).unwrap();
        assert_eq!(four.snapshot().wall_tiles_remaining, 69);
        assert_eq!(three.snapshot().wall_tiles_remaining, 54);
    }

    #[test]
    fn action_id_selection_avoids_consuming_red_fives_when_equivalent() {
        let red_pon = Action::new(
            crate::action::ActionType::Pon,
            Some(19),
            vec![16, 17],
            Some(0),
        );
        let plain_pon = Action::new(
            crate::action::ActionType::Pon,
            Some(19),
            vec![17, 18],
            Some(0),
        );
        let action_id = ActionEncoder::FourPlayer.encode(&red_pon).unwrap() as usize;

        let selected = select_encoded_action(&[red_pon, plain_pon.clone()], 4, action_id).unwrap();
        assert_eq!(selected, plain_pon);
    }
}
