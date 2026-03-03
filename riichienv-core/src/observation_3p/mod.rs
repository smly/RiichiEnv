#[cfg(feature = "python")]
mod encode;
#[cfg(feature = "python")]
pub(crate) mod helpers;
#[cfg(feature = "python")]
mod python;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};

use crate::action::{Action, Action3P};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::Meld;

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "riichienv._riichienv", get_all)
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation3P {
    pub player_id: u8,
    pub hands: [Vec<u32>; 3],
    pub melds: [Vec<Meld>; 3],
    pub discards: [Vec<u32>; 3],
    pub dora_indicators: Vec<u32>,
    pub scores: [i32; 3],
    pub riichi_declared: [bool; 3],

    pub(crate) _legal_actions: Vec<Action3P>,

    pub(crate) events: Vec<String>,

    pub honba: u8,
    pub riichi_sticks: u32,
    pub round_wind: u8,
    pub oya: u8,
    pub kyoku_index: u8,
    pub waits: Vec<u8>,
    pub is_tenpai: bool,
    pub tsumogiri_flags: [Vec<bool>; 3],
    pub riichi_sutehais: [Option<u8>; 3],
    pub last_tedashis: [Option<u8>; 3],
    pub last_discard: Option<u32>,
}

/// Pure Rust methods (no PyO3 dependency).
impl Observation3P {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player_id: u8,
        hands: [Vec<u8>; 3],
        melds: [Vec<Meld>; 3],
        discards: [Vec<u8>; 3],
        dora_indicators: Vec<u8>,
        scores: [i32; 3],
        riichi_declared: [bool; 3],
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
        riichi_sutehais: [Option<u8>; 3],
        last_tedashis: [Option<u8>; 3],
        last_discard: Option<u32>,
    ) -> Self {
        let hands_u32 = hands.map(|h| h.into_iter().map(|x| x as u32).collect());
        let discards_u32 = discards.map(|d| d.into_iter().map(|x| x as u32).collect());
        let dora_u32 = dora_indicators.iter().map(|&x| x as u32).collect();

        Self {
            player_id,
            hands: hands_u32,
            melds,
            discards: discards_u32,
            dora_indicators: dora_u32,
            scores,
            riichi_declared,
            _legal_actions: legal_actions
                .into_iter()
                .map(Action3P::from_action)
                .collect(),
            events,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            kyoku_index,
            waits,
            is_tenpai,
            tsumogiri_flags: Default::default(),
            riichi_sutehais,
            last_tedashis,
            last_discard,
        }
    }

    pub fn legal_actions_method(&self) -> Vec<Action3P> {
        self._legal_actions.clone()
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action3P> {
        self._legal_actions
            .iter()
            .find(|a| {
                if let Ok(idx) = a.encode() {
                    (idx as usize) == action_id
                } else {
                    false
                }
            })
            .cloned()
    }

    /// Return absolute player indices in relative order: [self, next, prev].
    pub(crate) fn rel_order(&self) -> [usize; 3] {
        let pid = self.player_id as usize;
        [pid, (pid + 1) % 3, (pid + 2) % 3]
    }

    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    /// Serialize this Observation3P to a base64-encoded JSON string.
    pub fn serialize_to_base64(&self) -> RiichiResult<String> {
        let json = serde_json::to_vec(self).map_err(|e| RiichiError::Serialization {
            message: format!("serialization failed: {e}"),
        })?;
        Ok(BASE64.encode(&json))
    }

    /// Deserialize an Observation3P from a base64-encoded JSON string.
    pub fn deserialize_from_base64(s: &str) -> RiichiResult<Self> {
        let bytes = BASE64.decode(s).map_err(|e| RiichiError::Serialization {
            message: format!("base64 decode failed: {e}"),
        })?;
        let obs: Observation3P =
            serde_json::from_slice(&bytes).map_err(|e| RiichiError::Serialization {
                message: format!("JSON deserialize failed: {e}"),
            })?;
        Ok(obs)
    }
}
