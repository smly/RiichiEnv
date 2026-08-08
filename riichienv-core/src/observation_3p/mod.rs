mod encode;
pub use encode::{OBS_3P_BASE_CHANNELS, OBS_3P_EXTENDED_CHANNELS, OBS_3P_TILE_TYPES};
pub(crate) mod helpers;
#[cfg(feature = "python")]
mod python;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};

use crate::action::{Action, Action3P, ActionType};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::{Meld, MeldType, TILES_4P, is_sanma_excluded_tile};

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "riichienv._riichienv", get_all, from_py_object)
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
    #[serde(default)]
    pub drawn_tile: Option<u8>,
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
        drawn_tile: Option<u8>,
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
            drawn_tile,
        }
    }

    pub fn legal_actions_method(&self) -> Vec<Action3P> {
        self._legal_actions.clone()
    }

    /// Validate indices and sanma physical tile IDs before encoding an
    /// externally supplied observation.
    pub fn validate(&self) -> RiichiResult<()> {
        self.validate_impl(false)
    }

    /// Preserve action-selection-only constructor/base64 observations while
    /// keeping feature paths strict.
    pub(crate) fn validate_action_selection_compat(&self) -> RiichiResult<()> {
        self.validate_impl(true)
    }

    fn validate_impl(&self, allow_action_only_empty_hand: bool) -> RiichiResult<()> {
        if self.player_id >= 3 {
            return Err(invalid_observation(format!(
                "player_id {} is out of range for 3 players",
                self.player_id
            )));
        }
        if self.oya >= 3 {
            return Err(invalid_observation(format!(
                "oya {} is out of range for 3 players",
                self.oya
            )));
        }
        if self.round_wind >= 4 {
            return Err(invalid_observation(format!(
                "round_wind {} is out of range; expected 0..=3",
                self.round_wind
            )));
        }
        let player = self.player_id as usize;
        if self
            .hands
            .iter()
            .enumerate()
            .any(|(index, hand)| index != player && !hand.is_empty())
        {
            return Err(invalid_observation(
                "opponent hands must be concealed in a player observation".to_string(),
            ));
        }
        if self.melds.iter().any(|melds| melds.len() > 4) {
            return Err(invalid_observation(
                "a player cannot have more than 4 melds".to_string(),
            ));
        }
        if self.discards.iter().any(|discards| discards.len() > 24) {
            return Err(invalid_observation(
                "a player discard list cannot exceed 24 tiles".to_string(),
            ));
        }
        if self.dora_indicators.len() > 5 {
            return Err(invalid_observation(
                "dora_indicators cannot exceed 5 tiles".to_string(),
            ));
        }
        let effective_hand_len = self.hands[player].len() + self.melds[player].len() * 3;
        let is_action_only = allow_action_only_empty_hand
            && self.hands[player].is_empty()
            && self.melds[player].is_empty()
            && !self._legal_actions.is_empty();
        if !is_action_only && !matches!(effective_hand_len, 13 | 14) {
            return Err(invalid_observation(format!(
                "effective self hand length is {effective_hand_len}; expected 13 or 14"
            )));
        }

        for (name, tile) in self
            .hands
            .iter()
            .flatten()
            .map(|&tile| ("hand", tile))
            .chain(
                self.discards
                    .iter()
                    .flatten()
                    .map(|&tile| ("discard", tile)),
            )
            .chain(
                self.dora_indicators
                    .iter()
                    .map(|&tile| ("dora indicator", tile)),
            )
        {
            validate_physical_tile_3p(tile, name)?;
        }
        for meld in self.melds.iter().flatten() {
            validate_meld_shape(meld)?;
            for &tile in &meld.tiles {
                validate_physical_tile_3p(u32::from(tile), "meld")?;
            }
            if let Some(tile) = meld.called_tile {
                validate_physical_tile_3p(u32::from(tile), "called meld")?;
            }
        }
        // Replay reconstruction may reuse a representative physical id for
        // identical MJAI strings, so enforce the shanten-safe type count.
        validate_tile_type_counts(self.hands[player].iter().copied(), 4, "self hand")?;
        validate_visible_tile_counts(
            self.hands
                .iter()
                .flatten()
                .copied()
                .chain(self.discards.iter().flatten().copied())
                .chain(
                    self.melds
                        .iter()
                        .flatten()
                        .flat_map(|meld| meld.tiles.iter().copied().map(u32::from)),
                )
                .chain(self.dora_indicators.iter().copied()),
        )?;
        for (name, tile) in self
            .riichi_sutehais
            .iter()
            .flatten()
            .map(|&tile| ("riichi discard", tile))
            .chain(
                self.last_tedashis
                    .iter()
                    .flatten()
                    .map(|&tile| ("last tedashi", tile)),
            )
            .chain(self.drawn_tile.map(|tile| ("drawn tile", tile)))
        {
            validate_physical_tile_3p(u32::from(tile), name)?;
        }
        if let Some(tile) = self.last_discard {
            validate_physical_tile_3p(tile, "last discard")?;
        }
        if let Some(&wait) = self.waits.iter().find(|&&wait| tile34_is_invalid_3p(wait)) {
            return Err(invalid_observation(format!(
                "wait tile type {wait} is out of range for sanma"
            )));
        }
        for action in &self._legal_actions {
            if let Some(tile) = action.0.tile {
                validate_physical_tile_3p(u32::from(tile), "action tile")?;
            }
            for &tile in &action.0.consume_tiles {
                validate_physical_tile_3p(u32::from(tile), "action consumed tile")?;
            }
            if let Some(actor) = action.0.actor
                && actor >= 3
            {
                return Err(invalid_observation(format!(
                    "action actor {actor} is out of range for 3 players"
                )));
            }
        }
        Ok(())
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action3P> {
        // Prefer non-red-five candidates so that 5m/5p/5s discards do not
        // accidentally drop the akadora when a normal 5 is also legal.
        let mut fallback: Option<&Action3P> = None;
        for action in &self._legal_actions {
            let Ok(idx) = action.encode() else {
                continue;
            };
            if (idx as usize) != action_id {
                continue;
            }
            if is_red_five_discard(&action.0) {
                fallback.get_or_insert(action);
            } else {
                return Some(action.clone());
            }
        }
        fallback.cloned()
    }

    /// Return absolute player indices in relative order: [self, next, prev].
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
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
        obs.validate_action_selection_compat()
            .map_err(|error| RiichiError::Serialization {
                message: format!("invalid observation payload: {error}"),
            })?;
        Ok(obs)
    }
}

fn invalid_observation(message: String) -> RiichiError {
    RiichiError::InvalidState { message }
}

fn tile34_is_invalid_3p(tile: u8) -> bool {
    tile >= 34 || (1..=7).contains(&tile)
}

fn validate_physical_tile_3p(tile: u32, context: &str) -> RiichiResult<()> {
    if tile as usize >= TILES_4P || is_sanma_excluded_tile(tile as u8) {
        return Err(invalid_observation(format!(
            "{context} contains invalid sanma physical tile id {tile}"
        )));
    }
    Ok(())
}

fn validate_meld_shape(meld: &Meld) -> RiichiResult<()> {
    let expected = match meld.meld_type {
        MeldType::Chi | MeldType::Pon => 3,
        MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan => 4,
    };
    if meld.tiles.len() != expected {
        return Err(invalid_observation(format!(
            "{:?} meld contains {} tiles; expected {expected}",
            meld.meld_type,
            meld.tiles.len()
        )));
    }
    if let Some(called_tile) = meld.called_tile
        && !meld.tiles.contains(&called_tile)
    {
        return Err(invalid_observation(format!(
            "called meld tile {called_tile} is not present in the meld"
        )));
    }
    Ok(())
}

fn validate_visible_tile_counts(tiles: impl IntoIterator<Item = u32>) -> RiichiResult<()> {
    let mut counts = [0u16; 34];
    for tile in tiles {
        let tile_type = tile as usize / 4;
        let Some(count) = counts.get_mut(tile_type) else {
            return Err(invalid_observation(format!(
                "visible zone contains invalid physical tile id {tile}"
            )));
        };
        *count += 1;
        if *count > 8 {
            return Err(invalid_observation(format!(
                "tile type {tile_type} appears too many times in visible zones"
            )));
        }
    }
    Ok(())
}

fn validate_tile_type_counts(
    tiles: impl IntoIterator<Item = u32>,
    maximum: u8,
    context: &str,
) -> RiichiResult<()> {
    let mut counts = [0u8; 34];
    for tile in tiles {
        let tile_type = tile as usize / 4;
        let Some(count) = counts.get_mut(tile_type) else {
            return Err(invalid_observation(format!(
                "{context} contains invalid physical tile id {tile}"
            )));
        };
        *count += 1;
        if *count > maximum {
            return Err(invalid_observation(format!(
                "tile type {tile_type} appears more than {maximum} times in {context}"
            )));
        }
    }
    Ok(())
}

fn is_red_five_discard(action: &Action) -> bool {
    matches!(action.action_type, ActionType::Discard)
        && matches!(action.tile, Some(16) | Some(52) | Some(88))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_with_actions(actions: Vec<Action>) -> Observation3P {
        Observation3P::new(
            0,
            [vec![], vec![], vec![]],
            [vec![], vec![], vec![]],
            [vec![], vec![], vec![]],
            vec![],
            [35000, 35000, 35000],
            [false, false, false],
            actions,
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None, None, None],
            [None, None, None],
            None,
            None,
        )
    }

    fn discard(tile: u8) -> Action {
        Action::new(ActionType::Discard, Some(tile), vec![], Some(0))
    }

    #[test]
    fn find_action_3p_prefers_non_red_5p() {
        let obs = obs_with_actions(vec![discard(52), discard(54)]);
        // 5p compact id matches the encoded discard id of either action.
        let id = obs._legal_actions[0].encode().unwrap() as usize;
        let chosen = obs.find_action(id).expect("discard 5p should resolve");
        assert_eq!(chosen.0.tile, Some(54), "non-red 5p must win over red 5p");
    }

    #[test]
    fn find_action_3p_prefers_non_red_5s() {
        let obs = obs_with_actions(vec![discard(88), discard(90)]);
        let id = obs._legal_actions[0].encode().unwrap() as usize;
        let chosen = obs.find_action(id).expect("discard 5s should resolve");
        assert_eq!(chosen.0.tile, Some(90));
    }

    #[test]
    fn find_action_3p_falls_back_to_red_when_only_red_legal() {
        let obs = obs_with_actions(vec![discard(52)]);
        let id = obs._legal_actions[0].encode().unwrap() as usize;
        let chosen = obs.find_action(id).expect("red-only 5p must still resolve");
        assert_eq!(chosen.0.tile, Some(52));
    }
}
