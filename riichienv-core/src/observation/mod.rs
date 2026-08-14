mod encode;
pub use encode::{OBS_BASE_CHANNELS, OBS_EXTENDED_CHANNELS, OBS_TILE_TYPES};
pub(crate) mod helpers;
#[cfg(feature = "python")]
pub(crate) mod mjai_select;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
pub(crate) mod sequence_features;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};

use crate::action::{Action, ActionEncoder, ActionEncoderV1, ActionType};
use crate::errors::{RiichiError, RiichiResult};
use crate::types::{Meld, MeldType, TILES_4P};

#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "riichienv._riichienv", get_all, from_py_object)
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub player_id: u8,
    pub hands: [Vec<u32>; 4],
    pub melds: [Vec<Meld>; 4],
    pub discards: [Vec<u32>; 4],
    pub dora_indicators: Vec<u32>,
    pub scores: [i32; 4],
    pub riichi_declared: [bool; 4],

    pub(crate) _legal_actions: Vec<Action>,

    pub(crate) events: Vec<String>,

    /// Pre-computed progression tuples (set by GameState for O(1) access).
    /// When Some, encode_seq_progression() returns this directly.
    #[serde(skip)]
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) cached_progression: Option<Vec<[u16; 5]>>,

    pub honba: u8,
    pub riichi_sticks: u32,
    pub round_wind: u8,
    pub oya: u8,
    pub kyoku_index: u8,
    pub waits: Vec<u8>,
    pub is_tenpai: bool,
    pub tsumogiri_flags: [Vec<bool>; 4],
    /// Complete per-discard tsumogiri history for DREV v2. The historical
    /// `tsumogiri_flags` field remains untouched to preserve the frozen
    /// Observation wire and legacy feature values.
    #[serde(skip)]
    pub(crate) public_tsumogiri_history: [Vec<bool>; 4],
    /// Per-discard riichi declaration flags, aligned with [`Self::discards`].
    /// Older observation payloads do not carry this history.
    // Runtime-only v2 sidecar. Keeping it out of the frozen Observation JSON
    // preserves byte-for-byte v0.4.x base64 payloads.
    #[serde(skip)]
    pub(crate) discard_is_riichi: [Vec<bool>; 4],
    /// Actor seat for every discard in public chronological order.
    #[serde(skip)]
    pub(crate) discard_actor_history: Vec<u8>,
    /// Number of chronological discards whose normal response window is over.
    #[serde(skip)]
    pub(crate) resolved_discard_count: usize,
    /// Per-seat tiles proven temporarily unable to deal in until that seat's
    /// next draw or call. This is a runtime-only consequence of the same-turn
    /// furiten rule and is never serialized in the frozen Observation wire.
    #[serde(skip)]
    pub(crate) public_temporary_safe_masks: [u64; 4],
    /// Whether the additive public-history fields cover the whole current hand.
    #[serde(skip)]
    pub(crate) public_history_complete: bool,
    pub riichi_sutehais: [Option<u8>; 4],
    pub last_tedashis: [Option<u8>; 4],
    pub last_discard: Option<u32>,
    #[serde(default)]
    pub drawn_tile: Option<u8>,
}

/// Pure Rust methods (no PyO3 dependency).
impl Observation {
    /// Complete public tedashi/tsumogiri history supplied by the live engine.
    pub fn public_tsumogiri_history(&self) -> &[Vec<bool>; 4] {
        &self.public_tsumogiri_history
    }

    /// Riichi-declaration markers aligned with [`Self::discards`].
    pub fn public_riichi_discard_history(&self) -> &[Vec<bool>; 4] {
        &self.discard_is_riichi
    }

    /// Absolute actors in chronological discard order.
    pub fn public_discard_actors(&self) -> &[u8] {
        &self.discard_actor_history
    }

    /// Number of chronological normal-discard response windows that resolved.
    pub fn resolved_public_discard_count(&self) -> usize {
        self.resolved_discard_count
    }

    /// Same-turn-furiten safety masks in absolute seat order.
    pub fn public_temporary_safe_masks(&self) -> &[u64; 4] {
        &self.public_temporary_safe_masks
    }

    /// Whether the runtime-only history sidecar is complete and trustworthy.
    pub fn has_complete_public_history(&self) -> bool {
        self.public_history_complete
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        player_id: u8,
        hands: [Vec<u8>; 4],
        melds: [Vec<Meld>; 4],
        discards: [Vec<u8>; 4],
        dora_indicators: Vec<u8>,
        scores: [i32; 4],
        riichi_declared: [bool; 4],
        legal_actions: Vec<Action>,
        events: Vec<String>,
        honba: u8,
        riichi_sticks: u32,
        round_wind: u8,
        oya: u8,
        kyoku_index: u8,
        waits: Vec<u8>,
        is_tenpai: bool,
        riichi_sutehais: [Option<u8>; 4],
        last_tedashis: [Option<u8>; 4],
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
            _legal_actions: legal_actions,
            events,
            cached_progression: None,
            honba,
            riichi_sticks,
            round_wind,
            oya,
            kyoku_index,
            waits,
            is_tenpai,
            tsumogiri_flags: Default::default(),
            public_tsumogiri_history: Default::default(),
            discard_is_riichi: Default::default(),
            discard_actor_history: Vec::new(),
            resolved_discard_count: 0,
            public_temporary_safe_masks: [0; 4],
            public_history_complete: false,
            riichi_sutehais,
            last_tedashis,
            last_discard,
            drawn_tile,
        }
    }

    pub fn legal_actions_method(&self) -> Vec<Action> {
        self._legal_actions.clone()
    }

    /// Validate indices and physical tile IDs before an externally supplied
    /// observation enters feature encoders or action selection.
    pub fn validate(&self) -> RiichiResult<()> {
        self.validate_impl(false)
    }

    /// Preserve the historical constructor/base64 use-case where callers
    /// build an action-selection-only observation with no hand. Feature
    /// encoders still call strict `validate` and reject that synthetic DTO.
    pub(crate) fn validate_action_selection_compat(&self) -> RiichiResult<()> {
        self.validate_impl(true)
    }

    fn validate_impl(&self, allow_action_only_empty_hand: bool) -> RiichiResult<()> {
        if self.player_id >= 4 {
            return Err(invalid_observation(format!(
                "player_id {} is out of range for 4 players",
                self.player_id
            )));
        }
        if self.oya >= 4 {
            return Err(invalid_observation(format!(
                "oya {} is out of range for 4 players",
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
            validate_physical_tile(tile, name)?;
        }
        for meld in self.melds.iter().flatten() {
            validate_meld_shape(meld)?;
            for &tile in &meld.tiles {
                validate_physical_tile(u32::from(tile), "meld")?;
            }
            if let Some(tile) = meld.called_tile {
                validate_physical_tile(u32::from(tile), "called meld")?;
            }
        }
        // Replay reconstruction may map repeated MJAI tile strings to the same
        // representative 136-id. The semantic invariant required by shanten
        // tables is therefore per tile type, not physical-id uniqueness.
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
            validate_physical_tile(u32::from(tile), name)?;
        }
        if let Some(tile) = self.last_discard {
            validate_physical_tile(tile, "last discard")?;
        }
        if let Some(&wait) = self
            .waits
            .iter()
            .find(|&&wait| wait as usize >= OBS_TILE_TYPES)
        {
            return Err(invalid_observation(format!(
                "wait tile type {wait} is out of range"
            )));
        }
        for action in &self._legal_actions {
            if let Some(tile) = action.tile {
                validate_physical_tile(u32::from(tile), "action tile")?;
            }
            for &tile in &action.consume_tiles {
                validate_physical_tile(u32::from(tile), "action consumed tile")?;
            }
            if let Some(actor) = action.actor
                && actor >= 4
            {
                return Err(invalid_observation(format!(
                    "action actor {actor} is out of range for 4 players"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_public_discard_history(&self) -> RiichiResult<()> {
        for seat in 0..4 {
            let discard_count = self.discards[seat].len();
            let tsumogiri_count = self.public_tsumogiri_history[seat].len();
            if tsumogiri_count != 0 && tsumogiri_count != discard_count {
                return Err(invalid_observation(format!(
                    "public_tsumogiri_history[{seat}] contains {tsumogiri_count} entries for {discard_count} discards"
                )));
            }
            let riichi_count = self.discard_is_riichi[seat].len();
            if riichi_count != 0 && riichi_count != discard_count {
                return Err(invalid_observation(format!(
                    "discard_is_riichi[{seat}] contains {riichi_count} entries for {discard_count} discards"
                )));
            }
            if self.public_history_complete
                && (tsumogiri_count != discard_count || riichi_count != discard_count)
            {
                return Err(invalid_observation(format!(
                    "complete public history for seat {seat} must align with all {discard_count} discards"
                )));
            }
            let declaration_count = self.discard_is_riichi[seat]
                .iter()
                .filter(|&&declared| declared)
                .count();
            if declaration_count > 1 {
                return Err(invalid_observation(format!(
                    "discard_is_riichi[{seat}] contains more than one declaration"
                )));
            }
            if self.public_history_complete
                && self.riichi_declared[seat] != (declaration_count == 1)
            {
                return Err(invalid_observation(format!(
                    "riichi state for seat {seat} is inconsistent with its public declaration history"
                )));
            }
        }

        if self.discard_actor_history.iter().any(|&actor| actor >= 4) {
            return Err(invalid_observation(
                "discard_actor_history contains an actor outside 0..=3".to_string(),
            ));
        }
        if !self.discard_actor_history.is_empty() || self.public_history_complete {
            let mut actor_counts = [0usize; 4];
            for &actor in &self.discard_actor_history {
                actor_counts[actor as usize] += 1;
            }
            for (seat, (&actual, discards)) in
                actor_counts.iter().zip(self.discards.iter()).enumerate()
            {
                if actual != discards.len() {
                    return Err(invalid_observation(format!(
                        "discard_actor_history contains {actual} entries for seat {seat}, expected {}",
                        discards.len()
                    )));
                }
            }
        }
        if self.resolved_discard_count > self.discard_actor_history.len() {
            return Err(invalid_observation(format!(
                "resolved_discard_count {} exceeds discard history length {}",
                self.resolved_discard_count,
                self.discard_actor_history.len()
            )));
        }
        let valid_tile_mask = (1u64 << OBS_TILE_TYPES) - 1;
        if self
            .public_temporary_safe_masks
            .iter()
            .any(|mask| mask & !valid_tile_mask != 0)
        {
            return Err(invalid_observation(
                "public_temporary_safe_masks contains an invalid tile type".to_string(),
            ));
        }
        Ok(())
    }

    pub fn find_action(&self, action_id: usize) -> Option<Action> {
        let encoder = ActionEncoder::FourPlayer;
        // Prefer non-red-five candidates so that 5m/5p/5s discards do not
        // accidentally drop the akadora when a normal 5 is also legal.
        let mut fallback: Option<&Action> = None;
        for action in &self._legal_actions {
            let Ok(idx) = encoder.encode(action) else {
                continue;
            };
            if (idx as usize) != action_id {
                continue;
            }
            if is_red_five_discard(action) {
                fallback.get_or_insert(action);
            } else {
                return Some(action.clone());
            }
        }
        fallback.cloned()
    }

    /// Resolve a red-aware v1 action ID without collapsing red and normal
    /// five choices. The legacy [`Self::find_action`] remains unchanged.
    pub fn find_action_v1(&self, action_id: usize) -> Option<Action> {
        self._legal_actions
            .iter()
            .find(|action| {
                ActionEncoderV1::FourPlayer
                    .encode(action)
                    .is_ok_and(|encoded| encoded >= 0 && encoded as usize == action_id)
            })
            .cloned()
    }

    /// Return absolute player indices in relative order: [self, shimocha, toimen, kamicha].
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub(crate) fn rel_order(&self) -> [usize; 4] {
        let pid = self.player_id as usize;
        [pid, (pid + 1) % 4, (pid + 2) % 4, (pid + 3) % 4]
    }

    pub fn new_events(&self) -> Vec<String> {
        self.events.clone()
    }

    /// Serialize this Observation to a base64-encoded JSON string.
    pub fn serialize_to_base64(&self) -> RiichiResult<String> {
        let json = serde_json::to_vec(self).map_err(|e| RiichiError::Serialization {
            message: format!("serialization failed: {e}"),
        })?;
        Ok(BASE64.encode(&json))
    }

    /// Deserialize an Observation from a base64-encoded JSON string.
    pub fn deserialize_from_base64(s: &str) -> RiichiResult<Self> {
        let bytes = BASE64.decode(s).map_err(|e| RiichiError::Serialization {
            message: format!("base64 decode failed: {e}"),
        })?;
        let obs: Observation =
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

fn validate_physical_tile(tile: u32, context: &str) -> RiichiResult<()> {
    if tile as usize >= TILES_4P {
        return Err(invalid_observation(format!(
            "{context} contains invalid physical tile id {tile}"
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

    let mut tile_types = meld
        .tiles
        .iter()
        .map(|tile| usize::from(*tile) / 4)
        .collect::<Vec<_>>();
    tile_types.sort_unstable();
    let valid_types = match meld.meld_type {
        MeldType::Chi => {
            tile_types[0] < 27
                && tile_types[0] / 9 == tile_types[2] / 9
                && tile_types[1] == tile_types[0] + 1
                && tile_types[2] == tile_types[1] + 1
        }
        MeldType::Pon | MeldType::Daiminkan | MeldType::Ankan | MeldType::Kakan => {
            tile_types.iter().all(|&tile| tile == tile_types[0])
        }
    };
    if !valid_types {
        return Err(invalid_observation(format!(
            "{:?} meld has an invalid tile-type composition",
            meld.meld_type
        )));
    }
    Ok(())
}

fn validate_visible_tile_counts(tiles: impl IntoIterator<Item = u32>) -> RiichiResult<()> {
    let mut counts = [0u16; OBS_TILE_TYPES];
    for tile in tiles {
        let tile_type = tile as usize / 4;
        let Some(count) = counts.get_mut(tile_type) else {
            return Err(invalid_observation(format!(
                "visible zone contains invalid physical tile id {tile}"
            )));
        };
        *count += 1;
        // Four physical copies plus at most four duplicated called tiles.
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
    let mut counts = [0u8; OBS_TILE_TYPES];
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

    fn obs_with_actions(actions: Vec<Action>) -> Observation {
        Observation::new(
            0,
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            [vec![], vec![], vec![], vec![]],
            vec![],
            [25000, 25000, 25000, 25000],
            [false, false, false, false],
            actions,
            vec![],
            0,
            0,
            0,
            0,
            0,
            vec![],
            false,
            [None, None, None, None],
            [None, None, None, None],
            None,
            None,
        )
    }

    fn discard(tile: u8) -> Action {
        Action::new(ActionType::Discard, Some(tile), vec![], Some(0))
    }

    #[test]
    fn find_action_prefers_non_red_5m() {
        // 5m action_id = 16 / 4 = 4; legal actions list red 5m first.
        let obs = obs_with_actions(vec![discard(16), discard(17), discard(18), discard(19)]);
        let chosen = obs.find_action(4).expect("discard 5m should resolve");
        assert_eq!(chosen.tile, Some(17), "non-red 5m must win over red 5m");
    }

    #[test]
    fn find_action_prefers_non_red_5p() {
        // 5p action_id = 52 / 4 = 13.
        let obs = obs_with_actions(vec![discard(52), discard(54)]);
        let chosen = obs.find_action(13).expect("discard 5p should resolve");
        assert_eq!(chosen.tile, Some(54));
    }

    #[test]
    fn find_action_prefers_non_red_5s() {
        // 5s action_id = 88 / 4 = 22.
        let obs = obs_with_actions(vec![discard(88), discard(90)]);
        let chosen = obs.find_action(22).expect("discard 5s should resolve");
        assert_eq!(chosen.tile, Some(90));
    }

    #[test]
    fn find_action_falls_back_to_red_when_only_red_legal() {
        // Only the red 5m is in the hand; we still need a working discard.
        let obs = obs_with_actions(vec![discard(16)]);
        let chosen = obs.find_action(4).expect("red-only 5m must still resolve");
        assert_eq!(chosen.tile, Some(16));
    }

    #[test]
    fn find_action_unaffected_for_non_five_tiles() {
        // 4m action_id = 12 / 4 = 3.
        let obs = obs_with_actions(vec![discard(12), discard(13)]);
        let chosen = obs.find_action(3).expect("discard 4m should resolve");
        assert_eq!(chosen.tile, Some(12), "first match wins for non-red ties");
    }

    #[test]
    fn legacy_public_history_fields_default_without_changing_wire() {
        let obs = obs_with_actions(vec![discard(0)]);
        let value = serde_json::to_value(&obs).expect("observation should serialize");
        let object = value.as_object().expect("observation should be an object");
        assert!(!object.contains_key("discard_is_riichi"));
        assert!(!object.contains_key("public_tsumogiri_history"));
        assert!(!object.contains_key("discard_actor_history"));
        assert!(!object.contains_key("resolved_discard_count"));
        assert!(!object.contains_key("public_temporary_safe_masks"));
        assert!(!object.contains_key("public_history_complete"));

        let restored: Observation =
            serde_json::from_value(value).expect("legacy observation should deserialize");
        assert_eq!(restored.discard_is_riichi, <[Vec<bool>; 4]>::default());
        assert_eq!(
            restored.public_tsumogiri_history,
            <[Vec<bool>; 4]>::default()
        );
        assert!(restored.discard_actor_history.is_empty());
        assert_eq!(restored.resolved_discard_count, 0);
        assert_eq!(restored.public_temporary_safe_masks, [0; 4]);
        assert!(!restored.public_history_complete);
    }

    #[test]
    fn complete_public_history_must_align_with_discards() {
        let mut obs = obs_with_actions(vec![discard(0)]);
        obs.discards[1] = vec![4, 8];
        obs.public_tsumogiri_history[1] = vec![false, true];
        obs.discard_is_riichi[1] = vec![true, false];
        obs.riichi_declared[1] = true;
        obs.discard_actor_history = vec![1, 1];
        obs.resolved_discard_count = 1;
        obs.public_history_complete = true;
        obs.validate_public_discard_history()
            .expect("aligned complete history should validate");

        obs.discard_actor_history.pop();
        assert!(obs.validate_public_discard_history().is_err());
    }

    #[test]
    fn meld_validation_rejects_non_sequence_chi_and_mixed_triplet() {
        let invalid_chi = Meld::new(MeldType::Chi, vec![0, 8, 12], true, 1, Some(0));
        assert!(validate_meld_shape(&invalid_chi).is_err());

        let invalid_pon = Meld::new(MeldType::Pon, vec![0, 1, 4], true, 1, Some(0));
        assert!(validate_meld_shape(&invalid_pon).is_err());

        let valid_chi = Meld::new(MeldType::Chi, vec![10, 2, 5], true, 1, Some(2));
        validate_meld_shape(&valid_chi).expect("physical copy order must not matter");
    }
}
