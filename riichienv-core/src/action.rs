#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{RiichiError, RiichiResult};
use crate::parser::tid_to_mjai;

pub const ACTION_SPACE_4P: usize = 82;
pub const ACTION_SPACE_3P: usize = 60;
/// Red-aware v1 action spaces. Each legacy semantic ID owns two slots:
/// `2 * v0` preserves/does not consume a red five and `2 * v0 + 1`
/// explicitly discards or consumes one. Unused odd slots remain masked out.
pub const ACTION_SPACE_4P_V1: usize = ACTION_SPACE_4P * 2;
pub const ACTION_SPACE_3P_V1: usize = ACTION_SPACE_3P * 2;

const TILE34_TO_COMPACT: [u8; 34] = [
    0, // type  0: 1m
    255, 255, 255, 255, 255, 255, 255, // type 1-7: 2m-8m (invalid)
    1,   // type  8: 9m
    2, 3, 4, 5, 6, 7, 8, 9, 10, // type  9-17: 1p-9p
    11, 12, 13, 14, 15, 16, 17, 18, 19, // type 18-26: 1s-9s
    20, 21, 22, 23, // type 27-30: ESWN
    24, 25, 26, // type 31-33: PFC
];

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", eq, eq_int, from_py_object)
)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    WaitAct = 0,
    WaitResponse = 1,
}

#[cfg(feature = "python")]
#[pymethods]
impl Phase {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

#[cfg_attr(
    feature = "python",
    pyclass(
        module = "riichienv._riichienv",
        eq,
        eq_int,
        from_py_object,
        rename_all = "SCREAMING_SNAKE_CASE"
    )
)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    Discard = 0,
    Chi = 1,
    Pon = 2,
    Daiminkan = 3,
    Ron = 4,
    Riichi = 5,
    Tsumo = 6,
    Pass = 7,
    Ankan = 8,
    Kakan = 9,
    KyushuKyuhai = 10,
    Kita = 11,
}

#[cfg(feature = "python")]
#[pymethods]
impl ActionType {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", from_py_object)
)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Action {
    pub action_type: ActionType,
    pub tile: Option<u8>,
    pub consume_tiles: Vec<u8>,
    pub actor: Option<u8>,
}

impl Action {
    pub fn new(
        r#type: ActionType,
        tile: Option<u8>,
        consume_tiles: Vec<u8>,
        actor: Option<u8>,
    ) -> Self {
        let mut sorted_consume = consume_tiles;
        sorted_consume.sort();
        Self {
            action_type: r#type,
            tile,
            consume_tiles: sorted_consume,
            actor,
        }
    }

    pub fn to_mjai(&self) -> String {
        let type_str = match self.action_type {
            ActionType::Discard => "dahai",
            ActionType::Chi => "chi",
            ActionType::Pon => "pon",
            ActionType::Daiminkan => "daiminkan",
            ActionType::Ankan => "ankan",
            ActionType::Kakan => "kakan",
            ActionType::Riichi => "reach",
            ActionType::Tsumo | ActionType::Ron => "hora",
            ActionType::KyushuKyuhai => "ryukyoku",
            ActionType::Kita => "kita",
            ActionType::Pass => "none",
        };

        let mut data = serde_json::Map::new();
        data.insert("type".to_string(), Value::String(type_str.to_string()));

        if let Some(actor) = self.actor {
            data.insert(
                "actor".to_string(),
                Value::Number(serde_json::Number::from(actor)),
            );
        }

        if let Some(t) = self.tile
            && self.action_type != ActionType::Tsumo
            && self.action_type != ActionType::Ron
            && self.action_type != ActionType::Riichi
        {
            data.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
        }

        if !self.consume_tiles.is_empty() {
            let cons: Vec<String> = self.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
            data.insert(
                "consumed".to_string(),
                serde_json::to_value(cons).expect("valid JSON"),
            );
        }

        Value::Object(data).to_string()
    }

    pub fn repr(&self) -> String {
        format!(
            "Action(action_type={:?}, tile={:?}, consume_tiles={:?}, actor={:?})",
            self.action_type, self.tile, self.consume_tiles, self.actor
        )
    }

    pub fn encode(&self) -> RiichiResult<i32> {
        match self.action_type {
            ActionType::Discard => {
                if let Some(tile) = self.tile {
                    Ok((tile as i32) / 4)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Discard action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Riichi => Ok(37),
            ActionType::Chi => {
                if let Some(target) = self.tile {
                    let target_34 = (target as i32) / 4;
                    let mut tiles_34: Vec<i32> =
                        self.consume_tiles.iter().map(|&x| (x as i32) / 4).collect();
                    tiles_34.push(target_34);
                    tiles_34.sort();
                    tiles_34.dedup();

                    if tiles_34.len() != 3 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Invalid Chi tiles: target={}, consumed={:?}",
                                target, self.consume_tiles
                            ),
                        });
                    }

                    if target_34 == tiles_34[0] {
                        Ok(38) // Low
                    } else if target_34 == tiles_34[1] {
                        Ok(39) // Mid
                    } else {
                        Ok(40) // High
                    }
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Chi action requires a target tile".to_string(),
                    })
                }
            }
            ActionType::Pon => Ok(41),
            ActionType::Daiminkan => {
                if let Some(tile) = self.tile {
                    Ok(42 + (tile as i32) / 4)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Daiminkan action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Ankan | ActionType::Kakan => {
                if let Some(first) = self.consume_tiles.first() {
                    Ok(42 + (*first as i32) / 4)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Ankan/Kakan action requires consumed tiles".to_string(),
                    })
                }
            }
            ActionType::Ron | ActionType::Tsumo => Ok(79),
            ActionType::KyushuKyuhai => Ok(80),
            ActionType::Pass => Ok(81),
            ActionType::Kita => Err(RiichiError::InvalidAction {
                message: "Kita action is not valid in 4-player mode".to_string(),
            }),
        }
    }

    /// Encode with the red-aware 4-player v1 model ABI.
    pub fn encode_v1(&self) -> RiichiResult<i32> {
        ActionEncoderV1::FourPlayer.encode(self)
    }
}

/// Separate encoder objects for 4P (default) and 3P action spaces.
///
/// `Action::encode()` always returns the 4P encoding (82 IDs, 0-81).
/// For 3P compact encoding (60 IDs), use `ActionEncoder::ThreePlayer`.
#[derive(Debug, Clone, Copy)]
pub enum ActionEncoder {
    FourPlayer,
    ThreePlayer,
}

impl ActionEncoder {
    pub fn from_num_players(n: u8) -> Self {
        match n {
            3 => Self::ThreePlayer,
            _ => Self::FourPlayer,
        }
    }

    pub fn action_space_size(&self) -> usize {
        match self {
            Self::FourPlayer => ACTION_SPACE_4P,
            Self::ThreePlayer => ACTION_SPACE_3P,
        }
    }

    pub fn encode(&self, action: &Action) -> RiichiResult<i32> {
        match self {
            Self::FourPlayer => action.encode(),
            Self::ThreePlayer => Self::encode_3p(action),
        }
    }

    fn encode_3p(action: &Action) -> RiichiResult<i32> {
        match action.action_type {
            ActionType::Discard => {
                if let Some(tile) = action.tile {
                    let tile_type = (tile / 4) as usize;
                    if tile_type >= 34 {
                        return Err(RiichiError::InvalidAction {
                            message: format!("Invalid tile type {} for 3P encode", tile_type),
                        });
                    }
                    let compact = TILE34_TO_COMPACT[tile_type];
                    if compact == 255 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Tile type {} (manzu 2-8) is not valid in 3P mode",
                                tile_type
                            ),
                        });
                    }
                    Ok(compact as i32)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Discard action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Riichi => Ok(27),
            ActionType::Chi => Err(RiichiError::InvalidAction {
                message: "Chi is not allowed in 3P mode".to_string(),
            }),
            ActionType::Pon => Ok(28),
            ActionType::Daiminkan => {
                if let Some(tile) = action.tile {
                    let tile_type = (tile / 4) as usize;
                    if tile_type >= 34 {
                        return Err(RiichiError::InvalidAction {
                            message: format!("Invalid tile type {} for 3P encode", tile_type),
                        });
                    }
                    let compact = TILE34_TO_COMPACT[tile_type];
                    if compact == 255 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Tile type {} (manzu 2-8) is not valid in 3P mode",
                                tile_type
                            ),
                        });
                    }
                    Ok(29 + compact as i32)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Daiminkan action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Ankan | ActionType::Kakan => {
                if let Some(first) = action.consume_tiles.first() {
                    let tile_type = (*first / 4) as usize;
                    if tile_type >= 34 {
                        return Err(RiichiError::InvalidAction {
                            message: format!("Invalid tile type {} for 3P encode", tile_type),
                        });
                    }
                    let compact = TILE34_TO_COMPACT[tile_type];
                    if compact == 255 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Tile type {} (manzu 2-8) is not valid in 3P mode",
                                tile_type
                            ),
                        });
                    }
                    Ok(29 + compact as i32)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Ankan/Kakan action requires consumed tiles".to_string(),
                    })
                }
            }
            ActionType::Ron | ActionType::Tsumo => Ok(56),
            ActionType::KyushuKyuhai => Ok(57),
            ActionType::Pass => Ok(58),
            ActionType::Kita => Ok(59),
        }
    }
}

/// Versioned model action encoder introduced for #210.
///
/// Unlike v0, physically distinct red-five discard/chi/pon choices receive
/// different IDs. Forced actions (ron, tsumo, kan, pass, kita, etc.) keep the
/// even slot because no red-preservation decision exists for the model.
#[derive(Debug, Clone, Copy)]
pub enum ActionEncoderV1 {
    FourPlayer,
    ThreePlayer,
}

impl ActionEncoderV1 {
    pub fn from_num_players(n: u8) -> Self {
        match n {
            3 => Self::ThreePlayer,
            _ => Self::FourPlayer,
        }
    }

    pub fn action_space_size(self) -> usize {
        match self {
            Self::FourPlayer => ACTION_SPACE_4P_V1,
            Self::ThreePlayer => ACTION_SPACE_3P_V1,
        }
    }

    pub fn encode(self, action: &Action) -> RiichiResult<i32> {
        let legacy = match self {
            Self::FourPlayer => ActionEncoder::FourPlayer.encode(action)?,
            Self::ThreePlayer => ActionEncoder::ThreePlayer.encode(action)?,
        };
        Ok(legacy * 2 + i32::from(red_choice_bit(action)))
    }
}

fn red_choice_bit(action: &Action) -> u8 {
    let is_red = |tile: u8| matches!(tile, 16 | 52 | 88);
    match action.action_type {
        ActionType::Discard => u8::from(action.tile.is_some_and(is_red)),
        ActionType::Chi | ActionType::Pon => {
            u8::from(action.consume_tiles.iter().copied().any(is_red))
        }
        _ => 0,
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (r#type=ActionType::Pass, tile=None, consume_tiles=vec![], actor=None))]
    pub fn py_new(
        r#type: ActionType,
        tile: Option<u8>,
        consume_tiles: Vec<u8>,
        actor: Option<u8>,
    ) -> Self {
        Self::new(r#type, tile, consume_tiles, actor)
    }

    #[pyo3(name = "to_dict")]
    pub fn to_dict_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("type", self.action_type as i32)?;
        dict.set_item("tile", self.tile)?;

        let cons: Vec<u32> = self.consume_tiles.iter().map(|&x| x as u32).collect();
        dict.set_item("consume_tiles", cons)?;
        dict.set_item("actor", self.actor)?;
        Ok(dict.unbind().into())
    }

    #[pyo3(name = "to_mjai")]
    pub fn to_mjai_py(&self) -> PyResult<String> {
        Ok(self.to_mjai())
    }

    fn __repr__(&self) -> String {
        self.repr()
    }

    fn __str__(&self) -> String {
        self.repr()
    }

    #[getter]
    fn get_action_type(&self) -> ActionType {
        self.action_type
    }

    #[setter]
    fn set_action_type(&mut self, action_type: ActionType) {
        self.action_type = action_type;
    }

    #[getter]
    fn get_tile(&self) -> Option<u8> {
        self.tile
    }

    #[setter]
    fn set_tile(&mut self, tile: Option<u8>) {
        self.tile = tile;
    }

    #[getter]
    fn get_consume_tiles(&self) -> Vec<u32> {
        self.consume_tiles.iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_consume_tiles(&mut self, value: Vec<u8>) {
        self.consume_tiles = value;
    }

    #[getter]
    fn get_actor(&self) -> Option<u8> {
        self.actor
    }

    #[setter]
    fn set_actor(&mut self, actor: Option<u8>) {
        self.actor = actor;
    }

    #[pyo3(name = "encode")]
    pub fn encode_py(&self) -> PyResult<i32> {
        self.encode().map_err(Into::into)
    }

    #[pyo3(name = "encode_v1")]
    pub fn encode_v1_py(&self) -> PyResult<i32> {
        self.encode_v1().map_err(Into::into)
    }
}

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", from_py_object)
)]
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::Into,
)]
#[serde(transparent)]
pub struct Action3P(pub Action);

impl Action3P {
    pub fn from_action(action: Action) -> Self {
        Action3P(action)
    }

    pub fn encode(&self) -> RiichiResult<i32> {
        ActionEncoder::ThreePlayer.encode(&self.0)
    }

    pub fn encode_v1(&self) -> RiichiResult<i32> {
        ActionEncoderV1::ThreePlayer.encode(&self.0)
    }

    pub fn repr(&self) -> String {
        format!(
            "Action3P(action_type={:?}, tile={:?}, consume_tiles={:?}, actor={:?})",
            self.0.action_type, self.0.tile, self.0.consume_tiles, self.0.actor
        )
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Action3P {
    #[new]
    #[pyo3(signature = (r#type=ActionType::Pass, tile=None, consume_tiles=vec![], actor=None))]
    pub fn py_new(
        r#type: ActionType,
        tile: Option<u8>,
        consume_tiles: Vec<u8>,
        actor: Option<u8>,
    ) -> Self {
        Action3P(Action::new(r#type, tile, consume_tiles, actor))
    }

    #[pyo3(name = "encode")]
    pub fn encode_py(&self) -> PyResult<i32> {
        self.encode().map_err(Into::into)
    }

    #[pyo3(name = "encode_v1")]
    pub fn encode_v1_py(&self) -> PyResult<i32> {
        self.encode_v1().map_err(Into::into)
    }

    #[pyo3(name = "to_dict")]
    pub fn to_dict_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.0.to_dict_py(py)
    }

    #[pyo3(name = "to_mjai")]
    pub fn to_mjai_py(&self) -> PyResult<String> {
        Ok(self.0.to_mjai())
    }

    #[getter]
    fn get_action_type(&self) -> ActionType {
        self.0.action_type
    }

    #[setter]
    fn set_action_type(&mut self, action_type: ActionType) {
        self.0.action_type = action_type;
    }

    #[getter]
    fn get_tile(&self) -> Option<u8> {
        self.0.tile
    }

    #[setter]
    fn set_tile(&mut self, tile: Option<u8>) {
        self.0.tile = tile;
    }

    #[getter]
    fn get_consume_tiles(&self) -> Vec<u32> {
        self.0.consume_tiles.iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_consume_tiles(&mut self, value: Vec<u8>) {
        self.0.consume_tiles = value;
    }

    #[getter]
    fn get_actor(&self) -> Option<u8> {
        self.0.actor
    }

    #[setter]
    fn set_actor(&mut self, actor: Option<u8>) {
        self.0.actor = actor;
    }

    fn __repr__(&self) -> String {
        self.repr()
    }

    fn __str__(&self) -> String {
        self.repr()
    }
}

#[cfg(test)]
mod red_aware_tests {
    use super::*;

    #[test]
    fn v1_separates_red_and_normal_five_discards_without_changing_v0() {
        for (encoder_v0, encoder_v1, red, normal) in [
            (
                ActionEncoder::FourPlayer,
                ActionEncoderV1::FourPlayer,
                16,
                17,
            ),
            (
                ActionEncoder::ThreePlayer,
                ActionEncoderV1::ThreePlayer,
                52,
                53,
            ),
        ] {
            let red = Action::new(ActionType::Discard, Some(red), vec![], None);
            let normal = Action::new(ActionType::Discard, Some(normal), vec![], None);
            assert_eq!(
                encoder_v0.encode(&red).unwrap(),
                encoder_v0.encode(&normal).unwrap()
            );
            assert_eq!(
                encoder_v1.encode(&red).unwrap(),
                encoder_v1.encode(&normal).unwrap() + 1
            );
        }
    }

    #[test]
    fn v1_separates_red_consuming_calls_but_not_forced_kans() {
        let normal_pon = Action::new(ActionType::Pon, Some(53), vec![54, 55], Some(1));
        let red_pon = Action::new(ActionType::Pon, Some(53), vec![52, 54], Some(1));
        assert_eq!(
            ActionEncoder::FourPlayer.encode(&normal_pon).unwrap(),
            ActionEncoder::FourPlayer.encode(&red_pon).unwrap()
        );
        assert_eq!(
            ActionEncoderV1::FourPlayer.encode(&red_pon).unwrap(),
            ActionEncoderV1::FourPlayer.encode(&normal_pon).unwrap() + 1
        );

        let normal_chi = Action::new(ActionType::Chi, Some(45), vec![49, 53], Some(1));
        let red_chi = Action::new(ActionType::Chi, Some(45), vec![49, 52], Some(1));
        assert_eq!(
            ActionEncoder::FourPlayer.encode(&normal_chi).unwrap(),
            ActionEncoder::FourPlayer.encode(&red_chi).unwrap()
        );
        assert_eq!(
            ActionEncoderV1::FourPlayer.encode(&red_chi).unwrap(),
            ActionEncoderV1::FourPlayer.encode(&normal_chi).unwrap() + 1
        );

        assert_eq!(
            ActionEncoderV1::ThreePlayer.encode(&red_pon).unwrap(),
            ActionEncoderV1::ThreePlayer.encode(&normal_pon).unwrap() + 1
        );

        let kan = Action::new(ActionType::Ankan, Some(52), vec![52, 53, 54, 55], None);
        assert_eq!(ActionEncoderV1::FourPlayer.encode(&kan).unwrap() % 2, 0);
    }

    #[test]
    fn v1_action_space_sizes_are_explicit() {
        assert_eq!(ActionEncoderV1::FourPlayer.action_space_size(), 164);
        assert_eq!(ActionEncoderV1::ThreePlayer.action_space_size(), 120);
    }
}
