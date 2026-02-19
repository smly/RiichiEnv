#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{RiichiError, RiichiResult};
use crate::parser::tid_to_mjai;

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", eq, eq_int)
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
    pyclass(module = "riichienv._riichienv", eq, eq_int)
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
}

#[cfg(feature = "python")]
#[pymethods]
impl ActionType {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

#[cfg_attr(feature = "python", pyclass(module = "riichienv._riichienv"))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Action {
    pub action_type: ActionType,
    pub tile: Option<u8>,
    pub consume_tiles: Vec<u8>,
}

impl Action {
    pub fn new(r#type: ActionType, tile: Option<u8>, consume_tiles: Vec<u8>) -> Self {
        let mut sorted_consume = consume_tiles;
        sorted_consume.sort();
        Self {
            action_type: r#type,
            tile,
            consume_tiles: sorted_consume,
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
            ActionType::Pass => "none",
        };

        let mut data = serde_json::Map::new();
        data.insert("type".to_string(), Value::String(type_str.to_string()));

        if let Some(t) = self.tile {
            if self.action_type != ActionType::Tsumo
                && self.action_type != ActionType::Ron
                && self.action_type != ActionType::Riichi
            {
                data.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
            }
        }

        if !self.consume_tiles.is_empty() {
            let cons: Vec<String> = self.consume_tiles.iter().map(|&t| tid_to_mjai(t)).collect();
            data.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
        }

        Value::Object(data).to_string()
    }

    pub fn repr(&self) -> String {
        format!(
            "Action(action_type={:?}, tile={:?}, consume_tiles={:?})",
            self.action_type, self.tile, self.consume_tiles
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
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (r#type=ActionType::Pass, tile=None, consume_tiles=vec![]))]
    pub fn py_new(r#type: ActionType, tile: Option<u8>, consume_tiles: Vec<u8>) -> Self {
        Self::new(r#type, tile, consume_tiles)
    }

    #[pyo3(name = "to_dict")]
    pub fn to_dict_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("type", self.action_type as i32)?;
        dict.set_item("tile", self.tile)?;

        let cons: Vec<u32> = self.consume_tiles.iter().map(|&x| x as u32).collect();
        dict.set_item("consume_tiles", cons)?;
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

    #[pyo3(name = "encode")]
    pub fn encode_py(&self) -> PyResult<i32> {
        self.encode().map_err(Into::into)
    }
}
