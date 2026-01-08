use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    #[pyo3(get, set)]
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    #[pyo3(get, set)]
    pub is_kokushi_musou_13machi_double: bool,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_tenhou()
    }
}

#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false))]
    pub fn new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
        }
    }

    #[staticmethod]
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
        }
    }

    #[staticmethod]
    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={})",
            self.allows_ron_on_ankan_for_kokushi_musou, self.is_kokushi_musou_13machi_double
        )
    }
}
