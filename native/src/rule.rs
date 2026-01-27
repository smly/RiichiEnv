use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[pyclass(module = "riichienv._riichienv", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KuikaeMode {
    None = 0,
    Basic = 1,
    StrictFlank = 2,
}

#[pyclass(module = "riichienv._riichienv")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    #[pyo3(get, set)]
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    #[pyo3(get, set)]
    pub is_kokushi_musou_13machi_double: bool,
    #[pyo3(get, set)]
    pub yakuman_pao_is_liability_only: bool,
    #[pyo3(get, set)]
    pub allow_double_ron: bool,
    #[pyo3(get, set)]
    pub kuikae_mode: KuikaeMode,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_mjsoul()
    }
}

#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, yakuman_pao_is_liability_only=false, allow_double_ron=true, kuikae_mode=KuikaeMode::None))]
    pub fn new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
        yakuman_pao_is_liability_only: bool,
        allow_double_ron: bool,
        kuikae_mode: Option<KuikaeMode>,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
            yakuman_pao_is_liability_only,
            allow_double_ron,
            kuikae_mode: kuikae_mode.unwrap_or(KuikaeMode::None),
        }
    }

    #[staticmethod]
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            yakuman_pao_is_liability_only: false,
            allow_double_ron: true,
            kuikae_mode: KuikaeMode::StrictFlank,
        }
    }

    #[staticmethod]
    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            yakuman_pao_is_liability_only: true,
            allow_double_ron: true,
            kuikae_mode: KuikaeMode::StrictFlank,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, yakuman_pao_is_liability_only={}, allow_double_ron={}, kuikae_mode={:?})",
            self.allows_ron_on_ankan_for_kokushi_musou, self.is_kokushi_musou_13machi_double, self.yakuman_pao_is_liability_only, self.allow_double_ron, self.kuikae_mode
        )
    }
}
