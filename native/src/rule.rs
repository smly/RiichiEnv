use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[pyclass(module = "riichienv._riichienv", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KuikaeMode {
    None = 0,
    Basic = 1,
    StrictFlank = 2,
}

#[pyclass(module = "riichienv._riichienv", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KanDoraTimingMode {
    /// Tenhou style: All kan types reveal dora after discard (後めくり)
    /// - Ankan: after discard
    /// - Daiminkan/Kakan: after discard
    TenhouImmediate = 0,
    /// Majsoul style: Ankan reveals immediately, Daiminkan/Kakan after discard or before next kan
    /// - Ankan: immediately after kan declaration (即めくり)
    /// - Daiminkan/Kakan: after discard, in chronological order of kan declarations
    MajsoulImmediate = 1,
    /// All kans reveal dora after discard (same as TenhouImmediate)
    AfterDiscard = 2,
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
    #[pyo3(get, set)]
    pub kan_dora_timing: KanDoraTimingMode,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_mjsoul()
    }
}

#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, yakuman_pao_is_liability_only=false, allow_double_ron=true, kuikae_mode=KuikaeMode::StrictFlank, kan_dora_timing=KanDoraTimingMode::TenhouImmediate))]
    pub fn new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
        yakuman_pao_is_liability_only: bool,
        allow_double_ron: bool,
        kuikae_mode: Option<KuikaeMode>,
        kan_dora_timing: Option<KanDoraTimingMode>,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
            yakuman_pao_is_liability_only,
            allow_double_ron,
            kuikae_mode: kuikae_mode.unwrap_or(KuikaeMode::StrictFlank),
            kan_dora_timing: kan_dora_timing.unwrap_or(KanDoraTimingMode::TenhouImmediate),
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
            kan_dora_timing: KanDoraTimingMode::TenhouImmediate,
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
            kan_dora_timing: KanDoraTimingMode::MajsoulImmediate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, yakuman_pao_is_liability_only={}, allow_double_ron={}, kuikae_mode={:?}, kan_dora_timing={:?})",
            self.allows_ron_on_ankan_for_kokushi_musou, self.is_kokushi_musou_13machi_double, self.yakuman_pao_is_liability_only, self.allow_double_ron, self.kuikae_mode, self.kan_dora_timing
        )
    }
}
