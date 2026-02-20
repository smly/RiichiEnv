#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", eq, eq_int)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KuikaeMode {
    None = 0,
    Basic = 1,
    StrictFlank = 2,
}

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", eq, eq_int)
)]
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

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", get_all, set_all)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    pub is_kokushi_musou_13machi_double: bool,
    pub yakuman_pao_is_liability_only: bool,
    pub allow_double_ron: bool,
    pub kuikae_mode: KuikaeMode,
    pub kan_dora_timing: KanDoraTimingMode,
    pub is_sanma: bool,
    pub allow_kita: bool,
    pub sanma_tsumo_zon: bool,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_mjsoul()
    }
}

impl GameRule {
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            yakuman_pao_is_liability_only: false,
            allow_double_ron: true,
            kuikae_mode: KuikaeMode::StrictFlank,
            kan_dora_timing: KanDoraTimingMode::TenhouImmediate,
            is_sanma: false,
            allow_kita: false,
            sanma_tsumo_zon: false,
        }
    }

    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            yakuman_pao_is_liability_only: true,
            allow_double_ron: true,
            kuikae_mode: KuikaeMode::StrictFlank,
            kan_dora_timing: KanDoraTimingMode::MajsoulImmediate,
            is_sanma: false,
            allow_kita: false,
            sanma_tsumo_zon: false,
        }
    }

    pub fn default_tenhou_sanma() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            yakuman_pao_is_liability_only: false,
            allow_double_ron: true,
            kuikae_mode: KuikaeMode::StrictFlank,
            kan_dora_timing: KanDoraTimingMode::TenhouImmediate,
            is_sanma: true,
            allow_kita: true,
            sanma_tsumo_zon: true,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, yakuman_pao_is_liability_only=false, allow_double_ron=true, kuikae_mode=KuikaeMode::StrictFlank, kan_dora_timing=KanDoraTimingMode::TenhouImmediate, is_sanma=false, allow_kita=false, sanma_tsumo_zon=false))]
    pub fn py_new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
        yakuman_pao_is_liability_only: bool,
        allow_double_ron: bool,
        kuikae_mode: Option<KuikaeMode>,
        kan_dora_timing: Option<KanDoraTimingMode>,
        is_sanma: bool,
        allow_kita: bool,
        sanma_tsumo_zon: bool,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
            yakuman_pao_is_liability_only,
            allow_double_ron,
            kuikae_mode: kuikae_mode.unwrap_or(KuikaeMode::StrictFlank),
            kan_dora_timing: kan_dora_timing.unwrap_or(KanDoraTimingMode::TenhouImmediate),
            is_sanma,
            allow_kita,
            sanma_tsumo_zon,
        }
    }

    #[staticmethod]
    #[pyo3(name = "default_tenhou")]
    pub fn py_default_tenhou() -> Self {
        Self::default_tenhou()
    }

    #[staticmethod]
    #[pyo3(name = "default_mjsoul")]
    pub fn py_default_mjsoul() -> Self {
        Self::default_mjsoul()
    }

    #[staticmethod]
    #[pyo3(name = "default_tenhou_sanma")]
    pub fn py_default_tenhou_sanma() -> Self {
        Self::default_tenhou_sanma()
    }

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, yakuman_pao_is_liability_only={}, allow_double_ron={}, kuikae_mode={:?}, kan_dora_timing={:?}, is_sanma={}, allow_kita={}, sanma_tsumo_zon={})",
            self.allows_ron_on_ankan_for_kokushi_musou, self.is_kokushi_musou_13machi_double, self.yakuman_pao_is_liability_only, self.allow_double_ron, self.kuikae_mode, self.kan_dora_timing, self.is_sanma, self.allow_kita, self.sanma_tsumo_zon
        )
    }
}
