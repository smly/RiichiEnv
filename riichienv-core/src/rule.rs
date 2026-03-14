#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", get_all, set_all, from_py_object)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    pub is_kokushi_musou_13machi_double: bool,
    pub is_suuankou_tanki_double: bool,
    pub is_junsei_chuurenpoutou_double: bool,
    pub is_daisuushii_double: bool,
    pub yakuman_pao_is_liability_only: bool,
    pub sanchaho_is_draw: bool,

    pub kuikae_forbidden: bool,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_tenhou()
    }
}

impl GameRule {
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: true,

            kuikae_forbidden: true,
        }
    }

    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            is_suuankou_tanki_double: true,
            is_junsei_chuurenpoutou_double: true,
            is_daisuushii_double: true,
            yakuman_pao_is_liability_only: true,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, is_suuankou_tanki_double=false, is_junsei_chuurenpoutou_double=false, is_daisuushii_double=false, yakuman_pao_is_liability_only=false, sanchaho_is_draw=false, kuikae_forbidden=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
        is_suuankou_tanki_double: bool,
        is_junsei_chuurenpoutou_double: bool,
        is_daisuushii_double: bool,
        yakuman_pao_is_liability_only: bool,
        sanchaho_is_draw: bool,
        kuikae_forbidden: bool,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
            is_suuankou_tanki_double,
            is_junsei_chuurenpoutou_double,
            is_daisuushii_double,
            yakuman_pao_is_liability_only,
            sanchaho_is_draw,
            kuikae_forbidden,
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

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, is_suuankou_tanki_double={}, is_junsei_chuurenpoutou_double={}, is_daisuushii_double={}, yakuman_pao_is_liability_only={}, sanchaho_is_draw={}, kuikae_forbidden={})",
            self.allows_ron_on_ankan_for_kokushi_musou,
            self.is_kokushi_musou_13machi_double,
            self.is_suuankou_tanki_double,
            self.is_junsei_chuurenpoutou_double,
            self.is_daisuushii_double,
            self.yakuman_pao_is_liability_only,
            self.sanchaho_is_draw,
            self.kuikae_forbidden
        )
    }
}
