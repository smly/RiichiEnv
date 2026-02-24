#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", get_all, set_all)
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

    /// Whether open kan (Daiminkan/Kakan) dora is revealed after the discard.
    /// - `true`: dora revealed after discard (Mahjong Soul style)
    /// - `false`: dora revealed before discard (Tenhou style)
    ///
    /// Note: Ankan (closed kan) always reveals dora immediately (before rinshan tsumo),
    /// regardless of this flag.
    pub open_kan_dora_after_discard: bool,
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
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: true,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: false,
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
            open_kan_dora_after_discard: true,
        }
    }

    pub fn default_mjsoul_sanma() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            is_suuankou_tanki_double: true,
            is_junsei_chuurenpoutou_double: true,
            is_daisuushii_double: true,
            yakuman_pao_is_liability_only: true,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: true,
        }
    }

    pub fn default_tenhou_sanma() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            open_kan_dora_after_discard: false,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, is_suuankou_tanki_double=false, is_junsei_chuurenpoutou_double=false, is_daisuushii_double=false, yakuman_pao_is_liability_only=false, sanchaho_is_draw=false, kuikae_forbidden=true, open_kan_dora_after_discard=false))]
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
        open_kan_dora_after_discard: bool,
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
            open_kan_dora_after_discard,
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
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, is_suuankou_tanki_double={}, is_junsei_chuurenpoutou_double={}, is_daisuushii_double={}, yakuman_pao_is_liability_only={}, sanchaho_is_draw={}, kuikae_forbidden={}, open_kan_dora_after_discard={})",
            self.allows_ron_on_ankan_for_kokushi_musou, self.is_kokushi_musou_13machi_double, self.is_suuankou_tanki_double, self.is_junsei_chuurenpoutou_double, self.is_daisuushii_double, self.yakuman_pao_is_liability_only, self.sanchaho_is_draw, self.kuikae_forbidden, self.open_kan_dora_after_discard
        )
    }
}
