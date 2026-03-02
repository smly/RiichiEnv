use pyo3::prelude::*;

mod env;

#[pyfunction]
#[pyo3(name = "calculate_score", signature = (han, fu, is_oya, is_tsumo, honba, num_players=4))]
fn calculate_score_py(
    han: u8,
    fu: u8,
    is_oya: bool,
    is_tsumo: bool,
    honba: u32,
    num_players: u8,
) -> riichienv_core::score::Score {
    riichienv_core::score::calculate_score(han, fu, is_oya, is_tsumo, honba, num_players)
}

#[pyfunction]
#[pyo3(name = "parse_hand")]
fn parse_hand_py(text: &str) -> PyResult<(Vec<u32>, Vec<riichienv_core::types::Meld>)> {
    riichienv_core::parser::parse_hand(text).map_err(Into::into)
}

#[pyfunction]
#[pyo3(name = "parse_tile")]
fn parse_tile_py(text: &str) -> PyResult<u8> {
    riichienv_core::parser::parse_tile(text).map_err(Into::into)
}

#[pyfunction]
#[pyo3(name = "check_riichi_candidates")]
fn check_riichi_candidates_py(tiles_136: Vec<u8>) -> Vec<u32> {
    riichienv_core::check_riichi_candidates(tiles_136)
}

#[pyfunction]
#[pyo3(name = "calculate_shanten")]
fn calculate_shanten_py(hand_tiles: Vec<u32>) -> i32 {
    riichienv_core::shanten::calculate_shanten(&hand_tiles)
}

#[pyfunction]
#[pyo3(name = "calculate_shanten_3p")]
fn calculate_shanten_3p_py(hand_tiles: Vec<u32>) -> i32 {
    riichienv_core::shanten::calculate_shanten_3p(&hand_tiles)
}

#[pymodule]
fn _riichienv(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<riichienv_core::types::Meld>()?;
    m.add_class::<riichienv_core::types::MeldType>()?;
    m.add_class::<riichienv_core::types::Wind>()?;
    m.add_class::<riichienv_core::types::Conditions>()?;
    m.add_class::<riichienv_core::types::WinResult>()?;
    m.add_class::<riichienv_core::score::Score>()?;
    m.add_class::<riichienv_core::hand_evaluator::HandEvaluator>()?;
    m.add_class::<riichienv_core::hand_evaluator_3p::HandEvaluator3P>()?;
    m.add_class::<riichienv_core::replay::MjSoulReplay>()?;
    m.add_class::<riichienv_core::replay::MjaiReplay>()?;
    m.add_class::<riichienv_core::replay::LogKyoku>()?;
    m.add_class::<riichienv_core::replay::mjsoul_replay::KyokuIterator>()?;
    m.add_class::<riichienv_core::replay::WinResultContext>()?;
    m.add_class::<riichienv_core::replay::WinResultContextIterator>()?;
    m.add_class::<riichienv_core::rule::GameRule>()?;
    m.add_class::<riichienv_core::yaku::Yaku>()?;

    // Env classes
    m.add_class::<riichienv_core::action::ActionType>()?;
    m.add_class::<riichienv_core::action::Phase>()?;
    m.add_class::<riichienv_core::action::Action>()?;
    m.add_class::<riichienv_core::action::Action3P>()?;
    m.add_class::<riichienv_core::observation::Observation>()?;
    m.add_class::<riichienv_core::observation_3p::Observation3P>()?;
    m.add_class::<env::RiichiEnv>()?;

    m.add_function(wrap_pyfunction!(calculate_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_hand_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tile_py, m)?)?;
    m.add_function(wrap_pyfunction!(check_riichi_candidates_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shanten_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_shanten_3p_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        riichienv_core::yaku::get_yaku_by_id_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(riichienv_core::yaku::get_all_yaku_py, m)?)?;
    Ok(())
}
