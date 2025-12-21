use pyo3::prelude::*;

mod agari;
mod agari_calculator;
mod score;
mod tests;
mod types;
mod yaku;

mod parser;
mod replay;

#[pymodule]
fn _riichienv(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<types::Meld>()?;
    m.add_class::<types::MeldType>()?;
    m.add_class::<types::Wind>()?;
    m.add_class::<types::Conditions>()?;
    m.add_class::<types::Agari>()?;
    m.add_class::<score::Score>()?;
    m.add_class::<agari_calculator::AgariCalculator>()?;
    m.add_class::<replay::ReplayGame>()?;
    m.add_class::<replay::Kyoku>()?;
    m.add_class::<replay::KyokuIterator>()?;
    m.add_class::<replay::AgariContext>()?;
    m.add_class::<replay::AgariContextIterator>()?;
    m.add_function(wrap_pyfunction!(score::calculate_score, m)?)?;
    m.add_function(wrap_pyfunction!(parser::parse_hand, m)?)?;
    m.add_function(wrap_pyfunction!(parser::parse_tile, m)?)?;
    Ok(())
}
