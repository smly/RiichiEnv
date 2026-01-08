use pyo3::prelude::*;

mod agari;
mod agari_calculator;
mod score;
mod tests;
mod types;
mod yaku;

mod env;
mod parser;
mod replay;
mod rule;

#[pyfunction]
fn check_riichi_candidates(tiles_136: Vec<u8>) -> Vec<u32> {
    let mut candidates = Vec::new();
    // Convert to 34-tile hand
    let mut tiles_34 = Vec::with_capacity(tiles_136.len());
    for t in &tiles_136 {
        tiles_34.push(t / 4);
    }

    // We iterate each tile in 136-tiles as a discard candidate.
    // However, since we care about valid discards that lead to Tenpai,
    // and multiple 136-tiles map to the same 34-tile, we can optimize or check simply.
    // For exact 136 return, we should check each unique 136 tile in hand.

    // Count uniqueness to avoid redundant checks?
    // Actually, simple iteration is fine for 14 tiles.

    for (i, &t_discard) in tiles_136.iter().enumerate() {
        // Construct hand effectively by removing this tile
        // Re-construct Hand from scratch is safer/easier than modifying mutable hand repeatedly?
        // Or remove from `tiles_34`.

        let mut hand = types::Hand::default();
        for (j, &t) in tiles_34.iter().enumerate() {
            if i != j {
                hand.add(t);
            }
        }

        if agari::is_tenpai(&mut hand) {
            candidates.push(t_discard as u32);
        }
    }
    candidates
}

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
    m.add_class::<rule::GameRule>()?;

    // Env classes
    m.add_class::<env::ActionType>()?;
    m.add_class::<env::Phase>()?;
    m.add_class::<env::Action>()?;
    m.add_class::<env::Observation>()?;
    m.add_class::<env::RiichiEnv>()?;

    m.add_function(wrap_pyfunction!(score::calculate_score, m)?)?;
    m.add_function(wrap_pyfunction!(parser::parse_hand, m)?)?;
    m.add_function(wrap_pyfunction!(parser::parse_tile, m)?)?;
    m.add_function(wrap_pyfunction!(check_riichi_candidates, m)?)?;
    Ok(())
}
