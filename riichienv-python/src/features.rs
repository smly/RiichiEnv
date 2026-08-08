use pyo3::prelude::*;
use pyo3::types::PyBytes;
use riichienv_core::features;
use riichienv_core::observation::Observation;
use riichienv_core::observation_3p::Observation3P;

fn floats_as_bytes<'py>(py: Python<'py>, values: &[f32]) -> Bound<'py, PyBytes> {
    let byte_len = std::mem::size_of_val(values);
    // f32 has no padding and the returned PyBytes copies the slice before
    // `values` is dropped.
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) };
    PyBytes::new(py, bytes)
}

#[pyfunction]
#[pyo3(name = "encode_base_batch")]
pub fn encode_base_batch_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_base_4p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}

#[pyfunction]
#[pyo3(name = "encode_extended_batch")]
pub fn encode_extended_batch_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_extended_4p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}

#[pyfunction]
#[pyo3(name = "encode_sp_batch")]
pub fn encode_sp_batch_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_sp_4p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}

#[pyfunction]
#[pyo3(name = "encode_drev_batch")]
pub fn encode_drev_batch_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_drev_4p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}

#[pyfunction]
#[pyo3(name = "encode_extended_with_sp_batch")]
pub fn encode_extended_with_sp_batch_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_extended_sp_drev_4p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}

#[pyfunction]
#[pyo3(name = "encode_base_batch_3p")]
pub fn encode_base_batch_3p_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation3P>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_base_3p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}

#[pyfunction]
#[pyo3(name = "encode_extended_batch_3p")]
pub fn encode_extended_batch_3p_py<'py>(
    py: Python<'py>,
    observations: Vec<Observation3P>,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = py
        .detach(move || features::encode_extended_3p_batch(&observations))
        .map_err(PyErr::from)?;
    Ok(floats_as_bytes(py, &values))
}
