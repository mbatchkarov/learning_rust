use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
mod clustering;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn cluster<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<&'py PyArray1<usize>> {
    let x = x.as_array();
    let res = clustering::cluster(&x, &k);

    Ok(res.into_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rsmeans(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    Ok(())
}
