use pyo3::prelude::*;

pub mod embedding_model;
pub mod vector_store;

use crate::embedding_model::FastembedEmbeddingModel;
use crate::vector_store::FastembedVectorstore;

#[pymodule]
fn fastembed_vectorstore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastembedVectorstore>()?;
    m.add_class::<FastembedEmbeddingModel>()?;
    Ok(())
}
