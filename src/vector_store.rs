use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;

use crate::embedding_model::FastembedEmbeddingModel;

#[pyclass]
pub struct FastembedVectorstore {
    model: EmbeddingModel,
    embeddings: HashMap<String, Vec<f32>>,
}

#[pymethods]
impl FastembedVectorstore {
    #[new]
    fn new(model: &FastembedEmbeddingModel) -> Self {
        FastembedVectorstore {
            model: model.to_embedding_model(),
            embeddings: HashMap::new(),
        }
    }

    #[classmethod]
    fn load(
        _: &Bound<'_, PyType>,
        model: &FastembedEmbeddingModel,
        path: String,
    ) -> PyResult<FastembedVectorstore> {
        let file_exists = fs::exists(&path).expect("Could not check if file exists");
        if file_exists {
            let json_string = fs::read_to_string(path).expect("Failed to read file");
            let parsed_json: HashMap<String, Vec<f32>> =
                serde_json::from_str(&json_string).expect("Failed to parse JSON");
            return Ok(FastembedVectorstore {
                model: model.to_embedding_model(),
                embeddings: parsed_json,
            });
        } else {
            return Err(PyException::new_err("File doesn't exist"));
        }
    }

    fn embed_documents(&mut self, mut documents: Vec<String>) -> PyResult<bool> {
        let model_result = TextEmbedding::try_new(
            InitOptions::new(self.model.clone()).with_show_download_progress(true),
        );
        if let Ok(model) = model_result {
            if let Ok(mut vec_embeddings) = model.embed(documents.clone(), None) {
                loop {
                    if documents.is_empty() {
                        break;
                    }
                    if let (Some(popped_document), Some(popped_embedding)) =
                        (documents.pop(), vec_embeddings.pop())
                    {
                        self.embeddings.insert(popped_document, popped_embedding);
                    }
                }
            }
        } else {
            println!("Could not embed documents");
            return Ok(false);
        }
        Ok(true)
    }

    fn search(&self, query: &str, n: usize) -> PyResult<Vec<(String, f32)>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(self.model.clone()).with_show_download_progress(true),
        )
        .expect("Could not initialize model");

        let query_embeddings = model
            .embed(vec![query.to_string()], None)
            .expect("Could not embed query");

        let mut similarities: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .map(|(document, embedding)| {
                let similarity = self.cosine_similarity(&query_embeddings[0][..], embedding);
                (document.clone(), similarity)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(n);

        Ok(similarities)
    }

    fn save(&self, path: &str) -> PyResult<bool> {
        if let Ok(serialized_json) = serde_json::to_string_pretty(&self.embeddings) {
            match fs::create_dir_all(path) {
                Ok(_) => {}
                Err(e) => {
                    println!("Error creating directory {}", e);
                    return Ok(false);
                }
            }
            if let Ok(mut save_file) = File::create(path) {
                match save_file.write_all(serialized_json.as_bytes()) {
                    Ok(_) => {}
                    Err(e) => {
                        println!("Error saving file {}", e);
                        return Ok(false);
                    }
                }
            }
        } else {
            println!("Error serializing HashMap");
            return Ok(false);
        }
        Ok(true)
    }
}

impl FastembedVectorstore {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}
