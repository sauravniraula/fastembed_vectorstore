use fastembed::EmbeddingModel;
use pyo3::prelude::pyclass;

#[pyclass]
pub enum FastembedEmbeddingModel {
    AllMiniLML6V2,
    AllMiniLML6V2Q,
    AllMiniLML12V2,
    AllMiniLML12V2Q,
    BGEBaseENV15,
    BGEBaseENV15Q,
    BGELargeENV15,
    BGELargeENV15Q,
    BGESmallENV15,
    BGESmallENV15Q,
    NomicEmbedTextV1,
    NomicEmbedTextV15,
    NomicEmbedTextV15Q,
    ParaphraseMLMiniLML12V2,
    ParaphraseMLMiniLML12V2Q,
    ParaphraseMLMpnetBaseV2,
    BGESmallZHV15,
    BGELargeZHV15,
    ModernBertEmbedLarge,
    MultilingualE5Small,
    MultilingualE5Base,
    MultilingualE5Large,
    MxbaiEmbedLargeV1,
    MxbaiEmbedLargeV1Q,
    GTEBaseENV15,
    GTEBaseENV15Q,
    GTELargeENV15,
    GTELargeENV15Q,
    ClipVitB32,
    JinaEmbeddingsV2BaseCode,
}

impl FastembedEmbeddingModel {
    pub fn to_embedding_model(&self) -> EmbeddingModel {
        match self {
            FastembedEmbeddingModel::AllMiniLML6V2 => EmbeddingModel::AllMiniLML6V2,
            FastembedEmbeddingModel::AllMiniLML6V2Q => EmbeddingModel::AllMiniLML6V2Q,
            FastembedEmbeddingModel::AllMiniLML12V2 => EmbeddingModel::AllMiniLML12V2,
            FastembedEmbeddingModel::AllMiniLML12V2Q => EmbeddingModel::AllMiniLML12V2Q,
            FastembedEmbeddingModel::BGEBaseENV15 => EmbeddingModel::BGEBaseENV15,
            FastembedEmbeddingModel::BGEBaseENV15Q => EmbeddingModel::BGEBaseENV15Q,
            FastembedEmbeddingModel::BGELargeENV15 => EmbeddingModel::BGELargeENV15,
            FastembedEmbeddingModel::BGELargeENV15Q => EmbeddingModel::BGELargeENV15Q,
            FastembedEmbeddingModel::BGESmallENV15 => EmbeddingModel::BGESmallENV15,
            FastembedEmbeddingModel::BGESmallENV15Q => EmbeddingModel::BGESmallENV15Q,
            FastembedEmbeddingModel::NomicEmbedTextV1 => EmbeddingModel::NomicEmbedTextV1,
            FastembedEmbeddingModel::NomicEmbedTextV15 => EmbeddingModel::NomicEmbedTextV15,
            FastembedEmbeddingModel::NomicEmbedTextV15Q => EmbeddingModel::NomicEmbedTextV15Q,
            FastembedEmbeddingModel::ParaphraseMLMiniLML12V2 => {
                EmbeddingModel::ParaphraseMLMiniLML12V2
            }
            FastembedEmbeddingModel::ParaphraseMLMiniLML12V2Q => {
                EmbeddingModel::ParaphraseMLMiniLML12V2Q
            }
            FastembedEmbeddingModel::ParaphraseMLMpnetBaseV2 => {
                EmbeddingModel::ParaphraseMLMpnetBaseV2
            }
            FastembedEmbeddingModel::BGESmallZHV15 => EmbeddingModel::BGESmallZHV15,
            FastembedEmbeddingModel::BGELargeZHV15 => EmbeddingModel::BGELargeZHV15,
            FastembedEmbeddingModel::ModernBertEmbedLarge => EmbeddingModel::ModernBertEmbedLarge,
            FastembedEmbeddingModel::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
            FastembedEmbeddingModel::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
            FastembedEmbeddingModel::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
            FastembedEmbeddingModel::MxbaiEmbedLargeV1 => EmbeddingModel::MxbaiEmbedLargeV1,
            FastembedEmbeddingModel::MxbaiEmbedLargeV1Q => EmbeddingModel::MxbaiEmbedLargeV1Q,
            FastembedEmbeddingModel::GTEBaseENV15 => EmbeddingModel::GTEBaseENV15,
            FastembedEmbeddingModel::GTEBaseENV15Q => EmbeddingModel::GTEBaseENV15Q,
            FastembedEmbeddingModel::GTELargeENV15 => EmbeddingModel::GTELargeENV15,
            FastembedEmbeddingModel::GTELargeENV15Q => EmbeddingModel::GTELargeENV15Q,
            FastembedEmbeddingModel::ClipVitB32 => EmbeddingModel::ClipVitB32,
            FastembedEmbeddingModel::JinaEmbeddingsV2BaseCode => {
                EmbeddingModel::JinaEmbeddingsV2BaseCode
            }
        }
    }
}
