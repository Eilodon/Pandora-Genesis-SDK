use crate::skills::information_retrieval_skill::{
    CognitiveOutput, Confidence, Document, ProgressiveSemanticEngine, RetrievalError,
};
use nalgebra::DVector;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum AnalogyError {
    #[error("Lỗi truy xuất thông tin nền tảng: {0}")]
    RetrievalError(#[from] RetrievalError),
    #[error("Không tìm thấy concept '{0}' trong bộ nhớ")]
    ConceptNotFound(String),
    #[error("Input không hợp lệ cho suy luận tương tự")]
    InvalidInput,
    #[error("Không thể lấy embedding cho concept: {0}")]
    EmbeddingNotFound(String),
}

/// Engine thực hiện suy luận tương tự
#[allow(dead_code)]
pub struct AnalogyEngine {
    /// Tham chiếu đến bộ nhớ ngữ nghĩa để thực hiện tìm kiếm vector và lấy thông tin concept
    retrieval_engine: Arc<RwLock<ProgressiveSemanticEngine>>,
}

impl AnalogyEngine {
    /// Khởi tạo một engine suy luận tương tự mới.
    pub fn new(retrieval_engine: Arc<RwLock<ProgressiveSemanticEngine>>) -> Self {
        Self { retrieval_engine }
    }

    /// Giải bài toán tương tự A:B :: C:?
    pub async fn solve_analogy(
        &self,
        a_query: &str,
        b_query: &str,
        c_query: &str,
    ) -> Result<CognitiveOutput, AnalogyError> {
        // 1) Lấy concepts A, B, C
        let concept_a = self.get_concept(a_query).await?;
        let concept_b = self.get_concept(b_query).await?;
        let concept_c = self.get_concept(c_query).await?;

        let v_a = DVector::from_vec(concept_a.embedding.clone());
        let v_b = DVector::from_vec(concept_b.embedding.clone());
        let v_c = DVector::from_vec(concept_c.embedding.clone());

        // 2) Vector mục tiêu
        let target_vector = v_c.clone() + (v_b.clone() - v_a.clone());

        // 3) Tìm ứng viên gần nhất bằng vector
        let candidates = self
            .retrieval_engine
            .read()
            .await
            .search_by_vector(target_vector.as_slice(), 10)
            .await?;

        // 4) Tính điểm tin cậy và chọn tốt nhất, loại trừ A/B/C
        let mut best: Option<(Document, f32)> = None;
        for cand in candidates {
            if cand.id == concept_a.id || cand.id == concept_b.id || cand.id == concept_c.id {
                continue;
            }
            let v_d = DVector::from_vec(cand.embedding.clone());
            let conf = self.calculate_multifactor_confidence(&v_a, &v_b, &v_c, &v_d);
            if best.as_ref().map(|(_, s)| conf > *s).unwrap_or(true) {
                best = Some((cand, conf));
            }
        }

        if let Some((winner, score)) = best {
            Ok(CognitiveOutput {
                content: winner.content.clone(),
                confidence: Confidence {
                    score,
                    epistemic_uncertainty: (1.0 - score).max(0.0),
                    aleatoric_uncertainty: 0.0,
                },
                reasoning_trace: vec![
                    format!(
                        "A:B :: C:? với A='{}', B='{}', C='{}'",
                        a_query, b_query, c_query
                    ),
                    format!(
                        "Ứng viên tốt nhất: '{}' với confidence {:.3}",
                        winner.id, score
                    ),
                ],
                documents: vec![winner],
            })
        } else {
            Err(AnalogyError::ConceptNotFound(
                "Không tìm thấy kết quả tương tự phù hợp.".to_string(),
            ))
        }
    }

    /// Hàm helper để lấy thông tin concept (document) từ retrieval engine
    async fn get_concept(&self, query: &str) -> Result<Document, AnalogyError> {
        let docs = self
            .retrieval_engine
            .read()
            .await
            .search_by_text(query, 1)
            .await?;
        docs.into_iter()
            .next()
            .ok_or_else(|| AnalogyError::ConceptNotFound(query.to_string()))
    }

    /// Tính toán điểm tin cậy đa yếu tố cho một ứng viên D
    fn calculate_multifactor_confidence(
        &self,
        v_a: &DVector<f32>,
        v_b: &DVector<f32>,
        v_c: &DVector<f32>,
        v_d: &DVector<f32>,
    ) -> f32 {
        const W_VEC: f32 = 0.5;
        const W_DIST: f32 = 0.3;
        const W_SEMANTIC: f32 = 0.2;

        let v_ab = v_b - v_a;
        let v_cd = v_d - v_c;
        let score_vec = cosine_similarity(&v_ab, &v_cd);

        let dist_ab = (v_a - v_b).norm();
        let dist_cd = (v_c - v_d).norm();
        let score_dist = if dist_ab > 0.0 && dist_cd > 0.0 {
            (dist_ab / dist_cd).min(dist_cd / dist_ab)
        } else {
            0.0
        };

        let score_semantic = (cosine_similarity(v_a, v_c) + cosine_similarity(v_b, v_d)) / 2.0;

        (W_VEC * score_vec + W_DIST * score_dist + W_SEMANTIC * score_semantic)
            .max(0.0)
            .min(1.0)
    }
}

fn cosine_similarity(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
    let denom = a.norm() * b.norm();
    if denom <= 0.0 {
        return 0.0;
    }
    a.dot(b) / denom
}
