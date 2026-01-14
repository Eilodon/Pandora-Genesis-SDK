use super::{DualIntrinsicReward, LearningEngine};
use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::world_model::WorldModel;
use std::sync::Arc;
use tracing::info;

/// `SkandhaProcessorWithLearning` là wrapper tích hợp SkandhaProcessor với LearningEngine.
/// Đây là cách để tránh vòng lặp dependency giữa pandora_core và pandora_learning_engine.
pub struct SkandhaProcessorWithLearning {
    processor: SkandhaProcessor,
    learning_engine: Arc<LearningEngine>,
}

impl SkandhaProcessorWithLearning {
    pub fn new(processor: SkandhaProcessor, learning_engine: Arc<LearningEngine>) -> Self {
        info!("✅ SkandhaProcessorWithLearning đã được khởi tạo với La Bàn Tuệ Giác.");
        Self {
            processor,
            learning_engine,
        }
    }

    /// Vận hành một chu trình nhận thức và tự đánh giá sự tiến bộ.
    pub async fn run_and_evaluate_cycle(
        &self,
        event: Vec<u8>,
        current_model: &dyn WorldModel,
        new_model: &dyn WorldModel,
    ) -> (Option<Vec<u8>>, DualIntrinsicReward) {
        info!("\n--- LUỒNG NHẬN THỨC LUẬN VÀ TỰ ĐÁNH GIÁ BẮT ĐẦU ---");

        // 1. Chạy chu trình nhận thức Ngũ Uẩn
        let reborn_event = self.processor.run_epistemological_cycle_async(event).await;

        // 2. Tạo một EpistemologicalFlow giả lập để tính toán phần thưởng
        let flow = EpistemologicalFlow::default();

        // 3. Tự Đánh giá (Self-Evaluation): Sử dụng LearningEngine để tính toán phần thưởng
        let reward = self
            .learning_engine
            .calculate_reward(current_model, new_model, &flow);

        info!("--- LUỒNG NHẬN THỨC LUẬN VÀ TỰ ĐÁNH GIÁ KẾT THÚC ---");

        (reborn_event, reward)
    }

    /// Truy cập trực tiếp vào SkandhaProcessor gốc.
    pub fn processor(&self) -> &SkandhaProcessor {
        &self.processor
    }

    /// Truy cập trực tiếp vào LearningEngine.
    pub fn learning_engine(&self) -> &Arc<LearningEngine> {
        &self.learning_engine
    }
}
