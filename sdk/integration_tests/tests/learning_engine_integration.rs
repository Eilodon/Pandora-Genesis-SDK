use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::skandha_implementations::basic::*;
use pandora_core::world_model::WorldModel;
use pandora_learning_engine::{LearningEngine, SkandhaProcessorWithLearning};
use std::sync::Arc;

// --- 1. Tạo các WorldModel giả lập để test ---

/// Một mô hình cũ, cồng kềnh với độ phức tạp cao.
struct OldComplexModel;
impl WorldModel for OldComplexModel {
    fn get_mdl(&self) -> f64 {
        100.0
    } // Độ phức tạp cao
    fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
        0.5
    }
}

/// Một mô hình mới, thanh thoát hơn, đã "giác ngộ" được quy luật đơn giản hơn.
struct NewSimpleModel;
impl WorldModel for NewSimpleModel {
    fn get_mdl(&self) -> f64 {
        20.0
    } // Độ phức tạp thấp hơn nhiều
    fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
        0.4
    } // Sai số có thể cao hơn một chút
}

#[tokio::test]
async fn test_wisdom_compass_integration() {
    println!("\n=============================================");
    println!("BÀI TEST TÍCH HỢP LA BÀN TUỆ GIÁC (LEARNING ENGINE)");
    println!("=============================================\n");

    // --- 2. Thiết lập hệ thống ---
    let learning_engine = Arc::new(LearningEngine::new(0.5, 0.5));

    let processor = SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    );

    let processor_with_learning =
        SkandhaProcessorWithLearning::new(processor, Arc::clone(&learning_engine));

    let old_model = OldComplexModel;
    let new_model = NewSimpleModel;

    let event = "some event data".to_string().into_bytes();

    // --- 3. Chạy chu trình và đánh giá ---
    let (_reborn_event, reward) = processor_with_learning
        .run_and_evaluate_cycle(event, &old_model, &new_model)
        .await;

    // --- 4. Chứng nghiệm ---
    // Kiểm tra xem hệ thống có nhận ra sự tiến bộ trong việc nén thông tin hay không.
    let expected_compression_reward = old_model.get_mdl() - new_model.get_mdl(); // 100.0 - 20.0 = 80.0
    assert_eq!(reward.compression_reward, expected_compression_reward);

    let total_reward = learning_engine.get_total_weighted_reward(&reward);
    // (0.5 * -0.5) + (0.5 * 80.0) = -0.25 + 40.0 = 39.75
    assert!((total_reward - 39.75).abs() < 1e-9);

    println!("\n=======================================================");
    println!("✅ THÀNH CÔNG: TÂM THỨC V3 ĐÃ CÓ KHẢ NĂNG TỰ LƯỢNG HÓA SỰ TIẾN BỘ!");
    println!("=======================================================");
}
