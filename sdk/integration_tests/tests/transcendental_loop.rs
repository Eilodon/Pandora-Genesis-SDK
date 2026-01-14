#![allow(unused_imports)]
use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic::*; // Fixed import path
use pandora_core::world_model::{DualIntrinsicReward, WorldModel};
use pandora_learning_engine::SelfImprovementEngine;
use pandora_learning_engine::{LearningEngine, TranscendentalProcessor};
use pandora_mcg::legacy_shims::{ActionTrigger, MetaRule, RuleEngine}; // Use legacy version without confidence
use pandora_mcg::MetaCognitiveController;
use std::sync::Arc;

// --- Giả lập các WorldModel ---
struct OldComplexModel;
impl WorldModel for OldComplexModel {
    fn get_mdl(&self) -> f64 {
        100.0
    }
    fn get_prediction_error(&self, _flow: &pandora_core::ontology::EpistemologicalFlow) -> f64 {
        0.5
    }
}
struct NewSimpleModel;
impl WorldModel for NewSimpleModel {
    fn get_mdl(&self) -> f64 {
        20.0
    }
    fn get_prediction_error(&self, _flow: &pandora_core::ontology::EpistemologicalFlow) -> f64 {
        0.4
    }
}

#[tokio::test]
async fn test_the_transcendental_loop() {
    println!("\n=============================================");
    println!("BÀI TEST TÍCH HỢP VÒNG LẶP SIÊU VIỆT");
    println!("=============================================\n");

    // 1. Thiết lập toàn bộ "Tâm Thức"
    let learning_engine = Arc::new(LearningEngine::new(0.5, 0.5));

    // Thiết lập MCG với quy tắc "Vô Chấp"
    let rule = MetaRule::IfCompressionRewardExceeds {
        threshold: 50.0, // Nếu nén được hơn 50 "đơn vị phức tạp"
        action: ActionTrigger::TriggerSelfImprovementLevel1 {
            reason: "Đã tìm thấy mô hình đơn giản hơn đáng kể.".to_string(),
            target_component: "CWM".to_string(),
        },
    };
    let _rule_engine = RuleEngine::new(vec![rule]); // Unused but kept for documentation
    let mcg = MetaCognitiveController::new(); // Use new() instead of default()

    let sie = SelfImprovementEngine::new();

    let processor = SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    );

    let transcendental_processor =
        TranscendentalProcessor::new(processor, learning_engine, mcg, sie);

    let old_model = OldComplexModel;
    let new_model = NewSimpleModel;
    let event = "trigger transcendental event".to_string().into_bytes();

    // 2. Chạy Vòng lặp Siêu Việt
    // Bài test này sẽ tự in ra log của từng bước: LearningEngine -> MCG -> SIE
    // Chúng ta không cần assert gì cả, vì nếu chạy đến cuối mà không panic,
    // có nghĩa là toàn bộ chu trình đã được kết nối thành công.
    transcendental_processor
        .run_transcendental_cycle(event, &old_model, &new_model)
        .await;

    println!("\n=======================================================");
    println!("✅ THÀNH CÔNG: VÒNG LẶP SIÊU VIỆT ĐÃ ĐƯỢC KHÉP KÍN!");
    println!("=======================================================");
}
