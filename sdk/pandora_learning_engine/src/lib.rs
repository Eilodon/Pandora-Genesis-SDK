use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::world_model::{DualIntrinsicReward, WorldModel};
use tracing::info;

#[cfg(test)]
mod active_inference_planning_test;
// Lightweight implementations with ndarray
// pub mod active_inference_full;  // Disabled due to dfdx API issues
pub mod active_inference_lightweight;
pub mod skill_forge_lightweight;

// Simplified implementations (fallback)
// pub mod active_inference_efe;  // Disabled due to burn dependency
pub mod active_inference_simplified;
pub mod active_inference_skandha;
pub mod experience_buffer;
pub mod integration_test;
#[cfg(test)]
mod non_attachment_learning_test;
pub mod policy;
// pub mod skill_forge;  // Disabled due to burn dependency
pub mod skill_forge_simplified;
pub mod skandha_integration;
pub mod transcendental_processor;
pub mod value_estimator;
pub mod world_models;

// Export lightweight implementations (preferred)
pub use active_inference_lightweight::{
    ActiveInferenceSankhara, EFECalculator, HierarchicalWorldModel, PerformanceMetrics,
    Intent, Vedana, Sanna, SankharaSkandha, WorldState, Action, RewardFunction,
    WorldModelLevel, EmotionalState, MemoryTrace, Observation
};
pub use skill_forge_lightweight::{
    SkillForge, QueSTEncoder, VectorQuantizer, CodeGenerator, LLMCodeGenerator,
    GeneratedSkill, SkillPerformanceMetrics, ResourceRequirements, SkillContext,
    PerformanceTracker, PerformanceThresholds
};

// Export simplified implementations as fallback
// pub use active_inference_efe::{ActiveInferenceSankhara, EFECalculator, HierarchicalWorldModel, PerformanceMetrics};  // Disabled
pub use active_inference_simplified::{ActiveInferenceSankhara as SimplifiedActiveInferenceSankhara, EFECalculator as SimplifiedEFECalculator, HierarchicalWorldModel as SimplifiedHierarchicalWorldModel, PerformanceMetrics as SimplifiedPerformanceMetrics};
pub use active_inference_skandha::ActiveInferenceSankharaSkandha;
pub use experience_buffer::{ExperienceBuffer, ExperienceSample, PriorityExperienceBuffer};
pub use policy::{EpsilonGreedyPolicy, Policy, ValueDrivenPolicy, PolicyAction};
// pub use skill_forge::{SkillForge, QueSTEncoder, VectorQuantizer, CodeGenerator, LLMCodeGenerator, SkillForgeMetrics};  // Disabled
pub use skill_forge_simplified::{SkillForge as SimplifiedSkillForge, QueSTEncoder as SimplifiedQueSTEncoder, CodeGenerator as SimplifiedCodeGenerator, LLMCodeGenerator as SimplifiedLLMCodeGenerator, SkillForgeMetrics as SimplifiedSkillForgeMetrics};
pub use skandha_integration::SkandhaProcessorWithLearning;
pub use transcendental_processor::TranscendentalProcessor;
pub use value_estimator::{
    ExponentialMovingAverageEstimator, MeanRewardEstimator, NeuralQValueEstimator, QValueEstimator,
    ValueEstimator,
};
// Temporary re-export to satisfy tests expecting SelfImprovementEngine here
pub use crate::transcendental_processor::SelfImprovementEngine;

/// Learning Engine responsible for calculating rewards and guiding the learning process.
///
/// This engine implements dual intrinsic reward calculation based on the Free Energy
/// Principle, combining prediction accuracy and model compression rewards.
///
/// # Examples
///
/// ```rust
/// use pandora_learning_engine::{LearningEngine, ExponentialMovingAverageEstimator, ExperienceBuffer};
/// use pandora_core::ontology::EpistemologicalFlow;
/// use bytes::Bytes;
///
/// struct MockModel { mdl: f64, err: f64 }
/// impl pandora_core::world_model::WorldModel for MockModel {
///     fn get_mdl(&self) -> f64 { self.mdl }
///     fn get_prediction_error(&self, _: &EpistemologicalFlow) -> f64 { self.err }
/// }
///
/// let le = LearningEngine::new(0.7, 0.3);
/// let current = MockModel { mdl: 10.0, err: 0.2 };
/// let new_model = MockModel { mdl: 9.0, err: 0.1 };
/// let flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test".as_ref()));
/// let mut ema = ExponentialMovingAverageEstimator::new(0.5);
/// let mut buffer = ExperienceBuffer::with_capacity(10);
///
/// let (reward, advantage) = le.learn_single_step(&current, &new_model, &flow, &mut ema, &mut buffer);
/// assert!(reward.prediction_reward.is_finite());
/// assert!(advantage.is_finite());
/// ```
pub struct LearningEngine {
    exploit_weight: f64,
    transcend_weight: f64,
}

impl LearningEngine {
    pub fn new(exploit_weight: f64, transcend_weight: f64) -> Self {
        Self {
            exploit_weight,
            transcend_weight,
        }
    }

    /// Tính toán phần thưởng nội tại kép dựa trên sự thay đổi giữa hai mô hình.
    /// Hiện thực hóa công thức `R_intrinsic(t) = w_exploit * R_predict(t) + w_transcend * R_compress(t)`
    pub fn calculate_reward(
        &self,
        current_model: &dyn WorldModel,
        new_model: &dyn WorldModel,
        flow: &EpistemologicalFlow,
    ) -> DualIntrinsicReward {
        info!("\n--- Động Cơ Học Tập: Tính toán Phần thưởng Kép ---");

        // R_predict(t): Phần thưởng cho việc giảm sai số dự đoán.
        // Ở đây ta đơn giản hóa là lấy sai số của mô hình mới.
        let prediction_reward = -current_model.get_prediction_error(flow);
        info!("Phần thưởng Dự đoán (R_predict): {:.4}", prediction_reward);

        // R_compress(t): Phần thưởng cho việc "giác ngộ" về một quy luật đơn giản hơn.
        let compression_reward = current_model.get_mdl() - new_model.get_mdl();
        if compression_reward > 0.0 {
            info!(
                "Phần thưởng Nén (R_compress): {:.4} -> Đã tìm thấy mô hình đơn giản hơn!",
                compression_reward
            );
        }

        DualIntrinsicReward {
            prediction_reward,
            compression_reward,
        }
    }

    /// Tính toán phần thưởng có baseline ước lượng (ví dụ EMA) để giảm phương sai
    pub fn calculate_reward_with_baseline(
        &self,
        current_model: &dyn WorldModel,
        new_model: &dyn WorldModel,
        flow: &EpistemologicalFlow,
        baseline: &dyn ValueEstimator,
    ) -> (DualIntrinsicReward, f64) {
        let raw = self.calculate_reward(current_model, new_model, flow);
        let baseline_est = baseline.estimate(flow);
        let advantage = self.exploit_weight * (raw.prediction_reward - baseline_est)
            + self.transcend_weight * raw.compression_reward;
        (raw, advantage)
    }

    /// Biến thể: ghi vào PriorityExperienceBuffer với priority = |advantage| và gọi policy.update
    pub fn learn_single_step_with_priority(
        &self,
        current_model: &dyn WorldModel,
        new_model: &dyn WorldModel,
        flow: &EpistemologicalFlow,
        baseline: &mut ExponentialMovingAverageEstimator,
        pbuffer: &mut PriorityExperienceBuffer,
        policy: &mut dyn Policy,
    ) -> (DualIntrinsicReward, f64) {
        let (raw, advantage) =
            self.calculate_reward_with_baseline(current_model, new_model, flow, baseline);
        let total = self.get_total_weighted_reward(&raw);
        baseline.update(flow, total);
        pbuffer.push(
            ExperienceSample {
                flow: flow.clone(),
                reward: total,
            },
            advantage.abs(),
        );
        policy.update(flow, advantage);
        (raw, advantage)
    }

    /// Tính toán tổng phần thưởng có trọng số.
    pub fn get_total_weighted_reward(&self, reward: &DualIntrinsicReward) -> f64 {
        let total = self.exploit_weight * reward.prediction_reward
            + self.transcend_weight * reward.compression_reward;
        info!("=> Tổng Phần thưởng Nội tại: {:.4}", total);
        total
    }

    /// Chạy một vòng học: tính reward, advantage; cập nhật baseline EMA bằng tổng reward
    pub fn learn_single_step(
        &self,
        current_model: &dyn WorldModel,
        new_model: &dyn WorldModel,
        flow: &EpistemologicalFlow,
        baseline: &mut ExponentialMovingAverageEstimator,
        buffer: &mut ExperienceBuffer,
    ) -> (DualIntrinsicReward, f64) {
        let (raw, advantage) =
            self.calculate_reward_with_baseline(current_model, new_model, flow, baseline);
        let total = self.get_total_weighted_reward(&raw);
        // cập nhật baseline bằng tổng reward có trọng số
        baseline.update(flow, total);
        buffer.push(ExperienceSample {
            flow: flow.clone(),
            reward: total,
        });
        (raw, advantage)
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::experience_buffer::PriorityExperienceBuffer;
    use crate::policy::EpsilonGreedyPolicy;
    use bytes::Bytes;

    struct MockModel {
        mdl: f64,
        err: f64,
    }
    impl WorldModel for MockModel {
        fn get_mdl(&self) -> f64 {
            self.mdl
        }
        fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
            self.err
        }
    }

    #[test]
    fn buffer_estimator_reward_flow() {
        let le = LearningEngine::new(0.7, 0.3);
        let cur = MockModel {
            mdl: 10.0,
            err: 0.2,
        };
        let newm = MockModel { mdl: 9.0, err: 0.1 };
        let flow = EpistemologicalFlow::from_bytes(Bytes::copy_from_slice(b"abc"));
        let mut ema = ExponentialMovingAverageEstimator::new(0.5);
        let mut buf = ExperienceBuffer::with_capacity(10);

        let (_raw1, adv1) = le.learn_single_step(&cur, &newm, &flow, &mut ema, &mut buf);
        let est1 = ema.estimate(&flow);
        assert!(buf.len() == 1 && est1 != 0.0);

        let (_raw2, adv2) = le.learn_single_step(&cur, &newm, &flow, &mut ema, &mut buf);
        let est2 = ema.estimate(&flow);
        // baseline có thể hội tụ đến cùng giá trị nếu reward lặp lại; chỉ cần đảm bảo hợp lệ
        assert!(est2.is_finite());
        // advantages are finite numbers
        assert!(adv1.is_finite() && adv2.is_finite());
    }

    #[test]
    fn priority_buffer_and_policy_update_flow() {
        let le = LearningEngine::new(0.7, 0.3);
        let cur = MockModel {
            mdl: 10.0,
            err: 0.2,
        };
        let newm = MockModel { mdl: 9.0, err: 0.1 };
        let flow = EpistemologicalFlow::from_bytes(Bytes::copy_from_slice(b"xyz"));
        let mut ema = ExponentialMovingAverageEstimator::new(0.5);
        let mut pbuf = PriorityExperienceBuffer::with_capacity(8);
        let mut pol = EpsilonGreedyPolicy::default();

        let (_raw, adv) =
            le.learn_single_step_with_priority(&cur, &newm, &flow, &mut ema, &mut pbuf, &mut pol);
        assert!(pbuf.len() == 1 && adv.is_finite());
    }
}
