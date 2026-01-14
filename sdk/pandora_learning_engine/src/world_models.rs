use super::LearningEngine;
use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::world_model::WorldModel;
use tracing::info;

/// Mô hình thế giới đơn giản để test động cơ học tập.
/// Mô hình này có độ phức tạp có thể điều chỉnh và khả năng dự đoán cơ bản.
#[derive(Debug, Clone)]
pub struct SimpleWorldModel {
    pub name: String,
    pub complexity: f64,      // Độ phức tạp của mô hình (MDL)
    pub accuracy: f64,        // Độ chính xác dự đoán (0.0 - 1.0)
    pub parameters: Vec<f64>, // Các tham số của mô hình
}

impl SimpleWorldModel {
    pub fn new(name: String, complexity: f64, accuracy: f64) -> Self {
        let parameters = vec![0.1; (complexity * 10.0) as usize];
        Self {
            name,
            complexity,
            accuracy,
            parameters,
        }
    }

    /// Tạo một mô hình "cải tiến" dựa trên mô hình hiện tại.
    /// Mô hình mới có thể có độ phức tạp khác và độ chính xác khác.
    pub fn evolve(&self, new_complexity: f64, new_accuracy: f64) -> Self {
        let new_name = format!("{}_evolved", self.name);
        Self::new(new_name, new_complexity, new_accuracy)
    }
}

impl WorldModel for SimpleWorldModel {
    fn get_mdl(&self) -> f64 {
        // MDL = số lượng tham số * log(độ phức tạp)
        self.parameters.len() as f64 * self.complexity.log2()
    }

    fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
        // Sai số dự đoán = 1 - độ chính xác + noise ngẫu nhiên
        let base_error = 1.0 - self.accuracy;
        let noise = (self.parameters.len() as f64 * 0.01) % 0.1;
        base_error + noise
    }
}

/// Mô hình thế giới phức tạp hơn, có khả năng học tập và thích ứng.
#[derive(Debug, Clone)]
pub struct AdaptiveWorldModel {
    pub name: String,
    pub base_complexity: f64,
    pub learning_rate: f64,
    pub adaptation_history: Vec<f64>,
    pub current_accuracy: f64,
}

impl AdaptiveWorldModel {
    pub fn new(name: String, base_complexity: f64, learning_rate: f64) -> Self {
        Self {
            name,
            base_complexity,
            learning_rate,
            adaptation_history: vec![base_complexity],
            current_accuracy: 0.5, // Bắt đầu với độ chính xác trung bình
        }
    }

    /// Thích ứng mô hình dựa trên phản hồi từ môi trường.
    pub fn adapt(&mut self, feedback: f64) {
        // Cập nhật độ chính xác dựa trên phản hồi
        self.current_accuracy =
            (self.current_accuracy + self.learning_rate * feedback).clamp(0.0, 1.0);

        // Điều chỉnh độ phức tạp dựa trên hiệu suất
        let complexity_change = if feedback > 0.0 {
            -self.learning_rate * 0.1 // Giảm phức tạp nếu hiệu suất tốt
        } else {
            self.learning_rate * 0.1 // Tăng phức tạp nếu hiệu suất kém
        };

        let new_complexity = (self.base_complexity + complexity_change).max(0.1);
        self.adaptation_history.push(new_complexity);
        self.base_complexity = new_complexity;
    }

    /// Tạo một mô hình con dựa trên kinh nghiệm học tập.
    pub fn spawn_child(&self) -> Self {
        let child_name = format!("{}_child", self.name);
        let child_complexity = self.base_complexity * 0.8; // Mô hình con đơn giản hơn
        let child_accuracy = self.current_accuracy * 1.1; // Nhưng chính xác hơn

        Self::new(child_name, child_complexity, child_accuracy)
    }
}

impl WorldModel for AdaptiveWorldModel {
    fn get_mdl(&self) -> f64 {
        // MDL dựa trên độ phức tạp hiện tại và lịch sử thích ứng
        let avg_complexity =
            self.adaptation_history.iter().sum::<f64>() / self.adaptation_history.len() as f64;
        avg_complexity * (1.0 + self.adaptation_history.len() as f64 * 0.1)
    }

    fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
        // Sai số dự đoán dựa trên độ chính xác hiện tại
        1.0 - self.current_accuracy
    }
}

/// Test động cơ học tập với các mô hình khác nhau.
pub fn test_learning_engine() {
    info!("\n=============================================");
    info!("BÀI TEST ĐỘNG CƠ HỌC TẬP VÔ CHẤP");
    info!("=============================================\n");

    // Tạo động cơ học tập với trọng số cân bằng
    let learning_engine = LearningEngine::new(0.6, 0.4); // 60% khai thác, 40% siêu việt

    // Tạo mô hình ban đầu (phức tạp, kém chính xác)
    let initial_model = SimpleWorldModel::new(
        "Initial Model".to_string(),
        5.0, // Độ phức tạp cao
        0.3, // Độ chính xác thấp
    );

    // Tạo mô hình cải tiến (đơn giản hơn, chính xác hơn)
    let improved_model = initial_model.evolve(
        3.0, // Độ phức tạp thấp hơn
        0.8, // Độ chính xác cao hơn
    );

    // Tạo một EpistemologicalFlow giả lập
    let flow = EpistemologicalFlow::default();

    // Tính toán phần thưởng kép
    let reward = learning_engine.calculate_reward(&initial_model, &improved_model, &flow);
    let total_reward = learning_engine.get_total_weighted_reward(&reward);

    info!("\n--- Kết quả Học tập Vô Chấp ---");
    info!(
        "Mô hình ban đầu: MDL = {:.4}, Accuracy = {:.4}",
        initial_model.get_mdl(),
        1.0 - initial_model.get_prediction_error(&flow)
    );
    info!(
        "Mô hình cải tiến: MDL = {:.4}, Accuracy = {:.4}",
        improved_model.get_mdl(),
        1.0 - improved_model.get_prediction_error(&flow)
    );
    info!("Tổng phần thưởng: {:.4}", total_reward);

    if total_reward > 0.0 {
        info!("✅ THÀNH CÔNG: Hệ thống đã học được cách 'vô chấp' - từ bỏ mô hình phức tạp để chọn mô hình đơn giản hơn!");
    } else {
        info!("⚠️  CẢNH BÁO: Hệ thống chưa học được cách 'vô chấp' - vẫn bám víu vào mô hình phức tạp.");
    }

    // Test với mô hình thích ứng
    info!("\n--- Test Mô hình Thích ứng ---");
    let mut adaptive_model = AdaptiveWorldModel::new(
        "Adaptive Model".to_string(),
        4.0, // Độ phức tạp ban đầu
        0.1, // Tốc độ học
    );

    // Mô phỏng quá trình học tập
    for i in 0..5 {
        let feedback = if i < 3 { 0.2 } else { -0.1 }; // Phản hồi tích cực rồi tiêu cực
        adaptive_model.adapt(feedback);

        let child_model = adaptive_model.spawn_child();
        let reward = learning_engine.calculate_reward(&adaptive_model, &child_model, &flow);
        let total_reward = learning_engine.get_total_weighted_reward(&reward);

        info!(
            "Lần {}: Feedback = {:.1}, Total Reward = {:.4}",
            i + 1,
            feedback,
            total_reward
        );
    }

    info!("\n=============================================");
    info!("✅ HOÀN THÀNH: Động cơ Học tập Vô Chấp đã hoạt động!");
    info!("=============================================");
}
