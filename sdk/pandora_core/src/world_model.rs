use crate::ontology::EpistemologicalFlow;

/// Trait đại diện cho một mô hình thế giới (World Model) mà hệ thống đang sử dụng.
/// Bất kỳ mô hình nào, từ một mạng nơ-ron đơn giản đến CWM phức tạp,
/// đều phải có khả năng tính toán độ phức tạp của chính nó.
pub trait WorldModel {
    /// Tính toán Độ dài Mô tả Tối thiểu (Minimum Description Length - MDL) của mô hình.
    /// Đây là thước đo cho sự phức tạp, "sự cồng kềnh" của mô hình.
    fn get_mdl(&self) -> f64;

    /// Tính toán sai số dự đoán cho một quan sát.
    fn get_prediction_error(&self, flow: &EpistemologicalFlow) -> f64;
}

/// Phần thưởng nội tại kép, cân bằng giữa việc "biết nhiều hơn" và "hiểu sâu hơn".
#[derive(Debug, Clone, PartialEq)]
pub struct DualIntrinsicReward {
    /// Phần thưởng cho việc giảm sai số dự đoán (khai thác tri thức hiện có).
    pub prediction_reward: f64,
    /// Phần thưởng cho việc tìm ra một mô hình đơn giản hơn (siêu việt, giác ngộ).
    pub compression_reward: f64,
}
