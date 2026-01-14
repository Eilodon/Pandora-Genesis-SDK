use async_trait::async_trait;

/// Giao diện trừu tượng cho một Tế bào FEP (Free Energy Principle).
/// Mọi thực thể sống trong B.ONE đều phải hiện thực hóa trait này.
/// Nó định nghĩa vòng lặp "Niềm tin - Giác quan - Hành động".
#[async_trait]
pub trait FepCell {
    /// Kiểu dữ liệu cho Niềm tin/Mô hình nội tại của Tế bào.
    type Belief;

    /// Kiểu dữ liệu cho Quan sát từ thế giới bên ngoài.
    type Observation;

    /// Kiểu dữ liệu cho Hành động mà Tế bào có thể thực hiện.
    type Action;

    /// Trả về một tham chiếu tới 'niềm tin' hay mô hình nội tại hiện tại của tế bào.
    fn get_internal_model(&self) -> &Self::Belief;

    /// Tiếp nhận quan sát, so sánh với niềm tin và trả về "sai số dự đoán" (Năng lượng Tự do).
    /// Đây là quá trình "Soi Chiếu", nơi thực tại được đối chiếu với kỳ vọng.
    async fn perceive(&mut self, observation: Self::Observation) -> f64;

    /// Dựa trên sai số dự đoán, quyết định và trả về một hành động (nếu cần).
    /// Đây là lúc "Ý niệm" được khởi sinh để thay đổi thế giới.
    async fn act(&mut self) -> Option<Self::Action>;
}
