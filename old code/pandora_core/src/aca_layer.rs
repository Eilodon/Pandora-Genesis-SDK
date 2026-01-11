//! Định nghĩa ACALayer - Tầng nhận thức
use async_trait::async_trait;

#[async_trait]
pub trait ACALayer: Send + Sync {
    /// Khởi tạo tầng
    async fn init(&mut self);
    /// Xử lý một chu kỳ nhận thức
    async fn perceive(&mut self);
    /// Dừng tầng
    async fn stop(&mut self);
}
