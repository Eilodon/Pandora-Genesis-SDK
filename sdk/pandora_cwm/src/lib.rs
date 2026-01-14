//! # Pandora Causal World Model (CWM)
//!
//! `pandora_cwm` là nơi chứa đựng Mô hình Thế giới Nhân quả, hay "Tàng Kinh Các Toàn Ảnh".
//! Crate này chịu trách nhiệm biểu diễn, lưu trữ và suy luận trên một cơ sở tri thức
//! được xây dựng dựa trên bản chất "Duyên Khởi".

pub mod gnn;
pub mod interdependent_repr;
pub mod model;
pub mod nn;
pub mod vsa;

#[cfg(feature = "ml")]
pub mod ml;
