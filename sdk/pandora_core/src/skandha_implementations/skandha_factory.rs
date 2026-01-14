use crate::interfaces::skandhas::*;
use crate::skandha_implementations::{advanced_skandhas::*, basic_skandhas::*};

/// Factory để tạo các Skandha variants khác nhau
pub struct SkandhaFactory;

impl SkandhaFactory {
    /// Tạo bộ Basic Skandhas
    #[allow(clippy::type_complexity)]
    pub fn create_basic_skandhas() -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        (
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        )
    }

    /// Tạo bộ Advanced Skandhas với cấu hình mặc định
    #[allow(clippy::type_complexity)]
    pub fn create_advanced_skandhas() -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        (
            Box::new(AdvancedRupaSkandha::new(true, true)),
            Box::new(AdvancedVedanaSkandha::new(0.5, true)),
            Box::new(AdvancedSannaSkandha::new(0.3, true)),
            Box::new(AdvancedSankharaSkandha::new(0.4, true)),
            Box::new(AdvancedVinnanaSkandha::new(0.5, true)),
        )
    }

    /// Tạo bộ Advanced Skandhas với cấu hình tùy chỉnh
    #[allow(clippy::type_complexity)]
    pub fn create_custom_advanced_skandhas(
        rupa_config: (bool, bool),    // (enable_metadata, enable_timestamp)
        vedana_config: (f32, bool),   // (karma_threshold, enable_context_analysis)
        sanna_config: (f64, bool),    // (pattern_threshold, enable_semantic_analysis)
        sankhara_config: (f64, bool), // (decision_threshold, enable_priority_system)
        vinnana_config: (f64, bool),  // (synthesis_threshold, enable_metacognition)
    ) -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        (
            Box::new(AdvancedRupaSkandha::new(rupa_config.0, rupa_config.1)),
            Box::new(AdvancedVedanaSkandha::new(vedana_config.0, vedana_config.1)),
            Box::new(AdvancedSannaSkandha::new(sanna_config.0, sanna_config.1)),
            Box::new(AdvancedSankharaSkandha::new(
                sankhara_config.0,
                sankhara_config.1,
            )),
            Box::new(AdvancedVinnanaSkandha::new(
                vinnana_config.0,
                vinnana_config.1,
            )),
        )
    }

    /// Tạo Skandha processor với preset configurations
    #[allow(clippy::type_complexity)]
    pub fn create_preset_processor(
        preset: SkandhaPreset,
    ) -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        match preset {
            SkandhaPreset::Basic => Self::create_basic_skandhas(),
            SkandhaPreset::Advanced => Self::create_advanced_skandhas(),
            SkandhaPreset::HighPerformance => Self::create_high_performance_skandhas(),
            SkandhaPreset::Debug => Self::create_debug_skandhas(),
            SkandhaPreset::Minimal => Self::create_minimal_skandhas(),
        }
    }

    /// Tạo bộ Skandhas tối ưu cho hiệu suất cao
    #[allow(clippy::type_complexity)]
    fn create_high_performance_skandhas() -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        (
            Box::new(AdvancedRupaSkandha::new(false, false)), // Tắt metadata để tăng tốc
            Box::new(AdvancedVedanaSkandha::new(0.3, false)), // Threshold thấp, tắt context analysis
            Box::new(AdvancedSannaSkandha::new(0.2, false)), // Threshold thấp, tắt semantic analysis
            Box::new(AdvancedSankharaSkandha::new(0.2, false)), // Threshold thấp, tắt priority system
            Box::new(AdvancedVinnanaSkandha::new(0.3, false)),  // Threshold thấp, tắt metacognition
        )
    }

    /// Tạo bộ Skandhas cho debug
    #[allow(clippy::type_complexity)]
    fn create_debug_skandhas() -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        (
            Box::new(AdvancedRupaSkandha::new(true, true)), // Bật tất cả metadata
            Box::new(AdvancedVedanaSkandha::new(0.1, true)), // Threshold rất thấp, bật context analysis
            Box::new(AdvancedSannaSkandha::new(0.1, true)), // Threshold rất thấp, bật semantic analysis
            Box::new(AdvancedSankharaSkandha::new(0.1, true)), // Threshold rất thấp, bật priority system
            Box::new(AdvancedVinnanaSkandha::new(0.1, true)), // Threshold rất thấp, bật metacognition
        )
    }

    /// Tạo bộ Skandhas tối giản
    #[allow(clippy::type_complexity)]
    fn create_minimal_skandhas() -> (
        Box<dyn RupaSkandha>,
        Box<dyn VedanaSkandha>,
        Box<dyn SannaSkandha>,
        Box<dyn SankharaSkandha>,
        Box<dyn VinnanaSkandha>,
    ) {
        Self::create_basic_skandhas() // Sử dụng basic skandhas cho minimal
    }
}

/// Các preset configurations cho Skandha processor
#[derive(Debug, Clone, Copy)]
pub enum SkandhaPreset {
    /// Basic Skandhas - đơn giản, ổn định
    Basic,
    /// Advanced Skandhas - đầy đủ tính năng
    Advanced,
    /// High Performance - tối ưu tốc độ
    HighPerformance,
    /// Debug - nhiều thông tin debug
    Debug,
    /// Minimal - tối thiểu
    Minimal,
}

impl SkandhaPreset {
    /// Lấy description của preset
    pub fn description(&self) -> &'static str {
        match self {
            SkandhaPreset::Basic => "Basic Skandhas - Simple and stable implementation",
            SkandhaPreset::Advanced => "Advanced Skandhas - Full-featured with complex algorithms",
            SkandhaPreset::HighPerformance => "High Performance - Optimized for speed",
            SkandhaPreset::Debug => "Debug - Maximum logging and analysis",
            SkandhaPreset::Minimal => "Minimal - Lightweight implementation",
        }
    }

    /// Lấy tất cả presets
    pub fn all() -> Vec<SkandhaPreset> {
        vec![
            SkandhaPreset::Basic,
            SkandhaPreset::Advanced,
            SkandhaPreset::HighPerformance,
            SkandhaPreset::Debug,
            SkandhaPreset::Minimal,
        ]
    }
}
