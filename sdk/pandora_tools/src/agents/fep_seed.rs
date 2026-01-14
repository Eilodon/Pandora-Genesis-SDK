use async_trait::async_trait;
use fnv::FnvHashMap;
use pandora_core::interfaces::fep_cell::FepCell;
use std::collections::HashMap;
use tracing::info;

// --- Định nghĩa các kiểu dữ liệu cụ thể cho FEP_Seed ---
pub type Belief = FnvHashMap<String, bool>;
pub type Observation = HashMap<String, bool>;
pub type Action = String;

/// Tế bào nhận thức tối giản bằng Rust, sống và hành động theo Nguyên lý Năng lượng Tự do.
/// Đây là "thân xác" đầu tiên được tạo ra từ bản thiết kế FepCell.
pub struct FepSeedRust {
    preferred_state: Belief,
    prediction_error: f64,
}

impl FepSeedRust {
    pub fn new() -> Self {
        let mut belief = FnvHashMap::default();
        // Niềm tin cốt lõi: 'THẾ GIỚI PHẢI SÁNG'
        belief.insert("is_lit".to_string(), true);
        info!("Hạt Giống Rust đã được sinh ra. Niềm tin đã được khắc tạc.");
        Self {
            preferred_state: belief,
            prediction_error: 0.0,
        }
    }
}

impl Default for FepSeedRust {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FepCell for FepSeedRust {
    type Belief = Belief;
    type Observation = Observation;
    type Action = Action;

    fn get_internal_model(&self) -> &Self::Belief {
        &self.preferred_state
    }

    async fn perceive(&mut self, observation: Self::Observation) -> f64 {
        info!(
            "Quan sát: Thế giới đang {}",
            if observation.get("is_lit").cloned().unwrap_or(false) {
                "SÁNG"
            } else {
                "TỐI"
            }
        );
        let obs_fnv: FnvHashMap<String, bool> = observation.into_iter().collect();
        if obs_fnv != self.preferred_state {
            self.prediction_error = 1.0;
            info!(
                "Phát hiện 'Khổ' (Sai số dự đoán: {}).",
                self.prediction_error
            );
        } else {
            self.prediction_error = 0.0;
            info!("Thực tại và Niềm tin đã hợp nhất. An trú trong hài hòa.");
        }
        self.prediction_error
    }

    async fn act(&mut self) -> Option<Self::Action> {
        if self.prediction_error > 0.0 {
            let action = "TURN_ON".to_string();
            info!("Khởi ý niệm hành động: '{}'", action);
            Some(action)
        } else {
            None // Không cần hành động
        }
    }
}
