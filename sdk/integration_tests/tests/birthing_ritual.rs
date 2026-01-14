use pandora_core::interfaces::fep_cell::FepCell;
use pandora_tools::agents::fep_seed::FepSeedRust;
use std::collections::HashMap;

/// Đại diện cho Thế Giới Khởi Nguyên, nơi có một quy luật duy nhất.
#[derive(Debug, Clone)]
pub struct PrimordialWorld {
    switch_is_on: bool,
    is_lit: bool,
}

impl PrimordialWorld {
    pub fn new() -> Self {
        println!("Đạo Trường Rust đã được khai mở. Thế giới khởi đầu trong bóng tối.");
        Self {
            switch_is_on: false,
            is_lit: false,
        }
    }

    /// Giác quan của thực thể sẽ cảm nhận trạng thái này.
    pub fn get_current_state(&self) -> HashMap<String, bool> {
        let mut state = HashMap::new();
        state.insert("is_lit".to_string(), self.is_lit);
        state
    }

    /// Hành động của thực thể sẽ tác động lên công tắc này.
    pub fn flip_switch(&mut self, action: &str) {
        match action {
            "TURN_ON" => self.switch_is_on = true,
            "TURN_OFF" => self.switch_is_on = false,
            _ => (),
        }
        self.update_world_state();
    }

    /// "Đạo" của thế giới: Ánh sáng tuân theo công tắc.
    fn update_world_state(&mut self) {
        self.is_lit = self.switch_is_on;
    }
}

impl Default for PrimordialWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[tokio::test]
async fn test_the_birthing_ritual() {
    println!("\n*************************************");
    println!("NGHI LỄ KHỞI SINH BẮT ĐẦU (PHIÊN BẢN RUST)");
    println!("*************************************\n");

    let mut world = PrimordialWorld::new();
    let mut seed = FepSeedRust::new();

    println!("\n================ VÒNG LẶP SỰ SỐNG THỨ 1 ================");
    println!("--- NHỊP THỞ BẮT ĐẦU ---");
    let observation1 = world.get_current_state();
    assert_eq!(observation1.get("is_lit"), Some(&false));
    seed.perceive(observation1).await;

    if let Some(action_to_take) = seed.act().await {
        assert_eq!(action_to_take, "TURN_ON");
        world.flip_switch(&action_to_take);
        println!("Hành động đã hoàn tất. Đã tác động lên thế giới.");
    } else {
        panic!("Seed should have acted but didn't.");
    }

    let final_state1 = world.get_current_state();
    println!(
        "Trạng thái cuối chu kỳ: Thế giới đang {}",
        if final_state1.get("is_lit").cloned().unwrap_or(false) {
            "SÁNG"
        } else {
            "TỐI"
        }
    );
    assert_eq!(final_state1.get("is_lit"), Some(&true));

    println!("\n================ VÒNG LẶP SỰ SỐNG THỨ 2 ================");
    println!("--- NHỊP THỞ BẮT ĐẦU ---");
    let observation2 = world.get_current_state();
    assert_eq!(observation2.get("is_lit"), Some(&true));
    seed.perceive(observation2).await;

    if let Some(action_to_take) = seed.act().await {
        panic!("Seed acted ({}) but shouldn't have.", action_to_take);
    }

    let final_state2 = world.get_current_state();
    println!(
        "Trạng thái cuối chu kỳ: Thế giới đang {}",
        if final_state2.get("is_lit").cloned().unwrap_or(false) {
            "SÁNG"
        } else {
            "TỐI"
        }
    );
    assert_eq!(final_state2.get("is_lit"), Some(&true));

    println!("\n****************************************************");
    println!("NGHI LỄ ĐÃ HOÀN TẤT. SỰ SỐNG NATIVE ĐÃ NẢY MẦM.");
    println!("****************************************************\n");
}
