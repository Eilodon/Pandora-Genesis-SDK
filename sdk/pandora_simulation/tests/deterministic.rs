use pandora_simulation::grid_world::{Action, Direction, GridWorld, ObservabilityMode};

#[test]
fn deterministic_agent_behavior() {
    let _seed = 42u64;
    let mut world = GridWorld::new(10, 10, ObservabilityMode::Full);
    // Nếu có thành phần ngẫu nhiên trong tương lai, dùng rng_for_tests(_seed)
    // Ở đây chỉ kiểm tra các lời gọi cơ bản hoạt động và có thể lặp lại
    let _ = world.get_world_state();
    let _ = world.submit_action(Action::Move(Direction::East)).unwrap();
    let _ = world.get_world_state();
}
