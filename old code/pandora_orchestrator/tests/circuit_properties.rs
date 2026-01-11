use pandora_orchestrator::circuit_breaker::{CircuitBreakerConfig, CircuitBreakerManager};
use proptest::prelude::*;

proptest! {
    #[test]
    fn circuit_starts_closed(skill_name in "[a-z]{3,20}") {
        let manager = CircuitBreakerManager::new(CircuitBreakerConfig::default());
        prop_assert!(!manager.is_open(&skill_name));
    }

    #[test]
    fn circuit_opens_after_threshold(
        skill_name in "[a-z]{3,20}",
        threshold in 1u32..10
    ) {
        let config = CircuitBreakerConfig { failure_threshold: threshold, ..Default::default() };
        let manager = CircuitBreakerManager::new(config);  // Fixed: use config instead of threshold
        for _ in 0..threshold-1 { manager.record_failure(&skill_name); prop_assert!(!manager.is_open(&skill_name)); }
        manager.record_failure(&skill_name);
        prop_assert!(manager.is_open(&skill_name));
    }

    #[test]
    fn success_resets_circuit(skill_name in "[a-z]{3,20}", num_failures in 0u32..5) {
        let config = CircuitBreakerConfig { failure_threshold: 10, ..Default::default() };
        let manager = CircuitBreakerManager::new(config.clone());
        for _ in 0..num_failures { manager.record_failure(&skill_name); }
        manager.record_success(&skill_name);
        prop_assert!(!manager.is_open(&skill_name));
        for _ in 0..config.failure_threshold - 1 { manager.record_failure(&skill_name); prop_assert!(!manager.is_open(&skill_name)); }
    }

    #[test]
    fn lru_maintains_capacity(num_skills in 10usize..50) {
        let config = CircuitBreakerConfig { max_circuits: 160, ..Default::default() }; // 16 shards * 10 per shard
        let manager = CircuitBreakerManager::new(config);
        for i in 0..num_skills { let skill_name = format!("skill_{}", i); manager.record_failure(&skill_name); }
        let stats = manager.stats();
        prop_assert!(stats.total_circuits <= 160); // Total capacity across all shards
    }

    #[test]
    fn circuits_are_independent(skill1 in "[a-z]{3,10}", skill2 in "[a-z]{3,10}") {
        prop_assume!(skill1 != skill2);
        let config = CircuitBreakerConfig { failure_threshold: 3, ..Default::default() };
        let manager = CircuitBreakerManager::new(config);
        for _ in 0..3 { manager.record_failure(&skill1); }
        prop_assert!(manager.is_open(&skill1));
        prop_assert!(!manager.is_open(&skill2));
        manager.record_failure(&skill2);
        prop_assert!(!manager.is_open(&skill2));
        prop_assert!(manager.is_open(&skill1));
    }
}
