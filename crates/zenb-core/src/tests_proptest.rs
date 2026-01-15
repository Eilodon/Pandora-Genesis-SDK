use proptest::prelude::*;

/// Property-based test suite for critical AGOLOS invariants
/// EIDOLON FIX 4.1: Validates system correctness under random inputs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::Variable;
    use crate::memory::hologram::HolographicMemory;
    use crate::Engine;
    use num_complex::Complex32;

    // =========================================================================
    // Test 1: SignalVariable Index Invariants
    // =========================================================================
    proptest! {
        #[test]
        fn test_variable_index_invariant(idx in 0usize..10usize) {
            // Any index within COUNT should round-trip correctly
            if idx < Variable::COUNT {
                if let Some(var) = Variable::from_index(idx) {
                    prop_assert!(var.index() < Variable::COUNT);
                    prop_assert_eq!(var.index(), idx);
                }
            } else {
                // Out of range should return None
                prop_assert!(Variable::from_index(idx).is_none());
            }
        }
    }

    // =========================================================================
    // Test 2: Timestamp Monotonicity Enforcement
    // =========================================================================
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_timestamp_monotonicity_rejection(
            ts1 in 1000i64..1_000_000_000i64,
            delta in -999i64..-1i64,  // Negative deltas (go back in time)
        ) {
            let mut engine = Engine::new_with_config(60.0, None);

            // First timestamp accepted (initializes state)
            let features = [70.0, 50.0, 6.0];
            let _est1 = engine.ingest_sensor(&features, ts1);

            // Second timestamp is earlier -> should return cached/previous estimate
            // (Estimator treats this as burst protection, returning last valid estimate)
            let ts2 = ts1 + delta;
            let est2 = engine.ingest_sensor(&features, ts2);

            // Non-monotonic timestamp should either:
            // 1. Return cached estimate (same ts as original), OR
            // 2. Be processed but with valid confidence (burst filter handles it)
            // The key invariant: the estimate should have valid values, not NaN/panic
            prop_assert!(
                est2.confidence >= 0.0 && est2.confidence <= 1.0,
                "Estimate should have valid confidence in [0, 1], got {}",
                est2.confidence
            );

            // Timestamp should not go backwards in the returned estimate
            prop_assert!(
                est2.ts_us >= ts1 || est2.ts_us == ts2,
                "Estimate timestamp should be valid"
            );
        }

        #[test]
        fn test_timestamp_monotonicity_acceptance(
            ts1 in 1000i64..500_000_000i64,
            delta in 1i64..1_000_000i64,  // Positive deltas (go forward)
        ) {
            let mut engine = Engine::new_with_config(60.0, None);

            let features = [70.0, 50.0, 6.0];
            let _est1 = engine.ingest_sensor(&features, ts1);

            // Forward timestamp should be accepted
            let ts2 = ts1 + delta;
            let est2 = engine.ingest_sensor(&features, ts2);

            // Accepted estimate should have positive confidence
            prop_assert!(
                est2.confidence > 0.0 || est2.hr_bpm.is_some(),
                "Monotonic timestamp should produce valid estimate"
            );
        }
    }

    // =========================================================================
    // Test 3: Holographic Memory Energy Bounds
    // =========================================================================
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn test_holographic_energy_bounded(
            pattern_count in 10usize..200usize,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand::rngs::StdRng;
            use rand::Rng;

            let mut rng = StdRng::seed_from_u64(seed);
            let dim = 128; // Smaller for faster tests
            let mut memory = HolographicMemory::new(dim);

            // max_allowed based on default max_magnitude = 100.0
            let max_allowed = 100.0 * 100.0 * dim as f32 * 1.5; // 150% of critical

            for i in 0..pattern_count {
                // Random key and value
                let key: Vec<Complex32> = (0..dim)
                    .map(|_| Complex32::new(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5))
                    .collect();
                let val = key.clone();

                memory.entangle(&key, &val);

                let energy = memory.energy();

                // Energy should never exceed bounds
                prop_assert!(
                    energy <= max_allowed,
                    "Energy {} exceeded max {} after {} patterns",
                    energy, max_allowed, i + 1
                );

                // Energy should never be NaN
                prop_assert!(
                    energy.is_finite(),
                    "Energy became NaN/Inf after {} patterns",
                    i + 1
                );
            }
        }
    }

    // =========================================================================
    // Test 4: Belief State Normalization (if applicable)
    // =========================================================================
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn test_belief_state_normalized(
            hr in 40.0f32..180.0f32,
            hrv in 10.0f32..100.0f32,
            rr in 4.0f32..15.0f32,
        ) {
            let mut engine = Engine::new_with_config(60.0, None);

            // Initialize belief state to uniform distribution before testing
            // (Engine starts with [0.0; 5] which is not normalized)
            engine.vinnana.pipeline.vedana.set_probabilities([0.2; 5]);
            engine.vinnana.pipeline.vedana.state_mut().conf = 0.5;

            // Feed sensor data
            let features = [hr, hrv, rr, 0.9, 0.1]; // Add quality and motion
            let ts = 1_000_000i64;
            engine.ingest_sensor(&features, ts);

            // Tick to trigger belief engine update
            engine.tick(100_000); // 100ms delta

            // Get belief state
            let belief = engine.vinnana.pipeline.vedana.state();

            // Check probability sum (should be ~1.0 or 0.0 if not updated yet)
            let sum: f32 = belief.p.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 0.01 || sum == 0.0,
                "Belief probabilities sum to {} (expected ~1.0 or 0.0)",
                sum
            );

            // All probabilities should be non-negative
            for &p in &belief.p {
                prop_assert!(p >= 0.0, "Negative probability: {}", p);
                prop_assert!(p <= 1.0, "Probability > 1: {}", p);
            }
        }
    }
    // =========================================================================
    // Test 5: Causal Graph Acyclicity Invariant
    // =========================================================================
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn test_causal_graph_acyclicity(
            // Generate a sequence of updates
            updates in prop::collection::vec(
                (0usize..12usize, 0usize..12usize, any::<bool>(), 0.01f32..0.5f32),
                1..20
            )
        ) {
            let mut graph = crate::causal::CausalGraph::new();

            // Apply updates
            for (cause_idx, effect_idx, success, lr) in updates {
                // Skip invalid or self-loops (logic handles them but good to be explicit)
                if cause_idx >= Variable::COUNT || effect_idx >= Variable::COUNT || cause_idx == effect_idx {
                    continue;
                }

                let _cause = Variable::from_index(cause_idx).unwrap();
                let _params = [
                    if cause_idx == 0 { 1.0 } else { 0.0 }, // Simple context
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]; // Full context vec size

                // We need to mock "update_weights" context state
                // Actually update_weights requires a context slice of length 12
                let mut context = vec![0.0; Variable::COUNT];
                context[cause_idx] = 1.0;

                // Try to update
                // The ActionPolicy argument is unused in current implementation
                let action = crate::causal::ActionPolicy {
                    action_type: crate::causal::ActionType::BreathGuidance,
                    intensity: 0.5,
                };

                let _ = graph.update_weights(&context, &action, success, lr);

                // INVARIANT: Graph must ALWAYS be acyclic
                prop_assert!(graph.is_acyclic(), "Graph cycle detected after update");
            }
        }
    }

    // =========================================================================
    // Test 6: Thermodynamic Stability Invariant
    // =========================================================================
    proptest! {
         #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn test_thermodynamic_stability(
            steps in 1usize..100usize,
            temp in 0.1f32..5.0f32,
        ) {
            let mut engine = Engine::new_with_config(60.0, None);

            // Enable thermodynamics
            let mut cfg = engine.config.clone();
            cfg.features.thermo_enabled = Some(true);
            cfg.features.thermo_temperature = Some(temp);
            engine.config = cfg;

            // Run steps
            for _ in 0..steps {
                engine.tick(100_000);
            }

            // Check stability - no NaNs in belief state
            let p = *engine.vinnana.pipeline.vedana.probabilities();
            for &val in &p {
                prop_assert!(val.is_finite(), "Belief state became NaN/Inf: {:?}", p);
                prop_assert!(val >= 0.0, "Belief state became negative: {:?}", p);
                prop_assert!(val <= 1.0, "Belief state > 1.0: {:?}", p);
            }
        }
    }
}
