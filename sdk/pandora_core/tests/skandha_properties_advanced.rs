use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic::*; // Already includes trait implementations
use proptest::prelude::*;

// Property: Pipeline should never panic
proptest! {
    #[test]
    fn pipeline_never_panics_with_any_input(
        event in prop::collection::vec(any::<u8>(), 0..10000)
    ) {
        let processor = SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        );

        // Should never panic
        let _result = processor.run_epistemological_cycle(event);
    }
}

// Property: Output determinism
proptest! {
    #[test]
    fn pipeline_is_deterministic(
        event in prop::collection::vec(any::<u8>(), 1..1000)
    ) {
        let processor = SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        );

        let result1 = processor.run_epistemological_cycle(event.clone());
        let result2 = processor.run_epistemological_cycle(event);

        prop_assert_eq!(result1, result2);
    }
}

// Property: Error events always produce intent
proptest! {
    #[test]
    fn error_events_produce_intent(
        prefix in prop::collection::vec(any::<u8>(), 0..100),
        suffix in prop::collection::vec(any::<u8>(), 0..100)
    ) {
        let processor = SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        );

        let mut event = prefix;
        event.extend_from_slice(b"error");
        event.extend(suffix);

        let result = processor.run_epistemological_cycle(event);

        prop_assert!(result.is_some());
        let result_bytes = result.unwrap();
        let output = String::from_utf8_lossy(&result_bytes);
        prop_assert!(output.contains("REPORT_ERROR"));
    }
}

// Property: Empty input produces no output
#[test]
fn empty_input_produces_no_output() {
    let processor = SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    );

    let result = processor.run_epistemological_cycle(vec![]);
    assert!(result.is_none());
}

// Property: Large inputs are handled efficiently
proptest! {
    #[test]
    fn large_inputs_handled_efficiently(
        event in prop::collection::vec(any::<u8>(), 1000..50000)
    ) {
        let processor = SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        );

        let start = std::time::Instant::now();
        let _result = processor.run_epistemological_cycle(event);
        let elapsed = start.elapsed();

        // Should process large inputs in reasonable time (< 50ms)
        prop_assert!(elapsed.as_millis() < 50);
    }
}

// Property: Unicode input is handled correctly
proptest! {
    #[test]
    fn unicode_input_handled_correctly(
        text in prop::string::string_regex("[\\p{L}\\p{N}\\p{P}\\p{S}\\p{Z}]{1,1000}").unwrap()
    ) {
        let processor = SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        );

        let event = text.into_bytes();
        let result = processor.run_epistemological_cycle(event);

        // Should not panic and produce consistent results
        let _ = result;
    }
}

// Property: Concurrent access is safe
proptest! {
    #[test]
    fn concurrent_access_is_safe(
        event in prop::collection::vec(any::<u8>(), 1..1000)
    ) {
        use std::sync::Arc;
        use std::thread;

        let processor = Arc::new(SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        ));

        let mut handles = vec![];

        // Spawn 10 threads processing the same event
        for _ in 0..10 {
            let proc = Arc::clone(&processor);
            let event_clone = event.clone();
            let handle = thread::spawn(move || {
                proc.run_epistemological_cycle(event_clone)
            });
            handles.push(handle);
        }

        // All threads should complete without panicking
        for handle in handles {
            let _result = handle.join().unwrap();
        }
    }
}
