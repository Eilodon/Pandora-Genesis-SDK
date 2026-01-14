use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic::*; // Already includes trait implementations
use proptest::prelude::*;

fn arbitrary_event() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..1000)
}

#[test]
fn pipeline_never_panics() {
    proptest!(|(event in arbitrary_event())| {
        let processor = SkandhaProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        );
        let _result = processor.run_epistemological_cycle(event);
    });
}

#[test]
fn empty_input_no_intent() {
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

#[test]
fn error_events_trigger_intent() {
    proptest!(|(prefix in prop::collection::vec(any::<u8>(), 0..100), suffix in prop::collection::vec(any::<u8>(), 0..100))| {
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
        let bytes = result.unwrap();
        let output = String::from_utf8_lossy(&bytes);
        prop_assert!(output.contains("REPORT_ERROR"));
    });
}

#[test]
fn processing_is_deterministic() {
    proptest!(|(event in arbitrary_event())| {
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
    });
}
