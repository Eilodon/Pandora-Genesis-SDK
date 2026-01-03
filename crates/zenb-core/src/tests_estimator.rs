//! Tests for P0.7: Estimator dt=0 Fix and Burst Filtering

use crate::estimator::Estimator;

#[test]
fn test_first_sample_initialization() {
    let mut est = Estimator::default();

    // First sample should use alpha=1.0 (direct initialization)
    let e1 = est.ingest(&[60.0, 40.0, 6.0], 0);

    assert_eq!(e1.hr_bpm, Some(60.0), "First HR should be exact");
    assert_eq!(e1.rmssd, Some(40.0), "First RMSSD should be exact");
    assert_eq!(e1.rr_bpm, Some(6.0), "First RR should be exact");
}

#[test]
fn test_burst_filtering() {
    let mut est = Estimator::default();

    // First sample
    let e1 = est.ingest(&[60.0, 40.0, 6.0], 0);
    assert_eq!(e1.hr_bpm, Some(60.0));

    // Second sample at 1ms (should be filtered as burst)
    let e2 = est.ingest(&[100.0, 10.0, 12.0], 1_000);
    assert_eq!(e2.hr_bpm, Some(60.0), "Burst should return cached estimate");
    assert_eq!(e2.rmssd, Some(40.0), "Burst should return cached estimate");

    // Third sample at 20ms (should be accepted)
    let e3 = est.ingest(&[65.0, 42.0, 6.2], 20_000);
    assert!(
        e3.hr_bpm.unwrap() > 60.0 && e3.hr_bpm.unwrap() < 100.0,
        "After burst filter, should update with EMA"
    );
}

#[test]
fn test_burst_filter_threshold() {
    let mut est = Estimator::default();

    est.ingest(&[60.0, 40.0, 6.0], 0);

    // Exactly at threshold (10ms) - should be filtered
    let e1 = est.ingest(&[70.0, 30.0, 7.0], 9_999);
    assert_eq!(e1.hr_bpm, Some(60.0), "9.999ms should be filtered");

    // Just above threshold - should be accepted
    let e2 = est.ingest(&[70.0, 30.0, 7.0], 10_001);
    assert!(e2.hr_bpm.unwrap() != 60.0, "10.001ms should be accepted");
}

#[test]
fn test_dt_zero_handling() {
    let mut est = Estimator::default();

    // First sample with dt=0 should initialize
    let e1 = est.ingest(&[60.0, 40.0, 6.0], 1000);
    assert_eq!(e1.hr_bpm, Some(60.0));

    // Subsequent samples should use exponential decay
    let e2 = est.ingest(&[70.0, 35.0, 7.0], 1_100_000); // 1.1 seconds later
    assert!(
        e2.hr_bpm.unwrap() > 60.0 && e2.hr_bpm.unwrap() < 70.0,
        "Should be EMA between old and new"
    );
}

#[test]
fn test_no_cached_estimate_on_first_burst() {
    let mut est = Estimator::default();

    // If first sample is a burst (unlikely but possible), should initialize anyway
    let e1 = est.ingest(&[60.0, 40.0, 6.0], 5_000);
    assert_eq!(
        e1.hr_bpm,
        Some(60.0),
        "Should initialize even if dt < threshold"
    );
}

#[test]
fn test_multiple_bursts() {
    let mut est = Estimator::default();

    let e1 = est.ingest(&[60.0, 40.0, 6.0], 0);

    // Multiple rapid samples (burst)
    let e2 = est.ingest(&[100.0, 10.0, 12.0], 1_000);
    let e3 = est.ingest(&[110.0, 5.0, 14.0], 2_000);
    let e4 = est.ingest(&[120.0, 3.0, 16.0], 3_000);

    // All should return the same cached estimate
    assert_eq!(e2.hr_bpm, e3.hr_bpm);
    assert_eq!(e3.hr_bpm, e4.hr_bpm);
    assert_eq!(e2.hr_bpm, Some(60.0));

    // After burst window, should update
    let e5 = est.ingest(&[65.0, 42.0, 6.2], 50_000);
    assert!(
        e5.hr_bpm.unwrap() != 60.0,
        "Should update after burst window"
    );
}

#[test]
fn test_confidence_calculation_unchanged() {
    let mut est = Estimator::default();

    let e1 = est.ingest(&[60.0, 40.0, 6.0], 0);
    assert!(
        e1.confidence > 0.0,
        "Should have confidence with all features"
    );

    let e2 = est.ingest(&[60.0, 40.0], 1_000_000); // Missing RR
    assert!(
        e2.confidence < e1.confidence,
        "Confidence should drop with missing feature"
    );
}

#[test]
fn test_ema_alpha_calculation() {
    let mut est = Estimator::default();

    // Initialize
    est.ingest(&[60.0, 40.0, 6.0], 0);

    // Short dt should have smaller alpha (less change)
    let e1 = est.ingest(&[100.0, 10.0, 12.0], 100_000); // 0.1s
    let hr1 = e1.hr_bpm.unwrap();

    // Reset and test longer dt
    let mut est2 = Estimator::default();
    est2.ingest(&[60.0, 40.0, 6.0], 0);
    let e2 = est2.ingest(&[100.0, 10.0, 12.0], 5_000_000); // 5s
    let hr2 = e2.hr_bpm.unwrap();

    // Longer dt should result in more change (higher alpha)
    assert!(hr2 > hr1, "Longer dt should allow more change");
}
