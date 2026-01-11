//! Basic skandha implementations (V1 - unchanged from original).

use crate::skandha_implementations::core::{
    RupaSkandha, SannaSkandha, SankharaSkandha, Skandha, VedanaSkandha, VinnanaSkandha,
};
use crate::ontology::{DataEidos, EpistemologicalFlow, Vedana};
use crate::intents::constants;
use bytes::Bytes;
use fnv::FnvHashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::{debug, trace};

// ============================================================================
// RUPA SKANDHA - Form
// ============================================================================

/// Basic Rūpa implementation: Simple byte-to-flow conversion.
///
/// # Performance
///
/// - Zero-copy when possible (uses `Bytes::from`)
/// - Target: <5µs for events <1KB
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicRupaSkandha;

impl Skandha for BasicRupaSkandha {
    fn name(&self) -> &'static str {
        "BasicRupa"
    }
}

impl RupaSkandha for BasicRupaSkandha {
    fn process_event(&self, event: Vec<u8>) -> EpistemologicalFlow {
        trace!("Rūpa: Converting {} bytes to flow", event.len());
        EpistemologicalFlow::from_bytes(Bytes::from(event))
    }
}

// ============================================================================
// VEDANA SKANDHA - Feeling
// ============================================================================

/// Basic Vedanā implementation: Keyword-based sentiment detection.
///
/// # Algorithm
///
/// Scans event text for keywords:
/// - Negative: "error", "fail", "critical", "exception"
/// - Positive: "success", "complete", "ok", "healthy"
/// - Neutral: everything else
///
/// # Performance
///
/// - Simple string contains checks
/// - Target: <3µs per event
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicVedanaSkandha;

impl Skandha for BasicVedanaSkandha {
    fn name(&self) -> &'static str {
        "BasicVedana"
    }
}

impl VedanaSkandha for BasicVedanaSkandha {
    fn feel(&self, flow: &mut EpistemologicalFlow) {
        if let Some(rupa_bytes) = &flow.rupa {
            // Convert to lowercase string for case-insensitive matching
            let content = String::from_utf8_lossy(rupa_bytes).to_lowercase();

            let vedana = if content.contains("error")
                || content.contains("fail")
                || content.contains("critical")
                || content.contains("exception")
                || content.contains("panic")
            {
                debug!("Vedanā: Detected UNPLEASANT feeling (negative keywords)");
                Vedana::Unpleasant { karma_weight: -0.7 }
            } else if content.contains("success")
                || content.contains("complete")
                || content.contains("ok")
                || content.contains("healthy")
                || content.contains("ready")
            {
                debug!("Vedanā: Detected PLEASANT feeling (positive keywords)");
                Vedana::Pleasant { karma_weight: 0.6 }
            } else {
                trace!("Vedanā: Neutral feeling (no strong keywords)");
                Vedana::Neutral
            };

            flow.vedana = Some(vedana);
        }
    }
}

// ============================================================================
// SANNA SKANDHA - Perception
// ============================================================================

/// Basic Saññā implementation: Hash-based pattern recognition.
///
/// # Algorithm
///
/// 1. Hash event content to get pattern ID
/// 2. Create DataEidos with hash as active index
/// 3. Dimensionality = 256 (arbitrary but consistent)
///
/// # Performance
///
/// - Fast hashing (DefaultHasher)
/// - Target: <10µs per event
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicSannaSkandha;

impl Skandha for BasicSannaSkandha {
    fn name(&self) -> &'static str {
        "BasicSanna"
    }
}

impl SannaSkandha for BasicSannaSkandha {
    fn perceive(&self, flow: &mut EpistemologicalFlow) {
        if let Some(rupa_bytes) = &flow.rupa {
            // Hash content to create pattern signature
            let mut hasher = DefaultHasher::new();
            rupa_bytes.hash(&mut hasher);
            let pattern_hash = hasher.finish();

            // Create eidos with hash as index
            let pattern_index = (pattern_hash % 256) as u32;
            let mut active_indices = FnvHashSet::default();
            active_indices.insert(pattern_index);

            let eidos = DataEidos {
                active_indices,
                dimensionality: 256,
            };

            debug!("Saññā: Recognized pattern at index {}", pattern_index);
            flow.sanna = Some(eidos);
        }
    }
}

// ============================================================================
// SANKHARA SKANDHA - Formations/Intent
// ============================================================================

/// Basic Saṅkhāra implementation: Simple decision tree.
///
/// # Algorithm
///
/// Maps vedana to intent:
/// - Unpleasant → REPORT_ERROR
/// - Pleasant → CONTINUE_SUCCESS
/// - Neutral → MAINTAIN_STATUS
///
/// # Performance
///
/// - Single match statement
/// - Target: <2µs per event
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicSankharaSkandha;

impl Skandha for BasicSankharaSkandha {
    fn name(&self) -> &'static str {
        "BasicSankhara"
    }
}

impl SankharaSkandha for BasicSankharaSkandha {
    fn form_intent(&self, flow: &mut EpistemologicalFlow) {
        let intent = match &flow.vedana {
            Some(Vedana::Unpleasant { karma_weight }) => {
                if *karma_weight < -0.5 {
                    debug!("Saṅkhāra: Forming REPORT_ERROR intent (strong negative)");
                    constants::REPORT_ERROR
                } else {
                    debug!("Saṅkhāra: Forming MONITOR_CLOSELY intent (weak negative)");
                    constants::MONITOR_CLOSELY
                }
            }
            Some(Vedana::Pleasant { .. }) => {
                debug!("Saṅkhāra: Forming CONTINUE_SUCCESS intent");
                constants::CONTINUE_SUCCESS
            }
            Some(Vedana::Neutral) | None => {
                trace!("Saṅkhāra: Forming MAINTAIN_STATUS intent");
                constants::MAINTAIN_STATUS
            }
        };

        flow.set_static_intent(intent);
    }
}

// ============================================================================
// VINNANA SKANDHA - Consciousness/Synthesis
// ============================================================================

/// Basic Viññāṇa implementation: Simple rebirth logic.
///
/// # Algorithm
///
/// Synthesize rebirth event if:
/// - Intent is REPORT_ERROR or other action intents
/// - Event contains critical keywords
///
/// # Performance
///
/// - Minimal logic
/// - Target: <5µs per event
#[derive(Debug, Clone, Copy, Default)]
pub struct BasicVinnanaSkandha;

impl Skandha for BasicVinnanaSkandha {
    fn name(&self) -> &'static str {
        "BasicVinnana"
    }
}

impl VinnanaSkandha for BasicVinnanaSkandha {
    fn synthesize(&self, flow: &EpistemologicalFlow) -> Option<Vec<u8>> {
        // Check if rebirth is needed based on intent
        let should_rebirth = flow.sankhara.as_ref().map_or(false, |intent| {
            let intent_str = intent.as_ref();
            intent_str == constants::REPORT_ERROR
                || intent_str == constants::TAKE_CORRECTIVE_ACTION
                || intent_str == constants::HIGH_PRIORITY_ANALYSIS
        });

        if should_rebirth {
            debug!("Viññāṇa: Synthesizing rebirth event");

            // Create new event with intent metadata
            let intent = flow.sankhara.as_ref()?;
            let event_data = format!(
                "{{\"type\":\"rebirth\",\"intent\":\"{}\",\"source\":\"vinnana\"}}",
                intent
            );

            Some(event_data.into_bytes())
        } else {
            trace!("Viññāṇa: No rebirth needed, cycle complete");
            None
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rupa() {
        let rupa = BasicRupaSkandha;
        let event = b"test event".to_vec();
        let flow = rupa.process_event(event);

        assert!(flow.rupa.is_some());
        assert_eq!(flow.rupa.unwrap(), Bytes::from("test event"));
    }

    #[test]
    fn test_basic_vedana_negative() {
        let vedana = BasicVedanaSkandha;
        let mut flow = EpistemologicalFlow::from_bytes(Bytes::from("system error detected"));

        vedana.feel(&mut flow);

        assert!(matches!(flow.vedana, Some(Vedana::Unpleasant { .. })));
    }

    #[test]
    fn test_basic_vedana_positive() {
        let vedana = BasicVedanaSkandha;
        let mut flow = EpistemologicalFlow::from_bytes(Bytes::from("operation success"));

        vedana.feel(&mut flow);

        assert!(matches!(flow.vedana, Some(Vedana::Pleasant { .. })));
    }

    #[test]
    fn test_basic_sankhara_error_intent() {
        let sankhara = BasicSankharaSkandha;
        let mut flow = EpistemologicalFlow::default();
        flow.vedana = Some(Vedana::Unpleasant { karma_weight: -0.8 });

        sankhara.form_intent(&mut flow);

        assert_eq!(
            flow.sankhara.as_ref().map(|s| s.as_ref()),
            Some(constants::REPORT_ERROR)
        );
    }

    #[test]
    fn test_basic_vinnana_rebirth() {
        let vinnana = BasicVinnanaSkandha;
        let mut flow = EpistemologicalFlow::default();
        flow.set_static_intent(constants::REPORT_ERROR);

        let result = vinnana.synthesize(&flow);

        assert!(result.is_some());
        let event = String::from_utf8(result.unwrap()).unwrap();
        assert!(event.contains("rebirth"));
        assert!(event.contains("REPORT_ERROR"));
    }
}
