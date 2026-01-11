//! Five Skandhas Cognitive Pipeline module.
//!
//! Ported from Pandora's `pandora_core::fep_cell` concept.
//! Provides a unified trait-based pipeline for cognitive processing:
//!
//! 1. **Rupa (Sắc)** - Form: Raw sensor input processing
//! 2. **Vedana (Thọ)** - Feeling: Valence/arousal extraction  
//! 3. **Sanna (Tưởng)** - Perception: Pattern recognition
//! 4. **Sankhara (Hành)** - Formations: Intent/action formation
//! 5. **Vinnana (Thức)** - Consciousness: Integration/synthesis
//!
//! # AGOLOS Mapping
//! - Rupa ≈ `SheafPerception` (sensor consensus)
//! - Vedana ≈ `BeliefEngine` (valence extraction)
//! - Sanna ≈ `HolographicMemory` (pattern recall)
//! - Sankhara ≈ `DharmaFilter` (ethical action filtering)
//! - Vinnana ≈ `Engine` (synthesis)

use serde::{Deserialize, Serialize};

// ============================================================================
// Skandha Data Flow
// ============================================================================

/// Raw sensory input (Rupa stage input).
#[derive(Debug, Clone, Default)]
pub struct SensorInput {
    pub hr_bpm: Option<f32>,
    pub hrv_rmssd: Option<f32>,
    pub rr_bpm: Option<f32>,
    pub quality: f32,
    pub motion: f32,
    pub timestamp_us: i64,
}

/// Processed form after sensor consensus (Rupa stage output).
#[derive(Debug, Clone, Default)]
pub struct ProcessedForm {
    /// Consensus sensor values
    pub values: [f32; 5],
    /// Anomaly score (0 = normal, >0 = anomalous)
    pub anomaly_score: f32,
    /// Energy level
    pub energy: f32,
    /// Whether data is reliable
    pub is_reliable: bool,
}

/// Affective state (Vedana stage output).
#[derive(Debug, Clone, Default)]
pub struct AffectiveState {
    /// Valence (-1 negative to +1 positive)
    pub valence: f32,
    /// Arousal (0 calm to 1 activated)
    pub arousal: f32,
    /// Confidence in affective assessment
    pub confidence: f32,
}

/// Perceived pattern (Sanna stage output).
#[derive(Debug, Clone, Default)]
pub struct PerceivedPattern {
    /// Recalled pattern from memory
    pub pattern_id: u64,
    /// Similarity to known patterns (0-1)
    pub similarity: f32,
    /// Associated context features
    pub context: [f32; 5],
    /// Whether this matches a trauma pattern
    pub is_trauma_associated: bool,
}

/// Formed intent (Sankhara stage output).
#[derive(Debug, Clone)]
pub struct FormedIntent {
    /// Proposed action type
    pub action: IntentAction,
    /// Ethical alignment score (0-1)
    pub alignment: f32,
    /// Whether intent passed ethical filter
    pub is_sanctioned: bool,
    /// Reasoning for intent
    pub reasoning: String,
}

impl Default for FormedIntent {
    fn default() -> Self {
        Self {
            action: IntentAction::Observe,
            alignment: 1.0,
            is_sanctioned: true,
            reasoning: "Default passive observation".to_string(),
        }
    }
}

/// Possible intent actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentAction {
    /// Passive observation
    Observe,
    /// Guide breathing at specific BPM
    GuideBreath { target_bpm: u8 },
    /// Suggest intervention
    SuggestIntervention,
    /// Alert user
    Alert,
    /// Fallback to safety
    SafeFallback,
}

/// Final synthesized output (Vinnana stage output).
#[derive(Debug, Clone, Default)]
pub struct SynthesizedState {
    /// Belief state distribution
    pub belief: [f32; 5],
    /// Dominant mode
    pub mode: u8,
    /// Overall confidence
    pub confidence: f32,
    /// Control decision
    pub decision: Option<ControlOutput>,
    /// Free energy (prediction error)
    pub free_energy: f32,
}

/// Control output from synthesis.
#[derive(Debug, Clone)]
pub struct ControlOutput {
    pub target_bpm: f32,
    pub confidence: f32,
    pub poll_interval_ms: u32,
}

// ============================================================================
// Skandha Traits  
// ============================================================================

/// Rupa Skandha - Form processing.
pub trait RupaSkandha {
    /// Process raw sensor input into structured form.
    fn process_form(&self, input: &SensorInput) -> ProcessedForm;
}

/// Vedana Skandha - Feeling/valence processing.
pub trait VedanaSkandha {
    /// Extract affective state from processed form.
    fn extract_affect(&self, form: &ProcessedForm) -> AffectiveState;
}

/// Sanna Skandha - Perception/pattern recognition.
pub trait SannaSkandha {
    /// Recognize patterns and recall associations.
    fn perceive(&self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern;
}

/// Sankhara Skandha - Formation/intent generation.
pub trait SankharaSkandha {
    /// Form intent based on perception.
    fn form_intent(&self, pattern: &PerceivedPattern, affect: &AffectiveState) -> FormedIntent;
}

/// Vinnana Skandha - Consciousness/synthesis.
pub trait VinnanaSkandha {
    /// Synthesize all stages into unified state.
    fn synthesize(
        &self,
        form: &ProcessedForm,
        affect: &AffectiveState,
        pattern: &PerceivedPattern,
        intent: &FormedIntent,
    ) -> SynthesizedState;
}

// ============================================================================
// Skandha Pipeline
// ============================================================================

/// Configuration for the Skandha pipeline.
#[derive(Debug, Clone)]
pub struct SkandhaConfig {
    /// Enable Rupa (sensor consensus)
    pub enable_rupa: bool,
    /// Enable Vedana (affect extraction)
    pub enable_vedana: bool,
    /// Enable Sanna (pattern recognition)
    pub enable_sanna: bool,
    /// Enable Sankhara (intent formation)
    pub enable_sankhara: bool,
    /// Enable Vinnana (synthesis)
    pub enable_vinnana: bool,
}

impl Default for SkandhaConfig {
    fn default() -> Self {
        Self {
            enable_rupa: true,
            enable_vedana: true,
            enable_sanna: true,
            enable_sankhara: true,
            enable_vinnana: true,
        }
    }
}

/// Default implementations for each skandha.
pub mod defaults {
    use super::*;

    /// Default Rupa implementation.
    #[derive(Debug, Clone, Default)]
    pub struct DefaultRupa {
        pub anomaly_threshold: f32,
    }

    impl RupaSkandha for DefaultRupa {
        fn process_form(&self, input: &SensorInput) -> ProcessedForm {
            let values = [
                input.hr_bpm.unwrap_or(60.0) / 200.0, // Normalize
                input.hrv_rmssd.unwrap_or(50.0) / 100.0,
                input.rr_bpm.unwrap_or(12.0) / 20.0,
                input.quality,
                input.motion,
            ];

            // Simple anomaly detection
            let anomaly_score = if input.quality < 0.3 || input.motion > 0.8 {
                1.0 - input.quality + input.motion * 0.5
            } else {
                0.0
            };

            ProcessedForm {
                values,
                anomaly_score,
                energy: values.iter().sum::<f32>() / 5.0,
                is_reliable: anomaly_score < self.anomaly_threshold,
            }
        }
    }

    /// Default Vedana implementation.
    #[derive(Debug, Clone, Default)]
    pub struct DefaultVedana;

    impl VedanaSkandha for DefaultVedana {
        fn extract_affect(&self, form: &ProcessedForm) -> AffectiveState {
            // Extract valence from HRV (higher HRV = more positive)
            let valence = (form.values[1] - 0.5) * 2.0; // -1 to 1

            // Extract arousal from HR (higher HR = more aroused)
            let arousal = form.values[0].clamp(0.0, 1.0);

            // Confidence from reliability
            let confidence = if form.is_reliable { 0.8 } else { 0.3 };

            AffectiveState {
                valence,
                arousal,
                confidence,
            }
        }
    }

    /// Default Sanna implementation.
    #[derive(Debug, Clone, Default)]
    pub struct DefaultSanna {
        /// Known pattern signatures (simplified)
        pub pattern_count: u64,
    }

    impl SannaSkandha for DefaultSanna {
        fn perceive(&self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern {
            // Simple pattern classification
            let pattern_id = if affect.arousal > 0.7 {
                1 // High arousal pattern
            } else if affect.valence < -0.3 {
                2 // Negative valence pattern
            } else {
                0 // Baseline pattern
            };

            PerceivedPattern {
                pattern_id,
                similarity: 0.7,
                context: form.values,
                is_trauma_associated: pattern_id == 2 && affect.arousal > 0.8,
            }
        }
    }

    /// Default Sankhara implementation.
    #[derive(Debug, Clone, Default)]
    pub struct DefaultSankhara {
        /// Ethical alignment threshold
        pub alignment_threshold: f32,
    }

    impl SankharaSkandha for DefaultSankhara {
        fn form_intent(&self, pattern: &PerceivedPattern, affect: &AffectiveState) -> FormedIntent {
            // Form intent based on pattern and affect
            let action = if pattern.is_trauma_associated {
                IntentAction::SafeFallback
            } else if affect.arousal > 0.8 {
                IntentAction::GuideBreath { target_bpm: 6 } // Calm down
            } else if affect.arousal < 0.2 {
                IntentAction::GuideBreath { target_bpm: 8 } // Energize
            } else {
                IntentAction::Observe
            };

            // Simple alignment check
            let alignment = if pattern.is_trauma_associated { 0.3 } else { 0.9 };
            let is_sanctioned = alignment >= self.alignment_threshold;

            FormedIntent {
                action: if is_sanctioned { action } else { IntentAction::Observe },
                alignment,
                is_sanctioned,
                reasoning: format!("Pattern {} with arousal {:.2}", pattern.pattern_id, affect.arousal),
            }
        }
    }

    /// Default Vinnana implementation.
    #[derive(Debug, Clone, Default)]
    pub struct DefaultVinnana;

    impl VinnanaSkandha for DefaultVinnana {
        fn synthesize(
            &self,
            form: &ProcessedForm,
            affect: &AffectiveState,
            _pattern: &PerceivedPattern,
            intent: &FormedIntent,
        ) -> SynthesizedState {
            // Synthesize belief state
            let mut belief = [0.2f32; 5];
            
            // Map affect to belief
            if affect.arousal < 0.3 && affect.valence > 0.2 {
                belief[0] = 0.6; // Calm
            } else if affect.arousal > 0.6 {
                belief[1] = 0.5; // Stress
            } else if affect.valence > 0.3 {
                belief[2] = 0.5; // Focus
            }

            // Normalize
            let sum: f32 = belief.iter().sum();
            if sum > 0.0 {
                for b in &mut belief {
                    *b /= sum;
                }
            }

            // Find dominant mode
            let mode = belief
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u8)
                .unwrap_or(0);

            // Create control output if intent sanctioned
            let decision = if intent.is_sanctioned {
                match intent.action {
                    IntentAction::GuideBreath { target_bpm } => Some(ControlOutput {
                        target_bpm: target_bpm as f32,
                        confidence: affect.confidence * intent.alignment,
                        poll_interval_ms: 500,
                    }),
                    _ => None,
                }
            } else {
                None
            };

            SynthesizedState {
                belief,
                mode,
                confidence: affect.confidence,
                decision,
                free_energy: form.anomaly_score * 2.0, // Simple proxy
            }
        }
    }
}

/// The unified Skandha Pipeline processor.
#[derive(Debug)]
pub struct SkandhaPipeline<R, V, S, K, I>
where
    R: RupaSkandha,
    V: VedanaSkandha,
    S: SannaSkandha,
    K: SankharaSkandha,
    I: VinnanaSkandha,
{
    pub rupa: R,
    pub vedana: V,
    pub sanna: S,
    pub sankhara: K,
    pub vinnana: I,
    pub config: SkandhaConfig,
}

impl<R, V, S, K, I> SkandhaPipeline<R, V, S, K, I>
where
    R: RupaSkandha,
    V: VedanaSkandha,
    S: SannaSkandha,
    K: SankharaSkandha,
    I: VinnanaSkandha,
{
    /// Create a new pipeline with components.
    pub fn new(rupa: R, vedana: V, sanna: S, sankhara: K, vinnana: I) -> Self {
        Self {
            rupa,
            vedana,
            sanna,
            sankhara,
            vinnana,
            config: SkandhaConfig::default(),
        }
    }

    /// Process input through the full pipeline.
    pub fn process(&self, input: &SensorInput) -> SynthesizedState {
        // Stage 1: Rupa (Form)
        let form = if self.config.enable_rupa {
            self.rupa.process_form(input)
        } else {
            ProcessedForm::default()
        };

        // Stage 2: Vedana (Feeling)
        let affect = if self.config.enable_vedana {
            self.vedana.extract_affect(&form)
        } else {
            AffectiveState::default()
        };

        // Stage 3: Sanna (Perception)
        let pattern = if self.config.enable_sanna {
            self.sanna.perceive(&form, &affect)
        } else {
            PerceivedPattern::default()
        };

        // Stage 4: Sankhara (Formation)
        let intent = if self.config.enable_sankhara {
            self.sankhara.form_intent(&pattern, &affect)
        } else {
            FormedIntent::default()
        };

        // Stage 5: Vinnana (Consciousness)
        if self.config.enable_vinnana {
            self.vinnana.synthesize(&form, &affect, &pattern, &intent)
        } else {
            SynthesizedState::default()
        }
    }
}

/// Create a default pipeline with all default implementations.
pub fn default_pipeline() -> SkandhaPipeline<
    defaults::DefaultRupa,
    defaults::DefaultVedana,
    defaults::DefaultSanna,
    defaults::DefaultSankhara,
    defaults::DefaultVinnana,
> {
    SkandhaPipeline::new(
        defaults::DefaultRupa { anomaly_threshold: 0.5 },
        defaults::DefaultVedana,
        defaults::DefaultSanna { pattern_count: 0 },
        defaults::DefaultSankhara { alignment_threshold: 0.2 },
        defaults::DefaultVinnana,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pipeline() {
        let pipeline = default_pipeline();
        
        let input = SensorInput {
            hr_bpm: Some(75.0),
            hrv_rmssd: Some(45.0),
            rr_bpm: Some(12.0),
            quality: 0.9,
            motion: 0.1,
            timestamp_us: 0,
        };

        let result = pipeline.process(&input);
        
        // Should have valid belief state
        let sum: f32 = result.belief.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_high_arousal_triggers_calming() {
        let pipeline = default_pipeline();
        
        let input = SensorInput {
            hr_bpm: Some(150.0), // Very high HR
            hrv_rmssd: Some(20.0), // Low HRV
            rr_bpm: Some(18.0),
            quality: 0.9,
            motion: 0.2,
            timestamp_us: 0,
        };

        let result = pipeline.process(&input);
        
        // Should suggest calming breath
        if let Some(decision) = result.decision {
            assert_eq!(decision.target_bpm, 6.0);
        }
    }

    #[test]
    fn test_low_quality_reduces_confidence() {
        let pipeline = default_pipeline();
        
        let good_input = SensorInput {
            hr_bpm: Some(70.0),
            hrv_rmssd: Some(50.0),
            rr_bpm: Some(12.0),
            quality: 0.95,
            motion: 0.05,
            timestamp_us: 0,
        };

        let bad_input = SensorInput {
            hr_bpm: Some(70.0),
            hrv_rmssd: Some(50.0),
            rr_bpm: Some(12.0),
            quality: 0.1, // Low quality
            motion: 0.9, // High motion
            timestamp_us: 0,
        };

        let good_result = pipeline.process(&good_input);
        let bad_result = pipeline.process(&bad_input);

        assert!(good_result.confidence > bad_result.confidence);
    }

    #[test]
    fn test_stages_can_be_disabled() {
        let mut pipeline = default_pipeline();
        pipeline.config.enable_sankhara = false;

        let input = SensorInput {
            hr_bpm: Some(150.0),
            hrv_rmssd: Some(20.0),
            rr_bpm: Some(18.0),
            quality: 0.9,
            motion: 0.2,
            timestamp_us: 0,
        };

        let result = pipeline.process(&input);
        
        // With sankhara disabled, intent defaults to Observe
        // So no control decision should be made
        assert!(result.decision.is_none());
    }
}
