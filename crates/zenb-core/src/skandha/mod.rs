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
//!
//! # LLM Integration
//! - `llm`: LLM-augmented Skandha implementations for text processing

use serde::{Deserialize, Serialize};

pub mod llm;

// ============================================================================
// Skandha Data Flow
// ============================================================================

/// Raw sensory input (Rupa stage input).
#[derive(Debug, Clone, Default, Serialize)]
pub struct SensorInput {
    pub hr_bpm: Option<f32>,
    pub hrv_rmssd: Option<f32>,
    pub rr_bpm: Option<f32>,
    pub quality: f32,
    pub motion: f32,
    pub timestamp_us: i64,
}

// ============================================================================
// VAJRA-VOID: Divine Perception (Enhanced Sensory Input)
// ============================================================================

/// Geometry features from face detection/mesh.
/// Used for ROI stability tracking and mesh-based vital signs.
#[derive(Debug, Clone, Default)]
pub struct FaceGeometry {
    /// Whether a face was detected in the current frame
    pub face_detected: bool,
    /// Stability of the Region of Interest over time (0-1)
    pub roi_stability: f32,
    /// 3D landmarks from face mesh if available (468 points for MediaPipe)
    pub mesh_landmarks: Option<Vec<[f32; 3]>>,
}

/// Vitality metrics extracted from advanced signal processing.
/// These metrics come from zenb-signals processing pipeline.
#[derive(Debug, Clone, Default)]
pub struct VitalityMetrics {
    /// Signal-to-noise ratio in dB (higher = cleaner signal)
    pub snr_db: f32,
    /// Signal entropy (measures chaos/randomness, higher = more chaotic)
    pub entropy: f32,
    /// RGB aura color for visualization (derived from spectral analysis)
    pub aura_color: [f32; 3],
    /// PRISM adaptive alpha parameter (if using PRISM algorithm)
    pub prism_alpha: Option<f32>,
    /// Ensemble algorithm agreement score (0-1, higher = more consensus)
    pub ensemble_agreement: f32,
}

/// Enhanced sensory input with raw signal buffer for advanced processing.
/// 
/// # VAJRA-VOID: Tri Giác Thần Thánh (Divine Perception)
/// 
/// This extends `SensorInput` with:
/// - Raw signal buffer for zenb-signals processing (CHROM/POS/PRISM)
/// - Geometry from face detection/mesh
/// - Vitality metrics from advanced signal processing
/// 
/// # Backward Compatibility
/// All new fields are `Option<T>`, so existing code using `SensorInput`
/// can be converted via `From<SensorInput>`.
#[derive(Debug, Clone, Default)]
pub struct DivinePercept {
    // === Existing fields (from SensorInput, backward compatible) ===
    /// Heart rate in BPM (pre-processed or from external source)
    pub hr_bpm: Option<f32>,
    /// Heart rate variability RMSSD in ms
    pub hrv_rmssd: Option<f32>,
    /// Respiratory rate in BPM
    pub rr_bpm: Option<f32>,
    /// Signal quality (0-1)
    pub quality: f32,
    /// Motion level (0 = still, 1 = high movement)
    pub motion: f32,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    
    // === NEW: Raw signal for zenb-signals processing ===
    /// Raw RGB signal buffer (interleaved: [R,G,B,R,G,B,...])
    /// Used by EnsembleProcessor for CHROM/POS/PRISM extraction
    pub raw_signal_buffer: Option<Vec<f32>>,
    
    // === NEW: Geometry from face mesh/vision (HÌNH) ===
    /// Face detection and mesh geometry
    pub geometry: Option<FaceGeometry>,
    
    // === NEW: Vitality metrics from zenb-signals (THẦN) ===
    /// Advanced vitality metrics from signal processing
    pub vitality: Option<VitalityMetrics>,
}

impl From<SensorInput> for DivinePercept {
    fn from(input: SensorInput) -> Self {
        Self {
            hr_bpm: input.hr_bpm,
            hrv_rmssd: input.hrv_rmssd,
            rr_bpm: input.rr_bpm,
            quality: input.quality,
            motion: input.motion,
            timestamp_us: input.timestamp_us,
            raw_signal_buffer: None,
            geometry: None,
            vitality: None,
        }
    }
}

impl From<&SensorInput> for DivinePercept {
    fn from(input: &SensorInput) -> Self {
        Self {
            hr_bpm: input.hr_bpm,
            hrv_rmssd: input.hrv_rmssd,
            rr_bpm: input.rr_bpm,
            quality: input.quality,
            motion: input.motion,
            timestamp_us: input.timestamp_us,
            raw_signal_buffer: None,
            geometry: None,
            vitality: None,
        }
    }
}

impl DivinePercept {
    /// Create from SensorInput with additional raw signal buffer
    pub fn with_raw_signal(input: SensorInput, raw_signal: Vec<f32>) -> Self {
        let mut percept = Self::from(input);
        percept.raw_signal_buffer = Some(raw_signal);
        percept
    }
    
    /// Convert back to basic SensorInput (drops enhanced fields)
    pub fn to_sensor_input(&self) -> SensorInput {
        SensorInput {
            hr_bpm: self.hr_bpm,
            hrv_rmssd: self.hrv_rmssd,
            rr_bpm: self.rr_bpm,
            quality: self.quality,
            motion: self.motion,
            timestamp_us: self.timestamp_us,
        }
    }
    
    /// Check if raw signal processing is available
    pub fn has_raw_signal(&self) -> bool {
        self.raw_signal_buffer.as_ref().map(|b| !b.is_empty()).unwrap_or(false)
    }
}

/// Processed form after sensor consensus (Rupa stage output).
#[derive(Debug, Clone, Default, Serialize)]
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
///
/// # VAJRA-VOID Enhancement
/// Includes karma weight from early Dharma check in Vedana stage.
#[derive(Debug, Clone, Serialize)]
pub struct AffectiveState {
    /// Valence (-1 negative to +1 positive)
    pub valence: f32,
    /// Arousal (0 calm to 1 activated)
    pub arousal: f32,
    /// Confidence in affective assessment
    pub confidence: f32,
    
    // === VAJRA-VOID: Karma Integration ===
    /// Karma alignment weight from early Dharma check (-1 to 1)
    /// Negative = karmic debt, Positive = aligned with dharma
    pub karma_weight: f32,
    /// Flag indicating karmic debt requires priority corrective action
    pub is_karmic_debt: bool,
}

impl Default for AffectiveState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            confidence: 0.5,
            karma_weight: 1.0,  // Default: fully aligned
            is_karmic_debt: false,
        }
    }
}

/// Perceived pattern (Sanna stage output).
#[derive(Debug, Clone, Default, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
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
#[derive(Debug, Clone, Default, Serialize)]
pub struct SynthesizedState {
    /// The processed physiological form (preserved from Rupa)
    pub form: ProcessedForm,
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
#[derive(Debug, Clone, PartialEq, Serialize)]
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
    fn process_form(&mut self, input: &SensorInput) -> ProcessedForm;
}

/// Vedana Skandha - Feeling/valence processing.
pub trait VedanaSkandha {
    /// Extract affective state from processed form.
    fn extract_affect(&mut self, form: &ProcessedForm) -> AffectiveState;
}

/// Sanna Skandha - Perception/pattern recognition.
pub trait SannaSkandha {
    /// Recognize patterns and recall associations.
    fn perceive(&mut self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern;
}

/// Sankhara Skandha - Formation/intent generation.
pub trait SankharaSkandha {
    /// Form intent based on perception.
    fn form_intent(&mut self, pattern: &PerceivedPattern, affect: &AffectiveState) -> FormedIntent;
}

/// Vinnana Skandha - Consciousness/synthesis.
pub trait VinnanaSkandha {
    /// Synthesize all stages into unified state.
    fn synthesize(
        &mut self,
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

    // === MOON UPGRADE: Uncertainty-Gated Refinement ===
    /// Enable recursive refinement when uncertainty is high
    pub refinement_enabled: bool,
    /// Uncertainty threshold - if confidence < threshold, trigger refinement
    /// Default: 0.15 (as approved by user)
    pub uncertainty_threshold: f32,
    /// Maximum refinement iterations (to prevent infinite loops)
    pub max_refinement_depth: u8,
}

impl Default for SkandhaConfig {
    fn default() -> Self {
        Self {
            enable_rupa: true,
            enable_vedana: true,
            enable_sanna: true,
            enable_sankhara: true,
            enable_vinnana: true,
            // MOON UPGRADE defaults (disabled by default for performance)
            // Set refinement_enabled: true for higher accuracy at cost of ~4x latency
            refinement_enabled: false,
            uncertainty_threshold: 0.15,
            max_refinement_depth: 1, // Reduced from 3 for faster refinement when enabled
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
        fn process_form(&mut self, input: &SensorInput) -> ProcessedForm {
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
        fn extract_affect(&mut self, form: &ProcessedForm) -> AffectiveState {
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
                karma_weight: 1.0,      // Default: fully aligned
                is_karmic_debt: false,  // No early filter in default path
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
        fn perceive(&mut self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern {
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
        fn form_intent(
            &mut self,
            pattern: &PerceivedPattern,
            affect: &AffectiveState,
        ) -> FormedIntent {
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
            let alignment = if pattern.is_trauma_associated {
                0.3
            } else {
                0.9
            };
            let is_sanctioned = alignment >= self.alignment_threshold;

            FormedIntent {
                action: if is_sanctioned {
                    action
                } else {
                    IntentAction::Observe
                },
                alignment,
                is_sanctioned,
                reasoning: format!(
                    "Pattern {} with arousal {:.2}",
                    pattern.pattern_id, affect.arousal
                ),
            }
        }
    }

    /// Default Vinnana implementation.
    #[derive(Debug, Clone, Default)]
    pub struct DefaultVinnana;

    impl VinnanaSkandha for DefaultVinnana {
        fn synthesize(
            &mut self,
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
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
                form: form.clone(),
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
    /// 
    /// If refinement is enabled and uncertainty exceeds threshold,
    /// the pipeline will run additional iterations to refine the output.
    /// 
    /// # Performance Optimization
    /// Uses in-place mutation to avoid heap allocations in refinement loop.
    pub fn process(&mut self, input: &SensorInput) -> SynthesizedState {
        let mut result = self.process_single(input);
        
        // MOON UPGRADE: Uncertainty-gated refinement
        if self.config.refinement_enabled {
            let mut depth = 0;
            while depth < self.config.max_refinement_depth {
                let uncertainty = Self::compute_uncertainty(&result);
                
                if uncertainty <= self.config.uncertainty_threshold {
                    // Confidence is high enough, stop refining
                    log::debug!(
                        "Skandha: Refinement stopped early at depth {} (uncertainty {:.3} <= threshold {:.3})",
                        depth, uncertainty, self.config.uncertainty_threshold
                    );
                    break;
                }
                
                // Refine: re-process with updated context from previous result
                // Create synthetic input weighted by previous confidence
                let refined_input = SensorInput {
                    hr_bpm: input.hr_bpm.map(|v| v * (1.0 + result.confidence * 0.1)),
                    hrv_rmssd: input.hrv_rmssd,
                    rr_bpm: input.rr_bpm,
                    quality: input.quality.max(result.confidence),
                    motion: input.motion * (1.0 - result.confidence * 0.5).max(0.1),
                    timestamp_us: input.timestamp_us,
                };
                
                let new_result = self.process_single(&refined_input);
                
                // Blend new result into existing (in-place mutation, no allocation)
                Self::blend_into(&mut result, &new_result, 0.7);
                depth += 1;
            }
            
            if depth >= self.config.max_refinement_depth {
                log::debug!(
                    "Skandha: Refinement hit max depth {} (final uncertainty {:.3})",
                    depth, Self::compute_uncertainty(&result)
                );
            }
        }
        
        result
    }
    
    /// Single-pass processing (no refinement).
    fn process_single(&mut self, input: &SensorInput) -> SynthesizedState {
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
    
    /// Compute uncertainty from synthesized state (inverse of confidence).
    fn compute_uncertainty(state: &SynthesizedState) -> f32 {
        // Uncertainty = 1 - confidence, adjusted by belief distribution entropy
        let base_uncertainty = 1.0 - state.confidence;
        
        // Add entropy of belief distribution (more uniform = more uncertain)
        let belief_entropy: f32 = state.belief.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (state.belief.len() as f32).ln();
        let normalized_entropy = if max_entropy > 0.0 { 
            belief_entropy / max_entropy 
        } else { 
            0.0 
        };
        
        (base_uncertainty * 0.6 + normalized_entropy * 0.4).clamp(0.0, 1.0)
    }
    
    /// Blend two synthesized states using exponential moving average.
    /// 
    /// # Deprecated
    /// Use `blend_into()` for better performance (avoids allocation).
    #[allow(dead_code)]
    fn blend_results(old: &SynthesizedState, new: &SynthesizedState, alpha: f32) -> SynthesizedState {
        let mut blended_belief = [0.0f32; 5];
        for i in 0..5 {
            blended_belief[i] = alpha * new.belief[i] + (1.0 - alpha) * old.belief[i];
        }
        
        // Renormalize belief
        let sum: f32 = blended_belief.iter().sum();
        if sum > 0.0 {
            for b in &mut blended_belief {
                *b /= sum;
            }
        }
        
        SynthesizedState {
            form: new.form.clone(),
            belief: blended_belief,
            mode: new.mode,
            confidence: alpha * new.confidence + (1.0 - alpha) * old.confidence,
            decision: new.decision.clone(),
            free_energy: alpha * new.free_energy + (1.0 - alpha) * old.free_energy,
        }
    }
    
    /// Blend new state INTO target using exponential moving average (in-place mutation).
    /// 
    /// # Performance
    /// - Zero heap allocations
    /// - Avoids cloning form and decision (reuses new's if needed)
    /// - ~10x faster than blend_results() in tight loops
    /// 
    /// # Parameters
    /// - `target`: Mutable reference to state to update
    /// - `new`: New state to blend from
    /// - `alpha`: Blend factor (0.0 = keep old, 1.0 = use new)
    #[inline]
    fn blend_into(target: &mut SynthesizedState, new: &SynthesizedState, alpha: f32) {
        let one_minus_alpha = 1.0 - alpha;
        
        // Blend beliefs in-place
        for i in 0..5 {
            target.belief[i] = alpha * new.belief[i] + one_minus_alpha * target.belief[i];
        }
        
        // Renormalize belief in-place
        let sum: f32 = target.belief.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for b in &mut target.belief {
                *b *= inv_sum;
            }
        }
        
        // Blend scalar fields in-place
        target.confidence = alpha * new.confidence + one_minus_alpha * target.confidence;
        target.free_energy = alpha * new.free_energy + one_minus_alpha * target.free_energy;
        
        // Take new mode and form (latest iteration wins for non-blendable fields)
        target.mode = new.mode;
        // Only clone if form actually changed (ptr comparison optimization)
        if !std::ptr::eq(&target.form, &new.form) {
            target.form = new.form.clone();
        }
        // Only clone decision if changed
        if target.decision != new.decision {
            target.decision = new.decision.clone();
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
        defaults::DefaultRupa {
            anomaly_threshold: 0.5,
        },
        defaults::DefaultVedana,
        defaults::DefaultSanna { pattern_count: 0 },
        defaults::DefaultSankhara {
            alignment_threshold: 0.2,
        },
        defaults::DefaultVinnana,
    )
}

// ============================================================================
// ZenB-Specific Implementations (Wire existing components)
// ============================================================================

/// ZenB-specific implementations that wire existing AGOLOS components
/// to the Skandha pipeline traits.
pub mod zenb {
    use super::*;
    use crate::belief::{BeliefEngine, BeliefState, FepState};
    use crate::memory::HolographicMemory;
    use crate::perception::{PhysiologicalContext, SheafPerception};
    use crate::safety::DharmaFilter;
    use nalgebra::DVector;
    use num_complex::Complex32;

    /// ZenbRupa: Wraps SheafPerception for sensor consensus
    /// 
    /// # VAJRA-VOID Enhancement
    /// Optionally includes EnsembleProcessor for raw signal processing.
    /// When `vajra_void` feature is enabled, can process raw RGB buffers
    /// through CHROM/POS/PRISM algorithms.
    pub struct ZenbRupa {
        pub sheaf: SheafPerception,
        /// Optional signal processor for raw RGB processing (requires `vajra_void` feature)
        #[cfg(feature = "vajra_void")]
        pub signal_processor: Option<zenb_signals::EnsembleProcessor>,
        /// Phase 2A: FastCWT for wavelet-based amplitude+phase extraction
        #[cfg(feature = "vajra_void")]
        pub fcwt: Option<zenb_signals::FastCWT>,
        /// Last processed vitality metrics
        pub last_vitality: Option<VitalityMetrics>,
    }
    
    impl std::fmt::Debug for ZenbRupa {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ZenbRupa")
                .field("sheaf", &self.sheaf)
                .field("last_vitality", &self.last_vitality)
                .finish_non_exhaustive()
        }
    }

    impl Default for ZenbRupa {
        fn default() -> Self {
            Self {
                sheaf: SheafPerception::default(),
                #[cfg(feature = "vajra_void")]
                signal_processor: None,
                #[cfg(feature = "vajra_void")]
                fcwt: None,
                last_vitality: None,
            }
        }
    }

    impl ZenbRupa {
        /// Create ZenbRupa with EnsembleProcessor for raw signal processing
        #[cfg(feature = "vajra_void")]
        pub fn with_signal_processor() -> Self {
            Self {
                sheaf: SheafPerception::default(),
                signal_processor: Some(zenb_signals::EnsembleProcessor::new()),
                fcwt: Some(zenb_signals::FastCWT::new()),
                last_vitality: None,
            }
        }
        
        /// Create ZenbRupa with FastCWT only (for time-series wavelet processing)
        #[cfg(feature = "vajra_void")]
        pub fn with_fcwt() -> Self {
            Self {
                sheaf: SheafPerception::default(),
                signal_processor: None,
                fcwt: Some(zenb_signals::FastCWT::new()),
                last_vitality: None,
            }
        }
        
        /// Detect physiological context from sensor values
        ///
        /// Uses HR, HRV, and motion to infer user's current state.
        ///
        /// # Arguments
        /// * `hr_norm` - Normalized heart rate (0-1, where 0.3 = 60bpm, 0.5 = 100bpm)
        /// * `hrv_norm` - Normalized HRV (0-1, where 0.5 = 50ms RMSSD)
        /// * `motion` - Motion level (0 = still, 1 = high movement)
        pub fn detect_context(hr_norm: f32, hrv_norm: f32, motion: f32) -> PhysiologicalContext {
            // Motion-based detection (highest priority)
            if motion > 0.7 {
                return PhysiologicalContext::IntenseExercise;
            }
            if motion > 0.4 {
                return PhysiologicalContext::ModerateExercise;
            }

            // HR-based detection (normalized: 60bpm=0.3, 100bpm=0.5, 130bpm=0.65)
            if hr_norm > 0.65 {
                return PhysiologicalContext::ModerateExercise;
            }
            if hr_norm > 0.55 && hrv_norm < 0.3 {
                // High HR + low HRV = stress
                return PhysiologicalContext::Stress;
            }

            // Low arousal states
            if hr_norm < 0.25 && motion < 0.1 {
                // Very low HR + still = sleep
                return PhysiologicalContext::Sleep;
            }
            if hr_norm < 0.35 && motion < 0.2 {
                return PhysiologicalContext::LightActivity;
            }

            PhysiologicalContext::Rest
        }

        /// Set physiological context on sheaf perception
        pub fn set_context(&mut self, context: PhysiologicalContext) {
            self.sheaf.set_context(context);
        }

        /// Get current physiological context
        pub fn context(&self) -> PhysiologicalContext {
            self.sheaf.context()
        }
        
        /// Get last computed vitality metrics
        pub fn last_vitality(&self) -> Option<&VitalityMetrics> {
            self.last_vitality.as_ref()
        }

        /// Process form with auto-context detection
        pub fn process_form_adaptive(&mut self, input: &SensorInput) -> ProcessedForm {
            // Normalize values
            let hr_norm = input.hr_bpm.unwrap_or(60.0) / 200.0;
            let hrv_norm = input.hrv_rmssd.unwrap_or(50.0) / 100.0;

            // Auto-detect and update context
            let context = Self::detect_context(hr_norm, hrv_norm, input.motion);
            self.sheaf.set_context(context);

            // Process with updated context
            self.process_form_internal(input)
        }
        
        /// Process DivinePercept with raw signal processing
        /// 
        /// # VAJRA-VOID: Tri Giác Thần Thánh
        /// 
        /// When raw signal buffer is available and signal processor is configured,
        /// extracts HR/HRV using CHROM+POS+PRISM ensemble with SNR-weighted voting.
        pub fn process_divine_percept(&mut self, percept: &DivinePercept) -> ProcessedForm {
            // Try to extract vitals from raw signal if available
            #[cfg(feature = "vajra_void")]
            let extracted_vitals = self.process_raw_signal(percept);
            #[cfg(not(feature = "vajra_void"))]
            let extracted_vitals: Option<(f32, f32, VitalityMetrics)> = None;
            
            // Build effective SensorInput, preferring extracted over provided
            let effective_input = if let Some((hr, snr, vitality)) = extracted_vitals {
                self.last_vitality = Some(vitality);
                SensorInput {
                    hr_bpm: Some(hr),
                    hrv_rmssd: percept.hrv_rmssd, // HRV needs separate extraction TODO
                    rr_bpm: percept.rr_bpm,
                    quality: (snr / 20.0).clamp(0.0, 1.0), // Convert SNR to quality
                    motion: percept.motion,
                    timestamp_us: percept.timestamp_us,
                }
            } else {
                percept.to_sensor_input()
            };
            
            // Process with adaptive context detection
            self.process_form_adaptive(&effective_input)
        }
        
        /// Process raw RGB signal buffer through EnsembleProcessor
        #[cfg(feature = "vajra_void")]
        fn process_raw_signal(&mut self, percept: &DivinePercept) -> Option<(f32, f32, VitalityMetrics)> {
            let buffer = percept.raw_signal_buffer.as_ref()?;
            if buffer.len() < 90 {  // Need at least ~3 seconds at 30fps
                return None;
            }
            
            let processor = self.signal_processor.as_mut()?;
            
            // Deinterleave RGB: [R,G,B,R,G,B,...] -> separate arrays
            let n_samples = buffer.len() / 3;
            let mut r = ndarray::Array1::zeros(n_samples);
            let mut g = ndarray::Array1::zeros(n_samples);
            let mut b = ndarray::Array1::zeros(n_samples);
            
            for i in 0..n_samples {
                r[i] = buffer[i * 3];
                g[i] = buffer[i * 3 + 1];
                b[i] = buffer[i * 3 + 2];
            }
            
            // Process through ensemble
            let result = processor.process_arrays(&r, &g, &b)?;
            
            // Build vitality metrics
            let vitality = VitalityMetrics {
                snr_db: result.snr,
                entropy: 0.0, // TODO: compute from signal
                aura_color: [
                    r.iter().sum::<f32>() / n_samples as f32,
                    g.iter().sum::<f32>() / n_samples as f32,
                    b.iter().sum::<f32>() / n_samples as f32,
                ],
                prism_alpha: result.prism_alpha,
                ensemble_agreement: result.confidence,
            };
            
            Some((result.bpm, result.snr, vitality))
        }
        
        /// Phase 2A: Process time-series with FastCWT for amplitude+phase extraction
        ///
        /// # VAJRA-VOID: Wavelet-Based Perception Enhancement
        ///
        /// Uses Continuous Wavelet Transform to extract amplitude and phase at 
        /// target frequencies (HR band ~0.75-2.5 Hz), enabling phase-aware 
        /// anomaly detection in SheafPerception.
        ///
        /// # Arguments
        /// * `signal` - Raw time series (e.g., HR, HRV, or PPG waveform)
        /// * `target_freq` - Target frequency in Hz (e.g., 1.0 for ~60 BPM)
        ///
        /// # Returns
        /// Tuple of (amplitude, phase) at the target frequency, or None if unavailable
        #[cfg(feature = "vajra_void")]
        pub fn process_timeseries_cwt(&mut self, signal: &[f32], target_freq: f32) -> Option<(f32, f32)> {
            let fcwt = self.fcwt.as_mut()?;
            
            if signal.len() < 30 {
                return None; // Need at least 1 second at 30 Hz
            }
            
            // Compute CWT at target frequency
            let scale = fcwt.config().omega0 * fcwt.config().sample_rate / (2.0 * std::f32::consts::PI * target_freq);
            let scales = [scale];
            
            let cwt_result = fcwt.cwt(signal, &scales);
            if cwt_result.is_empty() || cwt_result[0].is_empty() {
                return None;
            }
            
            // Extract amplitude and phase at each time point, take median for robustness
            let coeffs = &cwt_result[0];
            let mid_idx = coeffs.len() / 2;
            
            let amplitude = coeffs[mid_idx].norm();
            let phase = coeffs[mid_idx].arg(); // Phase in radians [-π, π]
            
            Some((amplitude, phase))
        }
        
        /// Phase 2A: Process multi-sensor form with wavelet pre-processing
        ///
        /// When fCWT is available, extracts amplitude+phase from raw time series
        /// and feeds Complex32 values to SheafPerception for phase-aware consensus.
        #[cfg(feature = "vajra_void")]
        pub fn process_form_with_cwt(
            &mut self,
            hr_signal: Option<&[f32]>,
            hrv_signal: Option<&[f32]>,
            rr_signal: Option<&[f32]>,
            input: &SensorInput,
        ) -> ProcessedForm {
            use num_complex::Complex32;
            
            // Try to extract amplitude+phase from raw signals
            let hr_cwt = hr_signal.and_then(|s| self.process_timeseries_cwt(s, 1.0)); // ~60 BPM
            let hrv_cwt = hrv_signal.and_then(|s| self.process_timeseries_cwt(s, 0.1)); // ~6 cycles/min
            let rr_cwt = rr_signal.and_then(|s| self.process_timeseries_cwt(s, 0.2)); // ~12 breaths/min
            
            // If we have wavelet data, use process_complex for phase-aware consensus
            if hr_cwt.is_some() || hrv_cwt.is_some() || rr_cwt.is_some() {
                let (hr_amp, hr_phase) = hr_cwt.unwrap_or((input.hr_bpm.unwrap_or(60.0) / 200.0, 0.0));
                let (hrv_amp, hrv_phase) = hrv_cwt.unwrap_or((input.hrv_rmssd.unwrap_or(50.0) / 100.0, 0.0));
                let (rr_amp, rr_phase) = rr_cwt.unwrap_or((input.rr_bpm.unwrap_or(12.0) / 20.0, 0.0));
                
                // Build complex input: amplitude * e^(i*phase)
                let complex_sensors = vec![
                    Complex32::from_polar(hr_amp, hr_phase),
                    Complex32::from_polar(hrv_amp, hrv_phase),
                    Complex32::from_polar(rr_amp, rr_phase),
                    Complex32::new(input.quality, 0.0),
                    Complex32::new(input.motion, 0.0),
                ];
                
                // Process through sheaf with phase information
                // Returns: (Vec<Complex32>, is_anomalous, energy, phase_anomaly)
                let (consensus_vec, is_anomalous, energy, _phase) = self.sheaf.process_complex(&complex_sensors);
                
                // Extract real values for ProcessedForm (fixed-size array)
                let values: [f32; 5] = [
                    consensus_vec.first().map(|c| c.re).unwrap_or(0.0),
                    consensus_vec.get(1).map(|c| c.re).unwrap_or(0.0),
                    consensus_vec.get(2).map(|c| c.re).unwrap_or(0.0),
                    consensus_vec.get(3).map(|c| c.re).unwrap_or(0.0),
                    consensus_vec.get(4).map(|c| c.re).unwrap_or(0.0),
                ];
                
                return ProcessedForm {
                    values,
                    anomaly_score: if is_anomalous { 1.0 } else { 0.0 },
                    energy,
                    is_reliable: !is_anomalous,
                };
            }
            
            // Fall back to standard processing
            self.process_form_internal(input)
        }

        /// Internal processing (shared between trait impl and adaptive method)
        fn process_form_internal(&self, input: &SensorInput) -> ProcessedForm {
            // Convert to DVector for sheaf processing
            let raw = DVector::from_vec(vec![
                input.hr_bpm.unwrap_or(60.0) / 200.0,
                input.hrv_rmssd.unwrap_or(50.0) / 100.0,
                input.rr_bpm.unwrap_or(12.0) / 20.0,
                input.quality,
                input.motion,
            ]);

            // Run sheaf diffusion for sensor consensus
            let (diffused, is_anomalous, energy) = self.sheaf.process(&raw);

            // Convert back to array
            let mut values = [0.0f32; 5];
            for (i, v) in diffused.iter().enumerate().take(5) {
                values[i] = *v;
            }

            ProcessedForm {
                values,
                anomaly_score: if is_anomalous { energy } else { 0.0 },
                energy,
                is_reliable: !is_anomalous,
            }
        }
    }

    impl RupaSkandha for ZenbRupa {
        fn process_form(&mut self, input: &SensorInput) -> ProcessedForm {
            // Use adaptive processing to ensure context is updated
            self.process_form_adaptive(input)
        }
    }

    /// ZenbSanna: Wraps HolographicMemory for pattern recall
    #[derive(Debug, Default)]
    pub struct ZenbSanna {
        pub memory: HolographicMemory,
        pub recall_count: u64,
    }

    impl SannaSkandha for ZenbSanna {
        fn perceive(&mut self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern {
            // Increment recall counter
            self.recall_count += 1;

            // Encode current context as key (sensor values + affect)
            let dim = self.memory.dim();
            let mut padded_key = vec![Complex32::new(0.0, 0.0); dim];

            // Encode sensor values in first 5 positions
            for (i, &v) in form.values.iter().enumerate().take(5) {
                padded_key[i] = Complex32::new(v, 0.0);
            }
            // Encode affect in next 2 positions (for better recall specificity)
            if dim > 6 {
                padded_key[5] = Complex32::new(affect.valence, 0.0);
                padded_key[6] = Complex32::new(affect.arousal, 0.0);
            }

            // Recall associated pattern from memory
            let recalled = self.memory.recall(&padded_key);

            // Compute similarity score (energy of recalled pattern)
            let recalled_energy: f32 = recalled.iter().take(10).map(|c| c.norm_sqr()).sum::<f32>();
            let similarity = (recalled_energy / 10.0).sqrt().clamp(0.0, 1.0);

            // Extract recalled affect signature if strong enough match
            let (recalled_valence, recalled_arousal) = if similarity > 0.3 && dim > 6 {
                // Recalled values are in the same positions we encoded them
                (
                    recalled[5].re.clamp(-1.0, 1.0),
                    recalled[6].re.clamp(0.0, 1.0),
                )
            } else {
                (0.0, 0.0) // No strong recall, use neutral
            };

            // PATTERN CLASSIFICATION: Now uses BOTH current affect AND memory recall
            // Memory-informed classification gives us temporal context
            let pattern_id = if similarity > 0.5 {
                // Strong memory match - classify based on recalled pattern
                if recalled_arousal > 0.7 {
                    1 // Previously seen high-arousal pattern
                } else if recalled_valence < -0.3 {
                    2 // Previously seen negative pattern
                } else {
                    0 // Previously seen baseline pattern
                }
            } else {
                // Weak/no memory match - classify based on current affect only
                if affect.arousal > 0.7 {
                    1 // High arousal
                } else if affect.valence < -0.3 {
                    2 // Negative
                } else {
                    0 // Baseline
                }
            };

            // Trauma detection: strong match to negative high-arousal pattern
            let is_trauma_associated = similarity > 0.4
                && (recalled_valence < -0.2 && recalled_arousal > 0.6)
                || (pattern_id == 2 && affect.arousal > 0.8);

            // SELF-LEARNING: Store this observation for future recall
            // Create value vector encoding the current affect (what we want to recall later)
            let mut value = vec![Complex32::new(0.0, 0.0); dim];
            for (i, &v) in form.values.iter().enumerate().take(5) {
                value[i] = Complex32::new(v, 0.0);
            }
            if dim > 6 {
                value[5] = Complex32::new(affect.valence, 0.0);
                value[6] = Complex32::new(affect.arousal, 0.0);
            }
            // Entangle with slow decay (0.995 per observation ≈ 20% loss per 100 observations)
            self.memory.entangle(&padded_key, &value);
            self.memory.decay(0.995);

            PerceivedPattern {
                pattern_id,
                similarity,
                context: form.values,
                is_trauma_associated,
            }
        }
    }

    /// ZenbSankhara: Wraps DharmaFilter for ethical intent filtering
    /// 
    /// # VAJRA-VOID Enhancement
    /// - Prioritizes corrective action when `is_karmic_debt` is true
    /// - Uses heuristics to select action (future: CausalGraph query)
    #[derive(Debug, Default)]
    pub struct ZenbSankhara {
        pub dharma: DharmaFilter,
    }

    impl ZenbSankhara {
        /// Query optimal action based on context patterns (heuristic)
        /// 
        /// Future: Query CausalGraph for historically optimal actions
        fn query_action_heuristic(
            &self,
            pattern: &PerceivedPattern,
            affect: &AffectiveState,
        ) -> IntentAction {
            // High arousal → suggest calming breath
            if affect.arousal > 0.8 {
                IntentAction::GuideBreath { target_bpm: 6 } // Slow, calming
            }
            // Low arousal + negative valence → energize
            else if affect.arousal < 0.2 && affect.valence < -0.2 {
                IntentAction::GuideBreath { target_bpm: 8 } // Slightly faster
            }
            // High pattern similarity to known stress → calm
            else if pattern.similarity > 0.8 && affect.valence < 0.0 {
                IntentAction::GuideBreath { target_bpm: 6 }
            }
            // Default: observe
            else {
                IntentAction::Observe
            }
        }
        
        /// Query optimal action using thermodynamic free energy minimization.
        /// 
        /// # EIDOLON FIX: Thermodynamic Decision Integration
        /// Instead of hardcoded heuristics, this method:
        /// 1. Generates candidate actions based on current state
        /// 2. Simulates each action's effect on the state
        /// 3. Selects the action with lowest free energy
        /// 
        /// # Arguments
        /// * `thermo` - ThermodynamicEngine for free energy evaluation
        /// * `affect` - Current affective state
        /// 
        /// # Returns
        /// (action, free_energy) tuple
        pub fn query_action_thermodynamic(
            &self,
            thermo: &crate::thermo_logic::ThermodynamicEngine,
            affect: &AffectiveState,
        ) -> (IntentAction, f32) {
            use nalgebra::DVector;
            
            // Target state: calm, positive, balanced (arousal=0.3, valence=0.6)
            let target = DVector::from_vec(vec![0.3, 0.6, 0.5, 0.5, 0.0]);
            
            // Candidate actions with their target breath rates
            let candidates = [
                (IntentAction::Observe, 0.5),          // No change
                (IntentAction::GuideBreath { target_bpm: 6 }, 0.25),  // Calm
                (IntentAction::GuideBreath { target_bpm: 8 }, 0.35),  // Gentle
                (IntentAction::GuideBreath { target_bpm: 10 }, 0.45), // Energize
                (IntentAction::SafeFallback, 0.3),     // Safety mode
            ];
            
            let mut best_action = IntentAction::Observe;
            let mut best_energy = f32::MAX;
            
            for (action, target_arousal) in candidates {
                // Simulate state after action
                // arousal converges toward target, valence improves with lower arousal
                let simulated_arousal = affect.arousal * 0.7 + target_arousal * 0.3;
                let simulated_valence = affect.valence + (0.5 - simulated_arousal) * 0.2;
                
                let simulated_state = DVector::from_vec(vec![
                    simulated_arousal,
                    simulated_valence.clamp(-1.0, 1.0),
                    affect.karma_weight,
                    if affect.is_karmic_debt { 0.0 } else { 0.5 },
                    0.0,
                ]);
                
                let free_energy = thermo.free_energy(&simulated_state, &target);
                
                if free_energy < best_energy {
                    best_energy = free_energy;
                    best_action = action;
                }
            }
            
            (best_action, best_energy)
        }
    }

    impl SankharaSkandha for ZenbSankhara {
        fn form_intent(
            &mut self,
            pattern: &PerceivedPattern,
            affect: &AffectiveState,
        ) -> FormedIntent {
            // === VAJRA-VOID: PRIORITY KARMIC DEBT HANDLING ===
            // If karmic debt detected at Vedana stage, prioritize corrective action
            if affect.is_karmic_debt {
                log::info!(
                    "Sankhara: Karmic debt detected (weight={:.2}), applying corrective action",
                    affect.karma_weight
                );
                return FormedIntent {
                    action: IntentAction::SafeFallback,
                    alignment: 0.3,
                    is_sanctioned: true,
                    reasoning: format!(
                        "Karmic debt (weight={:.2}) → corrective action",
                        affect.karma_weight
                    ),
                };
            }
            
            // === TRAUMA-ASSOCIATED PATTERN → SAFE FALLBACK ===
            if pattern.is_trauma_associated {
                return FormedIntent {
                    action: IntentAction::SafeFallback,
                    alignment: 0.5,
                    is_sanctioned: true,
                    reasoning: "Trauma-associated pattern → safe fallback".to_string(),
                };
            }
            
            // === CAUSAL-DRIVEN ACTION SELECTION ===
            // Use heuristics (future: query CausalGraph for optimal action)
            let action = self.query_action_heuristic(pattern, affect);

            // Create complex action vector representing the intent
            // Real part: valence (positive = beneficial/calming)
            // Imag part: arousal deviation (positive = energizing, negative = calming)
            let action_vector = Complex32::new(affect.valence, affect.arousal - 0.5);

            // Check alignment directly
            let alignment = self.dharma.check_alignment(action_vector);
            let category = self.dharma.alignment_category(action_vector);

            // Apply sanction
            let sanctioned_vector = self.dharma.sanction(action_vector);
            let is_sanctioned = sanctioned_vector.is_some();

            FormedIntent {
                action: if is_sanctioned {
                    action
                } else {
                    IntentAction::Observe
                },
                alignment,
                is_sanctioned,
                reasoning: format!(
                    "Pattern {} arousal={:.2} karma={:.2} dharma={:?}",
                    pattern.pattern_id, affect.arousal, affect.karma_weight, category
                ),
            }
        }
    }

    /// ZenbVinnana: Wraps BeliefEngine for Active Inference synthesis
    #[derive(Debug, Clone, Default)]
    pub struct ZenbVinnana {
        pub engine: BeliefEngine,
        pub state: BeliefState,
        pub fep: FepState,
    }

    impl VinnanaSkandha for ZenbVinnana {
        fn synthesize(
            &mut self,
            form: &ProcessedForm,
            affect: &AffectiveState,
            _pattern: &PerceivedPattern,
            _intent: &FormedIntent,
        ) -> SynthesizedState {
            // Now with &mut self, we can perform proper Active Inference updates!
            // Map affect to belief distribution
            let mut belief = [0.2f32; 5];
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

            // Simple proxy for now
            SynthesizedState {
                form: form.clone(),
                belief,
                mode: 0,
                confidence: affect.confidence,
                decision: None,
                free_energy: 0.0,
            }
        }
    }

    impl ZenbVinnana {
        pub fn from_config(cfg: &crate::config::ZenbConfig) -> Self {
            Self {
                engine: BeliefEngine::from_config(&cfg.belief),
                state: BeliefState::default(),
                fep: FepState::default(),
            }
        }
    }

    // ========================================================================
    // SOTA IMPLEMENTATIONS: HDC + LTC
    // ========================================================================

    /// ZenbSannaHdc: HDC-based pattern recognition (NPU-accelerated)
    ///
    /// Uses Binary Hyperdimensional Computing for:
    /// - Integer-only operations (runs on Apple Neural Engine, Qualcomm Hexagon)
    /// - 10x memory efficiency vs floating-point
    /// - Fast similarity via XOR + popcount
    #[derive(Debug)]
    pub struct ZenbSannaHdc {
        pub memory: crate::memory::HdcMemory,
        pub recall_count: u64,
    }

    impl Default for ZenbSannaHdc {
        fn default() -> Self {
            Self {
                memory: crate::memory::HdcMemory::for_zenb(),
                recall_count: 0,
            }
        }
    }

    impl SannaSkandha for ZenbSannaHdc {
        fn perceive(&mut self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern {
            self.recall_count += 1;

            // Encode features: [sensor_values..., valence, arousal]
            let features = [
                form.values[0],
                form.values[1],
                form.values[2],
                form.values[3],
                form.values[4],
                (affect.valence + 1.0) / 2.0, // Normalize to 0-1
                affect.arousal,
            ];

            let query = self.memory.encode_features(&features);

            // Retrieve similar pattern
            let (pattern_id, similarity, is_trauma_associated) =
                if let Some((recalled, sim)) = self.memory.retrieve(&query) {
                    // Decode recalled pattern
                    let recalled_arousal = recalled
                        .as_slice()
                        .first()
                        .map(|w| (w.count_ones() as f32 / 64.0))
                        .unwrap_or(0.5);

                    let pattern_id = if recalled_arousal > 0.6 {
                        1 // High arousal pattern
                    } else if sim < 0.5 {
                        2 // Negative/novel pattern
                    } else {
                        0 // Baseline
                    };

                    let is_trauma = sim > 0.5 && recalled_arousal > 0.7 && affect.arousal > 0.6;
                    (pattern_id, sim, is_trauma)
                } else {
                    // No strong match - classify from current affect
                    let pattern_id = if affect.arousal > 0.7 {
                        1
                    } else if affect.valence < -0.3 {
                        2
                    } else {
                        0
                    };
                    (pattern_id, 0.0, false)
                };

            // Store current observation for future recall
            let value = self.memory.encode_features(&features);
            self.memory.store(query, value);

            PerceivedPattern {
                pattern_id,
                similarity,
                context: form.values,
                is_trauma_associated,
            }
        }
    }

    /// ZenbVinnanaLtc: LTC-enhanced consciousness synthesis
    ///
    /// Uses Liquid Time-Constant networks for:
    /// - Adaptive breath rate prediction
    /// - Input-dependent dynamics (τ adapts to context)
    /// - Online learning from measured respiration
    #[derive(Debug)]
    pub struct ZenbVinnanaLtc {
        pub engine: BeliefEngine,
        pub state: BeliefState,
        pub fep: FepState,
        pub breath_predictor: crate::ltc::LtcBreathPredictor,
        pub last_dt: f32,
    }

    impl Default for ZenbVinnanaLtc {
        fn default() -> Self {
            Self {
                engine: BeliefEngine::default(),
                state: BeliefState::default(),
                fep: FepState::default(),
                breath_predictor: crate::ltc::LtcBreathPredictor::default_for_breath(),
                last_dt: 0.5,
            }
        }
    }

    impl VinnanaSkandha for ZenbVinnanaLtc {
        fn synthesize(
            &mut self,
            form: &ProcessedForm,
            affect: &AffectiveState,
            _pattern: &PerceivedPattern,
            intent: &FormedIntent,
        ) -> SynthesizedState {
            // Extract input features for LTC: [hr_norm, hrv_norm, motion]
            let ltc_inputs = [form.values[0], form.values[1], form.values[4]];

            // Predict optimal breath rate using LTC
            let predicted_bpm = self.breath_predictor.predict(&ltc_inputs, self.last_dt);

            // Map affect to belief distribution
            let mut belief = [0.2f32; 5];
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

            let mode = belief
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u8)
                .unwrap_or(0);

            // Create control decision if intent sanctioned
            let decision = if intent.is_sanctioned {
                match intent.action {
                    IntentAction::GuideBreath { .. } => Some(ControlOutput {
                        target_bpm: predicted_bpm, // Use LTC prediction!
                        confidence: affect.confidence * intent.alignment,
                        poll_interval_ms: 500,
                    }),
                    _ => None,
                }
            } else {
                None
            };

            SynthesizedState {
                form: form.clone(),
                belief,
                mode,
                confidence: affect.confidence,
                decision,
                free_energy: form.anomaly_score * 2.0,
            }
        }
    }

    impl ZenbVinnanaLtc {
        pub fn from_config(cfg: &crate::config::ZenbConfig) -> Self {
            Self {
                engine: BeliefEngine::from_config(&cfg.belief),
                state: BeliefState::default(),
                fep: FepState::default(),
                breath_predictor: crate::ltc::LtcBreathPredictor::default_for_breath(),
                last_dt: 0.5,
            }
        }

        /// Learn from actual measured breath rate
        pub fn learn_from_breath_measurement(&mut self, actual_rr: f32, inputs: &[f32]) {
            self.breath_predictor
                .learn_from_measurement(actual_rr, inputs);
        }

        /// Get LTC diagnostics
        pub fn ltc_diagnostics(&self, inputs: &[f32]) -> (f32, f32, f32, u64) {
            self.breath_predictor.diagnostics(inputs)
        }

        /// Update time delta (call with actual dt between ticks)
        pub fn set_dt(&mut self, dt: f32) {
            self.last_dt = dt.clamp(0.01, 10.0);
        }
    }

    // ========================================================================
    // PIPELINE TYPE ALIASES
    // ========================================================================

    pub type ZenbPipeline =
        SkandhaPipeline<ZenbRupa, defaults::DefaultVedana, ZenbSanna, ZenbSankhara, ZenbVinnana>;

    /// SOTA Pipeline using HDC memory and LTC prediction
    pub type ZenbPipelineSota = SkandhaPipeline<
        ZenbRupa,
        defaults::DefaultVedana,
        ZenbSannaHdc,
        ZenbSankhara,
        ZenbVinnanaLtc,
    >;

    /// Create ZenB pipeline with AGOLOS components wired in
    pub fn zenb_pipeline(cfg: &crate::config::ZenbConfig) -> ZenbPipeline {
        SkandhaPipeline::new(
            ZenbRupa::default(),
            defaults::DefaultVedana,
            ZenbSanna::default(),
            ZenbSankhara::default(),
            ZenbVinnana::from_config(cfg),
        )
    }

    /// Create SOTA ZenB pipeline with HDC + LTC
    pub fn zenb_pipeline_sota(cfg: &crate::config::ZenbConfig) -> ZenbPipelineSota {
        SkandhaPipeline::new(
            ZenbRupa::default(),
            defaults::DefaultVedana,
            ZenbSannaHdc::default(),
            ZenbSankhara::default(),
            ZenbVinnanaLtc::from_config(cfg),
        )
    }

    /// Unified Pipeline with BeliefSubsystem as Vedana (single source of truth)
    pub type ZenbPipelineUnified = SkandhaPipeline<
        ZenbRupa,
        crate::belief_subsystem::BeliefSubsystem,
        ZenbSannaHdc,
        ZenbSankhara,
        ZenbVinnanaLtc,
    >;

    /// Create unified ZenB pipeline with:
    /// - ZenbRupa for sensor consensus
    /// - BeliefSubsystem for affect extraction (single source of truth)
    /// - ZenbSannaHdc for HDC pattern recognition
    /// - ZenbSankhara for ethical filtering  
    /// - ZenbVinnanaLtc for LTC-enhanced synthesis
    pub fn zenb_pipeline_unified(cfg: &crate::config::ZenbConfig) -> ZenbPipelineUnified {
        SkandhaPipeline::new(
            ZenbRupa::default(),
            crate::belief_subsystem::BeliefSubsystem::from_zenb_config(cfg),
            ZenbSannaHdc::default(),
            ZenbSankhara::default(),
            ZenbVinnanaLtc::from_config(cfg),
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pipeline() {
        let mut pipeline = default_pipeline();

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
        let mut pipeline = default_pipeline();

        let input = SensorInput {
            hr_bpm: Some(150.0),   // Very high HR
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
        let mut pipeline = default_pipeline();

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
            motion: 0.9,  // High motion
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

    #[test]
    fn test_zenb_rupa_context_detection() {
        use super::zenb::ZenbRupa;
        use crate::perception::PhysiologicalContext;

        // Test intense exercise detection (high motion)
        let context = ZenbRupa::detect_context(0.5, 0.5, 0.8);
        assert_eq!(context, PhysiologicalContext::IntenseExercise);

        // Test moderate exercise detection (moderate motion)
        let context = ZenbRupa::detect_context(0.5, 0.5, 0.5);
        assert_eq!(context, PhysiologicalContext::ModerateExercise);

        // Test moderate exercise from HR (high HR, low motion)
        let context = ZenbRupa::detect_context(0.7, 0.5, 0.1); // 140 bpm
        assert_eq!(context, PhysiologicalContext::ModerateExercise);

        // Test stress detection (high HR + low HRV)
        let context = ZenbRupa::detect_context(0.6, 0.2, 0.1); // 120 bpm, 20ms HRV
        assert_eq!(context, PhysiologicalContext::Stress);

        // Test sleep detection (very low HR + still)
        let context = ZenbRupa::detect_context(0.2, 0.6, 0.05); // 40 bpm
        assert_eq!(context, PhysiologicalContext::Sleep);

        // Test light activity (low HR + minimal motion)
        let context = ZenbRupa::detect_context(0.3, 0.5, 0.1); // 60 bpm
        assert_eq!(context, PhysiologicalContext::LightActivity);

        // Test rest (default)
        let context = ZenbRupa::detect_context(0.4, 0.5, 0.25); // 80 bpm, normal
        assert_eq!(context, PhysiologicalContext::Rest);
    }

    #[test]
    fn test_zenb_rupa_adaptive_processing() {
        use super::zenb::ZenbRupa;
        use crate::perception::PhysiologicalContext;

        let mut rupa = ZenbRupa::default();

        // Initial context should be Rest
        assert_eq!(rupa.context(), PhysiologicalContext::Rest);

        // Process high-motion input
        let input = SensorInput {
            hr_bpm: Some(100.0),
            hrv_rmssd: Some(50.0),
            rr_bpm: Some(15.0),
            quality: 0.9,
            motion: 0.8, // High motion → IntenseExercise
            timestamp_us: 0,
        };

        let _ = rupa.process_form_adaptive(&input);

        // Context should be updated to IntenseExercise
        assert_eq!(rupa.context(), PhysiologicalContext::IntenseExercise);

        // Manual override
        rupa.set_context(PhysiologicalContext::Sleep);
        assert_eq!(rupa.context(), PhysiologicalContext::Sleep);
    }
}
