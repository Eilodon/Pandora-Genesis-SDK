//! Defines the core data structures for the Validation Sprint harness.
//! These structures allow for defining complex test scenarios in a declarative way.

use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::skandha_implementations::core::state_management::MoodState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Represents a single event to be processed in a scenario's input stream.
/// Includes content and an optional delay to simulate real-world timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEvent {
    pub content: String,
    #[serde(default)]
    pub delay_ms: u64,
}

/// Defines a specific behavior or state to be asserted at the end of a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExpectedBehavior {
    /// Asserts that the final karma_weight of the Vedana skandha falls within a given range.
    FinalKarmaWeightRange { min: f32, max: f32 },
    
    /// Asserts that the final mood state corresponds to a specific quadrant
    /// (e.g., "Pleasant-Activated").
    FinalMoodQuadrant { quadrant: String },
    
    /// Asserts that a specific volitional intent was formed by the Sankhara skandha.
    IntentFormed { intent: String },
}

/// Represents a complete test scenario, including its name, description,
/// a stream of input events, and a set of assertions to validate the outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub input_stream: Vec<TestEvent>,
    pub assertions: HashMap<String, ExpectedBehavior>,
}

/// Holds the results of running a single scenario on a processor.
/// This includes pass/fail status, performance metrics, and the final state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub processor_name: String,
    pub passed: bool,
    pub assertion_results: HashMap<String, Result<(), String>>,
    pub total_latency: Duration,
    // Memory usage would be populated by an external tool like Valgrind/DHAT
    // and added to the final report.
    // pub memory_usage_bytes: Option<usize>, 
    pub final_mood: Option<MoodState>,
    pub final_flow: EpistemologicalFlow,
}

impl TestScenario {
    /// Validates the final state of a processor against the scenario's assertions.
    pub fn validate_assertions(
        &self,
        final_flow: &EpistemologicalFlow,
        final_mood: Option<&MoodState>,
    ) -> HashMap<String, Result<(), String>> {
        let mut results = HashMap::new();
        for (assertion_name, behavior) in &self.assertions {
            let result = match behavior {
                ExpectedBehavior::FinalKarmaWeightRange { min, max } => {
                    match &final_flow.vedana {
                        Some(vedana) => {
                            let weight = vedana.get_karma_weight();
                            if weight >= *min && weight <= *max {
                                Ok(())
                            } else {
                                Err(format!(
                                    "Karma weight {} out of expected range [{}, {}]",
                                    weight, min, max
                                ))
                            }
                        }
                        None => Err("Vedana was not set in the final flow".to_string()),
                    }
                }
                ExpectedBehavior::FinalMoodQuadrant { quadrant: expected_quadrant } => {
                    match final_mood {
                        Some(mood) => {
                            let actual_quadrant = mood.quadrant();
                            if actual_quadrant == expected_quadrant {
                                Ok(())
                            } else {
                                Err(format!(
                                    "Mood quadrant '{}' does not match expected '{}'",
                                    actual_quadrant, expected_quadrant
                                ))
                            }
                        }
                        None => Err("No mood state available to check quadrant".to_string()),
                    }
                }
                ExpectedBehavior::IntentFormed { intent: expected_intent } => {
                    match &final_flow.sankhara {
                        Some(intent) if intent.as_ref() == expected_intent => Ok(()),
                        Some(intent) => Err(format!(
                            "Intent '{}' does not match expected '{}'",
                            intent.as_ref(),
                            expected_intent
                        )),
                        None => Err("Sankhara (intent) was not set".to_string()),
                    }
                }
            };
            results.insert(assertion_name.clone(), result);
        }
        results
    }
}
