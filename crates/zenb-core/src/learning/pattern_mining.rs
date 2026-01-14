//! Temporal Pattern Mining using PrefixSpan algorithm.
//!
//! Mines frequent sequential patterns from event sequences and predicts
//! next likely actions based on learned patterns.
//!
//! # Use Case
//! User behavior prediction: given a sequence of user actions,
//! predict the most likely next action.

use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

/// Errors that can occur during pattern mining.
#[derive(Debug, Error)]
pub enum PatternError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Minimum support must be > 0")]
    InvalidSupportThreshold,
}

/// A single event in a behavioral sequence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Event {
    /// Event type identifier (e.g., "open_app", "click_button")
    pub event_type: String,
    /// Timestamp in microseconds
    pub ts_us: i64,
}

impl Event {
    /// Create a new event with type and timestamp.
    pub fn new(event_type: impl Into<String>, ts_us: i64) -> Self {
        Self {
            event_type: event_type.into(),
            ts_us,
        }
    }
}

/// A sequence of events (e.g., one user session).
#[derive(Debug, Clone)]
pub struct Sequence {
    pub events: Vec<Event>,
}

impl Sequence {
    /// Create a new sequence from a vector of events.
    pub fn new(events: Vec<Event>) -> Self {
        Self { events }
    }

    /// Create sequence from just event types (timestamps auto-generated).
    pub fn from_types(types: &[&str]) -> Self {
        Self {
            events: types
                .iter()
                .enumerate()
                .map(|(i, t)| Event::new(*t, i as i64 * 1000))
                .collect(),
        }
    }
}

/// A mined temporal pattern.
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Sequence of event types forming this pattern
    pub items: Vec<String>,
    /// Number of sequences containing this pattern
    pub support: usize,
}

/// Predicted next action with confidence score.
#[derive(Debug, Clone)]
pub struct ActionPrediction {
    /// The predicted action/event type
    pub predicted_action: String,
    /// Confidence score (higher = more likely)
    pub confidence: f32,
}

/// Internal: projected sequence for PrefixSpan recursion.
struct ProjectedSequence<'a> {
    suffix: &'a [Event],
}

/// Sequential pattern mining engine using PrefixSpan algorithm.
///
/// # Algorithm
/// PrefixSpan works by:
/// 1. Finding all frequent 1-item patterns
/// 2. For each frequent item, project the database and find frequent extensions
/// 3. Recursively extend patterns until max length or no more frequent extensions
pub struct TemporalPrefixSpanEngine {
    min_support: usize,
    max_pattern_length: usize,
    mined_patterns: Vec<TemporalPattern>,
}

impl TemporalPrefixSpanEngine {
    /// Create a new engine with configuration.
    ///
    /// # Arguments
    /// * `min_support` - Minimum number of sequences a pattern must appear in
    /// * `max_pattern_length` - Maximum length of patterns to mine
    pub fn new(min_support: usize, max_pattern_length: usize) -> Result<Self, PatternError> {
        if min_support == 0 {
            return Err(PatternError::InvalidSupportThreshold);
        }
        Ok(Self {
            min_support,
            max_pattern_length,
            mined_patterns: Vec::new(),
        })
    }

    /// Mine patterns from a collection of sequences.
    ///
    /// Updates internal `mined_patterns` which can be used for prediction.
    pub fn mine_patterns(&mut self, sequences: &[Sequence]) -> Result<(), PatternError> {
        self.mined_patterns.clear();
        let mut frequent_patterns = Vec::new();

        // Step 1: Find frequent 1-item patterns (each sequence counts at most once)
        let mut item_counts: HashMap<String, usize> = HashMap::new();
        for seq in sequences {
            let mut seen_in_seq: HashMap<&str, bool> = HashMap::new();
            for event in &seq.events {
                if seen_in_seq.insert(&event.event_type, true).is_none() {
                    *item_counts.entry(event.event_type.clone()).or_insert(0) += 1;
                }
            }
        }

        let frequent_items: BTreeMap<String, usize> = item_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_support)
            .collect();

        // Step 2: For each frequent item, build projected database and recurse
        for (item, support) in frequent_items {
            let prefix = vec![item.clone()];
            frequent_patterns.push(TemporalPattern {
                items: prefix.clone(),
                support,
            });

            // Build projected database
            let mut projected_db = Vec::new();
            for seq in sequences {
                if let Some(pos) = seq.events.iter().position(|e| e.event_type == item) {
                    if pos + 1 < seq.events.len() {
                        projected_db.push(ProjectedSequence {
                            suffix: &seq.events[pos + 1..],
                        });
                    }
                }
            }

            if !projected_db.is_empty() {
                self.mine_recursive(projected_db, prefix, &mut frequent_patterns);
            }
        }

        self.mined_patterns = frequent_patterns;
        Ok(())
    }

    /// Recursive PrefixSpan mining.
    fn mine_recursive<'a>(
        &self,
        projected_db: Vec<ProjectedSequence<'a>>,
        prefix: Vec<String>,
        frequent_patterns: &mut Vec<TemporalPattern>,
    ) {
        if prefix.len() >= self.max_pattern_length {
            return;
        }

        // Count frequent items in projected database
        let mut item_counts: HashMap<String, usize> = HashMap::new();
        for p_seq in &projected_db {
            let mut seen_in_seq: HashMap<&str, bool> = HashMap::new();
            for event in p_seq.suffix {
                if seen_in_seq.insert(&event.event_type, true).is_none() {
                    *item_counts.entry(event.event_type.clone()).or_insert(0) += 1;
                }
            }
        }

        let frequent_items: BTreeMap<String, usize> = item_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_support)
            .collect();

        // Extend prefix with each frequent item
        for (item, support) in frequent_items {
            let mut new_prefix = prefix.clone();
            new_prefix.push(item.clone());

            frequent_patterns.push(TemporalPattern {
                items: new_prefix.clone(),
                support,
            });

            // Build next projected database
            let mut new_projected_db = Vec::new();
            for p_seq in &projected_db {
                if let Some(pos) = p_seq.suffix.iter().position(|e| e.event_type == item) {
                    if pos + 1 < p_seq.suffix.len() {
                        new_projected_db.push(ProjectedSequence {
                            suffix: &p_seq.suffix[pos + 1..],
                        });
                    }
                }
            }

            if !new_projected_db.is_empty() {
                self.mine_recursive(new_projected_db, new_prefix, frequent_patterns);
            }
        }
    }

    /// Predict next action(s) based on current sequence prefix.
    ///
    /// Returns predictions sorted by confidence (descending).
    pub fn predict_next_action(
        &self,
        current_sequence: &[Event],
    ) -> Result<Vec<ActionPrediction>, PatternError> {
        if current_sequence.is_empty() {
            return Ok(Vec::new());
        }

        let current_items: Vec<String> = current_sequence
            .iter()
            .map(|e| e.event_type.clone())
            .collect();

        let mut predictions: HashMap<String, usize> = HashMap::new();

        // Find patterns that start with current sequence
        for pattern in &self.mined_patterns {
            if pattern.items.len() > current_items.len()
                && pattern.items.starts_with(&current_items)
            {
                let next_action = &pattern.items[current_items.len()];
                *predictions.entry(next_action.clone()).or_insert(0) += pattern.support;
            }
        }

        let mut sorted_predictions: Vec<ActionPrediction> = predictions
            .into_iter()
            .map(|(action, total_support)| ActionPrediction {
                predicted_action: action,
                confidence: total_support as f32,
            })
            .collect();

        sorted_predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_predictions)
    }

    /// Get all mined patterns.
    pub fn patterns(&self) -> &Vec<TemporalPattern> {
        &self.mined_patterns
    }

    /// Get patterns of a specific length.
    pub fn patterns_of_length(&self, len: usize) -> Vec<&TemporalPattern> {
        self.mined_patterns
            .iter()
            .filter(|p| p.items.len() == len)
            .collect()
    }

    /// Clear all mined patterns.
    pub fn clear(&mut self) {
        self.mined_patterns.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mine_simple_sequences() {
        let sequences = vec![
            Sequence::from_types(&["A", "B", "C"]),
            Sequence::from_types(&["A", "B", "D"]),
            Sequence::from_types(&["A", "B", "C"]),
        ];

        let mut engine = TemporalPrefixSpanEngine::new(2, 3).unwrap();
        engine.mine_patterns(&sequences).unwrap();

        // "A" and "B" should be frequent, "A -> B" should be a pattern
        let patterns = engine.patterns();
        assert!(!patterns.is_empty());

        // Find "A -> B" pattern
        let ab_pattern = patterns.iter().find(|p| p.items == vec!["A", "B"]);
        assert!(ab_pattern.is_some(), "Should find A->B pattern");
        assert_eq!(ab_pattern.unwrap().support, 3);
    }

    #[test]
    fn predict_next_action_works() {
        let sequences = vec![
            Sequence::from_types(&["login", "browse", "purchase"]),
            Sequence::from_types(&["login", "browse", "purchase"]),
            Sequence::from_types(&["login", "browse", "logout"]),
            Sequence::from_types(&["login", "search", "purchase"]),
        ];

        let mut engine = TemporalPrefixSpanEngine::new(2, 4).unwrap();
        engine.mine_patterns(&sequences).unwrap();

        // Given "login, browse", predict next action
        let current = vec![Event::new("login", 1000), Event::new("browse", 2000)];
        let predictions = engine.predict_next_action(&current).unwrap();

        assert!(!predictions.is_empty());
        // "purchase" should be top prediction (2 occurrences vs 1 for logout)
        assert_eq!(predictions[0].predicted_action, "purchase");
    }

    #[test]
    fn min_support_filters_infrequent_patterns() {
        let sequences = vec![
            Sequence::from_types(&["A", "B"]),
            Sequence::from_types(&["A", "C"]),
            Sequence::from_types(&["D", "E"]),
        ];

        let mut engine = TemporalPrefixSpanEngine::new(2, 2).unwrap();
        engine.mine_patterns(&sequences).unwrap();

        // Only "A" should be frequent (appears in 2 sequences)
        let single_patterns = engine.patterns_of_length(1);
        assert_eq!(single_patterns.len(), 1);
        assert_eq!(single_patterns[0].items[0], "A");
    }

    #[test]
    fn invalid_support_returns_error() {
        let result = TemporalPrefixSpanEngine::new(0, 5);
        assert!(matches!(result, Err(PatternError::InvalidSupportThreshold)));
    }

    #[test]
    fn max_pattern_length_limits_recursion() {
        let sequences = vec![
            Sequence::from_types(&["A", "B", "C", "D", "E"]),
            Sequence::from_types(&["A", "B", "C", "D", "E"]),
        ];

        let mut engine = TemporalPrefixSpanEngine::new(2, 3).unwrap();
        engine.mine_patterns(&sequences).unwrap();

        // No patterns should be longer than 3
        for pattern in engine.patterns() {
            assert!(pattern.items.len() <= 3, "Pattern too long: {:?}", pattern.items);
        }
    }
}
