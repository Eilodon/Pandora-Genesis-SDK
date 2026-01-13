//! Propagating Effect Monad for Composable Causal Computations
//!
//! Implements a monad pattern for chaining causal operations with automatic
//! error propagation, logging, and intervention support.
//!
//! Inspired by DeepCausality's PropagatingEffect but tailored for AGOLOS.

use super::intervenable::Intervenable;
use super::Variable;

/// Propagating effect - a monadic container for causal computations
///
/// # Design Philosophy
/// Causal reasoning involves chains of transformations where:
/// - Errors should short-circuit the chain
/// - Each step should be auditable (logs)
/// - Interventions can override values mid-chain
///
/// This monad makes these patterns explicit and composable.
///
/// # Examples
/// ```
/// use zenb_core::causal::propagating_effect::PropagatingEffect;
///
/// let result = PropagatingEffect::pure(10.0)
///     .bind(|x| PropagatingEffect::pure(x * 2.0))
///     .map(|x| x + 5.0)
///     .extract();
///
/// assert_eq!(result, Ok(25.0));
/// ```
#[derive(Debug, Clone)]
pub struct PropagatingEffect<T> {
    /// The current value (None if error occurred)
    value: Option<T>,

    /// Audit trail of operations
    logs: Vec<String>,

    /// Error state (if any)
    error: Option<String>,
}

impl<T> PropagatingEffect<T> {
    /// Create a pure effect with no side effects
    ///
    /// This is the monadic `return` or `pure` operation.
    pub fn pure(value: T) -> Self {
        Self {
            value: Some(value),
            logs: Vec::new(),
            error: None,
        }
    }

    /// Create an error effect
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            value: None,
            logs: Vec::new(),
            error: Some(msg.into()),
        }
    }

    /// Monadic bind - chain computations
    ///
    /// If current effect is error, short-circuit and return error.
    /// Otherwise, apply function to value.
    ///
    /// # Examples
    /// ```ignore
    /// let result = PropagatingEffect::pure(5)
    ///     .bind(|x| PropagatingEffect::pure(x * 2))
    ///     .bind(|x| PropagatingEffect::pure(x + 3));
    /// ```
    pub fn bind<U, F>(self, f: F) -> PropagatingEffect<U>
    where
        F: FnOnce(T) -> PropagatingEffect<U>,
    {
        match (self.value, self.error) {
            (Some(val), None) => {
                let mut next = f(val);
                // Preserve logs from previous steps
                next.logs.splice(0..0, self.logs);
                next
            }
            (_, Some(err)) => {
                // Error state - short circuit
                PropagatingEffect {
                    value: None,
                    logs: self.logs,
                    error: Some(err),
                }
            }
            (None, None) => {
                // Should not happen, but handle gracefully
                PropagatingEffect {
                    value: None,
                    logs: self.logs,
                    error: Some("Unexpected None value without error".to_string()),
                }
            }
        }
    }

    /// Functor map - transform value without changing monad structure
    ///
    /// This is lighter than `bind` when you don't need to return a new effect.
    pub fn map<U, F>(self, f: F) -> PropagatingEffect<U>
    where
        F: FnOnce(T) -> U,
    {
        match (self.value, self.error) {
            (Some(val), None) => PropagatingEffect {
                value: Some(f(val)),
                logs: self.logs,
                error: None,
            },
            (_, Some(err)) => PropagatingEffect {
                value: None,
                logs: self.logs,
                error: Some(err),
            },
            (None, None) => PropagatingEffect {
                value: None,
                logs: self.logs,
                error: Some("Unexpected None value without error".to_string()),
            },
        }
    }

    /// Add log entry to audit trail
    pub fn log(mut self, msg: impl Into<String>) -> Self {
        self.logs.push(msg.into());
        self
    }

    /// Extract final value or error
    pub fn extract(self) -> Result<T, String> {
        match (self.value, self.error) {
            (Some(val), None) => Ok(val),
            (_, Some(err)) => Err(err),
            (None, None) => Err("No value and no error".to_string()),
        }
    }

    /// Get reference to value (if present)
    pub fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Get logs
    pub fn logs(&self) -> &[String] {
        &self.logs
    }

    /// Get error (if present)
    pub fn get_error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Check if effect is in error state
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Check if effect has value
    pub fn has_value(&self) -> bool {
        self.value.is_some() && self.error.is_none()
    }
}

/// Implement Intervenable for PropagatingEffect
///
/// This allows interventions mid-chain:
/// ```ignore
/// let result = PropagatingEffect::pure(state)
///     .bind(|s| process(s))
///     .intervene(Variable::HeartRate, 60.0)  // ← Force HR to 60
///     .bind(|s| continue_processing(s));
/// ```
impl<T> Intervenable for PropagatingEffect<T>
where
    T: Intervenable,
{
    fn intervene(self, variable: Variable, value: f32) -> Self {
        self.map(|inner| inner.intervene(variable, value))
            .log(format!("Intervention: {:?} = {}", variable, value))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BeliefState;

    #[test]
    fn test_pure_and_extract() {
        let effect = PropagatingEffect::pure(42);
        assert_eq!(effect.extract(), Ok(42));
    }

    #[test]
    fn test_error_propagation() {
        let effect: PropagatingEffect<i32> = PropagatingEffect::error("test error");
        assert!(effect.is_error());
        assert_eq!(effect.extract(), Err("test error".to_string()));
    }

    #[test]
    fn test_map() {
        let result = PropagatingEffect::pure(10).map(|x| x * 2).map(|x| x + 5);

        assert_eq!(result.extract(), Ok(25));
    }

    #[test]
    fn test_bind() {
        let result = PropagatingEffect::pure(5)
            .bind(|x| PropagatingEffect::pure(x * 2))
            .bind(|x| PropagatingEffect::pure(x + 3));

        assert_eq!(result.extract(), Ok(13));
    }

    #[test]
    fn test_bind_with_error() {
        let result = PropagatingEffect::pure(5)
            .bind(|x| PropagatingEffect::pure(x * 2))
            .bind(|_| PropagatingEffect::<i32>::error("computation failed"))
            .bind(|x| PropagatingEffect::pure(x + 100)); // Should not execute

        assert!(result.is_error());
        assert_eq!(result.extract(), Err("computation failed".to_string()));
    }

    #[test]
    fn test_logging() {
        let result = PropagatingEffect::pure(10)
            .log("Step 1: initialized")
            .map(|x| x * 2)
            .log("Step 2: doubled")
            .map(|x| x + 5)
            .log("Step 3: added 5");

        assert_eq!(result.logs().len(), 3);
        assert_eq!(result.logs()[0], "Step 1: initialized");
        assert_eq!(result.extract(), Ok(25));
    }

    #[test]
    fn test_logs_preserved_across_bind() {
        let result = PropagatingEffect::pure(5)
            .log("Initial value")
            .bind(|x| PropagatingEffect::pure(x * 2).log("Doubled"));

        assert_eq!(result.logs().len(), 2);
        assert_eq!(result.logs()[0], "Initial value");
        assert_eq!(result.logs()[1], "Doubled");
    }

    #[test]
    fn test_intervention() {
        let state = BeliefState::default();

        let result = PropagatingEffect::pure(state)
            .intervene(Variable::HeartRate, 0.9)
            .extract();

        assert!(result.is_ok());
        let final_state = result.unwrap();
        // Should have modified belief state (increased stress)
        assert!(final_state.p[1] > 0.0); // Stress mode
    }

    #[test]
    fn test_intervention_logs() {
        let state = BeliefState::default();

        let effect = PropagatingEffect::pure(state).intervene(Variable::HeartRate, 0.9);

        assert!(effect.logs().iter().any(|log| log.contains("Intervention")));
    }

    #[test]
    fn test_conditional_intervention() {
        let state = BeliefState::default();
        let emergency = true;

        let result = PropagatingEffect::pure(state)
            .map(|s| {
                if emergency {
                    s.intervene(Variable::HeartRate, 0.2) // Force calm
                } else {
                    s
                }
            })
            .extract();

        assert!(result.is_ok());
    }

    #[test]
    fn test_chaining_multiple_operations() {
        let result = PropagatingEffect::pure(1.0)
            .log("Start")
            .bind(|x| PropagatingEffect::pure(x + 1.0).log("Add 1"))
            .map(|x| x * 2.0)
            .log("Double")
            .bind(|x| PropagatingEffect::pure(x - 0.5).log("Subtract 0.5"))
            .extract();

        assert_eq!(result, Ok(3.5)); // (1 + 1) * 2 - 0.5 = 3.5
    }
}
