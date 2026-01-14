pub mod agents;
pub mod skills;
pub use skills::information_retrieval_skill::ProgressiveSemanticEngine;

// Shims for tests expecting these paths
pub mod skills_alias {
    pub use crate::skills::arithmetic_skill::AdaptiveArithmeticEngine as ArithmeticSkill;
    pub use crate::skills::logical_reasoning_skill::LogicalReasoningSkill;
    pub use crate::skills::pattern_matching_skill::TemporalPrefixSpanEngine as PatternMatchingSkill;
}
// Re-export unit-struct shims if defined
pub use crate::skills::arithmetic_skill::ArithmeticSkill;
pub use crate::skills::pattern_matching_skill::PatternMatchingSkill;
