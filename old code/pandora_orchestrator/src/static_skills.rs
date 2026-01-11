use pandora_core::interfaces::skills::{SkillDescriptor, SkillModule, SkillOutput};
use pandora_error::PandoraError;
use pandora_tools::skills::{
    analogy_reasoning_skill::AnalogyReasoningSkill, arithmetic_skill::ArithmeticSkill,
    logical_reasoning_skill::LogicalReasoningSkill, pattern_matching_skill::PatternMatchingSkill,
};
use serde_json::Value as SkillInput;

/// Static dispatch enum for built-in skills.
///
/// This enum allows compile-time dispatch to specific skill implementations,
/// eliminating vtable overhead and enabling inlining.
pub enum StaticSkill {
    Arithmetic(ArithmeticSkill),
    LogicalReasoning(LogicalReasoningSkill),
    PatternMatching(PatternMatchingSkill),
    AnalogyReasoning(AnalogyReasoningSkill),
}

impl StaticSkill {
    /// Create from skill name (for registration).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "arithmetic" => Some(Self::Arithmetic(ArithmeticSkill)),
            "logical_reasoning" => Some(Self::LogicalReasoning(LogicalReasoningSkill)),
            "pattern_matching" => Some(Self::PatternMatching(PatternMatchingSkill)),
            "analogy_reasoning" => Some(Self::AnalogyReasoning(AnalogyReasoningSkill)),
            _ => None,
        }
    }

    /// Get skill descriptor.
    ///
    /// Inlined for zero overhead.
    #[inline]
    pub fn descriptor(&self) -> SkillDescriptor {
        match self {
            Self::Arithmetic(s) => s.descriptor(),
            Self::LogicalReasoning(s) => s.descriptor(),
            Self::PatternMatching(s) => s.descriptor(),
            Self::AnalogyReasoning(s) => s.descriptor(),
        }
    }

    /// Execute skill with static dispatch.
    #[inline]
    pub async fn execute(&self, input: SkillInput) -> SkillOutput {
        match self {
            Self::Arithmetic(s) => s.execute(input).await,
            Self::LogicalReasoning(s) => s.execute(input).await,
            Self::PatternMatching(s) => s.execute(input).await,
            Self::AnalogyReasoning(s) => s.execute(input).await,
        }
    }
}

/// Hybrid registry that uses static dispatch for built-in skills
/// and dynamic dispatch for plugins.
pub struct HybridSkillRegistry {
    /// Built-in skills with static dispatch
    static_skills: fnv::FnvHashMap<String, StaticSkill>,

    /// Plugin skills with dynamic dispatch
    dynamic_skills: fnv::FnvHashMap<String, std::sync::Arc<dyn SkillModule>>,
}

impl HybridSkillRegistry {
    pub fn new() -> Self {
        Self {
            static_skills: fnv::FnvHashMap::default(),
            dynamic_skills: fnv::FnvHashMap::default(),
        }
    }

    /// Register a built-in skill (static dispatch).
    pub fn register_static(&mut self, name: &str) -> Result<(), PandoraError> {
        let skill = StaticSkill::from_name(name)
            .ok_or_else(|| PandoraError::config(format!("Unknown static skill: {}", name)))?;
        self.static_skills.insert(name.to_string(), skill);
        Ok(())
    }

    /// Register a plugin skill (dynamic dispatch).
    pub fn register_dynamic(&mut self, skill: std::sync::Arc<dyn SkillModule>) {
        let name = skill.descriptor().name;
        self.dynamic_skills.insert(name, skill);
    }

    /// Get skill (checks static first, then dynamic).
    pub fn get(&self, name: &str) -> Option<SkillRef<'_>> {
        if let Some(skill) = self.static_skills.get(name) {
            return Some(SkillRef::Static(skill));
        }
        if let Some(skill) = self.dynamic_skills.get(name) {
            return Some(SkillRef::Dynamic(skill));
        }
        None
    }

    /// List all skill names.
    pub fn list_names(&self) -> Vec<String> {
        self.static_skills
            .keys()
            .chain(self.dynamic_skills.keys())
            .cloned()
            .collect()
    }
}

impl Default for HybridSkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference to either a static or dynamic skill.
pub enum SkillRef<'a> {
    Static(&'a StaticSkill),
    Dynamic(&'a std::sync::Arc<dyn SkillModule>),
}

impl<'a> SkillRef<'a> {
    pub fn descriptor(&self) -> SkillDescriptor {
        match self {
            Self::Static(s) => s.descriptor(),
            Self::Dynamic(s) => s.descriptor(),
        }
    }

    pub async fn execute(&self, input: SkillInput) -> SkillOutput {
        match self {
            Self::Static(s) => s.execute(input).await,
            Self::Dynamic(s) => s.execute(input).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_skill_creation() {
        let skill = StaticSkill::from_name("arithmetic").unwrap();
        let desc = skill.descriptor();
        assert_eq!(desc.name, "arithmetic");
    }

    #[test]
    fn test_hybrid_registry() {
        let mut registry = HybridSkillRegistry::new();
        registry.register_static("arithmetic").unwrap();

        assert!(registry.get("arithmetic").is_some());
        assert!(registry.get("nonexistent").is_none());

        let names = registry.list_names();
        assert!(names.contains(&"arithmetic".to_string()));
    }

    #[tokio::test]
    async fn test_static_dispatch_execution() {
        let skill = StaticSkill::from_name("arithmetic").unwrap();
        let input = serde_json::json!({"expression": "2 + 2"});
        let result = skill.execute(input).await.unwrap();
        assert_eq!(result["result"], 4.0);
    }
}
