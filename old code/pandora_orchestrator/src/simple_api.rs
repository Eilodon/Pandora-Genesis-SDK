use pandora_core::error::{PResult, PandoraError};
use std::collections::HashMap;
use tracing::info;

type SkillFn = dyn Fn(&str) -> String + Send + Sync;

pub struct SkillRegistry {
    skills: HashMap<String, Box<SkillFn>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, skill: Box<SkillFn>) {
        self.skills.insert(name.to_string(), skill);
    }

    pub fn get_skill(&self, name: &str) -> Option<&SkillFn> {
        self.skills.get(name).map(|f| &**f as _)
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub trait OrchestratorTrait {
    fn process_request(&self, input: &str) -> PResult<String>;
}

pub struct Orchestrator {
    registry: SkillRegistry,
}

impl Orchestrator {
    pub fn new(registry: SkillRegistry) -> Self {
        Self { registry }
    }
}

impl OrchestratorTrait for Orchestrator {
    fn process_request(&self, input: &str) -> PResult<String> {
        info!(target: "pandora_orchestrator", request = %input, "Processing request");
        let Some(skill) = self.registry.get_skill("default") else {
            return Err(PandoraError::Orchestrator("default skill not found".into()));
        };
        Ok(skill(input))
    }
}
