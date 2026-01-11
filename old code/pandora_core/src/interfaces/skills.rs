use async_trait::async_trait;
use pandora_error::PandoraError;
use serde::{Deserialize, Serialize};
use serde_json::Value as SkillInput; // Đổi tên để rõ ràng

pub type SkillOutput = Result<serde_json::Value, PandoraError>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SkillDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: String,
    pub output_schema: String,
}

#[async_trait]
pub trait SkillModule: Send + Sync {
    fn descriptor(&self) -> SkillDescriptor;
    async fn execute(&self, input: SkillInput) -> SkillOutput;
}
