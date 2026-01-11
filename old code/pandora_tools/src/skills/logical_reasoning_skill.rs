#[cfg(test)]
#[allow(clippy::items_after_test_module, clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_simple_and_true() {
        let skill = LogicalReasoningSkill;
        let context = json!({"is_raining": true, "time": 18});
        let ast = json!({
            "type": "AND",
            "children": [
                {"type": "VAR", "name": "is_raining"},
                {"type": "CONST", "value": true}
            ]
        });
        let input = json!({"ast": ast, "context": context});
        assert_eq!(skill.execute(input).await.unwrap(), json!({"result": true}));
    }

    #[tokio::test]
    async fn test_simple_and_false() {
        let skill = LogicalReasoningSkill;
        let context = json!({"is_raining": false, "time": 18});
        let ast = json!({
            "type": "AND",
            "children": [
                {"type": "VAR", "name": "is_raining"},
                {"type": "CONST", "value": true}
            ]
        });
        let input = json!({"ast": ast, "context": context});
        assert_eq!(
            skill.execute(input).await.unwrap(),
            json!({"result": false})
        );
    }

    #[tokio::test]
    async fn test_simple_or_true() {
        let skill = LogicalReasoningSkill;
        let context = json!({"is_raining": false, "time": 18});
        let ast = json!({
            "type": "OR",
            "children": [
                {"type": "VAR", "name": "is_raining"},
                {"type": "CONST", "value": true}
            ]
        });
        let input = json!({"ast": ast, "context": context});
        assert_eq!(skill.execute(input).await.unwrap(), json!({"result": true}));
    }
}
use async_trait::async_trait;
use lru::LruCache;
use pandora_core::interfaces::skills::{SkillDescriptor, SkillModule, SkillOutput};
use pandora_error::PandoraError;
use serde_json::Value as SkillInput;
use serde_json::{json, Value};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use thiserror::Error;

pub struct LogicalReasoningSkill;

#[derive(Debug, Error)]
pub enum LogicalError {
    #[error("Lỗi parse JSON: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("Quy tắc không hợp lệ: {0}")]
    InvalidRule(String),
    #[error("Input không hợp lệ")]
    InvalidInput,
}

pub type CompiledRule = Box<dyn Fn(&Value) -> Result<bool, LogicalError> + Send + Sync>;

pub struct OptimizedJsonAstEngine {
    rule_cache: Arc<Mutex<LruCache<u64, CompiledRule>>>,
}

impl OptimizedJsonAstEngine {
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            rule_cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_capacity).unwrap(),
            ))),
        }
    }

    pub fn infer(&self, facts: &Value, rules_json: &str) -> Result<bool, LogicalError> {
        let rule_hash = seahash::hash(rules_json.as_bytes());
        let mut cache = self.rule_cache.lock().unwrap();

        if let Some(compiled_rule) = cache.get(&rule_hash) {
            return compiled_rule(facts);
        }

        let ast = serde_json::from_str(rules_json)?;
        let compiled_rule = self.compile_ast_to_closure(ast)?;
        let result = compiled_rule(facts);
        cache.put(rule_hash, compiled_rule);
        result
    }

    fn compile_ast_to_closure(&self, ast: Value) -> Result<CompiledRule, LogicalError> {
        let rule_logic = move |facts: &Value| -> Result<bool, LogicalError> {
            let ctx = facts.as_object().ok_or(LogicalError::InvalidInput)?;
            LogicalReasoningSkill::evaluate_node(&ast, ctx).map_err(LogicalError::InvalidRule)
        };
        Ok(Box::new(rule_logic))
    }
}

#[async_trait]
impl SkillModule for LogicalReasoningSkill {
    fn descriptor(&self) -> SkillDescriptor {
        SkillDescriptor {
			name: "logical_reasoning".to_string(),
			description: "Đánh giá các biểu thức logic dạng cây AST (AND, OR, NOT, VAR, CONST).".to_string(),
			input_schema: r#"{"type":"object","properties":{"ast":{"type":"object"},"context":{"type":"object"}},"required":["ast","context"]}"#.to_string(),
			output_schema: r#"{"type":"object","properties":{"result":{"type":"boolean"}}}"#.to_string(),
		}
    }

    async fn execute(&self, input: SkillInput) -> SkillOutput {
        let rules_json = input
            .get("ast")
            .and_then(|v| serde_json::to_string(v).ok())
            .ok_or_else(|| PandoraError::InvalidSkillInput {
                skill_name: "logical_reasoning".into(),
                message: "Missing 'ast' field".into(),
            })?;
        let facts = input
            .get("context")
            .ok_or_else(|| PandoraError::InvalidSkillInput {
                skill_name: "logical_reasoning".into(),
                message: "Missing or invalid 'context' field".into(),
            })?;
        let engine = OptimizedJsonAstEngine::new(128);
        engine
            .infer(facts, &rules_json)
            .map(|result| json!({"result": result}))
            .map_err(|e| PandoraError::skill_exec("logical_reasoning", e.to_string()))
    }
}

#[allow(dead_code)]
impl LogicalReasoningSkill {
    fn evaluate_node(
        node: &Value,
        context: &serde_json::Map<String, Value>,
    ) -> Result<bool, String> {
        let node_type = node
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Thiếu trường 'type'".to_string())?;
        match node_type {
            "CONST" => node
                .get("value")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| "Thiếu hoặc sai kiểu 'value' cho CONST".to_string()),
            "VAR" => {
                let var = node
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "Thiếu trường 'name' cho VAR".to_string())?;
                context
                    .get(var)
                    .and_then(|v| v.as_bool())
                    .ok_or_else(|| format!("Không tìm thấy biến '{}' trong context", var))
            }
            "NOT" => {
                let child = node
                    .get("child")
                    .ok_or_else(|| "Thiếu trường 'child' cho NOT".to_string())?;
                Ok(!Self::evaluate_node(child, context)?)
            }
            "AND" => {
                let children = node
                    .get("children")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| "Thiếu hoặc sai kiểu 'children' cho AND".to_string())?;
                for child in children {
                    if !Self::evaluate_node(child, context)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            "OR" => {
                let children = node
                    .get("children")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| "Thiếu hoặc sai kiểu 'children' cho OR".to_string())?;
                for child in children {
                    if Self::evaluate_node(child, context)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Err(format!("Không hỗ trợ node type: {}", node_type)),
        }
    }
}
