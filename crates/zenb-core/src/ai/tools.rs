//! AI Tool Registry - Safe function calling
//!
//! Reference: AIToolRegistry.ts (~250 lines)

use serde_json::Value;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Tool execution context
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub current_tempo: f32,
    pub last_tempo_change: u64,
    pub last_pattern_change: u64,
    pub session_duration: f32,
    pub user_confirmed: bool,
}

/// Tool execution result
#[derive(Debug)]
pub struct ToolResult {
    pub success: bool,
    pub message: String,
    pub needs_confirmation: bool,
}

/// AI Tool trait
pub trait AiTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    
    /// Check if tool can execute with given args
    fn can_execute(&self, args: &Value, ctx: &ToolContext) -> Result<(), String>;
    
    /// Execute the tool
    fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<String, String>;
}

/// Adjust Tempo Tool
pub struct AdjustTempoTool;

impl AiTool for AdjustTempoTool {
    fn name(&self) -> &str {
        "adjust_tempo"
    }
    
    fn description(&self) -> &str {
        "Adjust breathing tempo scale (0.8-1.4)"
    }
    
    fn can_execute(&self, args: &Value, ctx: &ToolContext) -> Result<(), String> {
        let scale = args["scale"]
            .as_f64()
            .ok_or("Missing 'scale' parameter")?
            as f32;
        
        // 1. Bounds check
        if !(0.8..=1.4).contains(&scale) {
            return Err(format!("Scale {} out of bounds [0.8, 1.4]", scale));
        }
        
        // 2. Rate limit: Max 1 per 5s
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        if now - ctx.last_tempo_change < 5000 {
            return Err("Rate limit: 5s cooldown required".into());
        }
        
        // 3. Max delta: 0.2
        if (scale - ctx.current_tempo).abs() > 0.2 {
            if !ctx.user_confirmed {
                return Err("Large change (>0.2) requires confirmation".into());
            }
        }
        
        Ok(())
    }
    
    fn execute(&self, args: &Value, _ctx: &ToolContext) -> Result<String, String> {
        let scale = args["scale"].as_f64().unwrap() as f32;
        let reason = args["reason"]
            .as_str()
            .unwrap_or("AI adjustment");
        
        Ok(format!("ADJUST_TEMPO:{:.2}:{}", scale, reason))
    }
}

/// Switch Pattern Tool
pub struct SwitchPatternTool;

impl AiTool for SwitchPatternTool {
    fn name(&self) -> &str {
        "switch_pattern"
    }
    
    fn description(&self) -> &str {
        "Switch breathing pattern (4-7-8, box, etc.)"
    }
    
    fn can_execute(&self, args: &Value, ctx: &ToolContext) -> Result<(), String> {
        let pattern_id = args["pattern_id"]
            .as_str()
            .ok_or("Missing 'pattern_id'")?;
        
        // Rate limit: Max 1 per 60s
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        if now - ctx.last_pattern_change < 60000 {
            return Err("Pattern changes limited to 1 per minute".into());
        }
        
        // Validate pattern exists
        let valid_patterns = ["4-7-8", "box", "coherent", "energize"];
        if !valid_patterns.contains(&pattern_id) {
            return Err(format!("Unknown pattern: {}", pattern_id));
        }
        
        Ok(())
    }
    
    fn execute(&self, args: &Value, _ctx: &ToolContext) -> Result<String, String> {
        let pattern_id = args["pattern_id"].as_str().unwrap();
        Ok(format!("SWITCH_PATTERN:{}", pattern_id))
    }
}

/// AI Tool Registry
pub struct AiToolRegistry {
    tools: HashMap<String, Box<dyn AiTool>>,
}

impl AiToolRegistry {
    pub fn new() -> Self {
        let mut tools: HashMap<String, Box<dyn AiTool>> = HashMap::new();
        
        tools.insert("adjust_tempo".into(), Box::new(AdjustTempoTool));
        tools.insert("switch_pattern".into(), Box::new(SwitchPatternTool));
        
        Self { tools }
    }
    
    pub fn execute(
        &self,
        tool_name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> ToolResult {
        let tool = match self.tools.get(tool_name) {
            Some(t) => t,
            None => {
                return ToolResult {
                    success: false,
                    message: format!("Unknown tool: {}", tool_name),
                    needs_confirmation: false,
                };
            }
        };
        
        // Check pre-conditions
        match tool.can_execute(args, ctx) {
            Ok(_) => {}
            Err(msg) => {
                let needs_conf = msg.contains("confirmation");
                return ToolResult {
                    success: false,
                    message: msg,
                    needs_confirmation: needs_conf,
                };
            }
        }
        
        // Execute
        match tool.execute(args, ctx) {
            Ok(result) => ToolResult {
                success: true,
                message: result,
                needs_confirmation: false,
            },
            Err(err) => ToolResult {
                success: false,
                message: err,
                needs_confirmation: false,
            },
        }
    }
    
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for AiToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_adjust_tempo_valid() {
        let registry = AiToolRegistry::new();
        let ctx = ToolContext {
            current_tempo: 1.0,
            last_tempo_change: 0,
            last_pattern_change: 0,
            session_duration: 60.0,
            user_confirmed: false,
        };
        
        let args = json!({"scale": 1.1, "reason": "test"});
        let result = registry.execute("adjust_tempo", &args, &ctx);
        
        assert!(result.success);
        assert!(result.message.contains("ADJUST_TEMPO"));
    }
    
    #[test]
    fn test_adjust_tempo_out_of_bounds() {
        let registry = AiToolRegistry::new();
        let ctx = ToolContext {
            current_tempo: 1.0,
            last_tempo_change: 0,
            last_pattern_change: 0,
            session_duration: 60.0,
            user_confirmed: false,
        };
        
        let args = json!({"scale": 2.0, "reason": "test"});
        let result = registry.execute("adjust_tempo", &args, &ctx);
        
        assert!(!result.success);
        assert!(result.message.contains("out of bounds"));
    }
    
    #[test]
    fn test_switch_pattern() {
        let registry = AiToolRegistry::new();
        let ctx = ToolContext {
            current_tempo: 1.0,
            last_tempo_change: 0,
            last_pattern_change: 0,
            session_duration: 60.0,
            user_confirmed: false,
        };
        
        let args = json!({"pattern_id": "4-7-8"});
        let result = registry.execute("switch_pattern", &args, &ctx);
        
        assert!(result.success);
        assert!(result.message.contains("SWITCH_PATTERN"));
    }
}
