use pandora_orchestrator::{Orchestrator, OrchestratorTrait, SkillRegistry};
use pandora_tools::skills::{
    // Temporarily disabled due to dependency conflicts
    // information_retrieval_skill::InformationRetrievalSkill,
    logical_reasoning_skill::LogicalReasoningSkill,
};
use pandora_tools::skills::arithmetic_skill::AdaptiveArithmeticEngine;
use pandora_tools::PatternMatchingSkill;
use std::io::{self, Write};
use std::sync::Arc;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};

fn init_logging() {
    let mut filter = EnvFilter::from_default_env();
    if let Ok(d) = "pandora_core=info".parse() {
        filter = filter.add_directive(d);
    }
    if let Ok(d) = "pandora_simulation=info".parse() {
        filter = filter.add_directive(d);
    }
    if let Ok(d) = "pandora_orchestrator=info".parse() {
        filter = filter.add_directive(d);
    }

    fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    info!("🔱 Pandora Genesis SDK - CLI Demo");
    info!("=====================================\n");

    // Khởi tạo Skill Registry
    let mut registry = SkillRegistry::new();

    // Đăng ký các skills
    registry.register_arc(Arc::new(AdaptiveArithmeticEngine::new()));
    registry.register_arc(Arc::new(LogicalReasoningSkill));
    registry.register_arc(Arc::new(PatternMatchingSkill));
    // registry.register_arc(Arc::new(AnalogyReasoningSkill));
    // Temporarily disabled due to dependency conflicts
    // registry.register(Arc::new(InformationRetrievalSkill));

    let orchestrator = Orchestrator::new(Arc::new(registry));

    info!("Available skills:");
    info!("- arithmetic: Perform arithmetic calculations");
    info!("- logical_reasoning: Evaluate logical expressions");
    info!("- pattern_matching: Match patterns in strings");
    // info!("- analogy_reasoning: Solve analogy problems");
    // Temporarily disabled due to dependency conflicts
    // info!("- information_retrieval: Search in text documents");
    info!("\nType 'help' for examples, 'quit' to exit.\n");

    loop {
        print!("pandora> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" {
            info!("Goodbye! 👋");
            break;
        }

        if input == "help" {
            show_help();
            continue;
        }

        if input.is_empty() {
            continue;
        }

        // Parse command: skill_name input_json
        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        if parts.len() != 2 {
            warn!("❌ Usage: <skill_name> <json_input>");
            info!("   Example: arithmetic '{{\"expression\": \"2 + 2\"}}'");
            continue;
        }

        let skill_name = parts[0];
        let json_input = parts[1];

        // Parse JSON input
        let input_value: serde_json::Value = match serde_json::from_str(json_input) {
            Ok(value) => value,
            Err(e) => {
                error!(error = %e, "❌ Invalid JSON");
                continue;
            }
        };

        // Execute skill
        match orchestrator.process_request(skill_name, input_value) {
            Ok(result) => {
                info!("✅ Result: {}", serde_json::to_string_pretty(&result)?);
            }
            Err(e) => {
                error!(error = %e, "❌ Error");
            }
        }
        info!("");
    }

    Ok(())
}

fn show_help() {
    info!("\n📖 Examples:");
    info!("arithmetic '{{\"expression\": \"2 + 3 * 4\"}}'");
    info!("logical_reasoning '{{\"ast\": {{\"type\": \"AND\", \"children\": [{{\"type\": \"CONST\", \"value\": true}}, {{\"type\": \"CONST\", \"value\": false}}]}}, \"context\": {{}}}}'");
    info!("pattern_matching '{{\"pattern\": \"a*b\", \"candidates\": [\"ab\", \"aab\", \"b\", \"acb\"]}}'");
    info!("analogy_reasoning '{{\"a\": \"man\", \"b\": \"king\", \"c\": \"woman\", \"candidates\": [\"queen\", \"prince\", \"duke\"]}}'");
    // Temporarily disabled due to dependency conflicts
    // info!("information_retrieval '{{\"query\": \"test\", \"documents\": [\"test document\", \"another doc\", \"test again\"]}}'");
    info!("");
}
