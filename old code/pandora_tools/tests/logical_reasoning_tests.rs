use pandora_tools::skills::logical_reasoning_skill::*;
use serde_json::json;

#[test]
fn test_simple_inference() {
    let _engine = OptimizedJsonAstEngine::new(10);
    let _facts = json!({ "temp": 80, "location": "indoor" });
    let _rule = r#"{ "and": [ { ">": [ { "var": "temp" }, 70 ] }, { "==": [ { "var": "location" }, "indoor" ] } ] }"#;
    // TODO: enable after compile_ast_to_closure is fully implemented
    // let result = engine.infer(&facts, rule).unwrap();
    // assert!(result);
}

#[test]
fn test_inference_failure() {
    let _engine = OptimizedJsonAstEngine::new(10);
    let _facts = json!({ "temp": 60 });
    let _rule = r#"{ ">": [ { "var": "temp" }, 70 ] }"#;
    // TODO: enable after compile_ast_to_closure is fully implemented
    // let result = engine.infer(&facts, rule).unwrap();
    // assert!(!result);
}

#[test]
fn test_caching_mechanism() {
    let _engine = OptimizedJsonAstEngine::new(10);
    let _facts = json!({ "value": 10 });
    let _rule = r#"{ "==": [ { "var": "value" }, 10 ] }"#;
    // TODO: add a counter to validate caching once compile is implemented
}
