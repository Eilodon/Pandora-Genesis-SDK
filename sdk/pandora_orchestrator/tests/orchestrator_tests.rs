use pandora_orchestrator::simple_api::{Orchestrator, OrchestratorTrait, SkillRegistry};

#[test]
fn registry_register_and_get() {
    let mut reg = SkillRegistry::new();
    reg.register("default", Box::new(|s| format!("echo:{s}")));
    let skill = reg.get_skill("default");
    assert!(skill.is_some());
    assert_eq!(skill.unwrap()("hi"), "echo:hi");
}

#[test]
fn orchestrator_processes_request() {
    let mut reg = SkillRegistry::new();
    reg.register("default", Box::new(|s| format!("ok:{s}")));
    let orch = Orchestrator::new(reg);
    let out = orch
        .process_request("default", serde_json::Value::String("ping".into()))
        .unwrap();
    assert_eq!(out, serde_json::json!("ok:ping"));
}

#[test]
fn orchestrator_missing_skill_errors() {
    let reg = SkillRegistry::new();
    let orch = Orchestrator::new(reg);
    let err = orch
        .process_request("unknown", serde_json::json!({}))
        .unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("unknown route"));
}
