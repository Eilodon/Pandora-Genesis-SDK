//! Quick test to verify YAML scenario parsing
#![allow(dead_code)]

#[cfg(test)]
mod yaml_parsing_test {
    use std::fs;
    use std::path::PathBuf;

    // Import the harness structures from the sibling module
    // Note: This assumes validation_harness.rs exports public structs
    
    #[test]
    fn test_parse_trauma_conditioning_scenario() {
        // Construct path to the YAML file
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let scenario_path = PathBuf::from(manifest_dir)
            .join("scenarios")
            .join("s01_trauma_conditioning.yaml");

        // Read the YAML file
        let yaml_content = fs::read_to_string(&scenario_path)
            .expect("Failed to read s01_trauma_conditioning.yaml");

        // Parse it using serde_yaml
        let scenario: serde_yaml::Value = serde_yaml::from_str(&yaml_content)
            .expect("Failed to parse YAML");

        // Basic structure validation
        assert!(scenario.get("name").is_some(), "Missing 'name' field");
        assert!(scenario.get("description").is_some(), "Missing 'description' field");
        assert!(scenario.get("input_stream").is_some(), "Missing 'input_stream' field");
        assert!(scenario.get("assertions").is_some(), "Missing 'assertions' field");

        // Validate input_stream has events
        let input_stream = scenario.get("input_stream").unwrap();
        assert!(input_stream.as_sequence().is_some(), "input_stream should be an array");
        let events = input_stream.as_sequence().unwrap();
        assert_eq!(events.len(), 4, "Should have 4 events");

        // Validate assertions
        let assertions = scenario.get("assertions").unwrap();
        assert!(assertions.as_mapping().is_some(), "assertions should be a map");
        let assertions_map = assertions.as_mapping().unwrap();
        assert_eq!(assertions_map.len(), 3, "Should have 3 assertions");

        println!("✅ YAML scenario structure is valid!");
        println!("Scenario name: {}", scenario.get("name").unwrap().as_str().unwrap());
    }
}
