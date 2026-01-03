#[cfg(test)]
mod tests {
    use crate::config::*;
    use std::env;
    use std::fs;
    use std::path::Path;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_valid() {
        let config = ZenbConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_fep() {
        let mut config = ZenbConfig::default();

        // Invalid process_noise
        config.fep.process_noise = 0.0;
        assert!(config.validate().is_err());

        config.fep.process_noise = 1.5;
        assert!(config.validate().is_err());

        // Invalid lr bounds
        config.fep.process_noise = 0.02;
        config.fep.lr_min = 0.9;
        config.fep.lr_max = 0.8;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_safety() {
        let mut config = ZenbConfig::default();

        // trauma_hard_th must be > trauma_soft_th
        config.safety.trauma_hard_th = 0.5;
        config.safety.trauma_soft_th = 0.7;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_belief() {
        let mut config = ZenbConfig::default();

        // pathway_weights must have 3 elements
        config.belief.pathway_weights = vec![1.0, 2.0];
        assert!(config.validate().is_err());

        // enter_threshold must be > exit_threshold
        config.belief.pathway_weights = vec![1.0, 1.0, 1.0];
        config.belief.enter_threshold = 0.4;
        config.belief.exit_threshold = 0.6;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_to_toml_string() {
        let config = ZenbConfig::default();
        let toml_str = config.to_toml_string().unwrap();

        assert!(toml_str.contains("[fep]"));
        assert!(toml_str.contains("[belief]"));
        assert!(toml_str.contains("process_noise"));
        assert!(toml_str.contains("pathway_weights"));
    }

    #[test]
    fn test_config_from_toml_string() {
        let toml_str = r#"
            [fep]
            process_noise = 0.03
            base_obs_var = 0.3
            lr_base = 0.7
            lr_min = 0.1
            lr_max = 0.9

            [resonance]
            window_size_sec = 15.0
            coherence_threshold = 0.4

            [safety]
            trauma_hard_th = 2.0
            trauma_soft_th = 0.8
            trauma_decay_default = 0.6

            [breath]
            default_target_bpm = 5.0

            [belief]
            pathway_weights = [1.5, 0.5, 1.0]
            prior_logits = [0.1, 0.0, 0.0, 0.0, 0.0]
            smooth_tau_sec = 3.0
            enter_threshold = 0.7
            exit_threshold = 0.3

            [performance]
            batch_size = 30
            flush_interval_ms = 100
            max_buffer_size = 150
            enable_burst_filter = true
            burst_filter_threshold_us = 5000
        "#;

        let config: ZenbConfig = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_ok());

        assert_eq!(config.fep.process_noise, 0.03);
        assert_eq!(config.belief.pathway_weights, vec![1.5, 0.5, 1.0]);
        assert_eq!(config.performance.batch_size, 30);
    }

    #[test]
    fn test_config_save_and_load() {
        let config = ZenbConfig::default();

        // Save to temp file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        config.save_to_file(path).unwrap();

        // Load back
        let loaded = ZenbConfig::from_file(path).unwrap();

        assert_eq!(config.fep.process_noise, loaded.fep.process_noise);
        assert_eq!(config.belief.pathway_weights, loaded.belief.pathway_weights);
    }

    #[test]
    fn test_config_env_overrides() {
        // Set environment variables
        env::set_var("ZENB_FEP_PROCESS_NOISE", "0.05");
        env::set_var("ZENB_BELIEF_SMOOTH_TAU_SEC", "7.0");
        env::set_var("ZENB_PERFORMANCE_BATCH_SIZE", "50");

        let mut config = ZenbConfig::default();
        config.apply_env_overrides().unwrap();

        assert_eq!(config.fep.process_noise, 0.05);
        assert_eq!(config.belief.smooth_tau_sec, 7.0);
        assert_eq!(config.performance.batch_size, 50);

        // Cleanup
        env::remove_var("ZENB_FEP_PROCESS_NOISE");
        env::remove_var("ZENB_BELIEF_SMOOTH_TAU_SEC");
        env::remove_var("ZENB_PERFORMANCE_BATCH_SIZE");
    }

    #[test]
    fn test_config_layered_loading() {
        // Create temp files
        let default_file = NamedTempFile::new().unwrap();
        let user_file = NamedTempFile::new().unwrap();

        // Write default config
        let default_config = ZenbConfig::default();
        default_config.save_to_file(default_file.path()).unwrap();

        // Write user config with overrides
        let mut user_config = ZenbConfig::default();
        user_config.fep.process_noise = 0.04;
        user_config.belief.smooth_tau_sec = 6.0;
        user_config.save_to_file(user_file.path()).unwrap();

        // Load layered
        let loaded =
            ZenbConfig::load_layered(Some(default_file.path()), Some(user_file.path())).unwrap();

        // User config should override
        assert_eq!(loaded.fep.process_noise, 0.04);
        assert_eq!(loaded.belief.smooth_tau_sec, 6.0);
    }

    #[test]
    fn test_belief_config_integration() {
        use crate::belief::BeliefEngine;

        let mut config = BeliefConfig::default();
        config.pathway_weights = vec![2.0, 1.0, 1.5];
        config.smooth_tau_sec = 5.0;
        config.enter_threshold = 0.7;

        let engine = BeliefEngine::from_config(&config);

        assert_eq!(engine.w, vec![2.0, 1.0, 1.5]);
        assert_eq!(engine.smooth_tau_sec, 5.0);
        assert_eq!(engine.enter_th, 0.7);
    }

    #[test]
    fn test_engine_with_custom_config() {
        use crate::engine::Engine;

        let mut config = ZenbConfig::default();
        config.breath.default_target_bpm = 5.0;
        config.belief.smooth_tau_sec = 3.0;

        let engine = Engine::new_with_config(0.0, Some(config.clone()));

        assert_eq!(engine.config.breath.default_target_bpm, 5.0);
        assert_eq!(engine.belief_engine.smooth_tau_sec, 3.0);
    }

    #[test]
    fn test_performance_config_values() {
        let config = PerformanceConfig::default();

        assert_eq!(config.batch_size, 20);
        assert_eq!(config.flush_interval_ms, 80);
        assert_eq!(config.max_buffer_size, 100);
        assert!(config.enable_burst_filter);
        assert_eq!(config.burst_filter_threshold_us, 10_000);
    }

    #[test]
    fn test_invalid_env_var_handling() {
        env::set_var("ZENB_FEP_PROCESS_NOISE", "invalid");

        let mut config = ZenbConfig::default();
        let result = config.apply_env_overrides();

        assert!(result.is_err());

        env::remove_var("ZENB_FEP_PROCESS_NOISE");
    }

    #[test]
    fn test_config_file_not_found() {
        let result = ZenbConfig::from_file("nonexistent.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_toml_syntax() {
        let temp_file = NamedTempFile::new().unwrap();
        fs::write(temp_file.path(), "invalid toml: syntax").unwrap();

        let result = ZenbConfig::from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_config_with_defaults() {
        let toml_str = r#"
            [fep]
            process_noise = 0.03
            
            [belief]
            smooth_tau_sec = 5.0
        "#;

        // This should fail because not all required fields are present
        let result: Result<ZenbConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_roundtrip() {
        let original = ZenbConfig::default();

        // Serialize to TOML
        let toml_str = original.to_toml_string().unwrap();

        // Deserialize back
        let roundtrip: ZenbConfig = toml::from_str(&toml_str).unwrap();

        // Validate
        assert!(roundtrip.validate().is_ok());

        // Check values match
        assert_eq!(original.fep.process_noise, roundtrip.fep.process_noise);
        assert_eq!(
            original.belief.pathway_weights,
            roundtrip.belief.pathway_weights
        );
    }
}
