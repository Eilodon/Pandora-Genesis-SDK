//! ML Trainer module for Pandora CWM
//! 
//! This module provides training capabilities for ML models.

#[cfg(feature = "ml")]
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;

/// Training configuration for ML models
#[cfg(feature = "ml")]
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub max_iterations: usize,
    pub learning_rate: f64,
    pub regularization: f64,
    pub validation_split: f64,
}

#[cfg(feature = "ml")]
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            learning_rate: 0.01,
            regularization: 0.001,
            validation_split: 0.2,
        }
    }
}

/// Result of data splitting operation
#[cfg(feature = "ml")]
#[derive(Debug, Clone)]
pub struct SplitDatasets {
    pub train_features: Array2<f64>,
    pub train_targets: Array1<f64>,
    pub val_features: Array2<f64>,
    pub val_targets: Array1<f64>,
}

/// ML Trainer for causal relationships
#[cfg(feature = "ml")]
pub struct MLTrainer {
    config: TrainingConfig,
    training_history: HashMap<String, Vec<f64>>,
}

#[cfg(feature = "ml")]
impl MLTrainer {
    /// Create a new ML Trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            training_history: HashMap::new(),
        }
    }

    /// Train a model with the given data
    pub fn train_model(
        &mut self,
        model_name: &str,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> Result<f64, String> {
        // Split data for validation
        let split = self.split_data(features, targets)?;

        // Simple training: calculate mean squared error on validation set
        let validation_error = MLTrainer::calculate_mse_static(&split.val_features, &split.val_targets);

        // Store training history
        self.training_history.insert(
            model_name.to_string(),
            vec![validation_error]
        );

        Ok(validation_error)
    }

    /// Split data into training and validation sets
    fn split_data(
        &self,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> Result<SplitDatasets, String> {
        let n_samples = features.nrows();
        let split_point = (n_samples as f64 * (1.0 - self.config.validation_split)) as usize;

        if split_point >= n_samples || split_point == 0 {
            return Err("Invalid validation split".to_string());
        }

        let train_features = features.slice(s![..split_point, ..]).to_owned();
        let train_targets = targets.slice(s![..split_point]).to_owned();
        let val_features = features.slice(s![split_point.., ..]).to_owned();
        let val_targets = targets.slice(s![split_point..]).to_owned();

        Ok(SplitDatasets {
            train_features,
            train_targets,
            val_features,
            val_targets,
        })
    }

    /// Get training history for a model
    pub fn get_training_history(&self, model_name: &str) -> Option<&Vec<f64>> {
        self.training_history.get(model_name)
    }

    /// Get all training histories
    pub fn get_all_training_histories(&self) -> &HashMap<String, Vec<f64>> {
        &self.training_history
    }

    /// Calculate mean squared error (static method)
    fn calculate_mse_static(features: &Array2<f64>, targets: &Array1<f64>) -> f64 {
        if features.nrows() != targets.len() || features.nrows() == 0 {
            return f64::INFINITY;
        }

        // Simple baseline: predict the mean
        let mean_target = targets.mean().unwrap_or(0.0);
        let mut mse = 0.0;
        
        for &target in targets.iter() {
            let error = target - mean_target;
            mse += error * error;
        }
        
        mse / targets.len() as f64
    }
}

#[cfg(feature = "ml")]
impl Default for MLTrainer {
    fn default() -> Self {
        Self::new(TrainingConfig::default())
    }
}