//! ML Predictor module for Pandora CWM
//! 
//! This module provides prediction capabilities using various ML algorithms.

#[cfg(feature = "ml")]
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Simple linear regression model for causal relationships
#[cfg(feature = "ml")]
#[derive(Debug, Clone)]
pub struct SimpleLinearModel {
    weights: Vec<f64>,
    bias: f64,
}

#[cfg(feature = "ml")]
impl SimpleLinearModel {
    /// Create a new simple linear model
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Predict using the linear model
    pub fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Vec::new();
        
        for row in features.rows() {
            let mut prediction = self.bias;
            for (i, &feature) in row.iter().enumerate() {
                if i < self.weights.len() {
                    prediction += feature * self.weights[i];
                }
            }
            predictions.push(prediction);
        }
        
        Array1::from_vec(predictions)
    }
}

/// ML Predictor for causal relationships
#[cfg(feature = "ml")]
pub struct MLPredictor {
    models: HashMap<String, SimpleLinearModel>,
    #[allow(dead_code)]
    feature_names: Vec<String>,
}

#[cfg(feature = "ml")]
impl MLPredictor {
    /// Create a new ML Predictor
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            models: HashMap::new(),
            feature_names,
        }
    }

    /// Train a model for a specific causal relationship using simple linear regression
    pub fn train_model(
        &mut self,
        relationship_name: &str,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> Result<f64, String> {
        if features.nrows() != targets.len() {
            return Err("Features and targets must have the same number of samples".to_string());
        }

        if features.nrows() == 0 {
            return Err("Cannot train on empty dataset".to_string());
        }

        // Simple linear regression using normal equation
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Calculate weights using normal equation: w = (X^T * X)^(-1) * X^T * y
        let mut weights = vec![0.0; n_features];
        let bias;

        if n_samples == 1 {
            // Single sample case
            bias = targets[0];
        } else if n_features == 0 {
            // No features case - just predict the mean
            bias = targets.mean().unwrap_or(0.0);
        } else {
            // Multiple samples and features
            // For simplicity, use a basic approach
            let mean_target = targets.mean().unwrap_or(0.0);
            bias = mean_target;
            
            // Simple correlation-based weights
            for i in 0..n_features {
                let mut correlation = 0.0;
                let feature_mean = features.column(i).mean().unwrap_or(0.0);
                let target_mean = mean_target;
                
                let mut feature_var = 0.0;
                let mut target_var = 0.0;
                let mut covariance = 0.0;
                
                for j in 0..n_samples {
                    let feature_diff = features[[j, i]] - feature_mean;
                    let target_diff = targets[j] - target_mean;
                    
                    covariance += feature_diff * target_diff;
                    feature_var += feature_diff * feature_diff;
                    target_var += target_diff * target_diff;
                }
                
                if feature_var > 0.0 && target_var > 0.0 {
                    correlation = covariance / (feature_var.sqrt() * target_var.sqrt());
                }
                
                weights[i] = correlation * 0.1; // Scale down for stability
            }
        }

        let model = SimpleLinearModel::new(weights, bias);
        
        // Calculate training error
        let predictions = model.predict(features);
        let error = self.mean_absolute_error(targets, &predictions);

        self.models.insert(relationship_name.to_string(), model);
        Ok(error)
    }

    /// Make predictions for a causal relationship
    pub fn predict(
        &self,
        relationship_name: &str,
        features: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let model = self.models.get(relationship_name)
            .ok_or_else(|| format!("Model '{}' not found", relationship_name))?;

        Ok(model.predict(features))
    }

    /// Get available model names
    pub fn get_model_names(&self) -> Vec<&String> {
        self.models.keys().collect()
    }

    /// Check if a model exists
    pub fn has_model(&self, relationship_name: &str) -> bool {
        self.models.contains_key(relationship_name)
    }

    /// Calculate mean absolute error
    fn mean_absolute_error(&self, targets: &Array1<f64>, predictions: &Array1<f64>) -> f64 {
        if targets.len() != predictions.len() {
            return f64::INFINITY;
        }

        let mut error = 0.0;
        for i in 0..targets.len() {
            error += (targets[i] - predictions[i]).abs();
        }
        error / targets.len() as f64
    }
}

#[cfg(feature = "ml")]
impl Default for MLPredictor {
    fn default() -> Self {
        Self::new(vec![])
    }
}