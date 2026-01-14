//! DECI (Deep End-to-End Causal Inference) Algorithm Implementation
//! 
//! Neural causal discovery using variational autoencoders and graph neural networks.
//! Based on "DECI: Differentiable End-to-end Causal Inference" (Geffner et al., 2022).

use ndarray::{Array2, Array1, ArrayView2, Axis};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyList, PyDict, PyTuple};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};

use crate::causal_discovery::{CausalHypothesis, CausalEdgeType, CausalDiscoveryConfig};

/// Configuration specific to DECI algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeciConfig {
    /// Number of latent dimensions for VAE
    pub latent_dim: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Learning rate for neural network training
    pub learning_rate: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Beta parameter for VAE (KL divergence weight)
    pub beta_kl: f32,
    /// Lambda parameter for sparsity regularization
    pub lambda_sparse: f32,
    /// Lambda parameter for acyclicity constraint
    pub lambda_dag: f32,
    /// Hidden dimensions for encoder/decoder networks
    pub hidden_dims: Vec<usize>,
    /// Whether to use GPU acceleration (if available)
    pub use_gpu: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Minimum edge weight threshold for causal graph
    pub edge_threshold: f32,
}

impl Default for DeciConfig {
    fn default() -> Self {
        Self {
            latent_dim: 64,
            num_epochs: 100,
            learning_rate: 1e-3,
            batch_size: 32,
            beta_kl: 1.0,
            lambda_sparse: 0.01,
            lambda_dag: 0.1,
            hidden_dims: vec![128, 64],
            use_gpu: false,
            random_seed: Some(42),
            edge_threshold: 0.1,
        }
    }
}

/// DECI algorithm implementation using Python PyTorch backend
pub fn discover_with_deci(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    debug!("CausalDiscovery: Using DECI (Deep End-to-End Causal Inference)");
    
    // Create DECI-specific config
    let deci_config = DeciConfig {
        latent_dim: 64,
        num_epochs: std::cmp::min(100, config.max_hypotheses * 10),
        learning_rate: 1e-3,
        batch_size: std::cmp::min(32, data_matrix.len() / 4),
        lambda_sparse: config.min_strength_threshold,
        lambda_dag: config.min_confidence_threshold * 10.0,
        edge_threshold: config.min_strength_threshold,
        ..Default::default()
    };
    
    // Check if we have enough data for deep learning
    if data_matrix.len() < 50 || data_matrix[0].len() < 3 {
        warn!("DECI: Insufficient data for deep learning approach, falling back to simplified version");
        return simplified_deci_fallback(data_matrix, config);
    }
    
    match run_deci_pytorch(py, data_matrix, &deci_config, config) {
        Ok(hypotheses) => Ok(hypotheses),
        Err(e) => {
            warn!("DECI: PyTorch implementation failed: {}, falling back to simplified version", e);
            simplified_deci_fallback(data_matrix.clone(), config)
        }
    }
}

/// Full DECI implementation using PyTorch
fn run_deci_pytorch(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    deci_config: &DeciConfig,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    // Import required Python modules
    let torch = match PyModule::import_bound(py, "torch") {
        Ok(module) => module,
        Err(_) => {
            return Err(PyErr::new::<pyo3::exceptions::PyImportError, _>(
                "PyTorch not available for DECI implementation"
            ));
        }
    };
    
    let deci_module = match PyModule::import_bound(py, "causica.models.deci") {
        Ok(module) => module,
        Err(_) => {
            warn!("DECI: Causica library not found, attempting custom implementation");
            return run_custom_deci_implementation(py, data_matrix, deci_config, config);
        }
    };
    
    info!("DECI: Starting deep causal inference on {}x{} data", 
          data_matrix.len(), data_matrix[0].len());
    
    // Convert data to PyTorch tensor
    let numpy = PyModule::import_bound(py, "numpy")?;
    let py_data = convert_to_numpy_array(py, &numpy, &data_matrix)?;
    let data_tensor = torch.call_method1("from_numpy", (py_data,))?;
    let data_tensor = data_tensor.call_method1("float", ())?;
    
    // Set up DECI model
    let n_vars = data_matrix[0].len();
    let model_kwargs = create_deci_model_config(py, n_vars, deci_config)?;
    
    let deci_model = deci_module.call_method("DECI", (), Some(&model_kwargs))?;
    
    // Set up optimizer
    let optim_module = PyModule::import_bound(py, "torch.optim")?;
    let optimizer = optim_module.call_method(
        "Adam", 
        (deci_model.getattr("parameters")?.call0()?,), 
        Some(&PyDict::from_iter([(py.get_type_bound::<pyo3::types::PyString>().call1(("lr",))?, deci_config.learning_rate)].iter()))
    )?;
    
    // Training loop
    info!("DECI: Training model for {} epochs", deci_config.num_epochs);
    
    for epoch in 0..deci_config.num_epochs {
        // Forward pass
        let loss_dict = deci_model.call_method1("compute_loss", (data_tensor.clone(),))?;
        let total_loss = loss_dict.get_item("loss")?;
        
        // Backward pass
        optimizer.call_method0("zero_grad")?;
        total_loss.call_method0("backward")?;
        optimizer.call_method0("step")?;
        
        if epoch % 20 == 0 {
            let loss_value: f32 = total_loss.call_method0("item")?.extract()?;
            debug!("DECI: Epoch {} - Loss: {:.4}", epoch, loss_value);
        }
    }
    
    // Extract learned causal graph
    info!("DECI: Extracting learned causal structure");
    let adjacency_matrix = deci_model.call_method0("get_adjacency_matrix")?;
    let adj_numpy = adjacency_matrix.call_method0("detach")?.call_method0("numpy")?;
    let adj_matrix: Vec<Vec<f32>> = adj_numpy.extract()?;
    
    // Convert to causal hypotheses
    convert_deci_adjacency_to_hypotheses(adj_matrix, deci_config, config)
}

/// Custom DECI implementation when causica library is not available
fn run_custom_deci_implementation(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    deci_config: &DeciConfig,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    info!("DECI: Using custom PyTorch implementation");
    
    let torch = PyModule::import_bound(py, "torch")?;
    let nn = PyModule::import_bound(py, "torch.nn")?;
    let functional = PyModule::import_bound(py, "torch.nn.functional")?;
    
    // Create custom DECI model in Python
    let deci_model_code = r#"
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDECI(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder (VAE encoder)
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Causal graph (learnable adjacency matrix)
        self.adjacency = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Apply causal structure
        adj_masked = torch.sigmoid(self.adjacency) * (1 - torch.eye(self.input_dim))
        z_causal = torch.matmul(z, adj_masked.T)
        
        # Decode
        reconstruction = self.decoder(z_causal)
        
        return reconstruction, mu, logvar, adj_masked
    
    def compute_loss(self, x, beta_kl=1.0, lambda_sparse=0.01, lambda_dag=0.1):
        reconstruction, mu, logvar, adj = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Sparsity regularization
        sparse_loss = torch.norm(adj, p=1)
        
        # DAG constraint (acyclicity)
        dag_loss = torch.trace(torch.matrix_exp(adj)) - self.input_dim
        
        total_loss = recon_loss + beta_kl * kl_loss + lambda_sparse * sparse_loss + lambda_dag * dag_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'sparse_loss': sparse_loss,
            'dag_loss': dag_loss
        }
    
    def get_adjacency_matrix(self):
        with torch.no_grad():
            return torch.sigmoid(self.adjacency) * (1 - torch.eye(self.input_dim))

def create_deci_model(input_dim, latent_dim, hidden_dims):
    return CustomDECI(input_dim, latent_dim, hidden_dims)
"#;
    
    // Execute the model definition
    py.run_bound(deci_model_code, None, None)?;
    
    // Get the model creation function
    let globals = py.eval_bound("globals()", None, None)?;
    let create_model_fn = globals.get_item("create_deci_model")?;
    
    // Create model instance
    let n_vars = data_matrix[0].len();
    let model = create_model_fn.call1((
        n_vars, 
        deci_config.latent_dim, 
        deci_config.hidden_dims.clone()
    ))?;
    
    // Convert data to tensor
    let numpy = PyModule::import_bound(py, "numpy")?;
    let py_data = convert_to_numpy_array(py, &numpy, &data_matrix)?;
    let data_tensor = torch.call_method1("from_numpy", (py_data,))?;
    let data_tensor = data_tensor.call_method1("float", ())?;
    
    // Set up optimizer
    let optim_module = PyModule::import_bound(py, "torch.optim")?;
    let params = model.call_method0("parameters")?;
    let optimizer = optim_module.call_method("Adam", (params,), Some(&PyDict::from_iter([
        ("lr", deci_config.learning_rate)
    ].iter())?))?;
    
    // Training loop
    for epoch in 0..deci_config.num_epochs {
        let loss_dict = model.call_method("compute_loss", (
            data_tensor.clone(),
            deci_config.beta_kl,
            deci_config.lambda_sparse,
            deci_config.lambda_dag,
        ), None)?;
        
        let total_loss = loss_dict.get_item("loss")?;
        
        optimizer.call_method0("zero_grad")?;
        total_loss.call_method0("backward")?;
        optimizer.call_method0("step")?;
        
        if epoch % 20 == 0 {
            let loss_value: f32 = total_loss.call_method0("item")?.extract()?;
            debug!("DECI Custom: Epoch {} - Loss: {:.4}", epoch, loss_value);
        }
    }
    
    // Extract adjacency matrix
    let adjacency_matrix = model.call_method0("get_adjacency_matrix")?;
    let adj_numpy = adjacency_matrix.call_method0("numpy")?;
    let adj_matrix: Vec<Vec<f32>> = adj_numpy.extract()?;
    
    convert_deci_adjacency_to_hypotheses(adj_matrix, deci_config, config)
}

/// Simplified DECI fallback using statistical methods
fn simplified_deci_fallback(
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    info!("DECI: Using simplified statistical fallback");
    
    let n_vars = data_matrix[0].len();
    let n_samples = data_matrix.len();
    
    // Simple correlation-based causal inference
    let mut hypotheses = Vec::new();
    
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i != j {
                // Calculate correlation
                let correlation = calculate_correlation(&data_matrix, i, j);
                
                if correlation.abs() > config.min_strength_threshold {
                    // Simple causality heuristic: temporal order + correlation
                    let causality_score = estimate_causality_direction(&data_matrix, i, j);
                    let confidence = (correlation.abs() * causality_score).min(1.0);
                    
                    if confidence >= config.min_confidence_threshold {
                        hypotheses.push(CausalHypothesis {
                            from_node_index: i,
                            to_node_index: j,
                            strength: correlation * causality_score,
                            confidence,
                            edge_type: determine_edge_type_from_correlation(correlation),
                        });
                    }
                }
            }
        }
    }
    
    // Sort and limit
    hypotheses.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    hypotheses.truncate(config.max_hypotheses);
    
    info!("DECI Fallback: Found {} causal hypotheses", hypotheses.len());
    Ok(hypotheses)
}

/// Helper functions
fn create_deci_model_config(
    py: Python,
    n_vars: usize,
    config: &DeciConfig,
) -> PyResult<pyo3::Bound<PyDict>> {
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("input_dim", n_vars)?;
    kwargs.set_item("latent_dim", config.latent_dim)?;
    kwargs.set_item("hidden_dims", config.hidden_dims.clone())?;
    kwargs.set_item("num_epochs", config.num_epochs)?;
    kwargs.set_item("learning_rate", config.learning_rate)?;
    kwargs.set_item("batch_size", config.batch_size)?;
    kwargs.set_item("beta_kl", config.beta_kl)?;
    kwargs.set_item("lambda_sparse", config.lambda_sparse)?;
    kwargs.set_item("lambda_dag", config.lambda_dag)?;
    
    if let Some(seed) = config.random_seed {
        kwargs.set_item("random_seed", seed)?;
    }
    
    Ok(kwargs)
}

fn convert_to_numpy_array(
    py: Python,
    numpy: &Bound<PyModule>,
    data_matrix: &[Vec<f32>]
) -> PyResult<Py<pyo3::PyAny>> {
    let py_data = PyList::new_bound(py, data_matrix.iter().map(|row| PyList::new_bound(py, row)));
    let np_array = numpy.call_method1("array", (py_data,))?;
    Ok(np_array.into())
}

fn convert_deci_adjacency_to_hypotheses(
    adjacency_matrix: Vec<Vec<f32>>,
    deci_config: &DeciConfig,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    let mut hypotheses = Vec::new();
    
    for (i, row) in adjacency_matrix.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate() {
            if weight > deci_config.edge_threshold && i != j {
                let confidence = calculate_deci_confidence(weight, &adjacency_matrix);
                
                if confidence >= config.min_confidence_threshold {
                    hypotheses.push(CausalHypothesis {
                        from_node_index: i,
                        to_node_index: j,
                        strength: weight,
                        confidence,
                        edge_type: determine_edge_type_from_weight(weight),
                    });
                }
            }
        }
    }
    
    hypotheses.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    hypotheses.truncate(config.max_hypotheses);
    
    info!("DECI: Converted adjacency matrix to {} hypotheses", hypotheses.len());
    Ok(hypotheses)
}

fn calculate_correlation(data_matrix: &[Vec<f32>], i: usize, j: usize) -> f32 {
    if data_matrix.is_empty() || i >= data_matrix[0].len() || j >= data_matrix[0].len() {
        return 0.0;
    }
    
    let n = data_matrix.len() as f32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for row in data_matrix {
        let x = row[i];
        let y = row[j];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

fn estimate_causality_direction(data_matrix: &[Vec<f32>], i: usize, j: usize) -> f32 {
    // Simple heuristic: assume variable with higher variance is more likely to be causal
    let var_i = calculate_variance(data_matrix, i);
    let var_j = calculate_variance(data_matrix, j);
    
    if var_i + var_j == 0.0 {
        0.5
    } else {
        var_i / (var_i + var_j)
    }
}

fn calculate_variance(data_matrix: &[Vec<f32>], var_index: usize) -> f32 {
    if data_matrix.is_empty() || var_index >= data_matrix[0].len() {
        return 0.0;
    }
    
    let values: Vec<f32> = data_matrix.iter().map(|row| row[var_index]).collect();
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance
}

fn calculate_deci_confidence(weight: f32, adjacency_matrix: &[Vec<f32>]) -> f32 {
    // Confidence based on relative strength in the learned graph
    let all_weights: Vec<f32> = adjacency_matrix.iter()
        .flat_map(|row| row.iter().cloned())
        .collect();
    
    let max_weight = all_weights.iter().fold(0.0, |a, &b| a.max(b));
    let mean_weight = all_weights.iter().sum::<f32>() / all_weights.len() as f32;
    
    if max_weight > 0.0 {
        ((weight - mean_weight) / (max_weight - mean_weight)).max(0.0).min(1.0)
    } else {
        0.0
    }
}

fn determine_edge_type_from_weight(weight: f32) -> CausalEdgeType {
    if weight > 0.7 {
        CausalEdgeType::Direct
    } else if weight > 0.4 {
        CausalEdgeType::Indirect
    } else if weight > 0.2 {
        CausalEdgeType::Conditional
    } else {
        CausalEdgeType::Inhibitory
    }
}

fn determine_edge_type_from_correlation(correlation: f32) -> CausalEdgeType {
    if correlation > 0.0 {
        CausalEdgeType::Direct
    } else {
        CausalEdgeType::Inhibitory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deci_config_default() {
        let config = DeciConfig::default();
        assert_eq!(config.latent_dim, 64);
        assert_eq!(config.num_epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_simplified_deci_fallback() {
        let data = vec![
            vec![1.0, 2.0, 1.5],
            vec![2.0, 4.0, 3.0],
            vec![3.0, 6.0, 4.5],
            vec![4.0, 8.0, 6.0],
        ];
        
        let config = CausalDiscoveryConfig::default();
        let result = simplified_deci_fallback(data, &config);
        
        assert!(result.is_ok());
        let hypotheses = result.unwrap();
        assert!(!hypotheses.is_empty());
    }

    #[test]
    fn test_calculate_correlation() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        
        let correlation = calculate_correlation(&data, 0, 1);
        assert!((correlation - 1.0).abs() < 0.001); // Should be perfectly correlated
    }

    #[test]
    fn test_estimate_causality_direction() {
        let data = vec![
            vec![1.0, 2.0],
            vec![10.0, 4.0], // Higher variance in first variable
            vec![3.0, 6.0],
            vec![15.0, 8.0],
        ];
        
        let direction = estimate_causality_direction(&data, 0, 1);
        assert!(direction > 0.5); // First variable should have higher causality score
    }
}