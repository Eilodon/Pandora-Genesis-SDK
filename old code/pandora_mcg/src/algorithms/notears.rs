//! NOTEARS (NO TEARS) Algorithm Implementation
//! 
//! Neural structure learning for causal discovery without combinatorial search.
//! Based on "DAGs with NO TEARS" (Zheng et al., 2018).

use ndarray::{Array2, Array1, ArrayView2, Axis};
use ndarray_linalg::{Trace, Eig, UPLO};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyList, PyDict};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::causal_discovery::{CausalHypothesis, CausalEdgeType, CausalDiscoveryConfig};

/// Configuration specific to NOTEARS algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotearsConfig {
    /// Regularization parameter for L1 sparsity
    pub lambda1: f32,
    /// Regularization parameter for acyclicity constraint  
    pub lambda2: f32,
    /// Maximum number of optimization iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Whether to use neural network (MLP) version
    pub use_mlp: bool,
    /// Hidden dimensions for MLP version [input_dim, hidden1, hidden2, ..., output_dim]
    pub mlp_hidden_dims: Vec<usize>,
}

impl Default for NotearsConfig {
    fn default() -> Self {
        Self {
            lambda1: 0.01,
            lambda2: 0.01, 
            max_iter: 100,
            tolerance: 1e-6,
            learning_rate: 0.1,
            use_mlp: false,
            mlp_hidden_dims: vec![64, 32],
        }
    }
}

/// NOTEARS algorithm implementation using Python backend
pub fn discover_with_notears(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    debug!("CausalDiscovery: Using NOTEARS algorithm");
    
    // Create NOTEARS-specific config from general config
    let notears_config = NotearsConfig {
        lambda1: config.min_strength_threshold,
        lambda2: config.min_confidence_threshold,
        max_iter: 100,
        tolerance: 1e-6,
        learning_rate: 0.1,
        use_mlp: false,
        mlp_hidden_dims: vec![64, 32],
    };
    
    if notears_config.use_mlp {
        discover_with_notears_mlp(py, data_matrix, &notears_config, config)
    } else {
        discover_with_notears_linear(py, data_matrix, &notears_config, config)
    }
}

/// Linear NOTEARS implementation
fn discover_with_notears_linear(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    notears_config: &NotearsConfig,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    // Import required Python modules
    let numpy = PyModule::import_bound(py, "numpy")?;
    let notears_module = match PyModule::import_bound(py, "notears") {
        Ok(module) => module,
        Err(_) => {
            warn!("NOTEARS: Python module not found, attempting to install...");
            return install_and_run_notears(py, data_matrix, notears_config, config);
        }
    };
    
    // Convert Rust data to numpy array
    let py_data = convert_to_numpy_array(py, &numpy, &data_matrix)?;
    
    // Set up NOTEARS parameters
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("lambda1", notears_config.lambda1)?;
    kwargs.set_item("lambda2", notears_config.lambda2)?; 
    kwargs.set_item("max_iter", notears_config.max_iter)?;
    kwargs.set_item("h_tol", notears_config.tolerance)?;
    kwargs.set_item("rho_max", 1e16)?;
    
    // Run NOTEARS algorithm
    info!("NOTEARS: Running linear NOTEARS on {}x{} data matrix", 
          data_matrix.len(), data_matrix[0].len());
    
    let result = notears_module.call_method("notears_linear", 
                                          (py_data,), 
                                          Some(&kwargs))?;
    
    // Extract weighted adjacency matrix
    let w_est: Vec<Vec<f32>> = result.extract()?;
    
    // Convert adjacency matrix to causal hypotheses
    convert_adjacency_to_hypotheses(w_est, config)
}

/// MLP NOTEARS implementation for nonlinear relationships
fn discover_with_notears_mlp(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    notears_config: &NotearsConfig,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    debug!("NOTEARS: Using MLP variant for nonlinear causal discovery");
    
    let notears_module = PyModule::import_bound(py, "notears")?;
    let numpy = PyModule::import_bound(py, "numpy")?;
    
    // Convert data to numpy
    let py_data = convert_to_numpy_array(py, &numpy, &data_matrix)?;
    
    // Set up MLP parameters
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("lambda1", notears_config.lambda1)?;
    kwargs.set_item("lambda2", notears_config.lambda2)?;
    kwargs.set_item("max_iter", notears_config.max_iter)?;
    kwargs.set_item("h_tol", notears_config.tolerance)?;
    kwargs.set_item("hidden_layers", notears_config.mlp_hidden_dims.clone())?;
    kwargs.set_item("use_A_connect_loss", true)?;
    kwargs.set_item("use_A_positional_loss", false)?;
    
    info!("NOTEARS: Running MLP NOTEARS with hidden dims {:?}", 
          notears_config.mlp_hidden_dims);
    
    let result = notears_module.call_method("notears_nonlinear", 
                                          (py_data,), 
                                          Some(&kwargs))?;
    
    // Extract adjacency matrix from MLP result
    let w_est: Vec<Vec<f32>> = result.extract()?;
    
    convert_adjacency_to_hypotheses(w_est, config)
}

/// Fallback: Install NOTEARS if not available and run
fn install_and_run_notears(
    py: Python,
    data_matrix: Vec<Vec<f32>>, 
    notears_config: &NotearsConfig,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    warn!("NOTEARS: Attempting to install notears package via pip");
    
    // Try to install notears via pip
    let subprocess = PyModule::import_bound(py, "subprocess")?;
    let install_result = subprocess.call_method1("run", 
        (vec!["pip", "install", "notears-py"],))?;
    
    // Check if installation succeeded
    let return_code: i32 = install_result.getattr("returncode")?.extract()?;
    if return_code != 0 {
        warn!("NOTEARS: Installation failed, falling back to simplified implementation");
        return simplified_notears_fallback(data_matrix, config);
    }
    
    info!("NOTEARS: Package installed successfully, retrying...");
    
    // Retry the algorithm after installation
    discover_with_notears_linear(py, data_matrix, notears_config, config)
}

/// Simplified NOTEARS implementation in pure Rust (fallback)
fn simplified_notears_fallback(
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    info!("NOTEARS: Using simplified Rust-only implementation");
    
    let n_vars = data_matrix[0].len();
    let n_samples = data_matrix.len();
    
    // Convert to ndarray for matrix operations
    let data = Array2::from_shape_vec(
        (n_samples, n_vars), 
        data_matrix.into_iter().flatten().collect()
    ).unwrap();
    
    // Initialize adjacency matrix
    let mut w = Array2::<f32>::zeros((n_vars, n_vars));
    
    // Simple gradient descent optimization
    let lambda1 = config.min_strength_threshold;
    let learning_rate = 0.01;
    let max_iter = 50;
    
    for iter in 0..max_iter {
        // Compute reconstruction loss gradient
        let pred = data.dot(&w);
        let residual = &data - &pred;
        let grad_likelihood = -data.t().dot(&residual) / (n_samples as f32);
        
        // Acyclicity constraint (simplified)
        let w_squared = w.mapv(|x| x * x);
        let trace_exp_approx = w_squared.sum(); // First-order approximation
        let grad_acyclicity = 2.0 * &w;
        
        // L1 regularization gradient (subgradient)
        let grad_l1 = w.mapv(|x| if x > 0.0 { lambda1 } else { -lambda1 });
        
        // Combined gradient
        let grad_total = grad_likelihood + 0.01 * grad_acyclicity + grad_l1;
        
        // Gradient descent step
        w = w - learning_rate * grad_total;
        
        // Project to ensure sparsity (threshold small values)
        w.mapv_inplace(|x| if x.abs() < config.min_strength_threshold { 0.0 } else { x });
        
        if iter % 10 == 0 {
            debug!("NOTEARS Fallback: Iteration {} completed", iter);
        }
    }
    
    // Convert final adjacency matrix to hypotheses
    let w_vec: Vec<Vec<f32>> = w.outer_iter()
        .map(|row| row.to_vec())
        .collect();
        
    convert_adjacency_to_hypotheses(w_vec, config)
}

/// Convert numpy data to Python format
fn convert_to_numpy_array(
    py: Python,
    numpy: &Bound<PyModule>,
    data_matrix: &[Vec<f32>]
) -> PyResult<Py<pyo3::PyAny>> {
    let py_data = PyList::new_bound(py, data_matrix.iter().map(|row| PyList::new_bound(py, row)));
    let np_array = numpy.call_method1("array", (py_data,))?;
    Ok(np_array.into())
}

/// Convert weighted adjacency matrix to causal hypotheses
fn convert_adjacency_to_hypotheses(
    adjacency_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    let mut hypotheses = Vec::new();
    
    for (i, row) in adjacency_matrix.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate() {
            if weight.abs() > config.min_strength_threshold && i != j {
                let confidence = calculate_notears_confidence(weight, &adjacency_matrix);
                
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
    
    // Sort by strength and limit results
    hypotheses.sort_by(|a, b| b.strength.abs().partial_cmp(&a.strength.abs()).unwrap());
    hypotheses.truncate(config.max_hypotheses);
    
    info!("NOTEARS: Found {} causal hypotheses", hypotheses.len());
    Ok(hypotheses)
}

fn calculate_notears_confidence(weight: f32, adjacency_matrix: &[Vec<f32>]) -> f32 {
    // Confidence based on relative strength compared to other edges
    let max_weight = adjacency_matrix.iter()
        .flat_map(|row| row.iter())
        .map(|w| w.abs())
        .fold(0.0, f32::max);
        
    if max_weight > 0.0 {
        (weight.abs() / max_weight).min(1.0)
    } else {
        0.0
    }
}

fn determine_edge_type_from_weight(weight: f32) -> CausalEdgeType {
    if weight > 0.0 {
        if weight > 0.7 { CausalEdgeType::Direct }
        else if weight > 0.3 { CausalEdgeType::Indirect }
        else { CausalEdgeType::Conditional }
    } else {
        CausalEdgeType::Inhibitory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notears_config_default() {
        let config = NotearsConfig::default();
        assert_eq!(config.lambda1, 0.01);
        assert_eq!(config.lambda2, 0.01);
        assert_eq!(config.max_iter, 100);
        assert!(!config.use_mlp);
    }

    #[test] 
    fn test_simplified_notears_fallback() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0], 
            vec![3.0, 6.0, 9.0],
            vec![4.0, 8.0, 12.0],
        ];
        
        let config = CausalDiscoveryConfig::default();
        let result = simplified_notears_fallback(data, &config);
        
        assert!(result.is_ok());
        let hypotheses = result.unwrap();
        assert!(!hypotheses.is_empty());
    }

    #[test]
    fn test_convert_adjacency_to_hypotheses() {
        let adj_matrix = vec![
            vec![0.0, 0.8, 0.1],
            vec![0.0, 0.0, 0.6],
            vec![0.0, 0.0, 0.0],
        ];
        
        let config = CausalDiscoveryConfig {
            min_strength_threshold: 0.2,
            min_confidence_threshold: 0.1,
            max_hypotheses: 10,
            algorithm: crate::causal_discovery::CausalAlgorithm::DirectLiNGAM,
        };
        
        let result = convert_adjacency_to_hypotheses(adj_matrix, &config);
        assert!(result.is_ok());
        
        let hypotheses = result.unwrap();
        assert_eq!(hypotheses.len(), 2); // Should find 0->1 and 1->2
    }
}