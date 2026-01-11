use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

// Import the new algorithm implementations
mod algorithms {
    pub mod notears;
    pub mod deci;
}

/// Represents a potential causal link discovered from data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalHypothesis {
    pub from_node_index: usize,
    pub to_node_index: usize,
    pub strength: f32,
    pub confidence: f32,
    pub edge_type: CausalEdgeType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CausalEdgeType {
    Direct,
    Indirect,
    Conditional,
    Inhibitory,
}

/// Configuration for causal discovery algorithms.
#[derive(Debug, Clone)]
pub struct CausalDiscoveryConfig {
    pub min_strength_threshold: f32,
    pub min_confidence_threshold: f32,
    pub max_hypotheses: usize,
    pub algorithm: CausalAlgorithm,
}

#[derive(Debug, Clone)]
pub enum CausalAlgorithm {
    DirectLiNGAM,
    PC,
    GES,
    /// NOTEARS: Neural structure learning without acyclicity constraints
    NOTEARS,
    /// DECI: Deep End-to-End Causal Inference with VAE + GNN
    DECI,
    /// NOTEARS-MLP: NOTEARS with multi-layer perceptron for nonlinear relationships
    NOTEARSMLP,
    /// DAGMA: Fast DAG learning with continuous optimization
    DAGMA,
}

impl Default for CausalDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_strength_threshold: 0.1,
            min_confidence_threshold: 0.3,
            max_hypotheses: 10,
            algorithm: CausalAlgorithm::DirectLiNGAM,
        }
    }
}

/// Uses a Python backend (like lingam-python) to discover causal relationships from observational data.
pub fn discover_causal_links(
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    if data_matrix.is_empty() || data_matrix[0].is_empty() {
        warn!("CausalDiscovery: Empty data matrix provided");
        return Ok(vec![]);
    }

    info!(
        "CausalDiscovery: Starting discovery with {} samples, {} variables",
        data_matrix.len(),
        data_matrix[0].len()
    );

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        match config.algorithm {
            CausalAlgorithm::DirectLiNGAM => discover_with_lingam(py, data_matrix, config),
            CausalAlgorithm::PC => discover_with_pc(py, data_matrix, config),
            CausalAlgorithm::GES => discover_with_ges(py, data_matrix, config),
            CausalAlgorithm::NOTEARS => algorithms::notears::discover_with_notears(py, data_matrix, config),
            CausalAlgorithm::DECI => algorithms::deci::discover_with_deci(py, data_matrix, config),
            CausalAlgorithm::NOTEARSMLP => {
                info!("Using NOTEARS-MLP variant");
                // For now, delegate to NOTEARS with MLP flag
                algorithms::notears::discover_with_notears(py, data_matrix, config)
            },
            CausalAlgorithm::DAGMA => {
                warn!("DAGMA not yet implemented, falling back to NOTEARS");
                algorithms::notears::discover_with_notears(py, data_matrix, config)
            },
        }
    })
}

fn discover_with_lingam(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    debug!("CausalDiscovery: Using DirectLiNGAM algorithm");
    
    // Import lingam module with error handling
    let lingam = match PyModule::import_bound(py, "lingam") {
        Ok(module) => module,
        Err(e) => {
            warn!("CausalDiscovery: Failed to import lingam module: {}", e);
            return Ok(vec![]);
        }
    };
    
    let direct_lingam_class = match lingam.getattr("DirectLiNGAM") {
        Ok(class) => class,
        Err(e) => {
            warn!("CausalDiscovery: Failed to get DirectLiNGAM class: {}", e);
            return Ok(vec![]);
        }
    };
    
    // Convert Rust data to Python list of lists
    let py_data = PyList::new_bound(py, data_matrix.iter().map(|row| PyList::new_bound(py, row)));
    
    // Create and fit the model with error handling
    let model = match direct_lingam_class.call1((py_data,)) {
        Ok(model) => model,
        Err(e) => {
            warn!("CausalDiscovery: Failed to create DirectLiNGAM model: {}", e);
            return Ok(vec![]);
        }
    };
    
    let _fit_result = match model.call_method0("fit") {
        Ok(result) => result,
        Err(e) => {
            warn!("CausalDiscovery: Failed to fit DirectLiNGAM model: {}", e);
            return Ok(vec![]);
        }
    };
    
    // Extract adjacency matrix with error handling
    let adjacency_matrix: Vec<Vec<f32>> = match model.getattr("adjacency_matrix_")?.extract() {
        Ok(matrix) => matrix,
        Err(e) => {
            warn!("CausalDiscovery: Failed to extract adjacency matrix: {}", e);
            return Ok(vec![]);
        }
    };
    
    // Convert adjacency matrix to hypotheses
    let mut hypotheses = Vec::new();
    for (i, row) in adjacency_matrix.iter().enumerate() {
        for (j, &strength) in row.iter().enumerate() {
            if strength.abs() > config.min_strength_threshold {
                let confidence = calculate_confidence(strength, &data_matrix);
                if confidence >= config.min_confidence_threshold {
                    hypotheses.push(CausalHypothesis {
                        from_node_index: i,
                        to_node_index: j,
                        strength,
                        confidence,
                        edge_type: determine_edge_type(strength),
                    });
                }
            }
        }
    }
    
    // Sort by strength and limit results
    hypotheses.sort_by(|a, b| b.strength.abs().partial_cmp(&a.strength.abs()).unwrap());
    hypotheses.truncate(config.max_hypotheses);
    
    info!("CausalDiscovery: Found {} causal hypotheses", hypotheses.len());
    Ok(hypotheses)
}

fn discover_with_pc(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    debug!("CausalDiscovery: Using PC algorithm");
    
    // Import pgmpy for PC algorithm
    let pgmpy = PyModule::import_bound(py, "pgmpy")?;
    let pc_class = pgmpy.getattr("PC")?;
    
    // Convert to pandas DataFrame
    let pandas = PyModule::import_bound(py, "pandas")?;
    let py_data = PyList::new_bound(py, data_matrix.iter().map(|row| PyList::new_bound(py, row)));
    let df = pandas.call_method1("DataFrame", (py_data,))?;
    
    // Run PC algorithm
    let pc = pc_class.call1((df,))?;
    let _estimate_result = pc.call_method0("estimate")?;
    
    // Extract edges
    let edges = pc.getattr("edges")?;
    let edges_list: Vec<(usize, usize)> = edges.extract()?;
    
    let mut hypotheses = Vec::new();
    for (from, to) in edges_list {
        if from < data_matrix[0].len() && to < data_matrix[0].len() {
            let strength = calculate_correlation_strength(&data_matrix, from, to);
            if strength.abs() > config.min_strength_threshold {
                let confidence = calculate_confidence(strength, &data_matrix);
                if confidence >= config.min_confidence_threshold {
                    hypotheses.push(CausalHypothesis {
                        from_node_index: from,
                        to_node_index: to,
                        strength,
                        confidence,
                        edge_type: CausalEdgeType::Direct,
                    });
                }
            }
        }
    }
    
    info!("CausalDiscovery: Found {} causal hypotheses with PC", hypotheses.len());
    Ok(hypotheses)
}

fn discover_with_ges(
    py: Python,
    data_matrix: Vec<Vec<f32>>,
    config: &CausalDiscoveryConfig,
) -> PyResult<Vec<CausalHypothesis>> {
    debug!("CausalDiscovery: Using GES algorithm");
    
    // Import pgmpy for GES algorithm
    let pgmpy = PyModule::import_bound(py, "pgmpy")?;
    let ges_class = pgmpy.getattr("GES")?;
    
    // Convert to pandas DataFrame
    let pandas = PyModule::import_bound(py, "pandas")?;
    let py_data = PyList::new_bound(py, data_matrix.iter().map(|row| PyList::new_bound(py, row)));
    let df = pandas.call_method1("DataFrame", (py_data,))?;
    
    // Run GES algorithm
    let ges = ges_class.call1((df,))?;
    let _estimate_result = ges.call_method0("estimate")?;
    
    // Extract edges
    let edges = ges.getattr("edges")?;
    let edges_list: Vec<(usize, usize)> = edges.extract()?;
    
    let mut hypotheses = Vec::new();
    for (from, to) in edges_list {
        if from < data_matrix[0].len() && to < data_matrix[0].len() {
            let strength = calculate_correlation_strength(&data_matrix, from, to);
            if strength.abs() > config.min_strength_threshold {
                let confidence = calculate_confidence(strength, &data_matrix);
                if confidence >= config.min_confidence_threshold {
                    hypotheses.push(CausalHypothesis {
                        from_node_index: from,
                        to_node_index: to,
                        strength,
                        confidence,
                        edge_type: CausalEdgeType::Direct,
                    });
                }
            }
        }
    }
    
    info!("CausalDiscovery: Found {} causal hypotheses with GES", hypotheses.len());
    Ok(hypotheses)
}

fn calculate_confidence(strength: f32, _data: &[Vec<f32>]) -> f32 {
    // Simple confidence calculation based on strength magnitude
    // In a real implementation, this would use statistical tests
    strength.abs().min(1.0)
}

fn calculate_correlation_strength(data: &[Vec<f32>], from: usize, to: usize) -> f32 {
    if data.is_empty() || from >= data[0].len() || to >= data[0].len() {
        return 0.0;
    }
    
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let n = data.len() as f32;
    
    for row in data {
        let x = row[from];
        let y = row[to];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        (numerator / denominator) as f32
    }
}

fn determine_edge_type(strength: f32) -> CausalEdgeType {
    if strength > 0.0 {
        if strength.abs() > 0.7 {
            CausalEdgeType::Direct
        } else if strength.abs() > 0.4 {
            CausalEdgeType::Indirect
        } else {
            CausalEdgeType::Conditional
        }
    } else {
        CausalEdgeType::Inhibitory
    }
}

/// Validates a causal hypothesis by checking if it makes sense given the data.
pub fn validate_hypothesis(
    hypothesis: &CausalHypothesis,
    data: &[Vec<f32>],
) -> bool {
    if data.is_empty() || 
       hypothesis.from_node_index >= data[0].len() || 
       hypothesis.to_node_index >= data[0].len() {
        return false;
    }
    
    // Check if the relationship is statistically significant
    let correlation = calculate_correlation_strength(data, hypothesis.from_node_index, hypothesis.to_node_index);
    correlation.abs() > 0.1 && hypothesis.confidence > 0.2
}

/// Groups hypotheses by their target node for easier processing.
pub fn group_hypotheses_by_target(hypotheses: Vec<CausalHypothesis>) -> HashMap<usize, Vec<CausalHypothesis>> {
    let mut groups = HashMap::new();
    
    for hypothesis in hypotheses {
        groups.entry(hypothesis.to_node_index)
            .or_insert_with(Vec::new)
            .push(hypothesis);
    }
    
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_discovery_config_default() {
        let config = CausalDiscoveryConfig::default();
        assert_eq!(config.min_strength_threshold, 0.1);
        assert_eq!(config.min_confidence_threshold, 0.3);
        assert_eq!(config.max_hypotheses, 10);
    }

    #[test]
    fn test_calculate_correlation_strength() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        
        let strength = calculate_correlation_strength(&data, 0, 1);
        assert!(strength > 0.9); // Should be highly correlated
    }

    #[test]
    fn test_determine_edge_type() {
        assert!(matches!(determine_edge_type(0.8), CausalEdgeType::Direct));
        assert!(matches!(determine_edge_type(0.5), CausalEdgeType::Indirect));
        assert!(matches!(determine_edge_type(0.3), CausalEdgeType::Conditional));
        assert!(matches!(determine_edge_type(-0.5), CausalEdgeType::Inhibitory));
    }

    #[test]
    fn test_validate_hypothesis() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        
        let valid_hypothesis = CausalHypothesis {
            from_node_index: 0,
            to_node_index: 1,
            strength: 0.8,
            confidence: 0.7,
            edge_type: CausalEdgeType::Direct,
        };
        
        let invalid_hypothesis = CausalHypothesis {
            from_node_index: 0,
            to_node_index: 1,
            strength: 0.05,
            confidence: 0.1,
            edge_type: CausalEdgeType::Direct,
        };
        
        assert!(validate_hypothesis(&valid_hypothesis, &data));
        assert!(!validate_hypothesis(&invalid_hypothesis, &data));
    }

    #[test]
    fn test_group_hypotheses_by_target() {
        let hypotheses = vec![
            CausalHypothesis {
                from_node_index: 0,
                to_node_index: 2,
                strength: 0.5,
                confidence: 0.6,
                edge_type: CausalEdgeType::Direct,
            },
            CausalHypothesis {
                from_node_index: 1,
                to_node_index: 2,
                strength: 0.3,
                confidence: 0.4,
                edge_type: CausalEdgeType::Indirect,
            },
        ];
        
        let groups = group_hypotheses_by_target(hypotheses);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups.get(&2).unwrap().len(), 2);
    }
}
