//! Types and structures for causal graph neural networks

use pandora_error::PandoraError;
use petgraph::graph::DiGraph;

/// Represents different types of causal relationships between nodes in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalEdgeKind {
    /// Represents a direct causal link (A causes B)
    Cause,
    /// Represents a necessary condition (A is a precondition for B)
    Precondition,
    /// Represents an inhibitory link (A prevents B)
    Inhibit,
    /// Represents an enabling link (A allows C to cause B)
    Enable,
    /// Represents a learned correlation, causality not yet determined
    Correlate,
}

impl CausalEdgeKind {
    /// Returns a default weight for this edge type based on its causal strength.
    pub fn default_weight(&self) -> f32 {
        match self {
            CausalEdgeKind::Cause => 1.0,
            CausalEdgeKind::Precondition => 0.8,
            CausalEdgeKind::Inhibit => -0.6,
            CausalEdgeKind::Enable => 0.7,
            CausalEdgeKind::Correlate => 0.3,
        }
    }

    /// Returns a string representation of the edge kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            CausalEdgeKind::Cause => "cause",
            CausalEdgeKind::Precondition => "precondition",
            CausalEdgeKind::Inhibit => "inhibit",
            CausalEdgeKind::Enable => "enable",
            CausalEdgeKind::Correlate => "correlate",
        }
    }
}

/// Represents a causal edge in the graph with its type and strength.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    /// The type of causal relationship
    pub kind: CausalEdgeKind,
    /// Strength of the causal link (can be negative for inhibitory relationships)
    pub weight: f32,
    /// Optional metadata about the edge
    pub metadata: Option<String>,
}

impl CausalEdge {
    /// Creates a new causal edge with the given type and weight.
    pub fn new(kind: CausalEdgeKind, weight: f32) -> Self {
        Self {
            kind,
            weight,
            metadata: None,
        }
    }

    /// Creates a new causal edge with default weight for the given type.
    pub fn with_default_weight(kind: CausalEdgeKind) -> Self {
        Self {
            weight: kind.default_weight(),
            kind,
            metadata: None,
        }
    }

    /// Creates a new causal edge with metadata.
    pub fn with_metadata(kind: CausalEdgeKind, weight: f32, metadata: String) -> Self {
        Self {
            kind,
            weight,
            metadata: Some(metadata),
        }
    }
}

/// Represents node data as embeddings (concepts or events).
pub type NodeData = Vec<f32>;

/// The main causal graph type using petgraph's DiGraph.
pub type CausalGraph = DiGraph<NodeData, CausalEdge>;

/// Configuration for the Graph Neural Network.
#[derive(Debug, Clone)]
pub struct GnnConfig {
    /// Input dimension for node features
    pub input_dims: usize,
    /// Hidden dimension for the neural network layers
    pub hidden_dims: usize,
    /// Number of GNN layers
    pub num_layers: usize,
    /// Learning rate for training
    pub learning_rate: f32,
    /// Dropout rate for regularization
    pub dropout_rate: f32,
    /// Whether to use attention mechanisms
    pub use_attention: bool,
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            input_dims: 64,
            hidden_dims: 128,
            num_layers: 3,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            use_attention: true,
        }
    }
}

impl GnnConfig {
    /// Creates a new GNN configuration with the given parameters.
    pub fn new(input_dims: usize, hidden_dims: usize, num_layers: usize) -> Self {
        Self {
            input_dims,
            hidden_dims,
            num_layers,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            use_attention: true,
        }
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<(), PandoraError> {
        if self.input_dims == 0 {
            return Err(PandoraError::config("input_dims must be greater than 0"));
        }
        if self.hidden_dims == 0 {
            return Err(PandoraError::config("hidden_dims must be greater than 0"));
        }
        if self.num_layers == 0 {
            return Err(PandoraError::config("num_layers must be greater than 0"));
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(PandoraError::config(
                "learning_rate must be between 0 and 1",
            ));
        }
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(PandoraError::config("dropout_rate must be between 0 and 1"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_edge_kind_default_weights() {
        assert_eq!(CausalEdgeKind::Cause.default_weight(), 1.0);
        assert_eq!(CausalEdgeKind::Precondition.default_weight(), 0.8);
        assert_eq!(CausalEdgeKind::Inhibit.default_weight(), -0.6);
        assert_eq!(CausalEdgeKind::Enable.default_weight(), 0.7);
        assert_eq!(CausalEdgeKind::Correlate.default_weight(), 0.3);
    }

    #[test]
    fn test_causal_edge_creation() {
        let edge = CausalEdge::new(CausalEdgeKind::Cause, 0.9);
        assert_eq!(edge.kind, CausalEdgeKind::Cause);
        assert_eq!(edge.weight, 0.9);
        assert!(edge.metadata.is_none());

        let edge_with_default = CausalEdge::with_default_weight(CausalEdgeKind::Cause);
        assert_eq!(edge_with_default.weight, 1.0);

        let edge_with_metadata = CausalEdge::with_metadata(
            CausalEdgeKind::Precondition,
            0.8,
            "test metadata".to_string(),
        );
        assert_eq!(
            edge_with_metadata.metadata,
            Some("test metadata".to_string())
        );
    }

    #[test]
    fn test_gnn_config_validation() {
        let valid_config = GnnConfig::new(64, 128, 3);
        assert!(valid_config.validate().is_ok());

        let invalid_config = GnnConfig {
            input_dims: 0,
            hidden_dims: 128,
            num_layers: 3,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            use_attention: true,
        };
        assert!(invalid_config.validate().is_err());
    }
}
