use crate::gnn::types::{CausalEdgeKind, CausalGraph};
#[cfg(feature = "ml")]
use ndarray::{Array1, Array2};
use pandora_error::PandoraError;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

/// A basic Graph Convolution Layer for Graph Neural Networks.
///
/// This layer performs message passing followed by a linear transformation,
/// implementing a simple form of graph convolution.
///
/// # Examples
///
/// ```rust
/// use pandora_cwm::gnn::layers::GraphConvLayer;
/// use ndarray::arr2;
///
/// let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
/// let layer = GraphConvLayer::new(weight);
/// let adj = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
/// let features = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
/// let output = layer.forward(&adj, &features);
/// assert_eq!(output.shape(), &[2, 2]);
/// ```
pub struct GraphConvLayer {
    pub weight: Array2<f32>, // (in_dim, out_dim)
}

#[cfg(feature = "ml")]
impl GraphConvLayer {
    /// Creates a new GraphConvLayer with the given weight matrix.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight matrix of shape (in_dim, out_dim)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandora_cwm::gnn::layers::GraphConvLayer;
    /// use ndarray::arr2;
    /// let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    /// let layer = GraphConvLayer::new(weight);
    /// ```
    pub fn new(weight: Array2<f32>) -> Self {
        Self { weight }
    }

    /// Performs forward pass through the graph convolution layer.
    ///
    /// # Arguments
    ///
    /// * `adj` - Adjacency matrix (n x n)
    /// * `x` - Input node features (n x in_dim)
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - Output features (n x out_dim)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandora_cwm::gnn::layers::GraphConvLayer;
    /// use ndarray::arr2;
    /// let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    /// let layer = GraphConvLayer::new(weight);
    /// let adj = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
    /// let x = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    /// let output = layer.forward(&adj, &x);
    /// ```
    pub fn forward(&self, adj: &Array2<f32>, x: &Array2<f32>) -> Array2<f32> {
        // simple: mean aggregate then linear
        let agg = crate::gnn::message_passing::aggregate_mean(adj, x);
        agg.dot(&self.weight)
    }
}

/// Graph Attention Layer for handling different causal edge types.
///
/// This layer implements a Graph Attention Network (GAT) that can learn to weigh
/// different causal edge types differently based on context. It uses attention
/// mechanisms to focus on the most relevant causal relationships.
///
/// # Examples
///
/// ```rust
/// use pandora_cwm::gnn::layers::GraphAttentionLayer;
/// use pandora_cwm::gnn::types::{CausalGraph, CausalEdgeKind};
/// use ndarray::arr2;
///
/// let input_dim = 64;
/// let hidden_dim = 128;
/// let layer = GraphAttentionLayer::new(input_dim, hidden_dim);
/// ```
pub struct GraphAttentionLayer {
    /// Linear transformation weights for source nodes
    pub w_src: Array2<f32>, // (input_dim, hidden_dim)
    /// Linear transformation weights for target nodes  
    pub w_tgt: Array2<f32>, // (input_dim, hidden_dim)
    /// Attention weights for different edge types
    pub edge_type_weights: std::collections::HashMap<CausalEdgeKind, Array1<f32>>, // (hidden_dim,)
    /// Attention mechanism weights
    pub attention_weights: Array1<f32>, // (2 * hidden_dim,)
    /// LeakyReLU negative slope
    pub leaky_relu_slope: f32,
}

#[cfg(feature = "ml")]
impl GraphAttentionLayer {
    /// Creates a new GraphAttentionLayer with the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dim` - Hidden feature dimension
    ///
    /// # Returns
    ///
    /// * `Self` - The initialized attention layer
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize weights with small random values
        let w_src = Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.gen_range(-0.1..0.1));
        let w_tgt = Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.gen_range(-0.1..0.1));

        // Initialize edge type weights for each causal edge kind
        let mut edge_type_weights = std::collections::HashMap::new();
        for edge_kind in [
            CausalEdgeKind::Cause,
            CausalEdgeKind::Precondition,
            CausalEdgeKind::Inhibit,
            CausalEdgeKind::Enable,
            CausalEdgeKind::Correlate,
        ] {
            edge_type_weights.insert(
                edge_kind,
                Array1::from_shape_fn(hidden_dim, |_| rng.gen_range(-0.1..0.1)),
            );
        }

        // Initialize attention weights (for concatenated features: input_dim + input_dim)
        let attention_weights = Array1::from_shape_fn(2 * input_dim, |_| rng.gen_range(-0.1..0.1));

        Self {
            w_src,
            w_tgt,
            edge_type_weights,
            attention_weights,
            leaky_relu_slope: 0.2,
        }
    }

    /// Performs forward pass through the graph attention layer.
    ///
    /// This method computes attention scores for each edge based on its type and context,
    /// then performs weighted aggregation of neighbor features.
    ///
    /// # Arguments
    ///
    /// * `graph` - The causal graph
    /// * `node_features` - Input node features (n x input_dim)
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f32>, PandoraError>` - Output features (n x hidden_dim)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandora_cwm::gnn::layers::GraphAttentionLayer;
    /// use pandora_cwm::gnn::types::{CausalGraph, CausalEdgeKind};
    /// use ndarray::arr2;
    ///
    /// let layer = GraphAttentionLayer::new(4, 8);
    /// let graph = CausalGraph::new();
    /// let features = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    /// let output = layer.forward(&graph, &features)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn forward(
        &self,
        graph: &CausalGraph,
        node_features: &Array2<f32>,
    ) -> Result<Array2<f32>, PandoraError> {
        let n_nodes = node_features.nrows();
        let hidden_dim = self.w_src.ncols();

        // Transform node features
        let src_features = node_features.dot(&self.w_src); // (n_nodes, hidden_dim)
        let tgt_features = node_features.dot(&self.w_tgt); // (n_nodes, hidden_dim)

        // Initialize output features
        let mut output_features = Array2::zeros((n_nodes, hidden_dim));

        // For each node, compute attention with its neighbors
        for (node_idx, _) in graph.node_indices().enumerate() {
            let node_index = NodeIndex::new(node_idx);

            // Get neighbors (both incoming and outgoing edges)
            let mut neighbor_features = Vec::new();
            let mut attention_scores = Vec::new();

            // Process incoming edges
            for edge_idx in graph.edges_directed(node_index, petgraph::Direction::Incoming) {
                let source_idx = edge_idx.source().index();
                let edge_data = &edge_idx.weight();

                // Compute attention score
                let score = self.compute_attention_score(
                    &src_features.row(source_idx).to_owned(),
                    &tgt_features.row(node_idx).to_owned(),
                    edge_data.kind,
                )?;

                neighbor_features.push(src_features.row(source_idx).to_owned());
                attention_scores.push(score);
            }

            // Process outgoing edges
            for edge_idx in graph.edges_directed(node_index, petgraph::Direction::Outgoing) {
                let target_idx = edge_idx.target().index();
                let edge_data = &edge_idx.weight();

                // Compute attention score
                let score = self.compute_attention_score(
                    &src_features.row(node_idx).to_owned(),
                    &tgt_features.row(target_idx).to_owned(),
                    edge_data.kind,
                )?;

                neighbor_features.push(src_features.row(target_idx).to_owned());
                attention_scores.push(score);
            }

            // Apply attention weights and aggregate
            if !neighbor_features.is_empty() {
                let attention_sum: f32 = attention_scores.iter().sum();
                if attention_sum > 0.0 {
                    let mut aggregated = Array1::zeros(hidden_dim);
                    for (features, score) in neighbor_features.iter().zip(attention_scores.iter()) {
                        let scaled_features = features * (*score / attention_sum);
                        aggregated = aggregated + scaled_features;
                    }
                    output_features.row_mut(node_idx).assign(&aggregated);
                }
            }
        }

        Ok(output_features)
    }

    /// Computes attention score between source and target features for a given edge type.
    fn compute_attention_score(
        &self,
        src_features: &Array1<f32>,
        tgt_features: &Array1<f32>,
        edge_kind: CausalEdgeKind,
    ) -> Result<f32, PandoraError> {
        // Get edge type specific weights
        let edge_weights = self
            .edge_type_weights
            .get(&edge_kind)
            .ok_or_else(|| PandoraError::config(format!("Unknown edge kind: {:?}", edge_kind)))?;

        // Concatenate source and target features
        let mut concat_features = Array1::zeros(src_features.len() + tgt_features.len());
        for i in 0..src_features.len() {
            concat_features[i] = src_features[i];
        }
        for i in 0..tgt_features.len() {
            concat_features[src_features.len() + i] = tgt_features[i];
        }

        // Apply edge type weights
        let weighted_features = &concat_features * edge_weights;

        // Compute attention score using the attention weights
        let score = weighted_features.dot(&self.attention_weights);

        // Apply LeakyReLU activation
        let activated_score = if score > 0.0 {
            score
        } else {
            self.leaky_relu_slope * score
        };

        Ok(activated_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::types::CausalEdgeKind;
    use ndarray::arr2;

    #[test]
    fn test_graph_conv_layer() {
        let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let layer = GraphConvLayer::new(weight);
        let adj = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let output = layer.forward(&adj, &x);
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_graph_attention_layer_creation() {
        let layer = GraphAttentionLayer::new(64, 128);
        assert_eq!(layer.w_src.shape(), &[64, 128]);
        assert_eq!(layer.w_tgt.shape(), &[64, 128]);
        assert_eq!(layer.attention_weights.len(), 128); // 2 * 64
        assert_eq!(layer.edge_type_weights.len(), 5); // 5 edge kinds
    }

    #[test]
    fn test_graph_attention_layer_forward() {
        let layer = GraphAttentionLayer::new(4, 8);
        let graph = CausalGraph::new();
        let features = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);

        let result = layer.forward(&graph, &features);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_attention_score_computation() {
        let layer = GraphAttentionLayer::new(4, 8);
        let src_features = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let tgt_features = Array1::from(vec![5.0, 6.0, 7.0, 8.0]);

        let score =
            layer.compute_attention_score(&src_features, &tgt_features, CausalEdgeKind::Cause);
        assert!(score.is_ok());

        let score_value = score.unwrap();
        assert!(score_value.is_finite());
    }
}
