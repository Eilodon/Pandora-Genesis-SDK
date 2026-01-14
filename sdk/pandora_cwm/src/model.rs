//! Interdependent-Causal World Model (CWM) v3.0
//!
//! This module contains the unified core model that combines interdependent representation
//! and graph neural networks to create a causality-aware world model.

use crate::gnn::{types::GnnConfig, GraphNeuralNetwork};
use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::world_model::WorldModel;
use pandora_error::PandoraError;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use tracing::debug;

/// Represents a causal hypothesis discovered through data analysis.
/// This is a simplified version for the CWM module.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalHypothesis {
    pub from_node_index: usize,
    pub to_node_index: usize,
    pub strength: f32,
    pub confidence: f32,
    pub edge_type: CausalEdgeType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CausalEdgeType {
    Direct,
    Indirect,
    Conditional,
    Inhibitory,
}

/// Represents the Interdependent-Causal World Model.
/// This model uses a Graph Neural Network to represent the dynamic,
/// relational, and causal structure of the world.
pub struct InterdependentCausalModel {
    gnn: GraphNeuralNetwork,
    predictor: StatePredictor,
}

/// Simple state predictor for forward modeling
pub struct StatePredictor {
    #[allow(dead_code)]
    input_dims: usize,
    output_dims: usize,
}

impl StatePredictor {
    /// Creates a new StatePredictor
    pub fn new(input_dims: usize, output_dims: usize) -> Self {
        Self {
            input_dims,
            output_dims,
        }
    }

    /// Predicts the next state embedding from current context
    pub fn predict(&self, context_embedding: &[f32]) -> Result<Vec<f32>, PandoraError> {
        // Simplified prediction: just add some noise to simulate state change
        // In a full implementation, this would be a trained neural network
        let mut prediction = context_embedding.to_vec();

        // Add small random changes to simulate state transition
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for val in prediction.iter_mut() {
            *val += rng.gen_range(-0.1..0.1);
        }

        // Ensure output has correct dimensions
        if prediction.len() != self.output_dims {
            prediction.resize(self.output_dims, 0.0);
        }

        Ok(prediction)
    }
}

impl InterdependentCausalModel {
    /// Creates a new InterdependentCausalModel with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the underlying Graph Neural Network
    ///
    /// # Returns
    ///
    /// * `Result<Self, PandoraError>` - The initialized model or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandora_cwm::model::InterdependentCausalModel;
    /// use pandora_cwm::gnn::types::GnnConfig;
    ///
    /// let config = GnnConfig::new(64, 128, 3);
    /// let model = InterdependentCausalModel::new(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: GnnConfig) -> Result<Self, PandoraError> {
        let gnn = GraphNeuralNetwork::new(config.clone())?;
        let predictor = StatePredictor::new(config.input_dims, config.hidden_dims);
        Ok(Self { gnn, predictor })
    }

    /// Updates the internal graph model based on a new cognitive flow.
    /// This method translates the flow into graph updates and triggers the GNN's learning mechanism.
    ///
    /// # Arguments
    ///
    /// * `flow` - The epistemological flow containing new information to learn from
    ///
    /// # Returns
    ///
    /// * `Result<(), PandoraError>` - Success or error
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use pandora_cwm::model::InterdependentCausalModel;
    /// # use pandora_cwm::gnn::types::GnnConfig;
    /// # use pandora_core::ontology::EpistemologicalFlow;
    /// # let config = GnnConfig::new(64, 128, 3);
    /// # let mut cwm = InterdependentCausalModel::new(config)?;
    /// # let flow = EpistemologicalFlow::default();
    /// cwm.learn_relations(&flow)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn learn_relations(&mut self, flow: &EpistemologicalFlow) -> Result<(), PandoraError> {
        // Extract causal information from the flow and update the GNN graph
        debug!("Learning relations from epistemological flow");

        // Create a new node for the current flow if it doesn't exist
        let flow_embedding = self.gnn.get_contextual_embedding(flow)?;
        let flow_node = self.gnn.add_node(flow_embedding);

        // Extract causal relationships from the flow's sankhara (intent) and sanna (perception)
        if let Some(ref sankhara) = flow.sankhara {
            let intent_str = sankhara.as_ref();

            // Create a node for the intent if it doesn't exist
            let intent_embedding = self.create_intent_embedding(intent_str);
            let intent_node = self.gnn.add_node(intent_embedding);

            // Add a causal edge from intent to flow
            use crate::gnn::types::{CausalEdge, CausalEdgeKind};
            let intent_edge = CausalEdge::with_metadata(
                CausalEdgeKind::Cause,
                0.8, // High confidence for intent->action relationship
                format!("Intent '{}' causes flow", intent_str),
            );
            self.gnn.add_edge(intent_node, flow_node, intent_edge)?;
        }

        // Extract relationships from sanna (perception)
        if let Some(ref sanna) = flow.sanna {
            for &active_index in &sanna.active_indices {
                // Create a node for each active concept
                let concept_embedding = self.create_concept_embedding(active_index);
                let concept_node = self.gnn.add_node(concept_embedding);

                // Add a correlation edge from concept to flow
                use crate::gnn::types::{CausalEdge, CausalEdgeKind};
                let correlation_edge = CausalEdge::with_metadata(
                    CausalEdgeKind::Correlate,
                    0.6, // Medium confidence for perception->flow relationship
                    format!("Concept {} correlates with flow", active_index),
                );
                self.gnn
                    .add_edge(concept_node, flow_node, correlation_edge)?;
            }
        }

        // Extract relationships from related_eidos
        if let Some(ref related_eidos) = flow.related_eidos {
            for eidos in related_eidos {
                for &active_index in &eidos.active_indices {
                    // Create a node for each related concept
                    let concept_embedding = self.create_concept_embedding(active_index);
                    let concept_node = self.gnn.add_node(concept_embedding);

                    // Add a correlation edge from related concept to flow
                    use crate::gnn::types::{CausalEdge, CausalEdgeKind};
                    let correlation_edge = CausalEdge::with_metadata(
                        CausalEdgeKind::Correlate,
                        0.4, // Lower confidence for related concepts
                        format!("Related concept {} correlates with flow", active_index),
                    );
                    self.gnn
                        .add_edge(concept_node, flow_node, correlation_edge)?;
                }
            }
        }

        // Update existing nodes with new information
        self.update_existing_nodes(flow)?;

        debug!(
            "Successfully learned relations from flow, added {} nodes, {} edges",
            self.gnn.node_count(),
            self.gnn.edge_count()
        );
        Ok(())
    }

    /// Creates an embedding for an intent string
    pub fn create_intent_embedding(&self, intent: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.gnn.config.hidden_dims];

        // Simple hash-based embedding for intent
        let mut hash = 0u32;
        for byte in intent.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }

        // Normalize to [-1, 1] range
        let normalized = (hash as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        embedding[0] = normalized;

        // Add some variation based on intent length
        embedding[1] = (intent.len() as f32) / 100.0; // Normalize length

        // Add some random noise for variation
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for val in embedding.iter_mut().skip(2) {
            *val = rng.gen_range(-0.1..0.1);
        }

        embedding
    }

    /// Creates an embedding for a concept index
    pub fn create_concept_embedding(&self, concept_index: u32) -> Vec<f32> {
        let mut embedding = vec![0.0; self.gnn.config.hidden_dims];

        // Use the concept index to create a deterministic embedding
        let normalized_index = (concept_index as f32) / 1000.0; // Normalize index
        embedding[0] = normalized_index;

        // Add some variation based on the index
        embedding[1] = (concept_index as f32).sin();
        embedding[2] = (concept_index as f32).cos();

        // Add deterministic variation based on the index for remaining dimensions
        for (i, val) in embedding.iter_mut().enumerate().skip(3) {
            let seed = (concept_index as f32) * (i as f32) * 0.1;
            *val = (seed.sin() * 0.05) + (seed.cos() * 0.03);
        }

        embedding
    }

    /// Updates existing nodes with new information from the flow
    fn update_existing_nodes(&mut self, _flow: &EpistemologicalFlow) -> Result<(), PandoraError> {
        // In a real implementation, this would update existing node embeddings
        // based on the new information in the flow
        // For now, we'll just log that we're updating
        debug!("Updating existing nodes with new flow information");

        // Simple update: add small random changes to existing nodes
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for node_idx in self.gnn.graph().node_indices() {
            if let Some(node_data) = self.gnn.graph_mut().node_weight_mut(node_idx) {
                for val in node_data.iter_mut() {
                    *val += rng.gen_range(-0.01..0.01); // Small random update
                }
            }
        }

        Ok(())
    }

    /// Infers a richer context for the given flow by reasoning over the graph.
    /// This method uses the GNN to perform message passing to enrich the
    /// `sanna` and `related_eidos` fields of the flow.
    ///
    /// # Arguments
    ///
    /// * `flow` - The epistemological flow to enrich with contextual information
    ///
    /// # Returns
    ///
    /// * `Result<(), PandoraError>` - Success or error
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use pandora_cwm::model::InterdependentCausalModel;
    /// # use pandora_cwm::gnn::types::GnnConfig;
    /// # use pandora_core::ontology::EpistemologicalFlow;
    /// # let config = GnnConfig::new(64, 128, 3);
    /// # let cwm = InterdependentCausalModel::new(config)?;
    /// # let mut flow = EpistemologicalFlow::default();
    /// cwm.infer_context(&mut flow)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn infer_context(&self, flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        // Use the GNN to perform multi-round message passing for richer context
        debug!("Inferring context for epistemological flow with multi-round reasoning");

        // Get the initial contextual embedding
        let mut current_embedding = self.gnn.get_contextual_embedding(flow)?;

        // Perform multiple rounds of message passing (reasoning)
        let num_rounds = 3; // Configurable number of reasoning rounds
        for round in 0..num_rounds {
            debug!("Message passing round {}/{}", round + 1, num_rounds);

            // Perform one round of message passing
            let enriched_embedding = self.perform_message_passing_round(&current_embedding)?;

            // Update the current embedding with the enriched version
            current_embedding = enriched_embedding;
        }

        // Decode the final enriched embedding back into the flow
        self.decode_enriched_embedding_to_flow(flow, &current_embedding)?;

        debug!("Completed multi-round context inference");
        Ok(())
    }

    /// Performs one round of message passing through the GNN
    pub fn perform_message_passing_round(
        &self,
        input_embedding: &[f32],
    ) -> Result<Vec<f32>, PandoraError> {
        let mut enriched_embedding = input_embedding.to_vec();

        // Simple message passing simulation
        // In a real implementation, this would use the actual GNN layers
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Simulate information propagation from neighboring nodes
        for i in 0..enriched_embedding.len() {
            // Add some influence from neighboring dimensions
            let neighbor_influence = if i > 0 {
                input_embedding[i - 1] * 0.1
            } else {
                0.0
            };

            let next_neighbor_influence = if i < input_embedding.len() - 1 {
                input_embedding[i + 1] * 0.1
            } else {
                0.0
            };

            // Update the embedding with neighbor influences
            enriched_embedding[i] =
                input_embedding[i] + neighbor_influence + next_neighbor_influence;

            // Add some random noise to simulate complex interactions
            enriched_embedding[i] += rng.gen_range(-0.05..0.05);

            // Apply a simple activation function (tanh)
            enriched_embedding[i] = enriched_embedding[i].tanh();
        }

        Ok(enriched_embedding)
    }

    /// Decodes the enriched embedding back into the flow's sanna and related_eidos
    fn decode_enriched_embedding_to_flow(
        &self,
        flow: &mut EpistemologicalFlow,
        embedding: &[f32],
    ) -> Result<(), PandoraError> {
        // Find the most significant dimensions in the enriched embedding
        let mut significant_dims = Vec::new();
        for (i, &value) in embedding.iter().enumerate() {
            if value.abs() > 0.3 {
                // Threshold for significance
                significant_dims.push((i, value));
            }
        }

        // Sort by absolute value (descending)
        significant_dims.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Update sanna with the most significant concepts
        if !significant_dims.is_empty() {
            let mut new_sanna = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };

            // Add the top 5 most significant dimensions
            for (dim, _) in significant_dims.iter().take(5) {
                new_sanna.active_indices.insert(*dim as u32);
            }

            // Merge with existing sanna if it exists
            if let Some(ref existing_sanna) = flow.sanna {
                for &index in &existing_sanna.active_indices {
                    new_sanna.active_indices.insert(index);
                }
            }

            flow.sanna = Some(new_sanna);
        }

        // Update related_eidos with additional significant concepts
        let mut related_eidos = SmallVec::new();

        // Create related eidos for the next most significant dimensions
        for (dim, value) in significant_dims.iter().skip(5).take(3) {
            let mut eidos = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };

            eidos.active_indices.insert(*dim as u32);

            // Add related dimensions based on value magnitude
            if value.abs() > 0.5 {
                // High magnitude - add neighboring dimensions
                eidos.active_indices.insert((*dim as u32).saturating_sub(1));
                eidos.active_indices.insert((*dim as u32).saturating_add(1));
            }

            related_eidos.push(eidos);
        }

        if !related_eidos.is_empty() {
            flow.related_eidos = Some(related_eidos);
        }

        Ok(())
    }

    /// Returns a reference to the underlying Graph Neural Network.
    /// This is useful for debugging and advanced operations.
    pub fn gnn(&self) -> &GraphNeuralNetwork {
        &self.gnn
    }

    /// Returns a mutable reference to the underlying Graph Neural Network.
    /// This allows for direct manipulation of the graph structure.
    pub fn gnn_mut(&mut self) -> &mut GraphNeuralNetwork {
        &mut self.gnn
    }

    /// Predicts the next state of the EpistemologicalFlow based on its current
    /// state and the `sankhara` (intent) it contains.
    ///
    /// This is the core of the forward model that enables Active Inference planning.
    /// The method uses the GNN to predict how the world state will change based
    /// on the current state and the intended action.
    ///
    /// # Arguments
    ///
    /// * `flow` - The current epistemological flow to predict from
    ///
    /// # Returns
    ///
    /// * `Result<(), PandoraError>` - Success or error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandora_core::ontology::EpistemologicalFlow;
    /// use pandora_cwm::model::InterdependentCausalModel;
    /// use pandora_cwm::gnn::types::GnnConfig;
    ///
    /// let config = GnnConfig::new(64, 128, 3);
    /// let mut model = InterdependentCausalModel::new(config).unwrap();
    /// let mut flow = EpistemologicalFlow::default();
    ///
    /// // Set an intent
    /// flow.sankhara = Some(std::sync::Arc::<str>::from("unlock_door"));
    ///
    /// // Predict the next state
    /// model.predict_next_state(&mut flow).unwrap();
    /// ```
    pub fn predict_next_state(&self, flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        // This is the core of the forward model.
        // 1. Get the embedding of the current flow's state and the intent.
        let context_embedding = self.gnn.get_contextual_embedding(flow)?;

        // 2. Use a predictive model (this could be a layer in your GNN or a separate NN)
        //    to predict the embedding of the *next* state.
        let next_state_embedding = self.predictor.predict(&context_embedding)?;

        // 3. Decode the next state embedding back into the `EpistemologicalFlow`.
        //    This is a complex step. For now, we can simplify: update the most
        //    salient features in the flow based on the prediction.
        //    For example, if the intent was "unlock_door" and the agent has a key,
        //    the GNN should predict a new state where the "door_is_locked" concept
        //    is no longer active.
        self.decode_and_update_flow(flow, &next_state_embedding)?;

        Ok(())
    }

    /// Decodes a predicted embedding back into changes in the EpistemologicalFlow.
    ///
    /// This is the inverse of `get_contextual_embedding` and is crucial for
    /// translating GNN predictions back into actionable state changes.
    ///
    /// # Arguments
    ///
    /// * `flow` - The flow to update based on the prediction
    /// * `embedding` - The predicted next state embedding
    ///
    /// # Returns
    ///
    /// * `Result<(), PandoraError>` - Success or error
    fn decode_and_update_flow(
        &self,
        flow: &mut EpistemologicalFlow,
        embedding: &[f32],
    ) -> Result<(), PandoraError> {
        // Get the current contextual embedding for comparison
        let current_embedding = self.gnn.get_contextual_embedding(flow)?;

        // Ensure embeddings have the same dimensions
        if current_embedding.len() != embedding.len() {
            return Err(PandoraError::config("Embedding dimension mismatch"));
        }

        // Calculate the difference between current and predicted embeddings
        let mut differences = Vec::with_capacity(embedding.len());
        for (current, predicted) in current_embedding.iter().zip(embedding.iter()) {
            differences.push(predicted - current);
        }

        // Find the most significant differences (threshold-based)
        let threshold = 0.1; // Configurable threshold for significant changes
        let mut significant_changes = Vec::new();

        for (i, diff) in differences.iter().enumerate() {
            if diff.abs() > threshold {
                significant_changes.push((i, *diff));
            }
        }

        // Sort by magnitude of change (descending)
        significant_changes.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Update the flow based on the most significant changes
        // This is where we map embedding dimensions back to specific concepts

        // Handle intent-specific state transitions
        if let Some(ref intent) = flow.sankhara {
            let intent_str = intent.as_ref();

            // Process significant changes based on intent
            match intent_str {
                "unlock_door" => {
                    // Look for changes that might indicate door state changes
                    for (dim, change) in &significant_changes[..significant_changes.len().min(3)] {
                        if *dim < 10 {
                            // First 10 dimensions represent door-related concepts
                            if *change > 0.0 {
                                // Positive change might indicate door becoming unlocked
                                debug!("Predicted: Door unlocking detected in dimension {}", dim);
                                // Update sanna to reflect door state change
                                self.update_sanna_for_door_state(flow, false); // false = unlocked
                            } else {
                                // Negative change might indicate door becoming locked
                                debug!("Predicted: Door locking detected in dimension {}", dim);
                                self.update_sanna_for_door_state(flow, true); // true = locked
                            }
                        }
                    }
                }
                "pick_up_key" => {
                    // Look for changes that might indicate key pickup
                    for (dim, change) in &significant_changes[..significant_changes.len().min(3)] {
                        if *dim >= 10 && *dim < 20 {
                            // Dimensions 10-19 represent key-related concepts
                            if *change > 0.0 {
                                debug!("Predicted: Key pickup detected in dimension {}", dim);
                                self.update_sanna_for_key_state(flow, true); // true = has key
                            }
                        }
                    }
                }
                "move_forward" => {
                    // Look for changes that might indicate movement
                    for (dim, change) in &significant_changes[..significant_changes.len().min(3)] {
                        if *dim >= 20 && *dim < 30 {
                            // Dimensions 20-29 represent position concepts
                            if *change > 0.0 {
                                debug!("Predicted: Forward movement detected in dimension {}", dim);
                                self.update_sanna_for_position(flow, 1); // 1 = forward
                            } else {
                                debug!(
                                    "Predicted: Backward movement detected in dimension {}",
                                    dim
                                );
                                self.update_sanna_for_position(flow, -1); // -1 = backward
                            }
                        }
                    }
                }
                _ => {
                    // Generic state change handling
                    debug!(
                        "Predicted: Generic state change for intent '{}'",
                        intent_str
                    );
                    self.update_sanna_generic(flow, &significant_changes);
                }
            }
        } else {
            // No specific intent, process changes generically
            self.update_sanna_generic(flow, &significant_changes);
        }

        // Update related_eidos based on significant changes
        self.update_related_eidos(flow, &significant_changes);

        Ok(())
    }

    /// Updates the sanna field to reflect door state changes
    fn update_sanna_for_door_state(&self, flow: &mut EpistemologicalFlow, is_locked: bool) {
        if let Some(ref mut sanna) = flow.sanna {
            if is_locked {
                // Add door_locked concept
                sanna.active_indices.insert(1); // Index 1 represents "door_locked"
                sanna.active_indices.remove(&2); // Remove "door_unlocked" if present
            } else {
                // Add door_unlocked concept
                sanna.active_indices.insert(2); // Index 2 represents "door_unlocked"
                sanna.active_indices.remove(&1); // Remove "door_locked" if present
            }
        } else {
            // Create new sanna if it doesn't exist
            let mut new_sanna = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };
            if is_locked {
                new_sanna.active_indices.insert(1);
            } else {
                new_sanna.active_indices.insert(2);
            }
            flow.sanna = Some(new_sanna);
        }
    }

    /// Updates the sanna field to reflect key state changes
    fn update_sanna_for_key_state(&self, flow: &mut EpistemologicalFlow, has_key: bool) {
        if let Some(ref mut sanna) = flow.sanna {
            if has_key {
                sanna.active_indices.insert(3); // Index 3 represents "has_key"
            } else {
                sanna.active_indices.remove(&3);
            }
        } else if has_key {
            let mut new_sanna = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };
            new_sanna.active_indices.insert(3);
            flow.sanna = Some(new_sanna);
        }
    }

    /// Updates the sanna field to reflect position changes
    fn update_sanna_for_position(&self, flow: &mut EpistemologicalFlow, direction: i32) {
        if let Some(ref mut sanna) = flow.sanna {
            // Remove old position indicators
            sanna.active_indices.remove(&4); // Index 4 represents "position_forward"
            sanna.active_indices.remove(&5); // Index 5 represents "position_backward"

            if direction > 0 {
                sanna.active_indices.insert(4);
            } else if direction < 0 {
                sanna.active_indices.insert(5);
            }
        } else if direction != 0 {
            let mut new_sanna = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };
            if direction > 0 {
                new_sanna.active_indices.insert(4);
            } else {
                new_sanna.active_indices.insert(5);
            }
            flow.sanna = Some(new_sanna);
        }
    }

    /// Updates the sanna field generically based on significant changes
    fn update_sanna_generic(
        &self,
        flow: &mut EpistemologicalFlow,
        significant_changes: &[(usize, f32)],
    ) {
        if significant_changes.is_empty() {
            return;
        }

        if let Some(ref mut sanna) = flow.sanna {
            // Add indices for significant changes
            for (dim, _) in significant_changes.iter().take(5) {
                // Limit to top 5 changes
                sanna.active_indices.insert(*dim as u32);
            }
        } else {
            let mut new_sanna = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };
            for (dim, _) in significant_changes.iter().take(5) {
                new_sanna.active_indices.insert(*dim as u32);
            }
            flow.sanna = Some(new_sanna);
        }
    }

    /// Updates the related_eidos field based on significant changes
    fn update_related_eidos(
        &self,
        flow: &mut EpistemologicalFlow,
        significant_changes: &[(usize, f32)],
    ) {
        if significant_changes.is_empty() {
            return;
        }

        // Create related eidos for significant changes
        let mut related_eidos = SmallVec::new();

        for (dim, change) in significant_changes.iter().take(3) {
            // Limit to top 3 changes
            let mut eidos = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };

            // Add the dimension as an active index
            eidos.active_indices.insert(*dim as u32);

            // Add related dimensions based on change magnitude
            if change.abs() > 0.5 {
                // High magnitude change - add neighboring dimensions
                eidos.active_indices.insert((*dim as u32).saturating_sub(1));
                eidos.active_indices.insert((*dim as u32).saturating_add(1));
            }

            related_eidos.push(eidos);
        }

        if !related_eidos.is_empty() {
            flow.related_eidos = Some(related_eidos);
        }
    }

    /// Gets embeddings for all nodes in the CWM graph.
    ///
    /// This method extracts the current state embeddings from all nodes
    /// in the graph, which can be used for causal discovery.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Vec<f32>>, PandoraError>` - A matrix where each row is a node embedding
    pub fn get_all_node_embeddings(&self) -> Result<Vec<Vec<f32>>, PandoraError> {
        let mut embeddings = Vec::new();

        // Iterate through all nodes in the graph
        for node_idx in self.gnn.graph().node_indices() {
            if let Some(node_data) = self.gnn.graph().node_weight(node_idx) {
                // Extract the embedding from the node data
                // In a full implementation, this would be more sophisticated
                let embedding = node_data.clone(); // Assuming node_data is Vec<f32>
                embeddings.push(embedding);
            }
        }

        if embeddings.is_empty() {
            // Return a default embedding if no nodes exist
            embeddings.push(vec![0.0; self.gnn.config.hidden_dims]);
        }

        Ok(embeddings)
    }

    /// Checks if a causal link exists between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from_node` - The source node index
    /// * `to_node` - The target node index
    ///
    /// # Returns
    ///
    /// * `bool` - True if a causal link exists, false otherwise
    pub fn has_causal_link(&self, from_node: usize, to_node: usize) -> bool {
        use petgraph::graph::NodeIndex;

        let from_idx = NodeIndex::new(from_node);
        let to_idx = NodeIndex::new(to_node);

        // Check if both nodes exist
        if from_idx.index() >= self.gnn.graph().node_count()
            || to_idx.index() >= self.gnn.graph().node_count()
        {
            return false;
        }

        // Check if there's an edge between the nodes
        self.gnn.graph().find_edge(from_idx, to_idx).is_some()
    }

    /// Permanently adds a verified causal link to the model's graph.
    /// This represents the "crystallization" of a discovered law.
    ///
    /// # Arguments
    ///
    /// * `hypothesis` - The causal hypothesis to crystallize
    ///
    /// # Returns
    ///
    /// * `Result<(), PandoraError>` - Success or error
    pub fn crystallize_causal_link(
        &mut self,
        hypothesis: &crate::model::CausalHypothesis,
    ) -> Result<(), PandoraError> {
        use crate::gnn::types::{CausalEdge, CausalEdgeKind};
        use petgraph::graph::NodeIndex;

        // Ensure nodes exist, create if necessary
        let from_node = if hypothesis.from_node_index < self.gnn.node_count() {
            NodeIndex::new(hypothesis.from_node_index)
        } else {
            self.gnn.add_node(vec![0.0; self.gnn.config.input_dims])
        };

        let to_node = if hypothesis.to_node_index < self.gnn.node_count() {
            NodeIndex::new(hypothesis.to_node_index)
        } else {
            self.gnn.add_node(vec![0.0; self.gnn.config.input_dims])
        };

        // Convert CausalEdgeType to CausalEdgeKind
        let edge_kind = match hypothesis.edge_type {
            crate::model::CausalEdgeType::Direct => CausalEdgeKind::Cause,
            crate::model::CausalEdgeType::Indirect => CausalEdgeKind::Enable,
            crate::model::CausalEdgeType::Conditional => CausalEdgeKind::Precondition,
            crate::model::CausalEdgeType::Inhibitory => CausalEdgeKind::Inhibit,
        };

        // Create a high-weight, permanent edge
        let edge_weight = hypothesis.strength.abs().min(1.0);
        let edge = CausalEdge::with_metadata(
            edge_kind,
            edge_weight,
            format!(
                "Crystallized causal link: confidence={:.3}, strength={:.3}",
                hypothesis.confidence, hypothesis.strength
            ),
        );

        // Add the permanent edge
        self.gnn.add_edge(from_node, to_node, edge).map(|_| ())
    }
}

impl WorldModel for InterdependentCausalModel {
    /// Calculates the Minimum Description Length (MDL) of the model.
    /// This represents the complexity of the causal graph structure.
    fn get_mdl(&self) -> f64 {
        let node_count = self.gnn.node_count() as f64;
        let edge_count = self.gnn.edge_count() as f64;

        // MDL is based on the complexity of the graph structure
        // More nodes and edges = higher complexity = higher MDL
        let structural_complexity = node_count * 2.0 + edge_count * 1.5;

        // Add a base complexity for the model itself
        let base_complexity = 10.0;

        base_complexity + structural_complexity
    }

    /// Calculates prediction error for a given observation.
    /// This is a simplified implementation that uses the graph structure
    /// to estimate prediction accuracy.
    fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
        // Simplified prediction error calculation
        // In a full implementation, this would use the GNN to make predictions
        // and compare them with the actual flow data

        let node_count = self.gnn.node_count() as f64;
        let edge_count = self.gnn.edge_count() as f64;

        // Base error when no graph structure exists
        if node_count == 0.0 {
            return 1.0;
        }

        // Error decreases as graph becomes more connected and complex
        let connectivity = edge_count / node_count.max(1.0);
        let base_error = 0.5;

        // More connected graphs should have lower prediction error
        base_error * (1.0 - connectivity.min(1.0) * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn::types::GnnConfig;

    #[test]
    fn test_model_creation() {
        let config = GnnConfig::new(64, 128, 3);

        let model = InterdependentCausalModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_learn_relations() {
        let config = GnnConfig::new(64, 128, 3);

        let mut model = InterdependentCausalModel::new(config).unwrap();
        let flow = EpistemologicalFlow::default();

        let result = model.learn_relations(&flow);
        assert!(result.is_ok());
    }

    #[test]
    fn test_infer_context() {
        let config = GnnConfig::new(64, 128, 3);

        let model = InterdependentCausalModel::new(config).unwrap();
        let mut flow = EpistemologicalFlow::default();

        let result = model.infer_context(&mut flow);
        assert!(result.is_ok());
    }
}
