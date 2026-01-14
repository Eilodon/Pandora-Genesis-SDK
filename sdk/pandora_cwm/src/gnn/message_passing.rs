#[cfg(feature = "ml")]
use ndarray::Array2;

/// Performs mean aggregation for message passing in Graph Neural Networks.
///
/// This function aggregates neighbor features using mean pooling, which is a common
/// operation in GNN message passing layers.
///
/// # Arguments
///
/// * `adj` - Adjacency matrix (n x n) where adj[i,j] > 0 indicates an edge
/// * `features` - Node features (n x feature_dim)
///
/// # Returns
///
/// * `Array2<f32>` - Aggregated features (n x feature_dim)
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "ml")]
/// # {
/// use pandora_cwm::gnn::message_passing::aggregate_mean;
/// use ndarray::arr2;
///
/// let adj = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
/// let features = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
/// let aggregated = aggregate_mean(&adj, &features);
/// assert_eq!(aggregated.shape(), &[2, 2]);
/// # }
/// ```
#[cfg(feature = "ml")]
pub fn aggregate_mean(adj: &Array2<f32>, features: &Array2<f32>) -> Array2<f32> {
    let n = adj.nrows();
    let mut out = Array2::<f32>::zeros((n, features.ncols()));
    for i in 0..n {
        let row = adj.row(i);
        let mut sum = vec![0.0f32; features.ncols()];
        let mut count = 0.0f32;
        for (j, w) in row.iter().enumerate() {
            if *w > 0.0 {
                let fj = features.row(j);
                for c in 0..sum.len() {
                    sum[c] += fj[c];
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for val in sum.iter_mut() {
                *val /= count;
            }
            for (c, &val) in sum.iter().enumerate() {
                out[[i, c]] = val;
            }
        } else {
            let fi = features.row(i);
            for c in 0..sum.len() {
                out[[i, c]] = fi[c];
            }
        }
    }
    out
}
