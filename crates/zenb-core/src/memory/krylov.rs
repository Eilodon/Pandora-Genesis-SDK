//! Krylov Subspace Acceleration for Holographic Memory
//!
//! Implements Lanczos algorithm for approximating matrix functions (like exp(iH))
//! in a low-dimensional Krylov subspace (~30D) instead of the full memory dimension (512D+).
//!
//! # Mathematical Foundation (Krylov Subspace)
//! Instead of computing e^A directly (O(N^3)), we project A onto a small subspace K_m,
//! compute the exponential there (O(m^3)), and project back (O(N*m)).
//!
//! Approximation: e^{tA}v ≈ V_m e^{tT_m} e_1
//! where V_m is the basis of K_m and T_m is the tridiagonal projection of A.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex32;

/// Projector for Krylov subspace operations.
pub struct KrylovProjector {
    /// Dimension of the subspace (m << N)
    k: usize,
    /// Lanczos vectors (basis V_m) [dim x k] flattened? Or vec of vectors?
    /// Storing as Vec<DVector> for clarity
    v: Vec<DVector<Complex32>>,
}

impl KrylovProjector {
    /// Create a new projector with subspace dimension k.
    ///
    /// # Arguments
    /// * `k` - Subspace dimension (typically 30)
    pub fn new(k: usize) -> Self {
        Self {
            k,
            v: Vec::with_capacity(k + 1),
        }
    }

    /// Default configuration
    pub fn default() -> Self {
        Self::new(30)
    }

    /// Approximate the matrix exponential action e^(iH) * v using Krylov subspace.
    ///
    /// # Arguments
    /// * `h_op` - Linear operator representing H (matrix-vector multiplication function)
    /// * `start_vec` - Initial vector v (memory state)
    /// * `dt` - Time step for evolution (or scaling factor)
    ///
    /// # Returns
    /// Approximated result e^(iH * dt) * v
    pub fn exp_time_evolution<F>(
        &mut self,
        h_op: F,
        start_vec: &[Complex32],
        dt: f32,
    ) -> Vec<Complex32>
    where
        F: Fn(&[Complex32]) -> Vec<Complex32>,
    {
        let dim = start_vec.len();
        self.v.clear();

        // 1. Initialize first Lanczos vector v1 = v / |v|
        let v0 = DVector::from_vec(start_vec.to_vec());
        let norm = v0.norm();
        if norm < 1e-10 {
            return start_vec.to_vec();
        }

        self.v.push(v0 / Complex32::new(norm, 0.0));

        // 2. Lanczos Iteration
        // Build basis V_m and tridiagonal matrix T_m
        // T_m is [alpha, beta]
        //        [beta, alpha]

        let mut alphas = Vec::with_capacity(self.k);
        let mut betas = Vec::with_capacity(self.k);

        for j in 0..self.k {
            // w = A * v_j
            let w_vec = h_op(self.v[j].as_slice());
            let mut w = DVector::from_vec(w_vec);

            // alpha_j = v_j^H * w
            let _alpha = self.v[j].dot(&w); // dot implies conjugate of first arg in nalgebra for complex?
                                           // nalgebra dot: a.dot(&b) = sum(a_i * b_i). Inner product usually requires conjugation.
                                           // For Complex, nalgebra's dot is standard dot product (bilinear), NOT inner product (sesquilinear)?
                                           // We need Hermitian inner product: v^H * w = conj(v) dot w.
                                           // Correction: nalgebra DVector::dot(&self, rhs) is unconjugated.
                                           // DVector::dotc(&self, rhs) is conjugated (self^H * rhs).
            let alpha_val = self.v[j].dotc(&w);
            alphas.push(alpha_val);

            // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
            w = w - self.v[j].clone() * alpha_val;
            if j > 0 {
                w = w - self.v[j - 1].clone() * betas[j - 1];
            }

            // beta_j = |w|
            let beta = w.norm();

            if beta < 1e-10 {
                // Invariant subspace found, stop early
                break;
            }

            if j < self.k - 1 {
                betas.push(Complex32::new(beta, 0.0)); // Beta is real-valued in theory, but we store as complex
                self.v.push(w / Complex32::new(beta, 0.0));
            }
        }

        // 3. Construct T matrix (Tridiagonal)
        // This small matrix exponentiation is cheap.
        let m = alphas.len();
        let mut t_m = DMatrix::zeros(m, m);
        for i in 0..m {
            t_m[(i, i)] = alphas[i];
            if i < m - 1 && i < betas.len() {
                t_m[(i + 1, i)] = betas[i];
                t_m[(i, i + 1)] = betas[i].conj(); // Hermitian symmetry
            }
        }

        // 4. Compute e^{i * T_m * dt}
        // Actually, we want e^{iH t}, and H is mapped to T_m?
        // Yes, if A = iH, then we compute e^{T_m}.
        // If we passed H directly, we compute e^{i * T_m * dt}.
        // Let's assume input H is the Hamiltonian itself, and we want unitary evolution e^{-i H dt}.
        // The argument `dt` helps scale.
        // We compute exp( -i * t_m * dt ).

        let evolution_op = t_m * Complex32::new(0.0, -dt);
        let exp_t = evolution_op.exp(); // Matrix exponential

        // 5. Project result back: res ≈ |v0| * V_m * exp_t * e1
        // e1 = [1, 0, ... 0]^T
        let mut e1 = DVector::zeros(m);
        e1[0] = Complex32::new(1.0, 0.0);

        let projected_coeffs = exp_t * e1;

        // Reconstruct vector: sum_i (coeff_i * v_i)
        let mut final_vec = DVector::zeros(dim);
        for (i, coeff) in projected_coeffs.iter().enumerate() {
            final_vec += &self.v[i] * *coeff;
        }

        final_vec = final_vec * Complex32::new(norm, 0.0);

        // Convert to Vec<Complex32>
        final_vec.as_slice().to_vec()
    }
}
