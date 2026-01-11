use num_complex::Complex;
use pandora_error::PandoraError;
use rustfft::FftPlanner;

/// Thực hiện phép toán "ràng buộc" (binding) hai vector bằng phép chập tròn (circular convolution).
/// Tương đương với phép nhân element-wise trong miền tần số.
/// Bind two vectors using circular convolution (element-wise multiplication in frequency domain)
///
/// Errors
/// - Returns InvalidSkillInput if vectors have different lengths or are empty
pub fn bind(x: &[f64], y: &[f64]) -> Result<Vec<f64>, PandoraError> {
    if x.len() != y.len() {
        return Err(PandoraError::InvalidSkillInput {
            skill_name: "vsa_hrr".into(),
            message: format!(
                "Vector dimension mismatch: x.len()={}, y.len()={}",
                x.len(),
                y.len()
            ),
        });
    }
    if x.is_empty() {
        return Err(PandoraError::InvalidSkillInput {
            skill_name: "vsa_hrr".into(),
            message: "Cannot bind empty vectors".into(),
        });
    }
    let n = x.len();

    let mut x_complex: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut y_complex: Vec<Complex<f64>> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    fft.process(&mut x_complex);
    fft.process(&mut y_complex);

    let mut result_complex: Vec<Complex<f64>> = x_complex
        .iter()
        .zip(y_complex.iter())
        .map(|(a, b)| a * b)
        .collect();

    ifft.process(&mut result_complex);

    Ok(result_complex.iter().map(|c| c.re / n as f64).collect())
}

/// Thực hiện phép toán "gộp" (bundling) nhiều vector bằng phép cộng.
pub fn bundle(vectors: &[Vec<f64>]) -> Result<Vec<f64>, PandoraError> {
    if vectors.is_empty() {
        return Err(PandoraError::InvalidSkillInput {
            skill_name: "vsa_hrr".into(),
            message: "Cannot bundle empty vector list".into(),
        });
    }
    let len = vectors[0].len();
    if len == 0 {
        return Err(PandoraError::InvalidSkillInput {
            skill_name: "vsa_hrr".into(),
            message: "Cannot bundle zero-length vectors".into(),
        });
    }
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != len {
            return Err(PandoraError::InvalidSkillInput {
                skill_name: "vsa_hrr".into(),
                message: format!(
                    "Vector dimension mismatch at index {}: expected {}, got {}",
                    i,
                    len,
                    v.len()
                ),
            });
        }
    }
    let mut sum = vec![0.0; len];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            sum[i] += val;
        }
    }
    Ok(sum)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_identity() {
        let _n = 4;
        let identity = vec![1.0, 0.0, 0.0, 0.0];
        let vec_a = vec![0.1, 0.2, 0.3, 0.4];
        let result = bind(&vec_a, &identity).unwrap();
        result.iter().zip(vec_a.iter()).for_each(|(a, b)| {
            assert!((a - b).abs() < 1e-9);
        });
    }

    #[test]
    fn test_bundle() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![4.0, 5.0, 6.0];
        let result = bundle(&[vec_a, vec_b]).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_bind_dimension_mismatch() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(bind(&x, &y).is_err());
    }

    #[test]
    fn test_bind_empty_vectors() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        assert!(bind(&x, &y).is_err());
    }

    #[test]
    fn test_bundle_empty() {
        let result = bundle(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_dimension_mismatch() {
        let vec_a = vec![1.0, 2.0];
        let vec_b = vec![1.0, 2.0, 3.0];
        let result = bundle(&[vec_a, vec_b]);
        assert!(result.is_err());
    }
}
