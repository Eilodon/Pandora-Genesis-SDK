use pandora_cwm::vsa::hrr::{bind, bundle};
use proptest::prelude::*;

fn vec_f64(len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1000.0..1000.0, len)
}

#[test]
fn bind_is_commutative() {
    proptest!(|(x in vec_f64(16), y in vec_f64(16))| {
        let xy = bind(&x, &y).unwrap();
        let yx = bind(&y, &x).unwrap();
        for (a, b) in xy.iter().zip(yx.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    });
}

#[test]
fn bind_identity_property() {
    proptest!(|(v in vec_f64(16))| {
        let mut identity = vec![0.0; 16];
        identity[0] = 1.0;
        let result = bind(&v, &identity).unwrap();
        for (a, b) in result.iter().zip(v.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    });
}

#[test]
fn bind_preserves_dimension() {
    proptest!(|(x in vec_f64(32), y in vec_f64(32))| {
        let result = bind(&x, &y).unwrap();
        prop_assert_eq!(result.len(), 32);
    });
}

#[test]
fn bundle_is_associative() {
    proptest!(|(a in vec_f64(8), b in vec_f64(8), c in vec_f64(8))| {
        let abc = bundle(&[a.clone(), b.clone(), c.clone()]).unwrap();
        let ab = bundle(&[a, b]).unwrap();
        let ab_c = bundle(&[ab, c]).unwrap();
        for (x, y) in abc.iter().zip(ab_c.iter()) {
            prop_assert!((x - y).abs() < 1e-6);
        }
    });
}

#[test]
fn bundle_single_is_identity() {
    proptest!(|(v in vec_f64(16))| {
        let result = bundle(std::slice::from_ref(&v)).unwrap();
        prop_assert_eq!(result, v);
    });
}

#[test]
fn bind_rejects_mismatched_dimensions() {
    proptest!(|(len1 in 1usize..100, len2 in 1usize..100)| {
        prop_assume!(len1 != len2);
        let x = vec![1.0; len1];
        let y = vec![1.0; len2];
        prop_assert!(bind(&x, &y).is_err());
    });
}

#[test]
fn bind_rejects_empty_vectors() {
    // No inputs needed; just one deterministic check per case
    let x: Vec<f64> = vec![];
    let y: Vec<f64> = vec![];
    assert!(bind(&x, &y).is_err());
}
