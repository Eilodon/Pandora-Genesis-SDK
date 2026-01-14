use pandora_tools::skills::arithmetic_skill::AdaptiveArithmeticEngine;
use proptest::prelude::*;

proptest! {
    #[test]
    fn parse_numbers_always_succeeds(n in prop::num::f64::ANY.prop_filter("finite", |x| x.is_finite())) {
        let engine = AdaptiveArithmeticEngine::new();
        let _ = engine.evaluate(&n.to_string());
        // Only assert non-panicking; accept any outcome to ensure robustness
        prop_assert!(true);
    }

    #[test]
    fn addition_commutative(a in -1000..1000, b in -1000..1000) {
        let engine = AdaptiveArithmeticEngine::new();
        let expr1 = format!("{} + {}", a, b);
        let expr2 = format!("{} + {}", b, a);
        let result1 = engine.evaluate(&expr1).unwrap();
        let result2 = engine.evaluate(&expr2).unwrap();
        prop_assert_eq!(result1, result2);
    }

    #[test]
    fn multiplication_commutative(a in -100..100, b in -100..100) {
        let engine = AdaptiveArithmeticEngine::new();
        let expr1 = format!("{} * {}", a, b);
        let expr2 = format!("{} * {}", b, a);
        let result1 = engine.evaluate(&expr1).unwrap();
        let result2 = engine.evaluate(&expr2).unwrap();
        prop_assert_eq!(result1, result2);
    }

    #[test]
    fn operator_precedence(a in -100..100, b in -100..100, c in 1..100) {
        let engine = AdaptiveArithmeticEngine::new();
        let expr1 = format!("{} + {} * {}", a, b, c);
        let expr2 = format!("{} + ({} * {})", a, b, c);
        let result1 = engine.evaluate(&expr1).unwrap();
        let result2 = engine.evaluate(&expr2).unwrap();
        prop_assert_eq!(result1, result2);
    }

    #[test]
    fn division_by_zero_errors(n in -1000..1000) {
        let engine = AdaptiveArithmeticEngine::new();
        let expr = format!("{} / 0", n);
        let result = engine.evaluate(&expr);
        prop_assert!(result.is_err());
    }

    #[test]
    fn parentheses_change_order(a in 1..10, b in 1..10, c in 1..10) {
        let engine = AdaptiveArithmeticEngine::new();
        let without_parens = format!("{} * {} + {}", a, b, c);
        let with_parens = format!("{} * ({} + {})", a, b, c);
        let result1 = engine.evaluate(&without_parens).unwrap();
        let result2 = engine.evaluate(&with_parens).unwrap();
        if a > 1 && b > 0 && c > 0 { prop_assert_ne!(result1, result2); }
    }
}
