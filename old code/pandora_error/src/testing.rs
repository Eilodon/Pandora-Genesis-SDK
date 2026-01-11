#[cfg(test)]
use crate::PandoraError;

#[cfg(test)]
pub fn assert_error_code<T>(result: Result<T, PandoraError>, expected_code: &str) {
    match result {
        Err(e) => assert_eq!(e.code(), expected_code),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[cfg(test)]
pub fn assert_retryable<T>(result: Result<T, PandoraError>) {
    match result {
        Err(e) => assert!(e.is_retryable(), "Error should be retryable: {}", e),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[cfg(test)]
pub fn assert_transient<T>(result: Result<T, PandoraError>) {
    match result {
        Err(e) => assert!(e.is_transient(), "Error should be transient: {}", e),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_error_code() {
        let result: Result<(), PandoraError> = Err(PandoraError::config("test"));
        assert_error_code(result, "CONFIG_ERROR");
    }

    #[test]
    #[should_panic]
    fn test_assert_error_code_wrong() {
        let result: Result<(), PandoraError> = Err(PandoraError::config("test"));
        assert_error_code(result, "WRONG_CODE");
    }

    #[test]
    fn test_assert_retryable() {
        let result: Result<(), PandoraError> = Err(PandoraError::Timeout { operation: "test".into(), timeout_ms: 1000 });
        assert_retryable(result);
    }
}


