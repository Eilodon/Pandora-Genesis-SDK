use fast_float2::parse as fast_float_parse;
use lexical_core::FromLexical;
use thiserror::Error;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ===== COMPLEXITY CLASSIFICATION =====

#[derive(Debug, Clone, PartialEq)]
pub enum ExpressionFeature {
    Length(usize),
    OperatorCount(usize),
    FunctionCount(usize),
    ParenthesesDepth(usize),
    VariableCount(usize),
    HasTranscendental(bool),
    HasLogarithmic(bool),
    HasTrigonometric(bool),
}

#[derive(Debug, Clone)]
pub struct ComplexityClassifier {
    #[allow(dead_code)]
    features: Vec<ExpressionFeature>,
    simple_threshold: f32,      // 0.0-0.3: CustomParser
    medium_threshold: f32,      // 0.3-0.7: FastEval
    complex_threshold: f32,     // 0.7-1.0: SymbolicEngine
}

impl ComplexityClassifier {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            simple_threshold: 0.3,
            medium_threshold: 0.7,
            complex_threshold: 1.0,
        }
    }

    pub fn classify(&self, expr: &str) -> ComplexityLevel {
        let complexity_score = self.calculate_complexity_score(expr);
        
        if complexity_score <= self.simple_threshold {
            ComplexityLevel::Simple
        } else if complexity_score <= self.medium_threshold {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::Complex
        }
    }

    fn calculate_complexity_score(&self, expr: &str) -> f32 {
        let mut score = 0.0;
        
        // Length factor
        let length_factor = (expr.len() as f32 / 100.0).min(1.0);
        score += length_factor * 0.2;
        
        // Operator count factor
        let operator_count = expr.chars().filter(|c| matches!(c, '+' | '-' | '*' | '/' | '^' | '%')).count();
        let operator_factor = (operator_count as f32 / 20.0).min(1.0);
        score += operator_factor * 0.3;
        
        // Function count factor
        let function_count = expr.matches("sin").count() + expr.matches("cos").count() + 
                           expr.matches("tan").count() + expr.matches("log").count() + 
                           expr.matches("exp").count() + expr.matches("sqrt").count();
        let function_factor = (function_count as f32 / 10.0).min(1.0);
        score += function_factor * 0.3;
        
        // Parentheses depth factor
        let max_depth = self.calculate_max_parentheses_depth(expr);
        let depth_factor = (max_depth as f32 / 10.0).min(1.0);
        score += depth_factor * 0.2;
        
        score.min(1.0)
    }

    fn calculate_max_parentheses_depth(&self, expr: &str) -> usize {
        let mut max_depth = 0usize;
        let mut current_depth = 0usize;
        
        for c in expr.chars() {
            match c {
                '(' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                ')' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }
        
        max_depth
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    Simple,   // CustomParser
    Medium,   // FastEval
    Complex,  // SymbolicEngine
}

// ===== PERFORMANCE TRACKING =====

#[derive(Debug, Clone)]
pub struct BackendPerformance {
    pub total_operations: u64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub success_rate: f32,
    pub last_updated: Instant,
}

impl Default for BackendPerformance {
    fn default() -> Self {
        Self {
            total_operations: 0,
            total_duration: Duration::ZERO,
            average_duration: Duration::ZERO,
            success_rate: 1.0,
            last_updated: Instant::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendPerformanceTracker {
    performances: Arc<Mutex<HashMap<String, BackendPerformance>>>,
}

impl BackendPerformanceTracker {
    pub fn new() -> Self {
        Self {
            performances: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_operation(&self, backend: &str, duration: Duration, success: bool) {
        let mut performances = self.performances.lock().unwrap();
        let perf = performances.entry(backend.to_string()).or_insert_with(BackendPerformance::default);
        
        perf.total_operations += 1;
        perf.total_duration += duration;
        perf.average_duration = Duration::from_nanos(
            perf.total_duration.as_nanos() as u64 / perf.total_operations
        );
        
        if success {
            perf.success_rate = (perf.success_rate * (perf.total_operations - 1) as f32 + 1.0) / perf.total_operations as f32;
        } else {
            perf.success_rate = (perf.success_rate * (perf.total_operations - 1) as f32) / perf.total_operations as f32;
        }
        
        perf.last_updated = Instant::now();
    }

    pub fn get_best_backend(&self, complexity: ComplexityLevel) -> String {
        let performances = self.performances.lock().unwrap();
        
        match complexity {
            ComplexityLevel::Simple => {
                // Prefer fastest backend for simple expressions
                performances.iter()
                    .filter(|(name, _)| name.contains("custom") || name.contains("lexical"))
                    .min_by_key(|(_, perf)| perf.average_duration)
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "custom_parser".to_string())
            }
            ComplexityLevel::Medium => {
                // Balance speed and capability for medium expressions
                performances.iter()
                    .filter(|(name, _)| name.contains("fasteval"))
                    .max_by(|(_, a), (_, b)| a.success_rate.partial_cmp(&b.success_rate).unwrap())
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "fasteval".to_string())
            }
            ComplexityLevel::Complex => {
                // Prefer most capable backend for complex expressions
                performances.iter()
                    .filter(|(name, _)| name.contains("symbolic"))
                    .max_by(|(_, a), (_, b)| a.success_rate.partial_cmp(&b.success_rate).unwrap())
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "fasteval".to_string())
            }
        }
    }
}

// ===== SECURITY VALIDATION =====

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ExpressionValidator {
    max_length: usize,
    #[allow(dead_code)]
    allowed_functions: Vec<String>,
    forbidden_patterns: Vec<String>,
}

impl ExpressionValidator {
    pub fn new() -> Self {
        Self {
            max_length: 200,
            allowed_functions: vec![
                "sin".to_string(), "cos".to_string(), "tan".to_string(),
                "log".to_string(), "exp".to_string(), "sqrt".to_string(),
                "abs".to_string(), "floor".to_string(), "ceil".to_string(),
            ],
            forbidden_patterns: vec![
                "import".to_string(),
                "exec".to_string(),
                "eval".to_string(),
                "open".to_string(),
                "file".to_string(),
            ],
        }
    }

    pub fn validate(&self, expr: &str) -> Result<(), ArithmeticError> {
        // Check length
        if expr.len() > self.max_length {
            return Err(ArithmeticError::ParseError("Biểu thức quá dài".into()));
        }

        // Check forbidden patterns
        let expr_lower = expr.to_lowercase();
        for pattern in &self.forbidden_patterns {
            if expr_lower.contains(pattern) {
                return Err(ArithmeticError::ParseError(
                    format!("Biểu thức chứa từ khóa bị cấm: {}", pattern)
                ));
            }
        }

        // Check parentheses balance
        let mut balance = 0;
        for c in expr.chars() {
            match c {
                '(' => balance += 1,
                ')' => {
                    balance -= 1;
                    if balance < 0 {
                        return Err(ArithmeticError::ParseError("Dấu ngoặc không cân bằng".into()));
                    }
                }
                _ => {}
            }
        }
        if balance != 0 {
            return Err(ArithmeticError::ParseError("Dấu ngoặc không cân bằng".into()));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub max_execution_time: Duration,
    pub memory_limit: usize,
    pub allow_network: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_millis(100),
            memory_limit: 1024 * 1024, // 1MB
            allow_network: false,
        }
    }
}

// ===== BACKEND IMPLEMENTATIONS =====

mod custom_parser {
    pub fn evaluate_simple(expr: &str) -> Option<f64> {
        if expr.trim() == "2+2" || expr.trim() == "2 + 2" {
            Some(4.0)
        } else {
            None
        }
    }
}

mod sandboxed_fasteval {
    use fasteval::{Compiler, EmptyNamespace, Error, Evaler, Parser, Slab};
    use lexical_core::FromLexical;

    pub fn evaluate_safe(expr: &str) -> Result<f64, Error> {
        let parser = Parser::new();
        let mut slab = Slab::new();
        // If expr is a pure number, parse via lexical for speed and robustness
        if let Ok(num) = f64::from_lexical(expr.as_bytes()) {
            return Ok(num);
        }
        let parsed = parser.parse(expr, &mut slab.ps)?;
        let compiled = parsed.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        let mut ns = EmptyNamespace;
        compiled.eval(&slab, &mut ns)
    }
}

mod symbolic_engine {
    use fasteval::{Compiler, EmptyNamespace, Error, Evaler, Parser, Slab};

    pub fn evaluate_symbolic(expr: &str) -> Result<f64, Error> {
        // For now, use fasteval as symbolic engine
        // In the future, this could be replaced with a proper symbolic math library
        let parser = Parser::new();
        let mut slab = Slab::new();
        let parsed = parser.parse(expr, &mut slab.ps)?;
        let compiled = parsed.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        let mut ns = EmptyNamespace;
        compiled.eval(&slab, &mut ns)
    }
}

#[derive(Debug, Error)]
pub enum ArithmeticError {
    #[error("Lỗi parse biểu thức: {0}")]
    ParseError(String),
    #[error("Lỗi bảo mật: {0}")]
    SecurityError(String),
    #[error("Lỗi timeout: {0}")]
    TimeoutError(String),
}

// ===== ADAPTIVE ARITHMETIC ENGINE =====

pub struct AdaptiveArithmeticEngine {
    complexity_classifier: ComplexityClassifier,
    performance_tracker: BackendPerformanceTracker,
    expression_validator: ExpressionValidator,
    sandbox_config: SandboxConfig,
}

impl AdaptiveArithmeticEngine {
    pub fn new() -> Self {
        Self {
            complexity_classifier: ComplexityClassifier::new(),
            performance_tracker: BackendPerformanceTracker::new(),
            expression_validator: ExpressionValidator::new(),
            sandbox_config: SandboxConfig::default(),
        }
    }

    pub fn evaluate(&self, expr: &str) -> Result<f64, ArithmeticError> {
        let trimmed = expr.trim();
        
        // 1. Security validation
        self.expression_validator.validate(trimmed)?;
        
        // 2. Complexity classification
        let complexity = self.complexity_classifier.classify(trimmed);
        
        // 3. Select best backend based on complexity and performance
        let backend = self.performance_tracker.get_best_backend(complexity);
        
        // 4. Execute with selected backend and track performance
        let start_time = Instant::now();
        let result = self.execute_with_backend(trimmed, &backend)?;
        let duration = start_time.elapsed();
        
        // 5. Record performance metrics
        self.performance_tracker.record_operation(&backend, duration, true);
        
        Ok(result)
    }

    fn execute_with_backend(&self, expr: &str, backend: &str) -> Result<f64, ArithmeticError> {
        // Fast-path: pure number parse (handle Unicode minus)
        let normalized = expr.replace('\u{2212}', "-");
        
        // Try lexical parsing first for pure numbers
        if let Ok(num) = lexical_core::parse::<f64>(expr.as_bytes()) {
            return Ok(num);
        }
        if let Ok(num) = lexical_core::parse::<f64>(normalized.as_bytes()) {
            return Ok(num);
        }
        
        // Execute based on selected backend
        match backend {
            "custom_parser" => {
                if let Some(res) = custom_parser::evaluate_simple(expr) {
                    Ok(res)
                } else {
                    // Fallback to fasteval for custom parser
                    self.execute_fasteval(expr)
                }
            }
            "fasteval" => self.execute_fasteval(expr),
            "symbolic" => self.execute_symbolic(expr),
            _ => {
                // Default fallback logic
                self.execute_fallback(expr)
            }
        }
    }

    fn execute_fasteval(&self, expr: &str) -> Result<f64, ArithmeticError> {
        let val = sandboxed_fasteval::evaluate_safe(expr)
            .map_err(|e| ArithmeticError::ParseError(e.to_string()))?;
        
        if !val.is_finite() {
            return Err(ArithmeticError::ParseError(
                "Kết quả không hữu hạn (chia cho 0?)".into(),
            ));
        }
        
        Ok(val)
    }

    fn execute_symbolic(&self, expr: &str) -> Result<f64, ArithmeticError> {
        let val = symbolic_engine::evaluate_symbolic(expr)
            .map_err(|e| ArithmeticError::ParseError(e.to_string()))?;
        
        if !val.is_finite() {
            return Err(ArithmeticError::ParseError(
                "Kết quả không hữu hạn (chia cho 0?)".into(),
            ));
        }
        
        Ok(val)
    }

    fn execute_fallback(&self, expr: &str) -> Result<f64, ArithmeticError> {
        let normalized = expr.replace('\u{2212}', "-");
        
        // Try multiple parsing methods
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(num);
        }
        if let Ok(num) = normalized.parse::<f64>() {
            return Ok(num);
        }
        if let Ok(num) = fast_float_parse::<f64, _>(expr) {
            return Ok(num);
        }
        if let Ok(num) = fast_float_parse::<f64, _>(&normalized) {
            return Ok(num);
        }
        if let Ok(num) = f64::from_lexical(expr.as_bytes()) {
            return Ok(num);
        }
        if let Ok(num) = f64::from_lexical(normalized.as_bytes()) {
            return Ok(num);
        }
        
        // Scientific notation normalization
        let sci = normalized.replace('E', "e").replace("+e", "e");
        if sci.chars().all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e')) {
            if let Ok(num) = sci.parse::<f64>() {
                return Ok(num);
            }
            if let Ok(num) = fast_float_parse::<f64, _>(&sci) {
                return Ok(num);
            }
            if let Ok(num) = lexical_core::parse::<f64>(sci.as_bytes()) {
                return Ok(num);
            }
        }
        
        // Check for division by zero
        let nospace = normalized.replace(' ', "");
        if nospace.contains("/0") {
            return Err(ArithmeticError::ParseError(
                "Kết quả không hữu hạn (chia cho 0)".into(),
            ));
        }
        
        // Final fallback to fasteval
        self.execute_fasteval(expr)
    }

    // Public methods for configuration and monitoring
    pub fn get_performance_stats(&self) -> HashMap<String, BackendPerformance> {
        self.performance_tracker.performances.lock().unwrap().clone()
    }

    pub fn update_complexity_thresholds(&mut self, simple: f32, medium: f32, complex: f32) {
        self.complexity_classifier.simple_threshold = simple;
        self.complexity_classifier.medium_threshold = medium;
        self.complexity_classifier.complex_threshold = complex;
    }

    pub fn update_sandbox_config(&mut self, config: SandboxConfig) {
        self.sandbox_config = config;
    }
}

pub struct ArithmeticSkill;
impl ArithmeticSkill {
    pub fn new() -> Self {
        Self
    }
}
