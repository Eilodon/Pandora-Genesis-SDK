use lazy_static::lazy_static;
use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Registry, TextEncoder};

// --- Định nghĩa các chỉ số (Metrics) theo đặc tả ---

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Performance Metrics
    pub static ref TASK_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("task_latency_seconds", "Độ trễ xử lý tác vụ")
    ).unwrap();
    pub static ref TASK_THROUGHPUT: Counter = Counter::new(
        "task_throughput_total", "Tổng số tác vụ đã xử lý"
    ).unwrap();
    pub static ref ERROR_RATE: Counter = Counter::new(
        "error_rate_total", "Tổng số lỗi xảy ra"
    ).unwrap();

    // Resource Metrics
    pub static ref CPU_UTILIZATION: Gauge = Gauge::new(
        "cpu_utilization_percent", "Tỷ lệ sử dụng CPU"
    ).unwrap();
    pub static ref MEMORY_USAGE: Gauge = Gauge::new(
        "memory_usage_mb", "Lượng bộ nhớ sử dụng (MB)"
    ).unwrap();

    // Quality Metrics
    pub static ref CONFIDENCE_SCORE: Histogram = Histogram::with_opts(
        HistogramOpts::new("confidence_score", "Phân phối điểm tự tin")
    ).unwrap();

    // Meta-Cognitive Metrics
    pub static ref SELF_CORRECTION_RATE: Counter = Counter::new(
        "self_correction_total", "Tổng số lần tự sửa lỗi được kích hoạt"
    ).unwrap();
}

/// Đăng ký tất cả các chỉ số với registry.
pub fn register_metrics() {
    REGISTRY.register(Box::new(TASK_LATENCY.clone())).unwrap();
    REGISTRY
        .register(Box::new(TASK_THROUGHPUT.clone()))
        .unwrap();
    REGISTRY.register(Box::new(ERROR_RATE.clone())).unwrap();
    REGISTRY
        .register(Box::new(CPU_UTILIZATION.clone()))
        .unwrap();
    REGISTRY.register(Box::new(MEMORY_USAGE.clone())).unwrap();
    REGISTRY
        .register(Box::new(CONFIDENCE_SCORE.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(SELF_CORRECTION_RATE.clone()))
        .unwrap();
}

/// Thu thập và trả về các chỉ số dưới dạng text cho Prometheus.
pub fn gather_metrics() -> String {
    let mut buffer = vec![];
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
