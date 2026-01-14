/// Pre-defined static intents to avoid allocations
pub mod constants {
    pub const REPORT_ERROR: &str = "REPORT_ERROR";
    pub const CONTINUE_SUCCESS: &str = "CONTINUE_SUCCESS";
    pub const TAKE_CORRECTIVE_ACTION: &str = "TAKE_CORRECTIVE_ACTION";
    pub const MONITOR_CLOSELY: &str = "MONITOR_CLOSELY";
    pub const MAINTAIN_STATUS: &str = "MAINTAIN_STATUS";
    pub const ANALYZE_PATTERN: &str = "ANALYZE_PATTERN";
    pub const INVESTIGATE_RELATIONS: &str = "INVESTIGATE_RELATIONS";
    pub const HIGH_PRIORITY_ANALYSIS: &str = "HIGH_PRIORITY_ANALYSIS";
    pub const MEDIUM_PRIORITY_ANALYSIS: &str = "MEDIUM_PRIORITY_ANALYSIS";
}
