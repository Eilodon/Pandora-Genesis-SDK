//! Education Vertical: Exam Proctoring

pub mod proctoring;

pub use proctoring::{
    ExamProctoring, ProctoringConfig, ProctoringResult, ProctoringAction, Violation, SessionStats,
};
