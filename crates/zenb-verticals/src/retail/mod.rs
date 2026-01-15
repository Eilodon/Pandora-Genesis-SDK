//! Retail Analytics Vertical

pub mod emotion_analytics;
pub mod engagement;
pub mod timeline;

pub use emotion_analytics::{
    EmotionAnalytics, EmotionAnalyticsConfig, CustomerInsights, EngagementLevel, SessionStats,
};
pub use engagement::{EngagementScorer, EngagementResult};
pub use timeline::{EmotionTimeline, EmotionSnapshot};
