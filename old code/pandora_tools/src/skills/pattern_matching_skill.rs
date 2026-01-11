use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PatternError {
    #[error("Input không hợp lệ: {0}")]
    InvalidInput(String),
    #[error("Ngưỡng hỗ trợ (support) phải lớn hơn 0")]
    InvalidSupportThreshold,
}

/// Đại diện cho một sự kiện đơn lẻ trong một chuỗi hành vi
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Event {
    /// Định danh của loại sự kiện (ví dụ: "open_app", "send_message")
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    // Có thể thêm dữ liệu khác nếu cần
}

/// Đại diện cho một chuỗi các sự kiện của một người dùng
#[derive(Debug, Clone)]
pub struct Sequence {
    pub events: Vec<Event>,
}

/// Đại diện cho một mẫu hành vi được tìm thấy
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Chuỗi các loại sự kiện tạo nên mẫu (ví dụ: ["open_app:music", "open_app:news"])
    pub items: Vec<String>,
    /// Số lần mẫu này xuất hiện trong dữ liệu
    pub support: usize,
}

/// Đại diện cho một hành động được dự đoán
#[derive(Debug, Clone)]
pub struct ActionPrediction {
    pub predicted_action: String,
    pub confidence: f32,
}

/// Một cấu trúc tạm thời để xây dựng projected database trong PrefixSpan
struct ProjectedSequence<'a> {
    // Suffix của chuỗi gốc
    suffix: &'a [Event],
}

/// Engine khai phá và dự báo mẫu hành vi
pub struct TemporalPrefixSpanEngine {
    min_support: usize,
    max_pattern_length: usize,
    /// Các mẫu đã được khai phá và lưu trữ
    mined_patterns: Vec<TemporalPattern>,
}

impl TemporalPrefixSpanEngine {
    /// Khởi tạo một engine mới với các tham số cấu hình.
    pub fn new(min_support: usize, max_pattern_length: usize) -> Result<Self, PatternError> {
        if min_support == 0 {
            return Err(PatternError::InvalidSupportThreshold);
        }
        Ok(Self {
            min_support,
            max_pattern_length,
            mined_patterns: Vec::new(),
        })
    }

    /// Khai phá các mẫu từ một tập hợp các chuỗi sự kiện.
    /// Phương thức này sẽ cập nhật `self.mined_patterns` bên trong.
    pub fn mine_patterns(&mut self, sequences: &[Sequence]) -> Result<(), PatternError> {
        self.mined_patterns.clear();
        let mut frequent_patterns = Vec::new();

        // Bước 1: Tìm các item (event_type) phổ biến ban đầu và support của chúng (mỗi chuỗi đếm tối đa 1 lần)
        let mut item_counts: HashMap<String, usize> = HashMap::new();
        for seq in sequences {
            let mut seen_in_seq: HashMap<&str, bool> = HashMap::new();
            for event in &seq.events {
                if seen_in_seq.insert(&event.event_type, true).is_none() {
                    *item_counts.entry(event.event_type.clone()).or_insert(0) += 1;
                }
            }
        }

        let frequent_items: BTreeMap<String, usize> = item_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_support)
            .collect();

        // Bước 2: Với mỗi item phổ biến, bắt đầu quá trình khai phá đệ quy
        for (item, support) in frequent_items {
            let prefix = vec![item.clone()];
            frequent_patterns.push(TemporalPattern {
                items: prefix.clone(),
                support,
            });

            // Xây dựng projected database cho item ban đầu
            let mut projected_db = Vec::new();
            for seq in sequences {
                if let Some(pos) = seq.events.iter().position(|e| e.event_type == item) {
                    if pos + 1 < seq.events.len() {
                        projected_db.push(ProjectedSequence {
                            suffix: &seq.events[pos + 1..],
                        });
                    }
                }
            }

            if !projected_db.is_empty() {
                self.mine_recursive(projected_db, prefix, &mut frequent_patterns);
            }
        }

        self.mined_patterns = frequent_patterns;
        Ok(())
    }

    /// Hàm đệ quy của thuật toán PrefixSpan
    fn mine_recursive<'a>(
        &self,
        projected_db: Vec<ProjectedSequence<'a>>,
        prefix: Vec<String>,
        frequent_patterns: &mut Vec<TemporalPattern>,
    ) {
        if prefix.len() >= self.max_pattern_length {
            return;
        }

        // Đếm các item phổ biến trong projected_db (mỗi sequence tối đa 1 lần)
        let mut item_counts: HashMap<String, usize> = HashMap::new();
        for p_seq in &projected_db {
            let mut seen_in_seq: HashMap<&str, bool> = HashMap::new();
            for event in p_seq.suffix {
                if seen_in_seq.insert(&event.event_type, true).is_none() {
                    *item_counts.entry(event.event_type.clone()).or_insert(0) += 1;
                }
            }
        }

        let frequent_items: BTreeMap<String, usize> = item_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.min_support)
            .collect();

        // Với mỗi item phổ biến, mở rộng prefix và tiếp tục đệ quy
        for (item, support) in frequent_items {
            let mut new_prefix = prefix.clone();
            new_prefix.push(item.clone());

            frequent_patterns.push(TemporalPattern {
                items: new_prefix.clone(),
                support,
            });

            // Xây dựng projected database tiếp theo
            let mut new_projected_db = Vec::new();
            for p_seq in &projected_db {
                if let Some(pos) = p_seq.suffix.iter().position(|e| e.event_type == item) {
                    if pos + 1 < p_seq.suffix.len() {
                        new_projected_db.push(ProjectedSequence {
                            suffix: &p_seq.suffix[pos + 1..],
                        });
                    }
                }
            }

            if !new_projected_db.is_empty() {
                self.mine_recursive(new_projected_db, new_prefix, frequent_patterns);
            }
        }
    }

    /// Dự đoán hành động tiếp theo dựa trên các mẫu đã học.
    pub fn predict_next_action(
        &self,
        current_sequence: &[Event],
    ) -> Result<Vec<ActionPrediction>, PatternError> {
        if current_sequence.is_empty() {
            return Ok(Vec::new());
        }

        let current_items: Vec<String> = current_sequence
            .iter()
            .map(|e| e.event_type.clone())
            .collect();
        let mut predictions: HashMap<String, usize> = HashMap::new();

        // Duyệt qua các mẫu đã học
        for pattern in &self.mined_patterns {
            if pattern.items.len() > current_items.len()
                && pattern.items.starts_with(&current_items)
            {
                let next_action = &pattern.items[current_items.len()];
                *predictions.entry(next_action.clone()).or_insert(0) += pattern.support;
            }
        }

        let mut sorted_predictions: Vec<ActionPrediction> = predictions
            .into_iter()
            .map(|(action, total_support)| ActionPrediction {
                predicted_action: action,
                confidence: total_support as f32,
            })
            .collect();

        sorted_predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_predictions)
    }

    /// Truy cập danh sách mẫu đã khai phá (hỗ trợ kiểm thử)
    pub fn patterns(&self) -> &Vec<TemporalPattern> {
        &self.mined_patterns
    }
}

pub struct PatternMatchingSkill;
impl PatternMatchingSkill {
    pub fn new() -> Self {
        Self
    }
}
