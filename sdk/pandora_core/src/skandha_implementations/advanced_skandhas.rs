use crate::interfaces::skandhas::*;
use crate::ontology::{DataEidos, EpistemologicalFlow, Vedana};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// Advanced RupaSkandha với khả năng phân tích metadata và timestamp
pub struct AdvancedRupaSkandha {
    pub enable_metadata: bool,
    pub enable_timestamp: bool,
}

impl AdvancedRupaSkandha {
    pub fn new(enable_metadata: bool, enable_timestamp: bool) -> Self {
        Self {
            enable_metadata,
            enable_timestamp,
        }
    }
}

impl Skandha for AdvancedRupaSkandha {
    fn name(&self) -> &'static str {
        "Advanced Rupa (Form)"
    }
}

impl RupaSkandha for AdvancedRupaSkandha {
    fn process_event(&self, event: Vec<u8>) -> EpistemologicalFlow {
        info!(
            "[{}] Tiếp nhận sự kiện với phân tích nâng cao.",
            self.name()
        );

        // Clone event để sử dụng trong metadata
        let event_clone = event.clone();

        let flow = EpistemologicalFlow::from_bytes(bytes::Bytes::from(event));

        // Thêm metadata nếu được bật
        if self.enable_metadata {
            let content = String::from_utf8_lossy(&event_clone);
            debug!(
                "[{}] Metadata: Kích thước={} bytes, Nội dung='{}'",
                self.name(),
                event_clone.len(),
                if content.len() > 50 {
                    &content[..50]
                } else {
                    &content
                }
            );
        }

        // Thêm timestamp nếu được bật
        if self.enable_timestamp {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            info!("[{}] Timestamp: {}", self.name(), timestamp);
        }

        flow
    }
}

/// Advanced VedanaSkandha với hệ thống scoring phức tạp hơn
pub struct AdvancedVedanaSkandha {
    pub karma_threshold: f32,
    pub enable_context_analysis: bool,
}

impl AdvancedVedanaSkandha {
    pub fn new(karma_threshold: f32, enable_context_analysis: bool) -> Self {
        Self {
            karma_threshold,
            enable_context_analysis,
        }
    }
}

impl Skandha for AdvancedVedanaSkandha {
    fn name(&self) -> &'static str {
        "Advanced Vedana (Feeling)"
    }
}

impl VedanaSkandha for AdvancedVedanaSkandha {
    fn feel(&self, flow: &mut EpistemologicalFlow) {
        info!(
            "[{}] Phân tích cảm thọ với hệ thống scoring nâng cao.",
            self.name()
        );

        let feeling = if let Some(rupa) = &flow.rupa {
            let content = String::from_utf8_lossy(rupa.as_ref());
            let mut karma_score = 0.0;

            // Phân tích từ khóa tích cực/tiêu cực
            let positive_keywords = ["success", "good", "excellent", "perfect", "great"];
            let negative_keywords = ["error", "fail", "bad", "terrible", "critical", "fatal"];

            for keyword in positive_keywords {
                if content.to_lowercase().contains(keyword) {
                    karma_score += 1.0;
                }
            }

            for keyword in negative_keywords {
                if content.to_lowercase().contains(keyword) {
                    karma_score -= 1.0;
                }
            }

            // Phân tích context nếu được bật
            if self.enable_context_analysis {
                karma_score += self.analyze_context(&content);
            }

            // Normalize karma score
            karma_score = karma_score.clamp(-2.0, 2.0);

            if karma_score > self.karma_threshold {
                info!(
                    "[{}] Cảm nhận 'Lạc Thọ' (karma: {:.2})",
                    self.name(),
                    karma_score
                );
                Vedana::Pleasant {
                    karma_weight: karma_score,
                }
            } else if karma_score < -self.karma_threshold {
                info!(
                    "[{}] Cảm nhận 'Khổ Thọ' (karma: {:.2})",
                    self.name(),
                    karma_score
                );
                Vedana::Unpleasant {
                    karma_weight: karma_score,
                }
            } else {
                info!(
                    "[{}] Cảm nhận 'Xả Thọ' (karma: {:.2})",
                    self.name(),
                    karma_score
                );
                Vedana::Neutral
            }
        } else {
            Vedana::Neutral
        };

        flow.vedana = Some(feeling);
    }
}

impl AdvancedVedanaSkandha {
    fn analyze_context(&self, content: &str) -> f32 {
        let mut context_score = 0.0;

        // Phân tích độ dài (quá ngắn hoặc quá dài có thể là dấu hiệu)
        let length = content.len();
        if length < 10 {
            context_score -= 0.5; // Có thể là lỗi
        } else if length > 1000 {
            context_score += 0.3; // Có thể là thông tin quan trọng
        }

        // Phân tích cấu trúc
        if content.contains(":") && content.contains(";") {
            context_score += 0.2; // Có cấu trúc log
        }

        context_score
    }
}

/// Advanced SannaSkandha với pattern recognition phức tạp
pub struct AdvancedSannaSkandha {
    pub pattern_threshold: f64,
    pub enable_semantic_analysis: bool,
}

impl AdvancedSannaSkandha {
    pub fn new(pattern_threshold: f64, enable_semantic_analysis: bool) -> Self {
        Self {
            pattern_threshold,
            enable_semantic_analysis,
        }
    }
}

impl Skandha for AdvancedSannaSkandha {
    fn name(&self) -> &'static str {
        "Advanced Sanna (Perception)"
    }
}

impl SannaSkandha for AdvancedSannaSkandha {
    fn perceive(&self, flow: &mut EpistemologicalFlow) {
        info!(
            "[{}] Nhận diện patterns với thuật toán nâng cao.",
            self.name()
        );

        let eidos = if let Some(rupa) = &flow.rupa {
            self.create_advanced_eidos(rupa)
        } else {
            DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 2048,
            }
        };

        flow.sanna = Some(eidos);

        // Tìm related patterns với thuật toán phức tạp hơn
        let related_eidos = match flow.sanna.as_ref() {
            Some(s) => self.find_advanced_patterns(s),
            None => Vec::new(),
        };
        flow.related_eidos = Some(smallvec::SmallVec::from_vec(related_eidos));

        let related_len = flow.related_eidos.as_ref().map(|v| v.len()).unwrap_or(0);
        info!(
            "[{}] Đã nhận diện {} patterns nâng cao.",
            self.name(),
            related_len
        );
    }
}

impl AdvancedSannaSkandha {
    fn create_advanced_eidos(&self, rupa: &[u8]) -> DataEidos {
        let content = String::from_utf8_lossy(rupa);
        let mut active_indices = HashSet::new();

        // Tạo indices dựa trên n-grams
        for i in 0..rupa.as_ref().len().saturating_sub(2) {
            let trigram = &rupa[i..i + 3];
            let hash = self.hash_bytes(trigram);
            active_indices.insert(hash % 2048);
        }

        // Thêm indices dựa trên semantic analysis
        if self.enable_semantic_analysis {
            for word in content.split_whitespace() {
                if word.len() > 3 {
                    let hash = self.hash_string(word);
                    active_indices.insert(hash % 2048);
                }
            }
        }

        DataEidos {
            active_indices: active_indices.into_iter().collect(),
            dimensionality: 2048,
        }
    }

    fn find_advanced_patterns(&self, eidos: &DataEidos) -> Vec<DataEidos> {
        let mut related = Vec::new();

        // Tạo patterns dựa trên similarity
        for similarity in [0.8, 0.6, 0.4] {
            let mut related_eidos = eidos.clone();
            let target_size = (eidos.active_indices.len() as f64 * similarity) as usize;

            // Thêm/bớt indices để tạo similarity
            while related_eidos.active_indices.len() < target_size {
                let new_idx = (related_eidos.active_indices.len() * 17) % 2048;
                related_eidos.active_indices.insert(new_idx as u32);
            }

            related.push(related_eidos);
        }

        related
    }

    fn hash_bytes(&self, bytes: &[u8]) -> u32 {
        let mut hash = 0u32;
        for &byte in bytes {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    fn hash_string(&self, s: &str) -> u32 {
        let mut hash = 0u32;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }
}

/// Advanced SankharaSkandha với decision tree phức tạp
pub struct AdvancedSankharaSkandha {
    pub decision_threshold: f64,
    pub enable_priority_system: bool,
}

impl AdvancedSankharaSkandha {
    pub fn new(decision_threshold: f64, enable_priority_system: bool) -> Self {
        Self {
            decision_threshold,
            enable_priority_system,
        }
    }
}

impl Skandha for AdvancedSankharaSkandha {
    fn name(&self) -> &'static str {
        "Advanced Sankhara (Formations)"
    }
}

impl SankharaSkandha for AdvancedSankharaSkandha {
    fn form_intent(&self, flow: &mut EpistemologicalFlow) {
        info!(
            "[{}] Khởi phát ý chỉ với hệ thống quyết định nâng cao.",
            self.name()
        );

        let intent = self.analyze_and_decide(flow);

        if let Some(intent) = intent {
            info!("[{}] Khởi phát ý chỉ: '{}'", self.name(), intent);
            flow.sankhara = Some(std::sync::Arc::<str>::from(intent));
        } else {
            info!("[{}] Không có ý chỉ nào được khởi phát.", self.name());
        }
    }
}

impl AdvancedSankharaSkandha {
    fn analyze_and_decide(&self, flow: &EpistemologicalFlow) -> Option<String> {
        let mut decision_score = 0.0;
        let mut intent = None;

        // Phân tích Vedana
        if let Some(vedana) = &flow.vedana {
            match vedana {
                Vedana::Pleasant { karma_weight } => {
                    decision_score += *karma_weight as f64;
                    if *karma_weight > 1.0 {
                        intent = Some("CONTINUE_SUCCESS".to_string());
                    }
                }
                Vedana::Unpleasant { karma_weight } => {
                    decision_score += *karma_weight as f64;
                    if *karma_weight < -1.0 {
                        intent = Some("TAKE_CORRECTIVE_ACTION".to_string());
                    } else {
                        intent = Some("MONITOR_CLOSELY".to_string());
                    }
                }
                Vedana::Neutral => {
                    decision_score += 0.1;
                    intent = Some("MAINTAIN_STATUS".to_string());
                }
            }
        }

        // Phân tích Sanna patterns
        if let Some(sanna) = &flow.sanna {
            let pattern_complexity =
                sanna.active_indices.len() as f64 / sanna.dimensionality as f64;
            decision_score += pattern_complexity * 2.0;

            if pattern_complexity > 0.1 {
                intent = Some("ANALYZE_PATTERN".to_string());
            }
        }

        // Phân tích related patterns
        if let Some(related) = &flow.related_eidos {
            if related.len() > 2 {
                decision_score += 0.5;
                intent = Some("INVESTIGATE_RELATIONS".to_string());
            }
        }

        // Hệ thống priority nếu được bật
        if self.enable_priority_system {
            intent = self.apply_priority_system(intent, decision_score);
        }

        if decision_score.abs() > self.decision_threshold {
            intent
        } else {
            None
        }
    }

    fn apply_priority_system(&self, intent: Option<String>, score: f64) -> Option<String> {
        match intent {
            Some(ref i) if i.contains("CRITICAL") || i.contains("CORRECTIVE") => {
                Some("HIGH_PRIORITY_ACTION".to_string())
            }
            Some(ref i) if i.contains("ANALYZE") || i.contains("INVESTIGATE") => {
                Some("MEDIUM_PRIORITY_ANALYSIS".to_string())
            }
            Some(i) if score > 0.5 => Some(i),
            _ => None,
        }
    }
}

/// Advanced VinnanaSkandha với synthesis phức tạp
pub struct AdvancedVinnanaSkandha {
    pub synthesis_threshold: f64,
    pub enable_metacognition: bool,
}

impl AdvancedVinnanaSkandha {
    pub fn new(synthesis_threshold: f64, enable_metacognition: bool) -> Self {
        Self {
            synthesis_threshold,
            enable_metacognition,
        }
    }
}

impl Skandha for AdvancedVinnanaSkandha {
    fn name(&self) -> &'static str {
        "Advanced Vinnana (Consciousness)"
    }
}

impl VinnanaSkandha for AdvancedVinnanaSkandha {
    fn synthesize(&self, flow: &EpistemologicalFlow) -> Option<Vec<u8>> {
        info!(
            "[{}] Tổng hợp nhận thức với thuật toán nâng cao.",
            self.name()
        );

        let synthesis_score = self.calculate_synthesis_score(flow);

        if synthesis_score > self.synthesis_threshold {
            let conscious_event = self.create_conscious_event(flow, synthesis_score);
            info!(
                "[{}] Tái sinh sự kiện với score: {:.2}",
                self.name(),
                synthesis_score
            );
            Some(conscious_event)
        } else {
            info!(
                "[{}] Vòng lặp kết thúc (score: {:.2})",
                self.name(),
                synthesis_score
            );
            None
        }
    }
}

impl AdvancedVinnanaSkandha {
    fn calculate_synthesis_score(&self, flow: &EpistemologicalFlow) -> f64 {
        let mut score = 0.0;

        // Score từ Vedana
        if let Some(vedana) = &flow.vedana {
            match vedana {
                Vedana::Pleasant { karma_weight } => score += *karma_weight as f64,
                Vedana::Unpleasant { karma_weight } => score += (*karma_weight as f64).abs(),
                Vedana::Neutral => score += 0.1,
            }
        }

        // Score từ Sanna complexity
        if let Some(sanna) = &flow.sanna {
            let complexity = sanna.active_indices.len() as f64 / sanna.dimensionality as f64;
            score += complexity * 3.0;
        }

        // Score từ related patterns
        if let Some(related) = &flow.related_eidos {
            score += related.len() as f64 * 0.2;
        }

        // Score từ intent
        if flow.sankhara.is_some() {
            score += 1.0;
        }

        // Metacognition bonus
        if self.enable_metacognition {
            score += 0.5;
        }

        score
    }

    fn create_conscious_event(&self, flow: &EpistemologicalFlow, score: f64) -> Vec<u8> {
        let mut event_parts = Vec::new();

        // Thêm intent nếu có
        if let Some(intent) = &flow.sankhara {
            event_parts.push(format!("Intent: {}", intent));
        }

        // Thêm synthesis score
        event_parts.push(format!("SynthesisScore: {:.2}", score));

        // Thêm pattern info
        if let Some(sanna) = &flow.sanna {
            event_parts.push(format!(
                "Patterns: {}/{}",
                sanna.active_indices.len(),
                sanna.dimensionality
            ));
        }

        // Thêm related patterns count
        if let Some(related) = &flow.related_eidos {
            event_parts.push(format!("RelatedPatterns: {}", related.len()));
        }

        // Thêm timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        event_parts.push(format!("Timestamp: {}", timestamp));

        let conscious_event = format!("AdvancedConsciousness: {}", event_parts.join(", "));
        conscious_event.into_bytes()
    }
}
