use crate::intents;
use crate::interfaces::skandhas::*;
use crate::ontology::{EpistemologicalFlow, Vedana};
use bytes::Bytes;
use tracing::{info, warn};

// --- 1. Sắc Uẩn ---
/// Basic implementation of Rupa (Form) Skandha for raw sensory input processing.
///
/// This skandha converts raw byte data into an `EpistemologicalFlow` for further
/// processing by subsequent skandhas in the cognitive pipeline.
///
/// # Examples
///
/// ```rust
/// use pandora_core::skandha_implementations::basic_skandhas::BasicRupaSkandha;
/// use pandora_core::interfaces::skandhas::RupaSkandha;
///
/// let rupa = BasicRupaSkandha;
/// let event = b"test event".to_vec();
/// let flow = rupa.process_event(event);
/// assert!(flow.rupa.is_some());
/// ```
pub struct BasicRupaSkandha;
impl Skandha for BasicRupaSkandha {
    fn name(&self) -> &'static str {
        "Basic Rupa (Form)"
    }
}
impl RupaSkandha for BasicRupaSkandha {
    #[inline]
    fn process_event(&self, event: Vec<u8>) -> EpistemologicalFlow {
        info!("[{}] Tiếp nhận sự kiện nguyên thủy.", self.name());
        EpistemologicalFlow::from_bytes(Bytes::from(event))
    }
}

// --- 2. Thọ Uẩn ---
/// Basic implementation of Vedana (Feeling) Skandha for moral/emotional valence assignment.
///
/// This skandha analyzes the content of events and assigns emotional valence:
/// - Events containing "error" → Unpleasant feeling (negative karma)
/// - Other events → Neutral feeling
///
/// # Examples
///
/// ```rust
/// use pandora_core::skandha_implementations::basic_skandhas::BasicVedanaSkandha;
/// use pandora_core::interfaces::skandhas::VedanaSkandha;
/// use pandora_core::ontology::EpistemologicalFlow;
/// use bytes::Bytes;
///
/// let vedana = BasicVedanaSkandha;
/// let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"error occurred".as_ref()));
/// vedana.feel(&mut flow);
/// assert!(matches!(flow.vedana, Some(pandora_core::ontology::Vedana::Unpleasant { .. })));
/// ```
pub struct BasicVedanaSkandha;
impl Skandha for BasicVedanaSkandha {
    fn name(&self) -> &'static str {
        "Basic Vedana (Feeling)"
    }
}
impl VedanaSkandha for BasicVedanaSkandha {
    fn feel(&self, flow: &mut EpistemologicalFlow) {
        // Logic đạo đức đơn giản: Nếu event chứa từ "error", gán "Khổ Thọ".
        let feeling = if let Some(rupa) = &flow.rupa {
            if String::from_utf8_lossy(rupa.as_ref()).contains("error") {
                info!("[{}] Cảm nhận 'Khổ Thọ' từ sự kiện.", self.name());
                Vedana::Unpleasant { karma_weight: -1.0 }
            } else {
                info!("[{}] Cảm nhận 'Xả Thọ' từ sự kiện.", self.name());
                Vedana::Neutral
            }
        } else {
            Vedana::Neutral
        };
        flow.vedana = Some(feeling);
    }
}

// --- 3. Tưởng Uẩn ---
/// Basic implementation of Sanna (Perception) Skandha for pattern recognition.
///
/// This skandha creates `DataEidos` (sparse binary patterns) from event content
/// and finds related patterns for knowledge retrieval.
///
/// # Examples
///
/// ```rust
/// use pandora_core::skandha_implementations::basic_skandhas::BasicSannaSkandha;
/// use pandora_core::interfaces::skandhas::SannaSkandha;
/// use pandora_core::ontology::EpistemologicalFlow;
/// use bytes::Bytes;
///
/// let sanna = BasicSannaSkandha;
/// let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"error warning".as_ref()));
/// sanna.perceive(&mut flow);
/// assert!(flow.sanna.is_some());
/// assert!(flow.related_eidos.is_some());
/// ```
pub struct BasicSannaSkandha;
impl Skandha for BasicSannaSkandha {
    fn name(&self) -> &'static str {
        "Basic Sanna (Perception)"
    }
}
impl SannaSkandha for BasicSannaSkandha {
    fn perceive(&self, flow: &mut EpistemologicalFlow) {
        info!("[{}] Đối chiếu sự kiện, nhận diện quy luật.", self.name());

        // Tạo DataEidos dựa trên nội dung sự kiện (an toàn, không panic)
        let eidos = self.create_eidos(flow);
        flow.sanna = Some(eidos.clone());

        // Tìm related eidos (giới hạn để tránh chi phí lớn)
        let related_eidos = self.find_related_patterns(&eidos);
        flow.related_eidos = Some(smallvec::SmallVec::from_vec(related_eidos));

        let related_len = flow.related_eidos.as_ref().map(|v| v.len()).unwrap_or(0);
        info!(
            "[{}] Đã nhận diện {} patterns liên quan.",
            self.name(),
            related_len
        );
    }
}

impl BasicSannaSkandha {
    fn create_eidos(&self, flow: &EpistemologicalFlow) -> crate::ontology::DataEidos {
        let default_eidos = crate::ontology::DataEidos {
            active_indices: Default::default(),
            dimensionality: 2048,
        };

        let Some(rupa) = &flow.rupa else {
            return default_eidos;
        };

        let content = String::from_utf8_lossy(rupa.as_ref());
        let mut active_indices = std::collections::HashSet::new();

        // Hash-based indices (giới hạn để tránh quá tải)
        for (i, byte) in rupa.as_ref().iter().enumerate().take(1000) {
            if *byte > 0 {
                active_indices.insert(((i * 7) as u32 + (*byte as u32)) % 2048);
            }
        }

        // Keyword-based indices (an toàn)
        for keyword in ["error", "warning", "success", "info", "critical"] {
            if content.to_lowercase().contains(keyword) {
                let hash = (keyword.len() as u32) * 13;
                active_indices.insert(hash % 2048);
            }
        }

        crate::ontology::DataEidos {
            active_indices: active_indices.into_iter().collect(),
            dimensionality: 2048,
        }
    }

    /// Tìm các patterns liên quan dựa trên DataEidos
    fn find_related_patterns(
        &self,
        eidos: &crate::ontology::DataEidos,
    ) -> Vec<crate::ontology::DataEidos> {
        let mut related = Vec::with_capacity(3);
        for i in 0..3 {
            let mut related_eidos = eidos.clone();
            for idx in eidos.active_indices.iter().take(100) {
                let new_idx = (idx.wrapping_add(i as u32 + 1)) % 2048;
                related_eidos.active_indices.insert(new_idx);
            }
            related.push(related_eidos);
        }
        related
    }
}

// --- 4. Hành Uẩn ---
/// Basic implementation of Sankhara (Formations) Skandha for intent formation.
///
/// This skandha forms intents based on perceived patterns and emotional valence.
/// Currently forms "REPORT_ERROR" intent for unpleasant feelings.
///
/// # Examples
///
/// ```rust
/// use pandora_core::skandha_implementations::basic_skandhas::BasicSankharaSkandha;
/// use pandora_core::interfaces::skandhas::SankharaSkandha;
/// use pandora_core::ontology::{EpistemologicalFlow, Vedana};
/// use bytes::Bytes;
///
/// let sankhara = BasicSankharaSkandha;
/// let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test".as_ref()));
/// flow.vedana = Some(Vedana::Unpleasant { karma_weight: -1.0 });
/// sankhara.form_intent(&mut flow);
/// assert!(flow.sankhara.is_some());
/// ```
pub struct BasicSankharaSkandha;
impl Skandha for BasicSankharaSkandha {
    fn name(&self) -> &'static str {
        "Basic Sankhara (Formations)"
    }
}
impl SankharaSkandha for BasicSankharaSkandha {
    fn form_intent(&self, flow: &mut EpistemologicalFlow) {
        // Logic đơn giản: Nếu cảm thấy "Khổ", khởi ý niệm "báo cáo lỗi".
        if let Some(Vedana::Unpleasant { .. }) = flow.vedana {
            info!(
                skandha = %self.name(),
                intent = %intents::constants::REPORT_ERROR,
                agent_pos = ?None::<(usize,usize)>,
                goal_pos = ?None::<(usize,usize)>,
                "Intent formed"
            );
            flow.set_static_intent(intents::constants::REPORT_ERROR);
        } else {
            warn!(skandha = %self.name(), "No intent formed");
        }
    }
}

// --- 5. Thức Uẩn ---
/// Basic implementation of Vinnana (Consciousness) Skandha for synthesis and rebirth.
///
/// This skandha synthesizes the cognitive flow and produces reborn events
/// when intents are present, enabling the cycle of consciousness.
///
/// # Examples
///
/// ```rust
/// use pandora_core::skandha_implementations::basic_skandhas::BasicVinnanaSkandha;
/// use pandora_core::interfaces::skandhas::VinnanaSkandha;
/// use pandora_core::ontology::EpistemologicalFlow;
/// use bytes::Bytes;
///
/// let vinnana = BasicVinnanaSkandha;
/// let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test".as_ref()));
/// flow.sankhara = Some("REPORT_ERROR".to_string().into());
/// let result = vinnana.synthesize(&flow);
/// assert!(result.is_some());
/// assert!(String::from_utf8_lossy(&result.unwrap()).contains("Synthesized consciousness"));
/// ```
pub struct BasicVinnanaSkandha;
impl Skandha for BasicVinnanaSkandha {
    fn name(&self) -> &'static str {
        "Basic Vinnana (Consciousness)"
    }
}
impl VinnanaSkandha for BasicVinnanaSkandha {
    fn synthesize(&self, flow: &EpistemologicalFlow) -> Option<Vec<u8>> {
        // Logic đơn giản: Nếu có "Ý Chỉ", tổng hợp nó thành một sự kiện mới để tái sinh.
        if let Some(intent) = &flow.sankhara {
            let conscious_event = format!("Synthesized consciousness: Intent is '{}'", intent);
            info!(
                "[{}] Tổng hợp nhận thức. Tái sinh sự kiện mới.",
                self.name()
            );
            Some(conscious_event.into_bytes())
        } else {
            info!("[{}] Tổng hợp nhận thức. Vòng lặp kết thúc.", self.name());
            None
        }
    }
}
