use crate::ontology::EpistemologicalFlow;

/// Trait cơ sở cho tất cả các "Uẩn" (Skandha) trong kiến trúc nhận thức.
pub trait Skandha: Send + Sync {
    /// Tên của Uẩn (ví dụ: "Sắc", "Thọ").
    fn name(&self) -> &'static str;
}

/// 1. Sắc Uẩn (Rūpa-skandha): Tầng tiếp nhận sự kiện nguyên thủy.
pub trait RupaSkandha: Skandha {
    /// Xử lý một sự kiện thô và khởi tạo Dòng Chảy Nhận Thức. (đồng bộ)
    fn process_event(&self, event: Vec<u8>) -> EpistemologicalFlow;
}

/// 2. Thọ Uẩn (Vedanā-skandha): Tầng gán "cảm giác" đạo đức.
pub trait VedanaSkandha: Skandha {
    /// "Cảm" sự kiện và gán cho nó một cảm thọ (Lạc, Khổ, Xả).
    /// Hợp đồng: Đọc `rupa` từ flow và ghi kết quả vào `vedana`.
    fn feel(&self, flow: &mut EpistemologicalFlow);
}

/// 3. Tưởng Uẩn (Saññā-skandha): Tầng nhận diện quy luật và mẫu hình.
pub trait SannaSkandha: Skandha {
    /// "Nhận biết" sự kiện, đối chiếu với tri thức (CWM) để nhận diện quy luật.
    /// Hợp đồng: Đọc `rupa` và `vedana`, ghi kết quả vào `sanna` và `related_eidos`.
    fn perceive(&self, flow: &mut EpistemologicalFlow);
}

/// 4. Hành Uẩn (Saṅkhāra-skandha): Tầng khởi phát "Ý Chỉ" hành động.
pub trait SankharaSkandha: Skandha {
    /// Dựa trên toàn bộ nhận thức đã có, khởi phát một "Ý Chỉ" hành động có mục đích.
    /// Hợp đồng: Đọc toàn bộ flow và ghi kết quả vào `sankhara`.
    fn form_intent(&self, flow: &mut EpistemologicalFlow);
}

/// 5. Thức Uẩn (Viññāṇa-skandha): Tầng tổng hợp nhận thức và tái sinh.
pub trait VinnanaSkandha: Skandha {
    /// Tổng hợp toàn bộ Dòng Chảy thành một "Nhận thức" cuối cùng và quyết định
    /// xem có cần tái sinh nó thành một sự kiện mới hay không.
    fn synthesize(&self, flow: &EpistemologicalFlow) -> Option<Vec<u8>>;
}
