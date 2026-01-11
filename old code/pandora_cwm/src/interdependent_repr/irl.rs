/// Interdependent Representation Learning (IRL)
///
/// IRL là kiến trúc học biểu diễn nhận biết được bản chất "tương tức, phụ thuộc lẫn nhau"
/// của vạn vật. Nó học cách biểu diễn các thực thể không phải như những đối tượng
/// độc lập, mà như những phần tử trong một mạng lưới quan hệ phức tạp.
use fnv::FnvHashMap;
use tracing::info;

/// Biểu diễn một thực thể trong mạng lưới duyên khởi.
/// Mỗi thực thể được định nghĩa bởi:
/// - Vị trí của nó trong mạng lưới (context)
/// - Các quan hệ với các thực thể khác (dependencies)
/// - Trạng thái hiện tại (state)
#[derive(Debug, Clone)]
pub struct InterdependentEntity {
    pub id: String,
    pub context: Vec<f64>,         // Vị trí trong không gian biểu diễn
    pub dependencies: Vec<String>, // Danh sách các thực thể phụ thuộc
    pub state: EntityState,
}

#[derive(Debug, Clone)]
pub enum EntityState {
    Active,
    Dormant,
    Transitioning,
}

/// Mạng lưới duyên khởi chứa tất cả các thực thể và quan hệ giữa chúng.
pub struct InterdependentNetwork {
    entities: FnvHashMap<String, InterdependentEntity>,
    relationships: FnvHashMap<(String, String), f64>, // Quan hệ với trọng số
}

impl InterdependentNetwork {
    pub fn new() -> Self {
        info!("IRL: Khởi tạo Mạng lưới Duyên khởi");
        Self {
            entities: FnvHashMap::default(),
            relationships: FnvHashMap::default(),
        }
    }

    /// Thêm một thực thể vào mạng lưới.
    pub fn add_entity(&mut self, entity: InterdependentEntity) {
        info!("IRL: Thêm thực thể '{}' vào mạng lưới", entity.id);
        self.entities.insert(entity.id.clone(), entity);
    }

    /// Thiết lập quan hệ giữa hai thực thể.
    /// Trọng số thể hiện mức độ phụ thuộc lẫn nhau.
    pub fn add_relationship(&mut self, from: &str, to: &str, weight: f64) {
        info!(
            "IRL: Thiết lập quan hệ '{}' -> '{}' (trọng số: {:.3})",
            from, to, weight
        );
        self.relationships
            .insert((from.to_string(), to.to_string()), weight);
    }

    /// Học biểu diễn duyên khởi cho một thực thể.
    /// Quá trình này cập nhật context của thực thể dựa trên:
    /// 1. Trạng thái của các thực thể phụ thuộc
    /// 2. Cấu trúc toàn cục của mạng lưới
    /// 3. Các quan hệ có trọng số cao
    pub fn learn_interdependent_representation(&mut self, entity_id: &str) {
        if let Some(entity) = self.entities.get(entity_id) {
            info!("IRL: Học biểu diễn duyên khởi cho '{}'", entity_id);

            // Cập nhật context dựa trên các thực thể phụ thuộc
            let mut new_context = entity.context.clone();

            for dep_id in &entity.dependencies {
                if let Some(dep_entity) = self.entities.get(dep_id) {
                    // Tính toán ảnh hưởng từ thực thể phụ thuộc
                    let influence = self.calculate_influence(entity_id, dep_id);

                    // Cập nhật context dựa trên ảnh hưởng
                    for (i, val) in dep_entity.context.iter().enumerate() {
                        if i < new_context.len() {
                            new_context[i] += val * influence;
                        }
                    }
                }
            }

            // Cập nhật context sau khi tính toán xong
            if let Some(entity) = self.entities.get_mut(entity_id) {
                entity.context = new_context;
                info!(
                    "IRL: Đã cập nhật context cho '{}' dựa trên {} thực thể phụ thuộc",
                    entity_id,
                    entity.dependencies.len()
                );
            }
        }
    }

    /// Tính toán mức độ ảnh hưởng giữa hai thực thể.
    fn calculate_influence(&self, from: &str, to: &str) -> f64 {
        // Lấy trọng số quan hệ trực tiếp
        let direct_weight = self
            .relationships
            .get(&(from.to_string(), to.to_string()))
            .copied()
            .unwrap_or(0.0);

        // Tính toán ảnh hưởng gián tiếp thông qua các thực thể trung gian
        let mut indirect_influence = 0.0;
        for intermediate in self.entities.keys() {
            if let Some(weight1) = self
                .relationships
                .get(&(from.to_string(), intermediate.clone()))
            {
                if let Some(weight2) = self
                    .relationships
                    .get(&(intermediate.clone(), to.to_string()))
                {
                    indirect_influence += weight1 * weight2 * 0.5; // Hệ số suy giảm
                }
            }
        }

        direct_weight + indirect_influence
    }

    /// Tìm các thực thể có ảnh hưởng mạnh nhất đến một thực thể cụ thể.
    pub fn find_key_influencers(&self, entity_id: &str) -> Vec<(String, f64)> {
        let mut influencers = Vec::new();

        for other_id in self.entities.keys() {
            if other_id != entity_id {
                let influence = self.calculate_influence(other_id, entity_id);
                if influence > 0.1 {
                    // Ngưỡng ảnh hưởng tối thiểu
                    influencers.push((other_id.clone().to_string(), influence));
                }
            }
        }

        // Sắp xếp theo mức độ ảnh hưởng giảm dần
        influencers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        influencers
    }

    /// Cập nhật trạng thái của một thực thể và lan truyền ảnh hưởng.
    pub fn update_entity_state(&mut self, entity_id: &str, new_state: EntityState) {
        if let Some(entity) = self.entities.get_mut(entity_id) {
            info!(
                "IRL: Cập nhật trạng thái '{}' -> {:?}",
                entity_id, new_state
            );
            entity.state = new_state;

            // Lan truyền ảnh hưởng đến các thực thể liên quan
            self.propagate_influence(entity_id);
        }
    }

    /// Lan truyền ảnh hưởng từ một thực thể đến các thực thể khác.
    fn propagate_influence(&mut self, source_id: &str) {
        let influencers = self.find_key_influencers(source_id);

        for (influenced_id, influence_strength) in influencers {
            if influence_strength > 0.5 {
                // Chỉ lan truyền đến các thực thể có ảnh hưởng mạnh
                info!(
                    "IRL: Lan truyền ảnh hưởng từ '{}' đến '{}' (cường độ: {:.3})",
                    source_id, influenced_id, influence_strength
                );

                // Cập nhật biểu diễn của thực thể bị ảnh hưởng
                self.learn_interdependent_representation(&influenced_id);
            }
        }
    }
}

impl Default for InterdependentNetwork {
    fn default() -> Self {
        Self::new()
    }
}
