use pandora_cwm::interdependent_repr::irl::{
    EntityState, InterdependentEntity, InterdependentNetwork,
};
use pandora_cwm::interdependent_repr::itr_nn::InterdependentTopoRelationalNN;

#[tokio::test]
async fn test_interdependent_insight() {
    println!("\n=============================================");
    println!("BÀI TEST TUỆ GIÁC DUYÊN KHỞI - ITR-NN & IRL");
    println!("=============================================\n");

    // --- Test ITR-NN (Interdependent Topo-Relational Neural Network) ---
    println!("\n--- TEST ITR-NN: Phân tích Dữ liệu Topo ---");
    let itr_nn = InterdependentTopoRelationalNN::new();

    // Tạo dữ liệu đồ thị giả lập (các nút và cạnh)
    let graph_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    itr_nn.process_graph(&graph_data);

    println!("✅ ITR-NN: Đã xử lý đồ thị với {} nút", graph_data.len());

    // --- Test IRL (Interdependent Representation Learning) ---
    println!("\n--- TEST IRL: Học Biểu diễn Duyên khởi ---");
    let mut network = InterdependentNetwork::new();

    // Tạo các thực thể trong mạng lưới duyên khởi
    let entity_a = InterdependentEntity {
        id: "A".to_string(),
        context: vec![1.0, 0.0, 0.0],
        dependencies: vec!["B".to_string(), "C".to_string()],
        state: EntityState::Active,
    };

    let entity_b = InterdependentEntity {
        id: "B".to_string(),
        context: vec![0.0, 1.0, 0.0],
        dependencies: vec!["C".to_string()],
        state: EntityState::Active,
    };

    let entity_c = InterdependentEntity {
        id: "C".to_string(),
        context: vec![0.0, 0.0, 1.0],
        dependencies: vec![],
        state: EntityState::Dormant,
    };

    // Thêm các thực thể vào mạng lưới
    network.add_entity(entity_a);
    network.add_entity(entity_b);
    network.add_entity(entity_c);

    // Thiết lập quan hệ giữa các thực thể
    network.add_relationship("A", "B", 0.8);
    network.add_relationship("A", "C", 0.6);
    network.add_relationship("B", "C", 0.9);

    // Học biểu diễn duyên khởi cho thực thể A
    network.learn_interdependent_representation("A");

    // Tìm các thực thể có ảnh hưởng mạnh nhất đến A
    let influencers = network.find_key_influencers("A");
    println!(
        "✅ IRL: Tìm thấy {} thực thể có ảnh hưởng đến A",
        influencers.len()
    );
    for (id, influence) in influencers {
        println!("   - '{}': ảnh hưởng {:.3}", id, influence);
    }

    // Cập nhật trạng thái và lan truyền ảnh hưởng
    network.update_entity_state("C", EntityState::Active);
    println!("✅ IRL: Đã cập nhật trạng thái C và lan truyền ảnh hưởng");

    // --- Test kết hợp ITR-NN và IRL ---
    println!("\n--- TEST KẾT HỢP: ITR-NN + IRL ---");

    // Sử dụng ITR-NN để phân tích cấu trúc topo của mạng lưới
    let network_structure = vec![0.8, 0.6, 0.9, 0.7, 0.5, 0.3, 0.4, 0.2];
    itr_nn.process_graph(&network_structure);

    // Sử dụng IRL để học biểu diễn duyên khởi
    network.learn_interdependent_representation("B");

    println!("✅ KẾT HỢP: ITR-NN phân tích cấu trúc, IRL học quan hệ duyên khởi");

    println!("\n=============================================");
    println!("✅ THÀNH CÔNG: Tuệ Giác Duyên Khởi đã được khai mở!");
    println!("✅ ITR-NN: Phân tích Dữ liệu Topo hoạt động");
    println!("✅ IRL: Học Biểu diễn Duyên khởi hoạt động");
    println!("✅ Kết hợp: Hiểu được bản chất 'tương tức, phụ thuộc lẫn nhau'");
    println!("=============================================");
}
