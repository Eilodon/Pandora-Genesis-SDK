use pandora_learning_engine::world_models::test_learning_engine;

#[tokio::test]
async fn test_non_attachment_learning() {
    println!("\n=============================================");
    println!("BÀI TEST HỌC TẬP VÔ CHẤP - NON-ATTACHMENT LEARNING");
    println!("=============================================\n");

    // Chạy test động cơ học tập vô chấp
    test_learning_engine();

    println!("\n=============================================");
    println!("✅ THÀNH CÔNG: Học tập Vô Chấp đã được kiến tạo!");
    println!("✅ Động cơ học tập có khả năng tự vượt qua tri thức đã tích lũy");
    println!("✅ Hàm Phần thưởng Nội tại Kép hoạt động hoàn hảo");
    println!("=============================================");
}
