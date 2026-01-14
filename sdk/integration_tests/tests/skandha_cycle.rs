use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic::*;

#[tokio::test]
async fn test_skandha_cycle_v3() {
    println!("\n=============================================");
    println!("BÀI TEST TÍCH HỢP LUỒNG NHẬN THỨC LUẬN V3");
    println!("=============================================\n");

    // 1. Lắp ráp một "Tâm thức" V3 hoàn chỉnh
    let processor = SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    );

    // --- Kịch bản 1: Sự kiện trung tính ---
    println!("\n--- KỊCH BẢN 1: SỰ KIỆN TRUNG TÍNH ---");
    let normal_event = "hello world".to_string().into_bytes();

    // Vận hành luồng nhận thức (biến thể async)
    let reborn_event_1 = processor
        .run_epistemological_cycle_async(normal_event)
        .await;

    // Kiểm tra: Vì sự kiện là trung tính, không có "Ý Chỉ" nào được khởi phát
    // và do đó, không có sự kiện nào được "tái sinh".
    assert!(reborn_event_1.is_none());
    println!(
        "\n✅ KẾT QUẢ KỊCH BẢN 1: Chính xác! Hệ thống an trú trong xả, không khởi phát hành động."
    );

    // --- Kịch bản 2: Sự kiện mang "phiền não" (lỗi) ---
    println!("\n--- KỊCH BẢN 2: SỰ KIỆN CÓ LỖI ---");
    let error_event = "system critical error detected".to_string().into_bytes();

    // Vận hành luồng nhận thức (biến thể async)
    let reborn_event_2 = processor.run_epistemological_cycle_async(error_event).await;

    // Kiểm tra:
    // 1. Phải có một sự kiện được tái sinh.
    // 2. Sự kiện đó phải chứa "Ý Chỉ" đã được khởi phát là "REPORT_ERROR".
    assert!(reborn_event_2.is_some());
    let reborn_bytes = reborn_event_2.expect("expected reborn event in error scenario");
    let reborn_content =
        String::from_utf8(reborn_bytes).expect("reborn event should be valid UTF-8");
    assert!(reborn_content.contains("REPORT_ERROR"));
    println!("\n✅ KẾT QUẢ KỊCH BẢN 2: Chính xác! Hệ thống cảm nhận 'Khổ', khởi phát 'Ý Chỉ' và tái sinh nhận thức.");

    println!("\n=======================================================");
    println!("✅ THÀNH CÔNG: TÂM THỨC V3 ĐÃ CHỨNG NGHIỆM THÀNH CÔNG!");
    println!("=======================================================");
}
