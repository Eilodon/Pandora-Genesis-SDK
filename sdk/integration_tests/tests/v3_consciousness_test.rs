use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic::*;

#[tokio::test]
async fn test_v3_consciousness_cycle() {
    println!("\n=============================================");
    println!("BÀI TEST TÂM THỨC V3 - LUỒNG NHẬN THỨC LUẬN");
    println!("=============================================\n");

    // --- 1. Khởi tạo SkandhaProcessor V3 ---
    let processor = SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    );

    // --- 2. Test Case 1: Sự kiện bình thường ---
    println!("\n--- TEST CASE 1: Sự kiện bình thường ---");
    let normal_event = b"Hello, world!".to_vec();
    let result1 = processor
        .run_epistemological_cycle_async(normal_event)
        .await;

    // Sự kiện bình thường không tạo ra ý chỉ, nên không có tái sinh
    assert!(result1.is_none());
    println!("✅ Test Case 1: Sự kiện bình thường - Không tái sinh (đúng)");

    // --- 3. Test Case 2: Sự kiện có lỗi ---
    println!("\n--- TEST CASE 2: Sự kiện có lỗi ---");
    let error_event = b"System error: connection failed".to_vec();
    let result2 = processor.run_epistemological_cycle_async(error_event).await;

    // Sự kiện có lỗi sẽ tạo ra ý chỉ "REPORT_ERROR", nên có tái sinh
    assert!(result2.is_some());
    let reborn_event = result2.expect("expected reborn event for error case");
    let reborn_text = String::from_utf8_lossy(&reborn_event);
    assert!(reborn_text.contains("REPORT_ERROR"));
    println!(
        "✅ Test Case 2: Sự kiện có lỗi - Tái sinh với ý chỉ '{}'",
        reborn_text
    );

    // --- 4. Test Case 3: Vòng lặp tái sinh ---
    println!("\n--- TEST CASE 3: Vòng lặp tái sinh ---");
    let reborn_result = processor
        .run_epistemological_cycle_async(reborn_event)
        .await;

    // Sự kiện tái sinh không chứa "error" nên không tạo ra ý chỉ mới
    assert!(reborn_result.is_none());
    println!("✅ Test Case 3: Vòng lặp tái sinh - Không tái sinh tiếp (đúng)");

    println!("\n=============================================");
    println!("✅ THÀNH CÔNG: Tâm thức V3 hoạt động hoàn hảo!");
    println!("✅ Ngũ Uẩn pipeline: Sắc -> Thọ -> Tưởng -> Hành -> Thức");
    println!("✅ Luồng Nhận Thức Luận: Hoạt động với khả năng tái sinh");
    println!("=============================================");
}
