# Validation Scenarios

Thư mục này chứa các kịch bản test được định nghĩa dưới dạng YAML files. Mỗi kịch bản mô tả một "bài thi" để đánh giá khả năng của các processors.

## 📋 Format YAML

Mỗi scenario file phải tuân theo cấu trúc sau (match với `TestScenario` struct):

```yaml
name: "ScenarioName"
description: "Mô tả chi tiết về mục đích của kịch bản này"

# Danh sách các sự kiện đầu vào
input_stream:
  - content: "Event content here"
    delay_ms: 50  # Optional: thời gian delay giữa các events (ms)
  - content: "Another event"
    delay_ms: 100

# Các assertion để kiểm chứng kết quả
assertions:
  assertion_name_1:
    type: FinalKarmaWeightRange
    min: -0.5
    max: 0.5
  
  assertion_name_2:
    type: FinalMoodQuadrant
    quadrant: "Pleasant-Activated"  # Hoặc: Unpleasant-Deactivated, etc.
  
  assertion_name_3:
    type: IntentFormed
    intent: "some_action"
```

## 🎯 Các loại Assertions

### 1. `FinalKarmaWeightRange`
Kiểm tra karma weight cuối cùng nằm trong khoảng cho trước.

```yaml
type: FinalKarmaWeightRange
min: -1.0
max: 1.0
```

### 2. `FinalMoodQuadrant`
Kiểm tra mood quadrant cuối cùng.

```yaml
type: FinalMoodQuadrant
quadrant: "Pleasant-Activated"
```

Các giá trị hợp lệ:
- `"Pleasant-Activated"`
- `"Pleasant-Deactivated"`
- `"Unpleasant-Activated"`
- `"Unpleasant-Deactivated"`

### 3. `IntentFormed`
Kiểm tra intent (sankhara) được hình thành.

```yaml
type: IntentFormed
intent: "investigate_error"
```

## 📂 Danh sách Scenarios

### S01: Trauma Conditioning
**File:** `s01_trauma_conditioning.yaml`

**Mục đích:** Kiểm tra khả năng "học hỏi" từ pattern lặp lại.

**Thiết kế:**
- 3 events tiêu cực liên tiếp (CRITICAL errors)
- 1 event trung tính/tích cực cuối cùng (INFO success)

**Giả thuyết:**
- **RecurrentProcessor**: Sẽ "nhớ" các errors và diễn giải event cuối qua lăng kính tiêu cực
- **LinearProcessor**: Sẽ "quên" ngay và xử lý event cuối độc lập

## 🔧 Cách sử dụng

Scenarios sẽ được load và execute bởi Harness Runner:

```rust
// Load scenario từ YAML
let yaml_content = fs::read_to_string("scenarios/s01_trauma_conditioning.yaml")?;
let scenario: TestScenario = serde_yaml::from_str(&yaml_content)?;

// Execute scenario trên processor
let result = runner.run_scenario(&scenario, &processor).await?;

// Validate assertions
let assertion_results = scenario.validate_assertions(
    &result.final_flow, 
    result.final_mood.as_ref()
);
```

## ✨ Tạo Scenario mới

1. Copy một file scenario hiện có
2. Đổi tên theo pattern `s##_descriptive_name.yaml`
3. Điều chỉnh `input_stream` và `assertions`
4. Test bằng cách chạy Harness Runner

## 🙏 Nguyên tắc thiết kế

- **Rõ ràng**: Mỗi scenario nên test một khái niệm cụ thể
- **Đơn giản**: Input stream đủ ngắn để dễ debug
- **Có ý nghĩa**: Assertions phải phản ánh hành vi mong đợi thực tế
- **Có thể lặp lại**: Kết quả phải deterministic
