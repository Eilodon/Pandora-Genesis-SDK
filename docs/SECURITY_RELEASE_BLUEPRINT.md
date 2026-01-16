# AGOLOS SDK — Security & Release Blueprint (Offline-first)

Tài liệu này mô tả blueprint khả thi để phát hành AGOLOS SDK theo mô hình:

- **Lite (public source)**: đủ dùng/demo, không chứa IP nhạy cảm.
- **Full (binary-only)**: chạy offline trên **iOS/Android/WASM** (lưu ý WASM không thể “giữ bí mật”).

Mục tiêu: bảo vệ **thuật toán** + **model weights** khỏi sao chép đại trà và tăng chi phí reverse/tamper, đồng thời đảm bảo **dữ liệu nhạy cảm** luôn ở offline và **mã hoá at-rest**.

---

## 0) Sự thật phải chấp nhận (để đặt kỳ vọng đúng)

1. **Không thể bảo mật tuyệt đối IP khi ship xuống máy khách** (đặc biệt là WASM). Attacker có thể debug, dump memory, patch binary, hook FFI, và bypass mọi check.
2. Vì vậy, chiến lược đúng là:
   - **Không ship “bí mật tĩnh”** (hardcoded key/token).
   - **Giảm bề mặt lộ** (API nhỏ, ít metadata).
   - **Tăng chi phí reverse** (hardening/strip/obfuscation mức hợp lý).
   - **Kiểm soát phân phối** (license ký số, watermark per-customer).
   - **Truy vết leak** (fingerprint trong artifacts/model pack).

---

## 1) Threat model (thực tế cho SDK offline)

**Attacker mục tiêu**:
- “Casual copier”: copy file `.so/.xcframework/.wasm` + model weights.
- “Reverse engineer”: dùng Ghidra/IDA, hook JNI/Swift, patch checks, dump memory.
- “Repackager”: bọc SDK trong app khác, bán lại.

**Bảo vệ ưu tiên**:
- IP: thuật toán + weights (ngăn copy đại trà, truy vết leak).
- Data: dữ liệu người dùng/biometric/logs (mã hoá at-rest, không rò rỉ).

**Không thể cam kết**:
- “Không ai trích được weights/logic” nếu attacker quyết tâm và có thời gian.

---

## 2) Kiến trúc phát hành Lite vs Full (khả thi với workspace hiện tại)

Workspace hiện có:
- `crates/zenb-core`: lõi engine
- `crates/zenb-signals`: rPPG/vision/dsp
- `crates/zenb-store`: SQLite + XChaCha20-Poly1305 (mã hoá at-rest)
- `crates/zenb-uniffi`: FFI iOS/Android (UniFFI)
- `crates/zenb-wasm-demo`: demo

### 2.1 Nguyên tắc tách Lite/Full (không leak IP qua repo public)

Không dùng “feature flag” để giấu full code trong repo public. Thay vào đó:

- **Repo public (Lite)**: chỉ chứa các crate/API “public” và/hoặc implementation đơn giản hơn.
- **Repo private (Full build)**: chứa thuật toán + model integration thật sự; xuất ra artifacts binary.

Nếu bắt buộc giữ monorepo nội bộ, quy trình publish Lite nên là **export subtree** (CI) sang repo public (đảm bảo không push nhầm code full).

### 2.2 API surface

Tối ưu bề mặt lộ IP bằng cách:
- Giữ UniFFI surface nhỏ, ổn định, không expose cấu trúc nội bộ.
- Tránh trả ra “debug introspection” của engine trong bản Full.
- Tách “diagnostics” thành feature/debug build riêng (chỉ cấp cho đối tác theo nhu cầu).

---

## 3) Release artifacts theo platform

### 3.1 iOS

**Artifacts đề xuất**:
- `ZenbSDK.xcframework` (chứa `zenb-uniffi` compiled).
- `ZenbSDK.swiftinterface` / UniFFI Swift bindings (SPM).
- `model.pack` (weights + metadata; ký số; có watermark).
- `license.lic` (ký số; bind theo bundle id + team id nếu cần).
- `dSYMs`/symbols: phát hành qua kênh nội bộ (crash symbolication), không public.

**Distribution**:
- SPM **binaryTarget** hoặc private repo/spec.

### 3.2 Android

**Artifacts đề xuất**:
- `zenb-sdk.aar` (wrapper Kotlin + resources).
- JNI libs: `libzenb_uniffi.so` cho `arm64-v8a`, `armeabi-v7a`, `x86_64`.
- `model.pack`, `license.lic`.
- mapping/proguard (nếu có Java/Kotlin code nhạy cảm).

**Distribution**:
- Private Maven repository (nexus/artifactory/github packages).

### 3.3 WASM

Khuyến nghị mạnh:
- **WASM chỉ phát hành Lite** (vì `.wasm` là “public binary”: rất dễ phân tích).

Nếu vẫn cần “Full on WASM”:
- Chấp nhận rủi ro leak IP; chỉ dùng cho demo/POC; coi như không bảo vệ được weights.

---

## 4) Data security (điều có thể “cam kết mạnh”)

### 4.1 Mã hoá at-rest

AGOLOS đã có `zenb-store` dùng **XChaCha20-Poly1305** per-event. Điều cần chuẩn hoá khi phát hành SDK:

- **Master key không bao giờ hardcode**.
- Master key tạo ngẫu nhiên **mỗi lần cài** và lưu trong:
  - iOS: Keychain (ThisDeviceOnly / AfterFirstUnlock)
  - Android: Keystore/StrongBox (wrap/unwrap)
  - WASM: WebCrypto + IndexedDB (best-effort)

### 4.2 Memory hygiene

- Zeroize key material sau khi dùng (đã có `zeroize` trong `zenb-store`).
- Tránh log ra dữ liệu nhạy cảm (JSON observation/raw signals).

---

## 5) Licensing offline (để kiểm soát phân phối, không phải để “giữ bí mật”)

### 5.1 Mục tiêu

- Chặn copy đại trà “cầm artifact dùng luôn”.
- Gating theo feature (Lite vs Full, modules).
- Hỗ trợ revoke/expiry (offline-friendly với grace period).

### 5.2 Định dạng license (đề xuất)

`license.lic` gồm:
- `payload` (JSON/CBOR):
  - `customer_id`, `license_id`, `issued_at`, `expires_at`
  - `features`: danh sách module (signals/core/verticals/…)
  - `platform_bindings`:
    - iOS: `bundle_id`, optional `team_id`
    - Android: `package_name`, `signing_cert_sha256`
    - WASM: `origin` (không tin cậy tuyệt đối)
  - `watermark_id` (để truy vết)
- `signature` (Ed25519) trên `payload` bytes

SDK chỉ cần embed **public key** để verify offline.

### 5.3 Clock tamper (offline)

Không thể chống 100%. Khuyến nghị:
- Grace period ngắn.
- Lưu “last_seen_time” + monotonic counters trong encrypted store; phát hiện rollback best-effort.

---

## 6) Model pack (weights) — ký số + watermark + (tuỳ chọn) mã hoá at-rest

### 6.1 “Model pack” tách khỏi binary

Không embed weights trực tiếp vào binary nếu mục tiêu là:
- update nhanh,
- watermark theo khách hàng,
- ký số độc lập.

### 6.2 Định dạng đề xuất: `model.pack`

`model.pack` gồm:
- Header (versioned): `model_id`, `algo`, `shape`, `quant`, `watermark_id`, `created_at`
- Payload: weights (có thể nén)
- `hash` (BLAKE3) của payload
- `signature` (Ed25519) trên header+hash

### 6.3 Mã hoá weights

Mã hoá chỉ là “tăng chi phí”, không phải tuyệt đối:
- iOS/Android: có thể dùng key từ Keychain/Keystore để decrypt payload (best-effort).
- WASM: coi như không có chỗ cất key an toàn.

Giá trị lớn nhất vẫn là **watermark per-customer** để truy vết leak.

---

## 7) Build hardening (tăng chi phí reverse)

### 7.1 Rust release profile (khuyến nghị)

- `lto = true`
- `codegen-units = 1`
- `panic = "abort"` (cho libs)
- `strip = "symbols"` (hoặc strip ở pipeline)
- hạn chế `debug` info trong release

### 7.2 Symbol strategy

- Public: không ship symbols.
- Nội bộ: lưu symbols theo version để debug crash.

### 7.3 Minimize metadata

- Tránh để lộ “tên thuật toán/đường đi logic” trong log/error string.
- Tách debug/diagnostic build riêng.

---

## 8) Tamper/root/jailbreak detection (best-effort)

Triển khai mức “signal” để:
- giảm abuse/repack phổ thông,
- bật chế độ hạn chế (disable full features, tăng logging nội bộ…).

Không coi đây là lớp bảo vệ chính vì attacker có thể patch.

---

## 9) Leak tracing (để xử lý khi bị leak)

### 9.1 Per-customer watermark

- Mỗi khách hàng build ra artifacts khác nhau (watermark id embedded):
  - trong license payload,
  - trong model pack header,
  - optional: trong một vài hằng số “không ảnh hưởng output” (canary).

### 9.2 Canary tokens

- Tạo “dummy endpoints/keys” chỉ dùng để phát hiện leak (nếu có online).
- Offline-only: canary chủ yếu phục vụ forensic sau khi thu được file leak.

---

## 10) Supply-chain & release pipeline

Checklist tối thiểu:
- Build từ CI sạch, khoá dependency (Cargo.lock).
- Ký artifacts (cosign/GPG) + publish checksum.
- Sinh SBOM (CycloneDX) cho B2B compliance.
- Lưu build provenance (commit hash → artifact).

---

## 11) Roadmap triển khai (đề xuất theo pha)

**P0 (1–2 tuần): Quick wins**
- Chuẩn hoá key management (iOS Keychain / Android Keystore) cho `zenb-store` master key.
- Tối ưu release profile + strip symbols.
- Thiết kế `license.lic` + verify offline.
- Thiết kế `model.pack` + ký số + watermark id.

**P1 (2–4 tuần): Full packaging**
- iOS: SPM binary target + XCFramework pipeline.
- Android: AAR + maven publish pipeline.
- Lite repo export pipeline.

**P2 (tuỳ chọn): “online nhưng không chạm dữ liệu nhạy cảm”**
- Attestation + short-lived unlock key (chỉ gửi attestation + license id).

---

## 12) Practical checklist (release-ready)

- [ ] Không hardcode bất kỳ key/token nào
- [ ] Master key lưu Keychain/Keystore
- [ ] License ký số + verify offline
- [ ] Model pack ký số + watermark per-customer
- [ ] Strip symbols + LTO + panic abort
- [ ] UniFFI API surface tối thiểu
- [ ] Tài liệu rõ ràng về “WASM không thể bảo mật IP”

