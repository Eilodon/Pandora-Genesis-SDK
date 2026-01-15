# 📚 VERTICAL MARKET EXPANSION - MASTER INDEX

## Tài liệu kế hoạch thực thi chi tiết

Kế hoạch này được tạo dựa trên **deep audit toàn bộ codebase AGOLOS**, bao gồm:
- `zenb-signals`: rPPG, DSP, Vision, Physio modules
- `zenb-core`: Engine, UKF, PhilosophicalState
- `zenb-uniffi`: FFI bindings

---

## 📖 MỤC LỤC TÀI LIỆU

| Part | Nội dung | File |
|------|----------|------|
| **PART 1** | Tổng quan, Infrastructure, Eye Metrics | [PART1](./VERTICAL_MARKET_PLAN_PART1.md) |
| **PART 2** | Gaze Estimator, Micro Expression | [PART2](./VERTICAL_MARKET_PLAN_PART2.md) |
| **PART 3** | Liveness Detection Module | [PART3](./VERTICAL_MARKET_PLAN_PART3.md) |
| **PART 4** | Driver Monitoring System | [PART4](./VERTICAL_MARKET_PLAN_PART4.md) |
| **PART 5** | Retail Analytics, Timeline | [PART5](./VERTICAL_MARKET_PLAN_PART5.md) |
| **PART 6** | Fintech, Education, Safety Framework | [PART6](./VERTICAL_MARKET_PLAN_PART6.md) |

---

## 🎯 QUICK REFERENCE

### Cấu trúc thư mục mới

```
crates/zenb-verticals/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── shared/
│   │   ├── mod.rs
│   │   ├── eye_metrics.rs      # EAR, PERCLOS, blink
│   │   ├── gaze_estimator.rs   # Head pose + eye gaze
│   │   ├── micro_expression.rs # AU detection
│   │   ├── safety_guard.rs     # Rate limiting, anti-replay
│   │   └── privacy.rs          # GDPR/CCPA compliance
│   ├── liveness/
│   │   ├── mod.rs
│   │   ├── detector.rs         # Core liveness logic
│   │   ├── texture_analyzer.rs # 3D vs 2D detection
│   │   ├── challenge_response.rs
│   │   └── temporal_consistency.rs
│   ├── automotive/
│   │   ├── mod.rs
│   │   ├── dms.rs              # Driver Monitoring System
│   │   ├── drowsiness.rs       # PERCLOS-based
│   │   ├── distraction.rs      # Gaze-based
│   │   └── cardiac_monitor.rs  # Emergency detection
│   ├── retail/
│   │   ├── mod.rs
│   │   ├── emotion_analytics.rs
│   │   ├── engagement.rs
│   │   └── timeline.rs
│   ├── fintech/
│   │   ├── mod.rs
│   │   └── fraud_detector.rs
│   └── education/
│       ├── mod.rs
│       └── proctoring.rs
└── tests/
```

### Timeline tổng quan

```
Week 1: Infrastructure + Shared Components
Week 2: Liveness Detection
Week 3-4: Driver Monitoring System
Week 5: Retail Analytics
Week 6+: Fintech & Education (optional)
```

### Ưu tiên thực thi

1. **🥇 Liveness Detection** - Fastest to market, unique rPPG differentiator
2. **🥈 Driver Monitoring** - EU regulatory tailwind, life-saving features
3. **🥉 Retail Analytics** - Proven market, quick pilots
4. **🏅 Fintech/Education** - Build on foundation

---

## 🔧 COMMANDS

```bash
# Build
cargo build -p zenb-verticals

# Test
cargo test -p zenb-verticals

# Specific feature
cargo build -p zenb-verticals --features liveness,automotive

# Documentation
cargo doc -p zenb-verticals --open
```

---

## 📊 REUSE MATRIX

| Existing Module | Reused In |
|-----------------|-----------|
| `zenb-signals::rppg::EnsembleProcessor` | Liveness, Fintech |
| `zenb-signals::physio::HrvEstimator` | DMS, Fintech |
| `zenb-signals::physio::RespirationEstimator` | DMS |
| `zenb-signals::dsp::MotionDetector` | Liveness, DMS |
| `zenb-signals::dsp::QualityScorer` | Liveness |
| `zenb-signals::beauty::landmarks` | All verticals |
| `zenb-core::estimators::UkfEstimator` | DMS, Retail |

---

## ⚠️ SAFETY CHECKLIST

- [ ] Rate limiting implemented
- [ ] Anti-replay (nonce validation) enabled
- [ ] Fail-safe defaults configured
- [ ] Confidence thresholds set
- [ ] Privacy policy defined
- [ ] Consent mechanisms in place
- [ ] Bias testing completed
- [ ] Human review for high-stakes decisions

---

**Tạo bởi:** Deep Audit của AGOLOS Codebase
**Ngày:** 2026-01-16
**Phiên bản:** 1.0
