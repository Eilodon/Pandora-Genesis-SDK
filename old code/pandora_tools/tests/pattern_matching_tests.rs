use chrono::{Duration, Utc};
use pandora_tools::skills::pattern_matching_skill::*;

fn create_test_sequences() -> Vec<Sequence> {
    vec![
        // Chuỗi 1: A -> B -> C
        Sequence {
            events: vec![
                Event {
                    event_type: "A".to_string(),
                    timestamp: Utc::now(),
                },
                Event {
                    event_type: "B".to_string(),
                    timestamp: Utc::now() + Duration::seconds(1),
                },
                Event {
                    event_type: "C".to_string(),
                    timestamp: Utc::now() + Duration::seconds(2),
                },
            ],
        },
        // Chuỗi 2: A -> B -> D
        Sequence {
            events: vec![
                Event {
                    event_type: "A".to_string(),
                    timestamp: Utc::now(),
                },
                Event {
                    event_type: "B".to_string(),
                    timestamp: Utc::now() + Duration::seconds(1),
                },
                Event {
                    event_type: "D".to_string(),
                    timestamp: Utc::now() + Duration::seconds(2),
                },
            ],
        },
        // Chuỗi 3: A -> C
        Sequence {
            events: vec![
                Event {
                    event_type: "A".to_string(),
                    timestamp: Utc::now(),
                },
                Event {
                    event_type: "C".to_string(),
                    timestamp: Utc::now() + Duration::seconds(1),
                },
            ],
        },
        // Chuỗi 4: B -> C
        Sequence {
            events: vec![
                Event {
                    event_type: "B".to_string(),
                    timestamp: Utc::now(),
                },
                Event {
                    event_type: "C".to_string(),
                    timestamp: Utc::now() + Duration::seconds(1),
                },
            ],
        },
        // Chuỗi 5: A -> B -> C (lặp lại)
        Sequence {
            events: vec![
                Event {
                    event_type: "A".to_string(),
                    timestamp: Utc::now(),
                },
                Event {
                    event_type: "B".to_string(),
                    timestamp: Utc::now() + Duration::seconds(1),
                },
                Event {
                    event_type: "C".to_string(),
                    timestamp: Utc::now() + Duration::seconds(2),
                },
            ],
        },
    ]
}

#[test]
fn test_mine_frequent_patterns_with_min_support_2() {
    let sequences = create_test_sequences();
    let mut engine = TemporalPrefixSpanEngine::new(2, 5).unwrap();
    engine.mine_patterns(&sequences).unwrap();

    let patterns_dbg = format!("{:?}", engine.patterns());

    assert!(patterns_dbg.contains("items: [\"A\"], support: 4"));
    assert!(patterns_dbg.contains("items: [\"B\"], support: 4"));
    assert!(patterns_dbg.contains("items: [\"C\"], support: 4"));
    assert!(patterns_dbg.contains("items: [\"A\", \"B\"], support: 3"));
    assert!(patterns_dbg.contains("items: [\"A\", \"C\"], support: 3"));
    assert!(patterns_dbg.contains("items: [\"B\", \"C\"], support: 3"));
    assert!(patterns_dbg.contains("items: [\"A\", \"B\", \"C\"], support: 2"));
}

#[test]
fn test_predict_next_action() {
    let sequences = create_test_sequences();
    let mut engine = TemporalPrefixSpanEngine::new(2, 5).unwrap();
    engine.mine_patterns(&sequences).unwrap();

    let current_sequence = vec![
        Event {
            event_type: "A".to_string(),
            timestamp: Utc::now(),
        },
        Event {
            event_type: "B".to_string(),
            timestamp: Utc::now() + Duration::seconds(1),
        },
    ];

    let predictions = engine.predict_next_action(&current_sequence).unwrap();
    assert!(!predictions.is_empty());
    assert_eq!(predictions.len(), 1);
    assert_eq!(predictions[0].predicted_action, "C");
    assert_eq!(predictions[0].confidence, 2.0);
}
