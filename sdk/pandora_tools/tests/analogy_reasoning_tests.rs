// sdk/pandora_tools/tests/analogy_reasoning_tests.rs

use pandora_tools::skills::analogy_reasoning_skill::*;
use pandora_tools::skills::information_retrieval_skill::*;
use std::sync::Arc;
use tokio::sync::RwLock;

async fn seed_engine() -> Arc<RwLock<ProgressiveSemanticEngine>> {
    let mut eng = ProgressiveSemanticEngine::new("memory://", "concepts", 2)
        .await
        .unwrap();
    let _ = eng
        .add_document(Document {
            id: "king".into(),
            content: "king".into(),
            embedding: vec![0.9, 0.1],
            metadata: Default::default(),
        })
        .await;
    let _ = eng
        .add_document(Document {
            id: "man".into(),
            content: "man".into(),
            embedding: vec![0.8, 0.1],
            metadata: Default::default(),
        })
        .await;
    let _ = eng
        .add_document(Document {
            id: "woman".into(),
            content: "woman".into(),
            embedding: vec![0.1, 0.8],
            metadata: Default::default(),
        })
        .await;
    let _ = eng
        .add_document(Document {
            id: "queen".into(),
            content: "queen".into(),
            embedding: vec![0.2, 0.8],
            metadata: Default::default(),
        })
        .await;
    Arc::new(RwLock::new(eng))
}

#[tokio::test]
async fn test_solve_analogy_king_queen() {
    let engine = seed_engine().await;
    let analogy = AnalogyEngine::new(engine);
    let out = analogy.solve_analogy("king", "man", "woman").await.unwrap();
    assert_eq!(out.content, "queen");
}

#[tokio::test]
async fn test_solve_analogy_geography() {
    let mut eng = ProgressiveSemanticEngine::new("memory://", "concepts", 2)
        .await
        .unwrap();
    let _ = eng
        .add_document(Document {
            id: "vietnam".into(),
            content: "vietnam".into(),
            embedding: vec![0.9, 0.1],
            metadata: Default::default(),
        })
        .await;
    let _ = eng
        .add_document(Document {
            id: "hanoi".into(),
            content: "hanoi".into(),
            embedding: vec![0.8, 0.1],
            metadata: Default::default(),
        })
        .await;
    let _ = eng
        .add_document(Document {
            id: "japan".into(),
            content: "japan".into(),
            embedding: vec![0.1, 0.9],
            metadata: Default::default(),
        })
        .await;
    let _ = eng
        .add_document(Document {
            id: "tokyo".into(),
            content: "tokyo".into(),
            embedding: vec![0.2, 0.9],
            metadata: Default::default(),
        })
        .await;
    let engine = Arc::new(RwLock::new(eng));
    let analogy = AnalogyEngine::new(engine);
    let out = analogy
        .solve_analogy("vietnam", "hanoi", "japan")
        .await
        .unwrap();
    assert_eq!(out.content, "tokyo");
}

#[tokio::test]
async fn test_analogy_fails_with_unrelated_concepts() {
    let engine = seed_engine().await;
    let analogy = AnalogyEngine::new(engine);
    let res = analogy.solve_analogy("car", "road", "fish").await;
    assert!(res.is_err() || res.as_ref().map(|o| o.confidence.score).unwrap_or(0.0) < 0.2);
}
