//! Simple test for LanceDB integration

use pandora_tools::skills::information_retrieval_skill::{InformationRetrievalSkill, Concept};
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::test]
async fn test_lancedb_basic_operations() {
    // Create a temporary directory for the database
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    
    // Create the InformationRetrievalSkill
    let skill = InformationRetrievalSkill::new(db_path, "concepts", 128)
        .await
        .expect("Failed to create InformationRetrievalSkill");

    // Create test concepts
    let concept1 = Concept {
        id: "concept_1".to_string(),
        text: "Machine learning is a subset of artificial intelligence".to_string(),
        embedding_vector: vec![0.1; 128],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::Value::String("AI".to_string()));
            meta.insert("confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.9).unwrap()));
            meta
        },
    };

    let concept2 = Concept {
        id: "concept_2".to_string(),
        text: "Deep learning uses neural networks with multiple layers".to_string(),
        embedding_vector: vec![0.2; 128],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::Value::String("ML".to_string()));
            meta.insert("confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()));
            meta
        },
    };

    // Add concepts to the database
    skill.add_concept(concept1.clone()).await
        .expect("Failed to add concept1");
    skill.add_concept(concept2.clone()).await
        .expect("Failed to add concept2");

    // Search for concepts using vector similarity
    let query_vector = vec![0.15; 128]; // Similar to both concepts
    let results = skill.search_by_vector(&query_vector, 2).await
        .expect("Failed to search by vector");

    // Verify results
    assert_eq!(results.len(), 2, "Should return 2 concepts");
    
    // Check that we got the concepts we added
    let result_ids: Vec<String> = results.iter().map(|c| c.id.clone()).collect();
    assert!(result_ids.contains(&"concept_1".to_string()), "Should contain concept_1");
    assert!(result_ids.contains(&"concept_2".to_string()), "Should contain concept_2");

    // Verify concept data integrity
    for result in &results {
        if result.id == "concept_1" {
            assert_eq!(result.text, concept1.text);
            assert_eq!(result.embedding_vector.len(), 128);
            assert_eq!(result.metadata.get("category").unwrap().as_str().unwrap(), "AI");
        } else if result.id == "concept_2" {
            assert_eq!(result.text, concept2.text);
            assert_eq!(result.embedding_vector.len(), 128);
            assert_eq!(result.metadata.get("category").unwrap().as_str().unwrap(), "ML");
        }
    }

    println!("✅ LanceDB integration test passed! Successfully wrote and read concepts from LanceDB.");
}
