// sdk/pandora_orchestrator/examples/edge_optimization_demo.rs
// Demo cho Edge Optimization với SkillForge và Active Inference

use pandora_orchestrator::{
    EdgeOptimizationManager, EdgeDeviceSpecs, EdgeDeviceType,
};
use pandora_core::ontology::{CognitiveRequest, TaskType};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    println!("🚀 EDGE OPTIMIZATION DEMO - Neural Skills Specifications");
    println!("{}", "=".repeat(80));

    // Test different device types
    test_mobile_device().await?;
    test_iot_device().await?;
    test_raspberry_pi_device().await?;
    test_microcontroller_device().await?;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

async fn test_mobile_device() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📱 Testing Mobile Device Optimization");
    println!("{}", "-".repeat(50));

    let device_specs = EdgeDeviceSpecs {
        device_type: EdgeDeviceType::Mobile,
        cpu_cores: 8,
        ram_mb: 4096,
        storage_mb: 32768,
        gpu_available: true,
        wasm_support: true,
        power_efficient: true,
    };

    let mut manager = EdgeOptimizationManager::new(device_specs);

    // Test various requests
    let requests = vec![
        create_test_request(TaskType::Arithmetic, "Calculate 2 + 2"),
        create_test_request(TaskType::LogicalReasoning, "If A then B, A is true, what is B?"),
        create_test_request(TaskType::PatternMatching, "Find pattern in sequence: 1,2,3,4,5"),
        create_test_request(TaskType::InformationRetrieval, "Search for information about AI"),
    ];

    for request in requests {
        let start = Instant::now();
        match manager.process_request(&request).await {
            Ok(response) => {
                let duration = start.elapsed();
                manager.update_metrics(true, duration.as_millis() as f32, 50.0);
                println!("✅ Processed: {:?} in {:.2}ms", request.task_type, duration.as_millis());
                println!("   Content: {:?}", response.content);
                println!("   Confidence: {:.2}", response.confidence);
            }
            Err(e) => {
                println!("❌ Failed: {:?} - {}", request.task_type, e);
                manager.update_metrics(false, 0.0, 0.0);
            }
        }
    }

    // Show performance metrics
    let metrics = manager.get_performance_metrics();
    println!("\n📊 Mobile Device Performance Metrics:");
    println!("   Total Requests: {}", metrics.total_requests);
    println!("   Success Rate: {:.1}%", (metrics.successful_requests as f32 / metrics.total_requests as f32) * 100.0);
    println!("   Average Latency: {:.2}ms", metrics.average_latency_ms);
    println!("   Memory Usage: {:.2}MB", metrics.memory_usage_mb);
    println!("   Energy Savings: {:.1}%", metrics.energy_savings * 100.0);

    // Show optimization recommendations
    let recommendations = manager.get_optimization_recommendations();
    if !recommendations.is_empty() {
        println!("\n💡 Optimization Recommendations:");
        for rec in recommendations {
            println!("   - {:?}", rec);
        }
    }

    Ok(())
}

async fn test_iot_device() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌐 Testing IoT Device Optimization");
    println!("{}", "-".repeat(50));

    let device_specs = EdgeDeviceSpecs {
        device_type: EdgeDeviceType::IoT,
        cpu_cores: 1,
        ram_mb: 64,
        storage_mb: 256,
        gpu_available: false,
        wasm_support: true,
        power_efficient: true,
    };

    let mut manager = EdgeOptimizationManager::new(device_specs);

    // Test lightweight requests
    let requests = vec![
        create_test_request(TaskType::Arithmetic, "Calculate sensor reading"),
        create_test_request(TaskType::PatternMatching, "Detect anomaly in sensor data"),
    ];

    for request in requests {
        let start = Instant::now();
        match manager.process_request(&request).await {
            Ok(_response) => {
                let duration = start.elapsed();
                manager.update_metrics(true, duration.as_millis() as f32, 10.0);
                println!("✅ Processed: {:?} in {:.2}ms", request.task_type, duration.as_millis());
            }
            Err(e) => {
                println!("❌ Failed: {:?} - {}", request.task_type, e);
                manager.update_metrics(false, 0.0, 0.0);
            }
        }
    }

    let metrics = manager.get_performance_metrics();
    println!("\n📊 IoT Device Performance Metrics:");
    println!("   Total Requests: {}", metrics.total_requests);
    println!("   Success Rate: {:.1}%", (metrics.successful_requests as f32 / metrics.total_requests as f32) * 100.0);
    println!("   Average Latency: {:.2}ms", metrics.average_latency_ms);
    println!("   Memory Usage: {:.2}MB", metrics.memory_usage_mb);
    println!("   Energy Savings: {:.1}%", metrics.energy_savings * 100.0);

    Ok(())
}

async fn test_raspberry_pi_device() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🍓 Testing Raspberry Pi Optimization");
    println!("{}", "-".repeat(50));

    let device_specs = EdgeDeviceSpecs {
        device_type: EdgeDeviceType::RaspberryPi,
        cpu_cores: 4,
        ram_mb: 1024,
        storage_mb: 8192,
        gpu_available: true,
        wasm_support: true,
        power_efficient: true,
    };

    let mut manager = EdgeOptimizationManager::new(device_specs);

    // Test medium complexity requests
    let requests = vec![
        create_test_request(TaskType::Arithmetic, "Complex mathematical calculation"),
        create_test_request(TaskType::LogicalReasoning, "Multi-step logical reasoning"),
        create_test_request(TaskType::PatternMatching, "Complex pattern recognition"),
        create_test_request(TaskType::InformationRetrieval, "Search and retrieve information"),
    ];

    for request in requests {
        let start = Instant::now();
        match manager.process_request(&request).await {
            Ok(_response) => {
                let duration = start.elapsed();
                manager.update_metrics(true, duration.as_millis() as f32, 30.0);
                println!("✅ Processed: {:?} in {:.2}ms", request.task_type, duration.as_millis());
            }
            Err(e) => {
                println!("❌ Failed: {:?} - {}", request.task_type, e);
                manager.update_metrics(false, 0.0, 0.0);
            }
        }
    }

    let metrics = manager.get_performance_metrics();
    println!("\n📊 Raspberry Pi Performance Metrics:");
    println!("   Total Requests: {}", metrics.total_requests);
    println!("   Success Rate: {:.1}%", (metrics.successful_requests as f32 / metrics.total_requests as f32) * 100.0);
    println!("   Average Latency: {:.2}ms", metrics.average_latency_ms);
    println!("   Memory Usage: {:.2}MB", metrics.memory_usage_mb);
    println!("   Energy Savings: {:.1}%", metrics.energy_savings * 100.0);

    Ok(())
}

async fn test_microcontroller_device() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Testing Microcontroller Optimization");
    println!("{}", "-".repeat(50));

    let device_specs = EdgeDeviceSpecs {
        device_type: EdgeDeviceType::Microcontroller,
        cpu_cores: 1,
        ram_mb: 32,
        storage_mb: 128,
        gpu_available: false,
        wasm_support: false,
        power_efficient: true,
    };

    let mut manager = EdgeOptimizationManager::new(device_specs);

    // Test very lightweight requests
    let requests = vec![
        create_test_request(TaskType::Arithmetic, "Simple calculation"),
        create_test_request(TaskType::PatternMatching, "Basic pattern detection"),
    ];

    for request in requests {
        let start = Instant::now();
        match manager.process_request(&request).await {
            Ok(_response) => {
                let duration = start.elapsed();
                manager.update_metrics(true, duration.as_millis() as f32, 5.0);
                println!("✅ Processed: {:?} in {:.2}ms", request.task_type, duration.as_millis());
            }
            Err(e) => {
                println!("❌ Failed: {:?} - {}", request.task_type, e);
                manager.update_metrics(false, 0.0, 0.0);
            }
        }
    }

    let metrics = manager.get_performance_metrics();
    println!("\n📊 Microcontroller Performance Metrics:");
    println!("   Total Requests: {}", metrics.total_requests);
    println!("   Success Rate: {:.1}%", (metrics.successful_requests as f32 / metrics.total_requests as f32) * 100.0);
    println!("   Average Latency: {:.2}ms", metrics.average_latency_ms);
    println!("   Memory Usage: {:.2}MB", metrics.memory_usage_mb);
    println!("   Energy Savings: {:.1}%", metrics.energy_savings * 100.0);

    Ok(())
}

fn create_test_request(task_type: TaskType, description: &str) -> CognitiveRequest {
    use pandora_core::ontology::{CognitiveInput, RequestContext, Priority, QualityPreference};
    
    CognitiveRequest {
        id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        user_id: Some("edge_demo".to_string()),
        session_id: Some(uuid::Uuid::new_v4()),
        task_type,
        input: CognitiveInput::Text(description.to_string()),
        context: RequestContext::default(),
        priority: Priority::Normal,
        deadline: None,
        quality_preference: QualityPreference::Fastest,
        resource_constraints: None,
        preferred_skills: None,
    }
}

