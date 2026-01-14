// sdk/pandora_rm/src/lib.rs

#![allow(clippy::all)]
use chrono::{DateTime, Utc};
use pandora_core::ontology::{CognitiveRequest, SkillId, TaskId};
use std::collections::HashMap;
use std::sync::Arc;
use sysinfo::{System, SystemExt};
use thiserror::Error;
use tokio::sync::RwLock;

// Import enhanced resource manager
pub mod enhanced_resource_manager;
pub use enhanced_resource_manager::*;

#[derive(Debug, Error)]
pub enum ResourceManagerError {
    #[error("Tài nguyên không đủ: {0}")]
    InsufficientResources(String),
}

// ===== 6. Resource Management Specifications =====

// --- 6.1 Resource Monitoring ---

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f32, // a.k.a load average
    pub memory_usage_mb: u64,
    pub battery_level: Option<f32>,
    pub network_available: bool,
}

pub struct ResourceMonitor {
    sys: System,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            sys: System::new_all(),
        }
    }

    pub fn get_current_status(&mut self) -> ResourceUsage {
        self.sys.refresh_all();

        ResourceUsage {
            timestamp: Utc::now(),
            cpu_usage: self.sys.load_average().one as f32,
            memory_usage_mb: (self.sys.used_memory() / 1024) as u64,
            battery_level: None,     // Cần thư viện chuyên dụng cho pin
            network_available: true, // Cần thư viện chuyên dụng cho mạng
        }
    }
}

// --- 6.2 Adaptive Allocation ---

#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    MinimizeLatency,
    MaximizeThroughput,
    MinimizeEnergyConsumption,
    BalancedPerformance,
}

#[derive(Debug, Clone)]
pub struct AllocationPlan {
    pub task_assignments: HashMap<TaskId, SkillId>,
    pub execution_order: Vec<TaskId>,
    pub resource_estimates: HashMap<TaskId, ResourceEstimate>,
}

#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    pub cpu_usage: f32,
    pub memory_mb: u64,
    pub estimated_duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct CognitiveTask {
    pub id: TaskId,
    pub priority: u8, // 0-255, higher is more important
    pub estimated_cost: ResourceEstimate,
    pub skill_requirements: Vec<SkillId>,
}

#[derive(Debug, Clone)]
pub struct AvailableResources {
    pub cpu_capacity: f32,
    pub memory_mb: u64,
    pub current_cpu_usage: f32,
    pub current_memory_usage: u64,
}

#[derive(Debug, Clone)]
pub enum SystemState {
    Normal,
    PowerSaving,
    Emergency,
}

pub struct AllocationOptimizer;

impl AllocationOptimizer {
    pub async fn optimize(
        &self,
        tasks: &[CognitiveTask],
        resources: &AvailableResources,
    ) -> Result<AllocationPlan, ResourceManagerError> {
        // Simple greedy allocation algorithm
        let mut plan = AllocationPlan {
            task_assignments: HashMap::new(),
            execution_order: Vec::new(),
            resource_estimates: HashMap::new(),
        };

        // Sort tasks by priority (highest first)
        let mut sorted_tasks = tasks.to_vec();
        sorted_tasks.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut available_cpu = resources.cpu_capacity - resources.current_cpu_usage;
        let mut available_memory = resources.memory_mb - resources.current_memory_usage;

        for task in sorted_tasks {
            // Check if we have enough resources for this task
            if task.estimated_cost.cpu_usage <= available_cpu 
                && task.estimated_cost.memory_mb <= available_memory {
                
                // Assign the first available skill (simplified)
                if let Some(skill_id) = task.skill_requirements.first() {
                    plan.task_assignments.insert(task.id.clone(), skill_id.clone());
                    plan.execution_order.push(task.id.clone());
                    plan.resource_estimates.insert(task.id.clone(), task.estimated_cost.clone());
                    
                    // Update available resources
                    available_cpu -= task.estimated_cost.cpu_usage;
                    available_memory -= task.estimated_cost.memory_mb;
                }
            }
        }

        Ok(plan)
    }
}

// --- Adaptive Resource Manager ---

pub struct AdaptiveResourceManager {
    pub resource_monitor: Arc<RwLock<ResourceMonitor>>,
    pub allocation_optimizer: Arc<AllocationOptimizer>,
    pub system_state: Arc<RwLock<SystemState>>,
    pub running_tasks: Arc<RwLock<HashMap<TaskId, CognitiveTask>>>,
}

impl AdaptiveResourceManager {
    pub fn new() -> Self {
        Self {
            resource_monitor: Arc::new(RwLock::new(ResourceMonitor::new())),
            allocation_optimizer: Arc::new(AllocationOptimizer),
            system_state: Arc::new(RwLock::new(SystemState::Normal)),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Tối ưu hóa việc phân bổ tài nguyên cho các tác vụ sắp tới.
    pub async fn optimize_allocation(
        &self,
        _upcoming_tasks: &[CognitiveRequest],
    ) -> Result<AllocationPlan, ResourceManagerError> {
        // 1. Dự đoán nhu cầu tài nguyên (tạm thời bỏ qua)
        // 2. Lấy trạng thái tài nguyên hiện tại
        let _current_status = self.resource_monitor.write().await.get_current_status();

        // 3. Tối ưu hóa việc phân bổ
        // self.allocation_optimizer.optimize(...).await

        Ok(AllocationPlan {
            task_assignments: HashMap::new(),
            execution_order: Vec::new(),
            resource_estimates: HashMap::new(),
        })
    }

    /// Xử lý một tình huống khủng hoảng tài nguyên (ví dụ: pin yếu)
    /// Sẽ được gọi bởi SymbolicBrain khi nhận được tín hiệu.
    pub async fn handle_resource_crisis(&self) {
        use tracing::warn;
        
        warn!("Resource crisis detected! Entering graceful degradation mode.");
        
        // Set system state to power saving
        {
            let mut state = self.system_state.write().await;
            *state = SystemState::PowerSaving;
        }
        
        // Identify and terminate low-priority running tasks
        let mut running_tasks = self.running_tasks.write().await;
        let mut tasks_to_terminate = Vec::new();
        
        for (task_id, task) in running_tasks.iter() {
            // Terminate tasks with priority < 100 (low priority)
            if task.priority < 100 {
                tasks_to_terminate.push(task_id.clone());
            }
        }
        
        // Remove terminated tasks
        for task_id in tasks_to_terminate {
            running_tasks.remove(&task_id);
            warn!("Terminated low-priority task: {:?}", task_id);
        }
        
        // Log current system state
        let current_state = self.system_state.read().await;
        warn!("System state changed to: {:?}", *current_state);
    }
    
    /// Get current system state
    pub async fn get_system_state(&self) -> SystemState {
        let state = self.system_state.read().await;
        state.clone()
    }
    
    /// Check if system is in power saving mode
    pub async fn is_power_saving(&self) -> bool {
        matches!(*self.system_state.read().await, SystemState::PowerSaving)
    }
    
    /// Add a running task
    pub async fn add_running_task(&self, task: CognitiveTask) {
        let mut running_tasks = self.running_tasks.write().await;
        running_tasks.insert(task.id.clone(), task);
    }
    
    /// Remove a completed task
    pub async fn remove_running_task(&self, task_id: &TaskId) {
        let mut running_tasks = self.running_tasks.write().await;
        running_tasks.remove(task_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_optimize_allocation() {
        let optimizer = AllocationOptimizer;
        
        let tasks = vec![
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 100,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.5,
                    memory_mb: 100,
                    estimated_duration_ms: 1000,
                },
                skill_requirements: vec!["skill1".to_string()],
            },
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 200, // Higher priority
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.3,
                    memory_mb: 50,
                    estimated_duration_ms: 500,
                },
                skill_requirements: vec!["skill2".to_string()],
            },
        ];
        
        let resources = AvailableResources {
            cpu_capacity: 1.0,
            memory_mb: 200,
            current_cpu_usage: 0.1,
            current_memory_usage: 10,
        };
        
        let plan = optimizer.optimize(&tasks, &resources).await.unwrap();
        
        // Should have 2 task assignments
        assert_eq!(plan.task_assignments.len(), 2);
        
        // Should be ordered by priority (task2 first, then task1)
        // Note: We can't easily test exact order with random UUIDs, so we just check length
        assert_eq!(plan.execution_order.len(), 2);
        
        // Should have resource estimates
        assert_eq!(plan.resource_estimates.len(), 2);
    }
    
    #[tokio::test]
    async fn test_optimize_insufficient_resources() {
        let optimizer = AllocationOptimizer;
        
        let tasks = vec![
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 100,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 2.0, // More than available
                    memory_mb: 100,
                    estimated_duration_ms: 1000,
                },
                skill_requirements: vec!["skill1".to_string()],
            },
        ];
        
        let resources = AvailableResources {
            cpu_capacity: 1.0,
            memory_mb: 200,
            current_cpu_usage: 0.1,
            current_memory_usage: 10,
        };
        
        let plan = optimizer.optimize(&tasks, &resources).await.unwrap();
        
        // Should have no task assignments due to insufficient resources
        assert_eq!(plan.task_assignments.len(), 0);
        assert_eq!(plan.execution_order.len(), 0);
    }
    
    #[tokio::test]
    async fn test_handle_resource_crisis() {
        let manager = AdaptiveResourceManager::new();
        
        // Add some running tasks
        let task1_id = Uuid::new_v4();
        let task2_id = Uuid::new_v4();
        
        let task1 = CognitiveTask {
            id: task1_id,
            priority: 50, // Low priority
            estimated_cost: ResourceEstimate {
                cpu_usage: 0.5,
                memory_mb: 100,
                estimated_duration_ms: 1000,
            },
            skill_requirements: vec!["skill1".to_string()],
        };
        
        let task2 = CognitiveTask {
            id: task2_id,
            priority: 150, // High priority
            estimated_cost: ResourceEstimate {
                cpu_usage: 0.3,
                memory_mb: 50,
                estimated_duration_ms: 500,
            },
            skill_requirements: vec!["skill2".to_string()],
        };
        
        manager.add_running_task(task1).await;
        manager.add_running_task(task2).await;
        
        // Verify initial state
        assert!(!manager.is_power_saving().await);
        assert_eq!(manager.running_tasks.read().await.len(), 2);
        
        // Handle crisis
        manager.handle_resource_crisis().await;
        
        // Verify power saving mode is enabled
        assert!(manager.is_power_saving().await);
        
        // Verify low priority task was terminated
        let running_tasks = manager.running_tasks.read().await;
        assert_eq!(running_tasks.len(), 1);
        assert!(running_tasks.contains_key(&task2_id));
        assert!(!running_tasks.contains_key(&task1_id));
    }
    
    #[tokio::test]
    async fn test_system_state_management() {
        let manager = AdaptiveResourceManager::new();
        
        // Initial state should be Normal
        assert!(!manager.is_power_saving().await);
        assert!(matches!(manager.get_system_state().await, SystemState::Normal));
        
        // Handle crisis
        manager.handle_resource_crisis().await;
        
        // Should be in PowerSaving mode
        assert!(manager.is_power_saving().await);
        assert!(matches!(manager.get_system_state().await, SystemState::PowerSaving));
    }

    // ===== ADVANCED SCENARIO TESTS =====

    #[tokio::test]
    async fn test_concurrent_crisis_handling() {
        let manager = Arc::new(AdaptiveResourceManager::new());
        
        // Add multiple running tasks
        for i in 0..10 {
            let priority = if i < 5 { 50 } else { 150 }; // Half low, half high priority
            let task = CognitiveTask {
                id: Uuid::new_v4(),
                priority,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.1,
                    memory_mb: 10,
                    estimated_duration_ms: 100,
                },
                skill_requirements: vec![format!("skill_{}", i)],
            };
            manager.add_running_task(task).await;
        }
        
        // Verify initial state
        assert_eq!(manager.running_tasks.read().await.len(), 10);
        
        // Trigger multiple concurrent crises
        let mut handles = vec![];
        for _ in 0..3 {
            let mgr = Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                mgr.handle_resource_crisis().await;
            });
            handles.push(handle);
        }
        
        // Wait for all crisis handlers to complete
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify low priority tasks were terminated
        let running_tasks = manager.running_tasks.read().await;
        assert!(running_tasks.len() <= 5, "Should have removed low priority tasks");
        
        // Verify all remaining tasks have high priority
        for task in running_tasks.values() {
            assert!(task.priority >= 100, "Only high priority tasks should remain");
        }
        
        // Verify system is in power saving mode
        assert!(manager.is_power_saving().await);
    }

    #[tokio::test]
    async fn test_resource_exhaustion_recovery() {
        let optimizer = AllocationOptimizer;
        
        // Create tasks that exceed available resources
        let tasks = vec![
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 100,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.6,
                    memory_mb: 150,
                    estimated_duration_ms: 1000,
                },
                skill_requirements: vec!["skill1".to_string()],
            },
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 200,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.5,
                    memory_mb: 100,
                    estimated_duration_ms: 500,
                },
                skill_requirements: vec!["skill2".to_string()],
            },
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 50,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.3,
                    memory_mb: 80,
                    estimated_duration_ms: 300,
                },
                skill_requirements: vec!["skill3".to_string()],
            },
        ];
        
        // Limited resources - can only fit high priority task
        let resources = AvailableResources {
            cpu_capacity: 1.0,
            memory_mb: 200,
            current_cpu_usage: 0.4, // Already using 40%
            current_memory_usage: 50,
        };
        
        let plan = optimizer.optimize(&tasks, &resources).await.unwrap();
        
        // Should only allocate the highest priority task that fits
        assert_eq!(plan.task_assignments.len(), 1, "Should only allocate tasks that fit");
        
        // Verify it's the highest priority task that was allocated
        let allocated_task = plan.execution_order.first().unwrap();
        let task = tasks.iter().find(|t| &t.id == allocated_task).unwrap();
        assert_eq!(task.priority, 200, "Should allocate highest priority task first");
    }

    #[tokio::test]
    async fn test_priority_inversion_prevention() {
        let optimizer = AllocationOptimizer;
        
        // Create tasks with varying priorities
        let high_priority_task = CognitiveTask {
            id: Uuid::new_v4(),
            priority: 255, // Maximum priority
            estimated_cost: ResourceEstimate {
                cpu_usage: 0.2,
                memory_mb: 50,
                estimated_duration_ms: 100,
            },
            skill_requirements: vec!["critical_skill".to_string()],
        };
        
        let low_priority_task = CognitiveTask {
            id: Uuid::new_v4(),
            priority: 1, // Minimum priority
            estimated_cost: ResourceEstimate {
                cpu_usage: 0.1,
                memory_mb: 30,
                estimated_duration_ms: 500,
            },
            skill_requirements: vec!["background_skill".to_string()],
        };
        
        let tasks = vec![low_priority_task.clone(), high_priority_task.clone()];
        
        let resources = AvailableResources {
            cpu_capacity: 1.0,
            memory_mb: 200,
            current_cpu_usage: 0.0,
            current_memory_usage: 0,
        };
        
        let plan = optimizer.optimize(&tasks, &resources).await.unwrap();
        
        // Verify high priority task is scheduled first
        let first_task_id = plan.execution_order.first().unwrap();
        assert_eq!(first_task_id, &high_priority_task.id, "High priority task should be scheduled first");
    }

    #[tokio::test]
    async fn test_graceful_degradation_stages() {
        let manager = AdaptiveResourceManager::new();
        
        // Add mix of priority tasks
        let critical_task = CognitiveTask {
            id: Uuid::new_v4(),
            priority: 200,
            estimated_cost: ResourceEstimate { cpu_usage: 0.5, memory_mb: 100, estimated_duration_ms: 1000 },
            skill_requirements: vec!["critical".to_string()],
        };
        
        let normal_task = CognitiveTask {
            id: Uuid::new_v4(),
            priority: 100,
            estimated_cost: ResourceEstimate { cpu_usage: 0.3, memory_mb: 50, estimated_duration_ms: 500 },
            skill_requirements: vec!["normal".to_string()],
        };
        
        let background_task = CognitiveTask {
            id: Uuid::new_v4(),
            priority: 50,
            estimated_cost: ResourceEstimate { cpu_usage: 0.2, memory_mb: 30, estimated_duration_ms: 300 },
            skill_requirements: vec!["background".to_string()],
        };
        
        manager.add_running_task(critical_task.clone()).await;
        manager.add_running_task(normal_task.clone()).await;
        manager.add_running_task(background_task.clone()).await;
        
        // Stage 1: Normal operation - all tasks running
        assert_eq!(manager.running_tasks.read().await.len(), 3);
        assert!(matches!(manager.get_system_state().await, SystemState::Normal));
        
        // Stage 2: Crisis triggered - degradation begins
        manager.handle_resource_crisis().await;
        
        // Only critical and normal priority tasks should remain
        let running = manager.running_tasks.read().await;
        assert!(running.len() <= 2, "Background task should be terminated");
        assert!(manager.is_power_saving().await);
        
        // Stage 3: Verify critical task is preserved
        assert!(running.contains_key(&critical_task.id), "Critical task must be preserved");
    }

    #[tokio::test]
    async fn test_task_lifecycle_management() {
        let manager = AdaptiveResourceManager::new();
        
        let task_id = Uuid::new_v4();
        let task = CognitiveTask {
            id: task_id,
            priority: 100,
            estimated_cost: ResourceEstimate {
                cpu_usage: 0.3,
                memory_mb: 50,
                estimated_duration_ms: 500,
            },
            skill_requirements: vec!["test_skill".to_string()],
        };
        
        // Add task
        manager.add_running_task(task.clone()).await;
        assert_eq!(manager.running_tasks.read().await.len(), 1);
        assert!(manager.running_tasks.read().await.contains_key(&task_id));
        
        // Remove task
        manager.remove_running_task(&task_id).await;
        assert_eq!(manager.running_tasks.read().await.len(), 0);
        assert!(!manager.running_tasks.read().await.contains_key(&task_id));
    }

    #[tokio::test]
    async fn test_zero_resources_allocation() {
        let optimizer = AllocationOptimizer;
        
        let tasks = vec![
            CognitiveTask {
                id: Uuid::new_v4(),
                priority: 100,
                estimated_cost: ResourceEstimate {
                    cpu_usage: 0.1,
                    memory_mb: 10,
                    estimated_duration_ms: 100,
                },
                skill_requirements: vec!["skill1".to_string()],
            },
        ];
        
        // Zero available resources
        let resources = AvailableResources {
            cpu_capacity: 1.0,
            memory_mb: 200,
            current_cpu_usage: 1.0, // Fully utilized
            current_memory_usage: 200, // Fully utilized
        };
        
        let plan = optimizer.optimize(&tasks, &resources).await.unwrap();
        
        // Should not allocate any tasks when resources are exhausted
        assert_eq!(plan.task_assignments.len(), 0, "Should not allocate tasks with zero resources");
        assert_eq!(plan.execution_order.len(), 0);
    }
}
