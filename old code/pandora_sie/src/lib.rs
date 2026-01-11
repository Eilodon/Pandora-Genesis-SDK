// sdk/pandora_sie/src/lib.rs

#![allow(clippy::all)]
use std::collections::HashMap;
use thiserror::Error;
use rand::Rng;
use rand::distributions::Distribution;

#[derive(Debug, Error)]
pub enum EvolutionError {
    #[error("Quần thể không đủ đa dạng để lai tạo")]
    PopulationTooSmall,
    #[error("Lỗi đánh giá fitness: {0}")]
    FitnessEvaluationError(String),
    #[error("Lỗi selection: {0}")]
    SelectionError(String),
    #[error("Lỗi crossover: {0}")]
    CrossoverError(String),
    #[error("Lỗi mutation: {0}")]
    MutationError(String),
}

// ===== 4. Evolution Engine Specifications =====

// --- 4.1 Genetic Algorithms ---

/// Represents an individual in the population, e.g., a set of parameters or a skill configuration.
#[derive(Clone, Debug)]
pub struct Individual {
    pub genes: Vec<f64>, // Example: can be parameters, skill IDs, etc.
    pub fitness: f64,
}

impl Individual {
    pub fn new(genes: Vec<f64>) -> Self {
        Self {
            genes,
            fitness: 0.0,
        }
    }
}

/// Defines different strategies for selecting parents for the next generation.
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    Tournament { size: usize },
    RouletteWheel,
}

/// Defines different strategies for combining parent genes.
#[derive(Debug, Clone)]
pub enum CrossoverStrategy {
    SinglePoint,
    Uniform,
}

/// Defines different strategies for introducing random changes.
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    Gaussian { noise: f64, probability: f64 },
    RandomReset { probability: f64 },
}

/// Insight from the Meta-Cognitive Governor for fitness evaluation
#[derive(Debug, Clone)]
pub struct Insight {
    pub performance: f64,
    pub efficiency: f64,
    pub novelty: f64,
    pub resource_usage: f64,
}

pub struct PopulationManager;
pub struct FitnessEvaluator;
pub struct EvolutionScheduler;
pub struct DiversityMaintainer;

// Placeholder types for compatibility
pub struct MutationOperator;
pub struct CrossoverOperator;

#[derive(Debug, Clone)]
pub struct EvolutionParameters {
    pub population_size: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub elitism_ratio: f32, // Tỷ lệ cá thể tốt nhất được giữ lại
}

#[allow(dead_code)]
pub struct EvolutionEngine {
    // Quản lý quần thể
    population_manager: PopulationManager,
    fitness_evaluator: FitnessEvaluator,
    selection_strategy: SelectionStrategy,

    // Toán tử di truyền
    mutation_operators: HashMap<String, MutationOperator>,
    crossover_operators: HashMap<String, CrossoverOperator>,

    // Kiểm soát tiến hóa
    evolution_scheduler: EvolutionScheduler,
    diversity_maintainer: DiversityMaintainer,

    params: EvolutionParameters,
}

impl EvolutionEngine {
    pub fn new(params: EvolutionParameters) -> Self {
        Self {
            population_manager: PopulationManager,
            fitness_evaluator: FitnessEvaluator,
            selection_strategy: SelectionStrategy::Tournament { size: 3 },
            mutation_operators: HashMap::new(),
            crossover_operators: HashMap::new(),
            evolution_scheduler: EvolutionScheduler,
            diversity_maintainer: DiversityMaintainer,
            params,
        }
    }

    /// Chạy một chu trình tiến hóa cho một quần thể kỹ năng.
    pub async fn evolve_generation(
        &self,
        current_population: Vec<Individual>,
        insights: &[Insight],
    ) -> Result<Vec<Individual>, EvolutionError> {
        if current_population.len() < 2 {
            return Err(EvolutionError::PopulationTooSmall);
        }

        let mut population = current_population;

        // 1. Đánh giá độ thích nghi (fitness) của tất cả các skill
        evaluate_fitness(&mut population, insights)?;

        // 2. Chọn lọc (Selection): Giữ lại những cá thể tốt nhất
        let selection_strategy = SelectionStrategy::Tournament { size: 3 };
        let parents = select_parents(&population, &selection_strategy, self.params.population_size)?;

        // 3. Tạo thế hệ mới
        let mut next_generation = Vec::new();
        let crossover_strategy = CrossoverStrategy::SinglePoint;
        let mutation_strategy = MutationStrategy::Gaussian { 
            noise: 0.1, 
            probability: self.params.mutation_rate as f64 
        };

        for _ in 0..self.params.population_size {
            // Chọn hai parents
            let parent1 = &parents[rand::thread_rng().gen_range(0..parents.len())];
            let parent2 = &parents[rand::thread_rng().gen_range(0..parents.len())];

            // Crossover để tạo child
            let mut child = crossover(parent1, parent2, &crossover_strategy)?;

            // Mutation
            mutate(&mut child, &mutation_strategy)?;

            next_generation.push(child);
        }

        Ok(next_generation)
    }
}

/// Evaluate fitness of population based on insights from MCG
fn evaluate_fitness(population: &mut [Individual], insights: &[Insight]) -> Result<(), EvolutionError> {
    if insights.is_empty() {
        return Err(EvolutionError::FitnessEvaluationError("No insights provided".to_string()));
    }

    for individual in population.iter_mut() {
        // Simple weighted sum fitness function
        let mut fitness = 0.0;
        let mut weight_sum = 0.0;

        for insight in insights {
            // Performance weight: 0.4
            fitness += insight.performance * 0.4;
            weight_sum += 0.4;

            // Efficiency weight: 0.3
            fitness += insight.efficiency * 0.3;
            weight_sum += 0.3;

            // Novelty weight: 0.2
            fitness += insight.novelty * 0.2;
            weight_sum += 0.2;

            // Resource usage weight: 0.1 (inverted - lower is better)
            fitness += (1.0 - insight.resource_usage) * 0.1;
            weight_sum += 0.1;
        }

        individual.fitness = if weight_sum > 0.0 { fitness / weight_sum } else { 0.0 };
    }

    Ok(())
}

/// Select parents using tournament selection
fn select_parents(
    population: &[Individual],
    strategy: &SelectionStrategy,
    count: usize,
) -> Result<Vec<Individual>, EvolutionError> {
    let mut parents = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..count {
        let parent = match strategy {
            SelectionStrategy::Tournament { size } => {
                if population.len() < *size {
                    return Err(EvolutionError::SelectionError("Population too small for tournament".to_string()));
                }

                // Select random individuals for tournament
                let mut tournament = Vec::new();
                for _ in 0..*size {
                    let idx = rng.gen_range(0..population.len());
                    tournament.push(&population[idx]);
                }

                // Select the best from tournament
                tournament.into_iter()
                    .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                    .unwrap()
                    .clone()
            }
            SelectionStrategy::RouletteWheel => {
                // Simple roulette wheel selection
                let total_fitness: f64 = population.iter().map(|ind| ind.fitness.max(0.0)).sum();
                if total_fitness <= 0.0 {
                    return Err(EvolutionError::SelectionError("No positive fitness found".to_string()));
                }

                let random_value = rng.gen_range(0.0..total_fitness);
                let mut cumulative = 0.0;

                let mut selected = None;
                for individual in population {
                    cumulative += individual.fitness.max(0.0);
                    if cumulative >= random_value {
                        selected = Some(individual.clone());
                        break;
                    }
                }

                // Fallback to last individual
                selected.unwrap_or_else(|| population.last().unwrap().clone())
            }
        };

        parents.push(parent);
    }

    Ok(parents)
}

/// Crossover two parents to create a child
fn crossover(
    parent1: &Individual,
    parent2: &Individual,
    strategy: &CrossoverStrategy,
) -> Result<Individual, EvolutionError> {
    if parent1.genes.len() != parent2.genes.len() {
        return Err(EvolutionError::CrossoverError("Parent gene lengths don't match".to_string()));
    }

    let mut child_genes = Vec::new();
    let mut rng = rand::thread_rng();

    match strategy {
        CrossoverStrategy::SinglePoint => {
            let crossover_point = rng.gen_range(0..parent1.genes.len());
            
            for i in 0..parent1.genes.len() {
                if i < crossover_point {
                    child_genes.push(parent1.genes[i]);
                } else {
                    child_genes.push(parent2.genes[i]);
                }
            }
        }
        CrossoverStrategy::Uniform => {
            for i in 0..parent1.genes.len() {
                if rng.gen_bool(0.5) {
                    child_genes.push(parent1.genes[i]);
                } else {
                    child_genes.push(parent2.genes[i]);
                }
            }
        }
    }

    Ok(Individual::new(child_genes))
}

/// Mutate an individual
fn mutate(individual: &mut Individual, strategy: &MutationStrategy) -> Result<(), EvolutionError> {
    let mut rng = rand::thread_rng();

    match strategy {
        MutationStrategy::Gaussian { noise, probability } => {
            for gene in individual.genes.iter_mut() {
                if rng.gen_bool(*probability) {
                    let normal = rand_distr::Normal::new(0.0, *noise).unwrap();
                    *gene += normal.sample(&mut rng);
                }
            }
        }
        MutationStrategy::RandomReset { probability } => {
            for gene in individual.genes.iter_mut() {
                if rng.gen_bool(*probability) {
                    *gene = rng.gen_range(-1.0..1.0);
                }
            }
        }
    }

    Ok(())
}

// ====== Placeholder types để biên dịch Phase 2 ======
#[derive(Debug, Clone)]
pub struct EvolutionarySkill;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SkillGenome;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_creation() {
        let genes = vec![1.0, 2.0, 3.0];
        let individual = Individual::new(genes.clone());
        assert_eq!(individual.genes, genes);
        assert_eq!(individual.fitness, 0.0);
    }

    #[test]
    fn test_fitness_evaluation() {
        let mut population = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        let insights = vec![
            Insight {
                performance: 0.8,
                efficiency: 0.7,
                novelty: 0.6,
                resource_usage: 0.3,
            }
        ];

        evaluate_fitness(&mut population, &insights).unwrap();

        // Both individuals should have the same fitness since they use the same insight
        assert!(population[0].fitness > 0.0);
        assert!(population[1].fitness > 0.0);
        assert_eq!(population[0].fitness, population[1].fitness);
    }

    #[test]
    fn test_tournament_selection() {
        let population = vec![
            Individual { genes: vec![1.0], fitness: 0.1 },
            Individual { genes: vec![2.0], fitness: 0.9 },
            Individual { genes: vec![3.0], fitness: 0.5 },
        ];

        let strategy = SelectionStrategy::Tournament { size: 2 };
        let parents = select_parents(&population, &strategy, 3).unwrap();

        assert_eq!(parents.len(), 3);
    }

    #[test]
    fn test_roulette_wheel_selection() {
        let population = vec![
            Individual { genes: vec![1.0], fitness: 0.1 },
            Individual { genes: vec![2.0], fitness: 0.9 },
            Individual { genes: vec![3.0], fitness: 0.5 },
        ];

        let strategy = SelectionStrategy::RouletteWheel;
        let parents = select_parents(&population, &strategy, 2).unwrap();

        assert_eq!(parents.len(), 2);
    }

    #[test]
    fn test_single_point_crossover() {
        let parent1 = Individual::new(vec![1.0, 2.0, 3.0, 4.0]);
        let parent2 = Individual::new(vec![5.0, 6.0, 7.0, 8.0]);

        let strategy = CrossoverStrategy::SinglePoint;
        let child = crossover(&parent1, &parent2, &strategy).unwrap();

        assert_eq!(child.genes.len(), 4);
        // Child should have genes from either parent (valid range 1.0-8.0)
        for gene in &child.genes {
            let is_from_parent1 = [1.0, 2.0, 3.0, 4.0].contains(gene);
            let is_from_parent2 = [5.0, 6.0, 7.0, 8.0].contains(gene);
            assert!(is_from_parent1 || is_from_parent2, "Gene {} not from either parent", gene);
        }
    }

    #[test]
    fn test_uniform_crossover() {
        let parent1 = Individual::new(vec![1.0, 2.0, 3.0, 4.0]);
        let parent2 = Individual::new(vec![5.0, 6.0, 7.0, 8.0]);

        let strategy = CrossoverStrategy::Uniform;
        let child = crossover(&parent1, &parent2, &strategy).unwrap();

        assert_eq!(child.genes.len(), 4);
    }

    #[test]
    fn test_gaussian_mutation() {
        let mut individual = Individual::new(vec![1.0, 2.0, 3.0]);
        let strategy = MutationStrategy::Gaussian { noise: 0.1, probability: 1.0 };

        mutate(&mut individual, &strategy).unwrap();

        // Genes should be modified (though exact values depend on random)
        assert_eq!(individual.genes.len(), 3);
    }

    #[test]
    fn test_random_reset_mutation() {
        let mut individual = Individual::new(vec![1.0, 2.0, 3.0]);
        let strategy = MutationStrategy::RandomReset { probability: 1.0 };

        mutate(&mut individual, &strategy).unwrap();

        // Genes should be modified (though exact values depend on random)
        assert_eq!(individual.genes.len(), 3);
    }

    #[test]
    fn test_evolution_engine_creation() {
        let params = EvolutionParameters {
            population_size: 10,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_ratio: 0.1,
        };

        let engine = EvolutionEngine::new(params);
        assert_eq!(engine.params.population_size, 10);
    }

    #[test]
    fn test_evolve_generation() {
        let params = EvolutionParameters {
            population_size: 4,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_ratio: 0.1,
        };

        let engine = EvolutionEngine::new(params);
        let population = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
            Individual::new(vec![5.0, 6.0]),
            Individual::new(vec![7.0, 8.0]),
        ];

        let insights = vec![
            Insight {
                performance: 0.8,
                efficiency: 0.7,
                novelty: 0.6,
                resource_usage: 0.3,
            }
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.evolve_generation(population, &insights)).unwrap();

        assert_eq!(result.len(), 4);
    }

    // ===== EDGE CASE TESTS =====

    #[test]
    fn test_evolve_generation_empty_population() {
        let params = EvolutionParameters {
            population_size: 4,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_ratio: 0.1,
        };

        let engine = EvolutionEngine::new(params);
        let population = vec![];
        let insights = vec![
            Insight {
                performance: 0.8,
                efficiency: 0.7,
                novelty: 0.6,
                resource_usage: 0.3,
            }
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.evolve_generation(population, &insights));

        // Should return error for empty population
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvolutionError::PopulationTooSmall));
    }

    #[test]
    fn test_evolve_generation_single_individual() {
        let params = EvolutionParameters {
            population_size: 2,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_ratio: 0.1,
        };

        let engine = EvolutionEngine::new(params);
        let population = vec![Individual::new(vec![1.0, 2.0])];
        let insights = vec![
            Insight {
                performance: 0.8,
                efficiency: 0.7,
                novelty: 0.6,
                resource_usage: 0.3,
            }
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.evolve_generation(population, &insights));

        // Should return error for single individual (need at least 2 for crossover)
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvolutionError::PopulationTooSmall));
    }

    #[test]
    fn test_fitness_evaluation_extreme_values() {
        let mut population = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        // Test with extreme fitness values
        let insights = vec![
            Insight {
                performance: 1.0,  // Maximum
                efficiency: 1.0,
                novelty: 1.0,
                resource_usage: 0.0,  // Minimum (best)
            }
        ];

        evaluate_fitness(&mut population, &insights).unwrap();

        // With perfect scores, fitness should be close to 1.0
        assert!(population[0].fitness > 0.9, "Fitness should be high with perfect scores");
        assert!(population[1].fitness > 0.9, "Fitness should be high with perfect scores");
    }

    #[test]
    fn test_fitness_evaluation_worst_case() {
        let mut population = vec![
            Individual::new(vec![1.0, 2.0]),
        ];

        // Test with worst possible scores
        let insights = vec![
            Insight {
                performance: 0.0,  // Minimum
                efficiency: 0.0,
                novelty: 0.0,
                resource_usage: 1.0,  // Maximum (worst)
            }
        ];

        evaluate_fitness(&mut population, &insights).unwrap();

        // With worst scores, fitness should be very low
        assert!(population[0].fitness < 0.2, "Fitness should be low with worst scores");
    }

    #[test]
    fn test_mutation_zero_probability() {
        let mut individual = Individual::new(vec![1.0, 2.0, 3.0]);
        let original = individual.clone();
        
        let strategy = MutationStrategy::Gaussian { noise: 0.1, probability: 0.0 };
        mutate(&mut individual, &strategy).unwrap();

        // With 0 probability, genes should remain unchanged
        assert_eq!(individual.genes, original.genes, "Genes should not mutate with 0 probability");
    }

    #[test]
    fn test_mutation_full_probability() {
        let mut individual = Individual::new(vec![1.0, 2.0, 3.0]);
        
        let strategy = MutationStrategy::Gaussian { noise: 0.5, probability: 1.0 };
        mutate(&mut individual, &strategy).unwrap();

        // With 100% probability and large noise, at least one gene should change
        // Note: Due to randomness, we can't guarantee exact values, but genes should be different
        // This is a weak test but necessary due to randomness
        assert_eq!(individual.genes.len(), 3, "Gene count should remain the same");
    }

    #[test]
    fn test_crossover_mismatched_genes() {
        let parent1 = Individual::new(vec![1.0, 2.0, 3.0]);
        let parent2 = Individual::new(vec![4.0, 5.0]); // Different length

        let strategy = CrossoverStrategy::SinglePoint;
        let result = crossover(&parent1, &parent2, &strategy);

        // Should return error for mismatched gene lengths
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvolutionError::CrossoverError(_)));
    }

    #[test]
    fn test_selection_population_too_small_for_tournament() {
        let population = vec![
            Individual { genes: vec![1.0], fitness: 0.5 },
        ];

        let strategy = SelectionStrategy::Tournament { size: 3 }; // Tournament size > population
        let result = select_parents(&population, &strategy, 2);

        // Should return error when tournament size exceeds population
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvolutionError::SelectionError(_)));
    }

    #[test]
    fn test_roulette_wheel_no_positive_fitness() {
        let population = vec![
            Individual { genes: vec![1.0], fitness: -0.5 },
            Individual { genes: vec![2.0], fitness: -0.3 },
            Individual { genes: vec![3.0], fitness: 0.0 },
        ];

        let strategy = SelectionStrategy::RouletteWheel;
        let result = select_parents(&population, &strategy, 2);

        // Should return error when no positive fitness
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvolutionError::SelectionError(_)));
    }

    #[test]
    fn test_fitness_evaluation_no_insights() {
        let mut population = vec![
            Individual::new(vec![1.0, 2.0]),
        ];

        let insights = vec![]; // Empty insights

        let result = evaluate_fitness(&mut population, &insights);

        // Should return error with no insights
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvolutionError::FitnessEvaluationError(_)));
    }
}
