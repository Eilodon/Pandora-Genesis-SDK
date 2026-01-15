# **🎯 MASTER EXECUTION PLAN**

## **From Current Codebase → Causal AGI Foundation**

**Premise:** Fen là founder/lead engineer. Plan này assume 1-2 core engineers \+ potential collaborators.

---

# **📋 PHASE 0: FOUNDATION AUDIT (Weeks 1-2)**

## **Week 1: Technical Debt Mapping**

### **Task 0.1: Test Coverage Analysis**

// Current state audit  
CRITICAL ISSUES:  
├── dagma.rs  
│   └── test\_dagma\_simple\_chain() \[IGNORED\] ⚠️  
├── async\_dagma.rs  
│   └── Coverage: 60% (missing error cases)  
├── hypergraph.rs  
│   └── Coverage: 70% (missing edge cases)  
└── graph\_change\_detector.rs  
    └── Coverage: 40% (primitive implementation)

ACTION ITEMS:  
1\. Create test\_status.md documenting ALL ignored/missing tests  
2\. Prioritize by criticality (P0: core algos, P1: features, P2: nice-to-have)  
3\. Estimate effort (each test \= 0.5-2 days)

**Deliverable:** `docs/test_audit.md` with full coverage map

### **Task 0.2: Performance Baseline**

\# Create benchmark suite  
cargo new \--lib benchmarks/dagma\_bench

\# Benchmark targets:  
1\. DAGMA convergence time (varying n\_vars: 5, 10, 50, 100\)  
2\. AsyncDagma throughput (requests/sec)  
3\. Hypergraph prediction latency  
4\. Memory usage under load

\# Deliverable: benchmarks/baseline\_results.json  
{  
  "dagma\_fit\_10vars": { "mean\_ms": 45, "std\_ms": 5 },  
  "async\_submit\_latency": { "mean\_us": 150, "std\_us": 20 },  
  ...  
}

**Deliverable:** `benchmarks/baseline_report.md`

### **Task 0.3: Dependency Audit**

\# Cargo.toml analysis  
CURRENT DEPENDENCIES:  
├── nalgebra (matrix ops) \- KEEP  
├── serde (serialization) \- KEEP  
├── rand (random init) \- KEEP  
└── \[missing critical deps\]

NEEDED ADDITIONS:  
├── statrs (statistical tests) \- for distribution drift  
├── approx (float comparison) \- for robust testing    
├── criterion (benchmarking) \- for performance tracking  
└── proptest (property testing) \- for algorithm validation

**Deliverable:** `docs/dependency_strategy.md`

## **Week 2: Architecture Documentation**

### **Task 0.4: Create Architecture Decision Records (ADRs)**

\# ADR-001: Why DAGMA over NOTEARS  
Status: Accepted  
Context: Need fast causal discovery  
Decision: Use DAGMA (log-det acyclicity)  
Consequences: 20x speedup, but newer algorithm (less battle-tested)

\# ADR-002: Why Hypergraph over Pairwise  
Status: Accepted  
Context: Physiological systems have multi-way interactions  
Decision: Implement hypergraph support  
Consequences: More expressive, but computationally expensive

\# ADR-003: Why Async Architecture  
Status: Accepted  
Context: Real-time constraints (1-5ms latency)  
Decision: Background causal learning thread  
Consequences: Non-blocking, but complexity in state sync

**Deliverable:** `docs/adr/` folder with 10+ ADRs

### **Task 0.5: API Stabilization Plan**

// Current API surface analysis  
pub struct Dagma { ... }           // STABLE (core algorithm)  
pub struct AsyncDagma { ... }      // UNSTABLE (needs backpressure)  
pub struct CausalHypergraph { ... } // UNSTABLE (needs cycle detection)  
pub struct GraphChangeDetector { ... } // NEEDS REWRITE

// Define stability tiers:  
Tier 1 (Stable): Won't change without major version bump  
Tier 2 (Evolving): May change with minor version bump    
Tier 3 (Experimental): May change anytime

// Deliverable: docs/api\_stability.md

---

# **📋 PHASE 1: SOLIDIFY CORE (Months 1-3)**

## **Month 1: Fix Critical Path**

### **Sprint 1.1: DAGMA Hyperparameter Tuning (Weeks 1-2)**

**Engineer Assignment:** Lead engineer

**Tasks:**

// Task 1.1.1: Implement adaptive hyperparameters  
pub struct DagmaConfig {  
    // OLD: Fixed values  
    // pub lambda: f32,  
    // pub lr: f32,  
      
    // NEW: Adaptive based on data  
    pub fn adaptive(data: \&DMatrix\<f32\>) \-\> Self {  
        let n\_vars \= data.ncols() as f32;  
        let n\_samples \= data.nrows() as f32;  
        let data\_variance \= compute\_variance(data);  
          
        Self {  
            // Scale sparsity with data variance  
            lambda: 0.02 \* data\_variance.sqrt(),  
              
            // Scale learning rate with graph size  
            lr: 0.02 / n\_vars.sqrt(),  
              
            // Scale penalty based on sample size  
            rho\_init: (n\_samples / 100.0).max(1.0),  
              
            // More iterations for larger graphs  
            max\_iter: (50.0 \* n\_vars.log2()).ceil() as usize,  
              
            ..Default::default()  
        }  
    }  
}

// Task 1.1.2: Un-ignore and fix test\_dagma\_simple\_chain  
\#\[test\]  
fn test\_dagma\_simple\_chain() {  
    let data \= generate\_chain\_data(n\_samples=500); // Increased samples  
    let config \= DagmaConfig::adaptive(\&data);     // Use adaptive config  
    let dagma \= Dagma::new(3, Some(config));  
    let w \= dagma.fit(\&data);  
      
    // More lenient thresholds for discovery  
    assert\!(w\[(0, 1)\].abs() \> 0.05, "Should detect X1-\>X2");  
    assert\!(w\[(1, 2)\].abs() \> 0.05, "Should detect X2-\>X3");  
      
    // But strict on acyclicity  
    let h \= dagma.h\_logdet(\&w);  
    assert\!(h.abs() \< 1e-6, "Must be acyclic");  
}

**Success Criteria:**

* \[ \] All DAGMA tests pass without \#\[ignore\]  
* \[ \] Adaptive config works for n\_vars ∈ \[3, 100\]  
* \[ \] Convergence rate improved by 30%+

**Deliverable:** `git tag v0.2.0-dagma-stable`

### **Sprint 1.2: AsyncDagma Backpressure (Week 3\)**

**Engineer Assignment:** Lead engineer

**Tasks:**

// Task 1.2.1: Add backpressure mechanism  
pub struct AsyncDagma {  
    // NEW: Bounded channel instead of unbounded  
    tx: SyncSender\<DagmaRequest\>,  // Blocks when full  
    rx: Receiver\<DagmaResult\>,  
      
    // NEW: Request queue management  
    max\_pending\_requests: usize,  
    pending\_count: Arc\<AtomicUsize\>,  
}

impl AsyncDagma {  
    pub fn spawn(n\_vars: usize, config: Option\<DagmaConfig\>) \-\> Self {  
        // Bounded channel (max 2 pending requests)  
        let (tx, rx) \= sync\_channel::\<DagmaRequest\>(2);  
          
        // Rest of implementation...  
    }  
      
    // Task 1.2.2: Non-blocking submit with error handling  
    pub fn try\_submit(  
        \&mut self,   
        data: DMatrix\<f32\>,   
        warm\_start: Option\<\&DMatrix\<f32\>\>  
    ) \-\> Result\<u64, SubmitError\> {  
        // Check if queue is full  
        if self.pending\_count.load(Ordering::Relaxed) \>= self.max\_pending\_requests {  
            return Err(SubmitError::QueueFull);  
        }  
          
        let request\_id \= self.next\_request\_id;  
        self.next\_request\_id \+= 1;  
          
        let request \= DagmaRequest { data, warm\_start: warm\_start.cloned(), request\_id };  
          
        // Non-blocking send  
        match self.tx.try\_send(request) {  
            Ok(\_) \=\> {  
                self.pending\_count.fetch\_add(1, Ordering::Relaxed);  
                Ok(request\_id)  
            }  
            Err(TrySendError::Full(\_)) \=\> Err(SubmitError::QueueFull),  
            Err(TrySendError::Disconnected(\_)) \=\> Err(SubmitError::WorkerDied),  
        }  
    }  
}

// Task 1.2.3: Add metrics  
pub struct AsyncDagmaMetrics {  
    pub total\_requests: u64,  
    pub completed\_requests: u64,  
    pub failed\_requests: u64,  
    pub avg\_processing\_time\_ms: f32,  
    pub queue\_depth: usize,  
}

impl AsyncDagma {  
    pub fn metrics(\&self) \-\> AsyncDagmaMetrics {  
        // Return current metrics for monitoring  
    }  
}

**Success Criteria:**

* \[ \] No unbounded queue growth under load  
* \[ \] Graceful degradation when overloaded  
* \[ \] Metrics for monitoring health

**Deliverable:** `git tag v0.2.1-async-backpressure`

### **Sprint 1.3: Hypergraph Cycle Detection (Week 4\)**

**Engineer Assignment:** Lead engineer

**Tasks:**

// Task 1.3.1: Implement topological sort  
impl CausalHypergraph {  
    pub fn add\_hyperedge(\&mut self, edge: HyperEdge) \-\> Result\<(), GraphError\> {  
        // Temporarily add edge  
        self.hyperedges.push(edge.clone());  
          
        // Check for cycles  
        if self.has\_cycle()? {  
            // Rollback  
            self.hyperedges.pop();  
            return Err(GraphError::CycleDetected {  
                edge: edge,  
                cycle\_path: self.find\_cycle\_path(),  
            });  
        }  
          
        Ok(())  
    }  
      
    fn has\_cycle(\&self) \-\> Result\<bool, GraphError\> {  
        // Build adjacency list  
        let mut graph: HashMap\<Variable, Vec\<Variable\>\> \= HashMap::new();  
          
        for edge in \&self.hyperedges {  
            for \&source in \&edge.sources {  
                graph.entry(source)  
                    .or\_default()  
                    .push(edge.target);  
            }  
        }  
          
        // DFS with recursion stack  
        let mut visited \= HashSet::new();  
        let mut rec\_stack \= HashSet::new();  
          
        for \&node in graph.keys() {  
            if self.dfs\_has\_cycle(\&graph, node, \&mut visited, \&mut rec\_stack) {  
                return Ok(true);  
            }  
        }  
          
        Ok(false)  
    }  
      
    fn dfs\_has\_cycle(  
        \&self,  
        graph: \&HashMap\<Variable, Vec\<Variable\>\>,  
        node: Variable,  
        visited: \&mut HashSet\<Variable\>,  
        rec\_stack: \&mut HashSet\<Variable\>,  
    ) \-\> bool {  
        if rec\_stack.contains(\&node) {  
            return true; // Cycle found  
        }  
          
        if visited.contains(\&node) {  
            return false; // Already processed  
        }  
          
        visited.insert(node);  
        rec\_stack.insert(node);  
          
        if let Some(neighbors) \= graph.get(\&node) {  
            for \&neighbor in neighbors {  
                if self.dfs\_has\_cycle(graph, neighbor, visited, rec\_stack) {  
                    return true;  
                }  
            }  
        }  
          
        rec\_stack.remove(\&node);  
        false  
    }  
      
    fn find\_cycle\_path(\&self) \-\> Vec\<Variable\> {  
        // Return the actual cycle for debugging  
        // Implementation: backtrack from cycle detection  
    }  
}

// Task 1.3.2: Add comprehensive tests  
\#\[test\]  
fn test\_cycle\_detection() {  
    let mut graph \= CausalHypergraph::new();  
      
    // Valid DAG: A-\>B-\>C  
    assert\!(graph.add\_interaction(vec\!\[Variable::A\], Variable::B, ...).is\_ok());  
    assert\!(graph.add\_interaction(vec\!\[Variable::B\], Variable::C, ...).is\_ok());  
      
    // Invalid: C-\>A creates cycle  
    let result \= graph.add\_interaction(vec\!\[Variable::C\], Variable::A, ...);  
    assert\!(result.is\_err());  
      
    match result {  
        Err(GraphError::CycleDetected { cycle\_path, .. }) \=\> {  
            assert\_eq\!(cycle\_path, vec\!\[Variable::A, Variable::B, Variable::C, Variable::A\]);  
        }  
        \_ \=\> panic\!("Expected CycleDetected error"),  
    }  
}

**Success Criteria:**

* \[ \] Cycle detection works for all graph sizes  
* \[ \] Performance: O(V+E) complexity  
* \[ \] Helpful error messages with cycle path

**Deliverable:** `git tag v0.2.2-hypergraph-safe`

## **Month 2: Smart Distribution Drift Detection**

### **Sprint 2.1: Statistical Tests (Weeks 5-6)**

**Engineer Assignment:** Lead \+ Junior/Collaborator

**Tasks:**

// Task 2.1.1: Add statrs dependency  
// Cargo.toml  
\[dependencies\]  
statrs \= "0.16"

// Task 2.1.2: Implement KS test  
use statrs::statistics::Statistics;  
use statrs::distribution::{ContinuousCDF, Normal};

pub struct DistributionDriftDetector {  
    // Window of old samples  
    reference\_window: VecDeque\<Vec\<f32\>\>,  
    window\_size: usize,  
      
    // Thresholds  
    ks\_threshold: f32,  
    chi2\_threshold: f32,  
}

impl DistributionDriftDetector {  
    pub fn new(window\_size: usize) \-\> Self {  
        Self {  
            reference\_window: VecDeque::with\_capacity(window\_size),  
            window\_size,  
            ks\_threshold: 0.1,  // Moderate sensitivity  
            chi2\_threshold: 0.05, // p-value threshold  
        }  
    }  
      
    pub fn detect\_drift(  
        \&mut self,   
        new\_samples: &\[Vec\<f32\>\]  
    ) \-\> DriftReport {  
        if self.reference\_window.len() \< self.window\_size / 2 {  
            // Not enough data yet  
            return DriftReport::Insufficient;  
        }  
          
        let mut variable\_drifts \= Vec::new();  
          
        // Check each variable independently  
        for var\_idx in 0..new\_samples\[0\].len() {  
            let old\_values: Vec\<f32\> \= self.reference\_window  
                .iter()  
                .map(|sample| sample\[var\_idx\])  
                .collect();  
              
            let new\_values: Vec\<f32\> \= new\_samples  
                .iter()  
                .map(|sample| sample\[var\_idx\])  
                .collect();  
              
            // Kolmogorov-Smirnov test  
            let ks\_stat \= self.ks\_statistic(\&old\_values, \&new\_values);  
              
            // Chi-square test for independence  
            let chi2\_pvalue \= self.chi\_square\_test(\&old\_values, \&new\_values);  
              
            variable\_drifts.push(VariableDrift {  
                variable: Variable::from\_index(var\_idx),  
                ks\_statistic: ks\_stat,  
                chi2\_pvalue: chi2\_pvalue,  
                drift\_detected: ks\_stat \> self.ks\_threshold || chi2\_pvalue \< self.chi2\_threshold,  
            });  
        }  
          
        DriftReport::Analyzed {  
            overall\_drift: variable\_drifts.iter().any(|d| d.drift\_detected),  
            per\_variable: variable\_drifts,  
            recommendation: self.recommend\_action(\&variable\_drifts),  
        }  
    }  
      
    fn ks\_statistic(\&self, sample1: &\[f32\], sample2: &\[f32\]) \-\> f32 {  
        // Two-sample Kolmogorov-Smirnov test  
        let mut s1\_sorted \= sample1.to\_vec();  
        let mut s2\_sorted \= sample2.to\_vec();  
        s1\_sorted.sort\_by(|a, b| a.partial\_cmp(b).unwrap());  
        s2\_sorted.sort\_by(|a, b| a.partial\_cmp(b).unwrap());  
          
        let n1 \= s1\_sorted.len() as f32;  
        let n2 \= s2\_sorted.len() as f32;  
          
        let mut max\_diff \= 0.0;  
        let mut i1 \= 0;  
        let mut i2 \= 0;  
          
        while i1 \< s1\_sorted.len() && i2 \< s2\_sorted.len() {  
            let cdf1 \= i1 as f32 / n1;  
            let cdf2 \= i2 as f32 / n2;  
              
            max\_diff \= max\_diff.max((cdf1 \- cdf2).abs());  
              
            if s1\_sorted\[i1\] \< s2\_sorted\[i2\] {  
                i1 \+= 1;  
            } else {  
                i2 \+= 1;  
            }  
        }  
          
        max\_diff  
    }  
      
    fn chi\_square\_test(\&self, sample1: &\[f32\], sample2: &\[f32\]) \-\> f32 {  
        // Chi-square test for independence  
        // Bin the data and compare distributions  
          
        let n\_bins \= 10;  
        let min\_val \= sample1.iter().chain(sample2.iter())  
            .cloned()  
            .fold(f32::INFINITY, f32::min);  
        let max\_val \= sample1.iter().chain(sample2.iter())  
            .cloned()  
            .fold(f32::NEG\_INFINITY, f32::max);  
          
        let bin\_width \= (max\_val \- min\_val) / n\_bins as f32;  
          
        let hist1 \= self.histogram(sample1, min\_val, bin\_width, n\_bins);  
        let hist2 \= self.histogram(sample2, min\_val, bin\_width, n\_bins);  
          
        // Chi-square statistic  
        let mut chi2 \= 0.0;  
        for i in 0..n\_bins {  
            let expected \= (hist1\[i\] \+ hist2\[i\]) / 2.0;  
            if expected \> 0.0 {  
                chi2 \+= (hist1\[i\] \- expected).powi(2) / expected;  
                chi2 \+= (hist2\[i\] \- expected).powi(2) / expected;  
            }  
        }  
          
        // Convert to p-value (simplified)  
        let df \= (n\_bins \- 1\) as f32;  
        self.chi2\_pvalue(chi2, df)  
    }  
      
    fn recommend\_action(\&self, drifts: &\[VariableDrift\]) \-\> RelearningRecommendation {  
        let drift\_count \= drifts.iter().filter(|d| d.drift\_detected).count();  
        let total\_vars \= drifts.len();  
          
        let drift\_ratio \= drift\_count as f32 / total\_vars as f32;  
          
        if drift\_ratio \> 0.5 {  
            RelearningRecommendation::Immediate  
        } else if drift\_ratio \> 0.2 {  
            RelearningRecommendation::Soon  
        } else if drift\_ratio \> 0.0 {  
            RelearningRecommendation::Monitor  
        } else {  
            RelearningRecommendation::NoAction  
        }  
    }  
}

// Task 2.1.3: Integrate with GraphChangeDetector  
pub struct GraphChangeDetector {  
    drift\_detector: DistributionDriftDetector,  
    sample\_count: usize,  
    min\_samples: usize,  
    last\_trigger\_count: usize,  
      
    // Adaptive trigger interval  
    base\_interval: usize,  
    current\_interval: usize,  
}

impl GraphChangeDetector {  
    pub fn should\_trigger\_learning(  
        \&mut self,   
        observations: &\[ObservationSnapshot\]  
    ) \-\> TriggerDecision {  
        self.sample\_count \= observations.len();  
          
        if self.sample\_count \< self.min\_samples {  
            return TriggerDecision::InsufficientData;  
        }  
          
        // Extract features from observations  
        let features \= self.extract\_features(observations);  
          
        // Check for distribution drift  
        let split\_point \= self.last\_trigger\_count;  
        let new\_features \= \&features\[split\_point..\];  
          
        let drift\_report \= self.drift\_detector.detect\_drift(new\_features);  
          
        match drift\_report {  
            DriftReport::Analyzed { overall\_drift: true, recommendation, .. } \=\> {  
                // Drift detected \- trigger learning  
                self.last\_trigger\_count \= self.sample\_count;  
                  
                // Adapt interval based on drift frequency  
                if matches\!(recommendation, RelearningRecommendation::Immediate) {  
                    // Frequent drift \- check more often  
                    self.current\_interval \= (self.current\_interval \* 0.8) as usize;  
                }  
                  
                TriggerDecision::Trigger {   
                    reason: TriggerReason::DistributionDrift(drift\_report),  
                }  
            }  
            \_ \=\> {  
                // No drift \- check if time-based trigger  
                let samples\_since \= self.sample\_count \- self.last\_trigger\_count;  
                  
                if samples\_since \>= self.current\_interval {  
                    self.last\_trigger\_count \= self.sample\_count;  
                      
                    // Stable environment \- check less often  
                    self.current\_interval \= (self.current\_interval \* 1.2) as usize  
                        .min(self.base\_interval \* 4);  
                      
                    TriggerDecision::Trigger {  
                        reason: TriggerReason::TimeElapsed,  
                    }  
                } else {  
                    TriggerDecision::Wait  
                }  
            }  
        }  
    }  
}

**Success Criteria:**

* \[ \] KS test detects distribution shifts correctly  
* \[ \] Adaptive interval reduces unnecessary relearning by 60%+  
* \[ \] No false negatives on Sachs protein dataset

**Deliverable:** `git tag v0.3.0-smart-triggering`

### **Sprint 2.2: Benchmarking Infrastructure (Weeks 7-8)**

**Engineer Assignment:** Junior/Collaborator

**Tasks:**

// Task 2.2.1: Create benchmark datasets  
pub mod datasets {  
    pub struct BenchmarkDataset {  
        pub name: String,  
        pub n\_vars: usize,  
        pub n\_samples: usize,  
        pub ground\_truth\_graph: CausalGraph,  
        pub data: DMatrix\<f32\>,  
    }  
      
    pub fn sachs\_protein() \-\> BenchmarkDataset {  
        // Real biological dataset (11 proteins, 853 samples)  
        // Ground truth known from domain knowledge  
    }  
      
    pub fn alarm\_network() \-\> BenchmarkDataset {  
        // Medical diagnosis network (37 variables)  
        // Classic benchmark  
    }  
      
    pub fn child\_network() \-\> BenchmarkDataset {  
        // Pediatric diagnosis (20 variables)  
        // Standard benchmark  
    }  
      
    pub fn synthetic\_chain(n\_vars: usize) \-\> BenchmarkDataset {  
        // X1 \-\> X2 \-\> X3 \-\> ... \-\> Xn  
    }  
      
    pub fn synthetic\_fork(n\_vars: usize) \-\> BenchmarkDataset {  
        // X1 \-\> X2, X3, X4, ...  
    }  
      
    pub fn synthetic\_collider(n\_vars: usize) \-\> BenchmarkDataset {  
        // X1, X2, X3 \-\> X4  
    }  
}

// Task 2.2.2: Implement evaluation metrics  
pub struct EvaluationMetrics {  
    /// Structural Hamming Distance  
    pub shd: usize,  
      
    /// True Positives, False Positives, False Negatives  
    pub tp: usize,  
    pub fp: usize,  
    pub fn\_count: usize,  
      
    /// Precision, Recall, F1  
    pub precision: f32,  
    pub recall: f32,  
    pub f1: f32,  
      
    /// Runtime  
    pub runtime\_ms: u64,  
      
    /// Memory usage  
    pub peak\_memory\_mb: usize,  
}

pub fn evaluate\_causal\_discovery(  
    learned: \&CausalGraph,  
    ground\_truth: \&CausalGraph,  
) \-\> EvaluationMetrics {  
    // Compute metrics  
}

// Task 2.2.3: Criterion-based benchmarks  
use criterion::{black\_box, criterion\_group, criterion\_main, Criterion};

fn benchmark\_dagma\_scaling(c: \&mut Criterion) {  
    let mut group \= c.benchmark\_group("dagma\_scaling");  
      
    for n\_vars in \[5, 10, 20, 50, 100\] {  
        let dataset \= datasets::synthetic\_chain(n\_vars);  
          
        group.bench\_function(format\!("dagma\_{}vars", n\_vars), |b| {  
            b.iter(|| {  
                let dagma \= Dagma::new(n\_vars, None);  
                dagma.fit(black\_box(\&dataset.data))  
            });  
        });  
    }  
      
    group.finish();  
}

fn benchmark\_hypergraph\_prediction(c: \&mut Criterion) {  
    let mut group \= c.benchmark\_group("hypergraph\_prediction");  
      
    for order in \[2, 3, 4, 5\] {  
        let mut graph \= CausalHypergraph::new();  
          
        // Add edges of varying order  
        for \_ in 0..100 {  
            let sources \= (0..order).map(|\_| random\_variable()).collect();  
            graph.add\_interaction(sources, random\_variable(), ...);  
        }  
          
        let state \= random\_state();  
          
        group.bench\_function(format\!("predict\_order{}", order), |b| {  
            b.iter(|| {  
                graph.predict\_effect(black\_box(\&state), black\_box(random\_variable()))  
            });  
        });  
    }  
      
    group.finish();  
}

criterion\_group\!(benches, benchmark\_dagma\_scaling, benchmark\_hypergraph\_prediction);  
criterion\_main\!(benches);

**Success Criteria:**

* \[ \] Benchmark suite covers 10+ scenarios  
* \[ \] Automated CI runs benchmarks on every PR  
* \[ \] Regression detection (\>10% slowdown fails CI)

**Deliverable:** `benchmarks/` folder with comprehensive suite

## **Month 3: Documentation & Stability**

### **Sprint 3.1: API Documentation (Weeks 9-10)**

**Engineer Assignment:** Lead engineer (documentation time)

**Tasks:**

// Task 3.1.1: Comprehensive rustdoc  
/// DAGMA: Fast DAG structure learning via log-det acyclicity  
///  
/// \# Overview  
///  
/// DAGMA learns directed acyclic graphs (DAGs) from observational data  
/// using a novel acyclicity constraint based on log-determinant instead  
/// of matrix exponential (NOTEARS).  
///  
/// \# Performance  
///  
/// \- \*\*20.6x faster\*\* than NOTEARS for d=100 variables  
/// \- \*\*76.5% better SHD\*\* on benchmark datasets  
///  
/// \# Example  
///  
/// \`\`\`rust  
/// use zenb\_core::causal::Dagma;  
/// use nalgebra::DMatrix;  
///  
/// // Generate data: X1 \-\> X2 \-\> X3  
/// let data \= DMatrix::from\_row\_slice(100, 3, &\[  
///     // ... your data  
/// \]);  
///  
/// // Learn causal structure  
/// let dagma \= Dagma::new(3, None);  
/// let graph \= dagma.fit(\&data);  
///  
/// // Interpret results  
/// println\!("Causal edge X1-\>X2: {:.3}", graph\[(0, 1)\]);  
/// \`\`\`  
///  
/// \# Algorithm Details  
///  
/// DAGMA minimizes:  
/// \`\`\`text  
/// minimize\_W: score(W) \+ λ||W||\_1  
/// subject to: h(W) \= \-log det(sI \- W⊙W) \+ d log(s) \= 0  
/// \`\`\`  
///  
/// where:  
/// \- \`W\` is the weighted adjacency matrix (W\[i\]\[j\] \= effect of i on j)  
/// \- \`λ\` controls sparsity (higher λ \= sparser graph)  
/// \- \`h(W) \= 0\` enforces acyclicity (DAG constraint)  
///  
/// \# Hyperparameter Tuning  
///  
/// For best results, use adaptive configuration:  
/// \`\`\`rust  
/// let config \= DagmaConfig::adaptive(\&data);  
/// let dagma \= Dagma::new(n\_vars, Some(config));  
/// \`\`\`  
///  
/// \# References  
///  
/// Bello et al. "DAGs with NO TEARS: Smooth Optimization for Structure Learning"  
/// NeurIPS 2022\. \[Paper\](https://arxiv.org/abs/2209.08037)  
pub struct Dagma { ... }

// Task 3.1.2: Tutorial documentation  
// docs/tutorials/01\_basic\_usage.md  
// docs/tutorials/02\_hyperparameter\_tuning.md  
// docs/tutorials/03\_hypergraphs.md  
// docs/tutorials/04\_interventions.md  
// docs/tutorials/05\_async\_learning.md

// Task 3.1.3: Migration guides  
// docs/migrations/v0.1\_to\_v0.2.md  
// docs/migrations/v0.2\_to\_v0.3.md

**Deliverable:**

* 100% public API documented  
* 5+ tutorials  
* Migration guides

### **Sprint 3.2: Property-Based Testing (Weeks 11-12)**

**Engineer Assignment:** Lead \+ Junior

**Tasks:**

// Task 3.2.1: Add proptest  
\[dev-dependencies\]  
proptest \= "1.0"

// Task 3.2.2: Property tests for DAGMA  
use proptest::prelude::\*;

proptest\! {  
    \#\[test\]  
    fn dagma\_always\_produces\_dag(  
        n\_vars in 3usize..20,  
        n\_samples in 50usize..200,  
        seed in any::\<u64\>()  
    ) {  
        let data \= generate\_random\_data(n\_vars, n\_samples, seed);  
        let dagma \= Dagma::new(n\_vars, None);  
        let w \= dagma.fit(\&data);  
          
        // Property: Result must be acyclic  
        let h \= dagma.h\_logdet(\&w);  
        prop\_assert\!(h.abs() \< 1e-5, "Result must be DAG (h={:.6e})", h);  
    }  
      
    \#\[test\]  
    fn hypergraph\_never\_has\_cycles(  
        edges in prop::collection::vec(  
            (1usize..5, 1usize..10), // (order, target)  
            10..50  
        )  
    ) {  
        let mut graph \= CausalHypergraph::new();  
          
        for (order, target\_idx) in edges {  
            let sources: Vec\<Variable\> \= (0..order)  
                .map(|i| Variable::from\_index(i))  
                .collect();  
            let target \= Variable::from\_index(target\_idx);  
              
            // Property: Adding edge should either succeed or fail cleanly  
            let result \= graph.add\_interaction(sources, target, CausalEdge::zero());  
              
            // If it succeeds, graph must still be acyclic  
            if result.is\_ok() {  
                prop\_assert\!(\!graph.has\_cycle().unwrap());  
            }  
        }  
    }  
      
    \#\[test\]  
    fn intervention\_preserves\_probability\_sum(  
        initial\_p in prop::array::uniform5(0.0f32..1.0),  
        variable in 0usize..5,  
        value in 0.0f32..1.0  
    ) {  
        let sum: f32 \= initial\_p.iter().sum();  
        let normalized: \[f32; 5\] \= initial\_p.map(|p| p / sum);  
          
        let mut state \= BeliefState::default();  
        state.p \= normalized;  
          
        let var \= Variable::from\_index(variable);  
        let intervened \= state.intervene(var, value);  
          
        // Property: Probabilities must still sum to 1.0  
        let new\_sum: f32 \= intervened.p.iter().sum();  
        prop\_assert\!((new\_sum \- 1.0).abs() \< 1e-5,   
                    "Probabilities must sum to 1.0, got {}", new\_sum);  
    }  
}

**Success Criteria:**

* \[ \] Property tests cover all core algorithms  
* \[ \] 10,000+ random test cases pass  
* \[ \] Found and fixed 3+ edge case bugs

**Deliverable:** `git tag v0.3.1-stable`

---

# **📋 PHASE 2: NOVEL RESEARCH (Months 4-9)**

## **Month 4-5: Physics Discovery Prototype**

### **Sprint 4.1: Physics Simulation Infrastructure (Weeks 13-16)**

**Engineer Assignment:** Lead \+ Specialist (Physics/Graphics)

**Tasks:**

// Task 4.1.1: Choose physics engine  
// Option A: Bevy (if targeting games/visualization)  
// Option B: Rapier (pure physics, no rendering)  
// Option C: Custom (minimal, educational)

// Recommendation: Rapier (focused, no bloat)  
\[dependencies\]  
rapier2d \= "0.17"  // For 2D (simpler to start)

// Task 4.1.2: Implement trajectory extraction  
pub struct PhysicsDataGenerator {  
    physics\_world: RapierPhysicsWorld,  
      
    pub fn simulate\_falling\_objects(  
        \&mut self,  
        n\_samples: usize  
    ) \-\> Vec\<PhysicsTrajectory\> {  
        let mut trajectories \= Vec::new();  
          
        for \_ in 0..n\_samples {  
            // Randomize initial conditions  
            let mass \= rand::uniform(0.1, 10.0);  
            let height \= rand::uniform(1.0, 10.0);  
            let gravity \= 9.8; // Fixed for now  
              
            // Simulate  
            let trajectory \= self.simulate\_single\_drop(mass, height, gravity);  
            trajectories.push(trajectory);  
        }  
          
        trajectories  
    }  
      
    fn simulate\_single\_drop(  
        \&mut self,  
        mass: f32,  
        initial\_height: f32,  
        gravity: f32  
    ) \-\> PhysicsTrajectory {  
        // Create rigid body  
        let body \= RigidBodyBuilder::dynamic()  
            .translation(vector\!\[0.0, initial\_height\])  
            .build();  
          
        let collider \= ColliderBuilder::ball(0.1)  
            .density(mass)  
            .build();  
          
        // Insert into world  
        let body\_handle \= self.physics\_world.bodies.insert(body);  
        self.physics\_world.colliders.insert\_with\_parent(  
            collider, body\_handle, \&mut self.physics\_world.bodies  
        );  
          
        // Simulate for 2 seconds at 60 FPS  
        let mut trajectory \= PhysicsTrajectory::new();  
          
        for frame in 0..120 {  
            self.physics\_world.step();  
              
            let body \= \&self.physics\_world.bodies\[body\_handle\];  
            let pos \= body.translation();  
            let vel \= body.linvel();  
              
            trajectory.add\_frame(PhysicsFrame {  
                time: frame as f32 / 60.0,  
                position: \*pos,  
                velocity: \*vel,  
                acceleration: vector\!\[0.0, \-gravity\],  
            });  
        }  
          
        trajectory  
    }  
}

// Task 4.1.3: Convert trajectories to DAGMA input  
impl PhysicsTrajectory {  
    pub fn to\_causal\_data(\&self) \-\> DMatrix\<f32\> {  
        // Variables: position, velocity, acceleration  
        // Each frame is a sample  
          
        let n\_samples \= self.frames.len();  
        let n\_vars \= 3; // \[pos, vel, acc\]  
          
        let mut data \= DMatrix::zeros(n\_samples, n\_vars);  
          
        for (i, frame) in self.frames.iter().enumerate() {  
            data\[(i, 0)\] \= frame.position.y;  
            data\[(i, 1)\] \= frame.velocity.y;  
            data\[(i, 2)\] \= frame.acceleration.y;  
        }  
          
        data  
    }  
}

// Task 4.1.4: Dataset generation pipeline  
pub fn generate\_physics\_dataset() \-\> PhysicsDataset {  
    let mut generator \= PhysicsDataGenerator::new();  
      
    let datasets \= vec\!\[  
        ("falling\_objects", generator.simulate\_falling\_objects(1000)),  
        ("elastic\_collisions", generator.simulate\_collisions(1000)),  
        ("pendulum", generator.simulate\_pendulums(1000)),  
        ("projectile", generator.simulate\_projectiles(1000)),  
    \];  
      
    PhysicsDataset { datasets }  
}

**Success Criteria:**

* \[ \] Generate 4 types of physics scenarios  
* \[ \] Extract clean trajectories  
* \[ \] Ground truth equations documented

**Deliverable:** `examples/physics_dataset/` with 10k+ trajectories

### **Sprint 4.2: Symbolic Regression (Weeks 17-20)**

**Engineer Assignment:** Lead \+ ML Specialist

**Tasks:**

// Task 4.2.1: Implement genetic programming for equation discovery  
pub struct SymbolicRegression {  
    population\_size: usize,  
    n\_generations: usize,  
    mutation\_rate: f32,  
}

\#\[derive(Clone, Debug)\]  
pub enum Expr {  
    Var(usize),           // x\[i\]  
    Const(f32),           // constant  
    Add(Box\<Expr\>, Box\<Expr\>),  // \+  
    Sub(Box\<Expr\>, Box\<Expr\>),  // \-  
    Mul(Box\<Expr\>, Box\<Expr\>),  // \*  
    Div(Box\<Expr\>, Box\<Expr\>),  // /  
    Pow(Box\<Expr\>, f32),        // ^n  
}

impl Expr {  
    pub fn evaluate(\&self, x: &\[f32\]) \-\> f32 {  
        match self {  
            Expr::Var(i) \=\> x\[\*i\],  
            Expr::Const(c) \=\> \*c,  
            Expr::Add(a, b) \=\> a.evaluate(x) \+ b.evaluate(x),  
            Expr::Sub(a, b) \=\> a.evaluate(x) \- b.evaluate(x),  
            Expr::Mul(a, b) \=\> a.evaluate(x) \* b.evaluate(x),  
            Expr::Div(a, b) \=\> {  
                let denom \= b.evaluate(x);  
                if denom.abs() \< 1e-6 { 0.0 } else { a.evaluate(x) / denom }  
            }  
            Expr::Pow(a, n) \=\> a.evaluate(x).powf(\*n),  
        }  
    }  
      
    pub fn complexity(\&self) \-\> usize {  
        match self {  
            Expr::Var(\_) | Expr::Const(\_) \=\> 1,  
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) \=\> {  
                1 \+ a.complexity() \+ b.complexity()  
            }  
            Expr::Pow(a, \_) \=\> 1 \+ a.complexity(),  
        }  
    }  
      
    pub fn simplify(\&self) \-\> Expr {  
        // Algebraic simplification rules  
        match self {  
            Expr::Add(a, b) \=\> {  
                let a\_simp \= a.simplify();  
                let b\_simp \= b.simplify();  
                  
                // x \+ 0 \= x  
                if matches\!(b\_simp, Expr::Const(c) if c \== 0.0) {  
                    return a\_simp;  
                }  
                  
                Expr::Add(Box::new(a\_simp), Box::new(b\_simp))  
            }  
            // ... more rules  
            \_ \=\> self.clone(),  
        }  
    }  
}

impl SymbolicRegression {  
    pub fn fit(\&mut self, x: \&DMatrix\<f32\>, y: &\[f32\]) \-\> Expr {  
        // Initialize population  
        let mut population: Vec\<Expr\> \= (0..self.population\_size)  
            .map(|\_| self.random\_expr(3)) // Max depth 3  
            .collect();  
          
        for generation in 0..self.n\_generations {  
            // Evaluate fitness  
            let fitness: Vec\<f32\> \= population.iter()  
                .map(|expr| self.fitness(expr, x, y))  
                .collect();  
              
            // Select best  
            let mut indexed: Vec\<(usize, f32)\> \= fitness.iter()  
                .enumerate()  
                .map(|(i, \&f)| (i, f))  
                .collect();  
            indexed.sort\_by(|a, b| b.1.partial\_cmp(\&a.1).unwrap());  
              
            // Elitism: keep top 10%  
            let elite\_count \= self.population\_size / 10;  
            let mut next\_population: Vec\<Expr\> \= indexed.iter()  
                .take(elite\_count)  
                .map(|(i, \_)| population\[\*i\].clone())  
                .collect();  
              
            // Crossover and mutation  
            while next\_population.len() \< self.population\_size {  
                let parent1 \= self.tournament\_select(\&population, \&fitness);  
                let parent2 \= self.tournament\_select(\&population, \&fitness);  
                  
                let mut child \= self.crossover(\&parent1, \&parent2);  
                  
                if rand::random::\<f32\>() \< self.mutation\_rate {  
                    child \= self.mutate(\&child);  
                }  
                  
                next\_population.push(child);  
            }  
              
            population \= next\_population;  
              
            if generation % 10 \== 0 {  
                let best\_fitness \= fitness\[indexed\[0\].0\];  
                println\!("Generation {}: best fitness \= {:.6}", generation, best\_fitness);  
            }  
        }  
          
        // Return best individual  
        let fitness: Vec\<f32\> \= population.iter()  
            .map(|expr| self.fitness(expr, x, y))  
            .collect();  
          
        let best\_idx \= fitness.iter()  
            .enumerate()  
            .max\_by(|(\_, a), (\_, b)| a.partial\_cmp(b).unwrap())  
            .unwrap().0;  
          
        population\[best\_idx\].simplify()  
    }  
      
    fn fitness(\&self, expr: \&Expr, x: \&DMatrix\<f32\>, y: &\[f32\]) \-\> f32 {  
        // Fitness \= accuracy / complexity (prefer simple equations)  
          
        let mut mse \= 0.0;  
        for i in 0..y.len() {  
            let row \= x.row(i);  
            let x\_vals: Vec\<f32\> \= row.iter().cloned().collect();  
            let pred \= expr.evaluate(\&x\_vals);  
            mse \+= (pred \- y\[i\]).powi(2);  
        }  
        mse /= y.len() as f32;  
          
        let accuracy \= 1.0 / (1.0 \+ mse);  
        let simplicity \= 1.0 / (1.0 \+ expr.complexity() as f32);  
          
        // Weighted combination  
        0.7 \* accuracy \+ 0.3 \* simplicity  
    }  
      
    // More methods: random\_expr, crossover, mutate, tournament\_select...  
}

// Task 4.2.2: Integrate DAGMA \+ Symbolic Regression  
pub struct PhysicsDiscoveryEngine {  
    dagma: Dagma,  
    symbolic\_regressor: SymbolicRegression,  
}

impl PhysicsDiscoveryEngine {  
    pub fn discover\_equations(\&mut self, trajectory: \&PhysicsTrajectory) \-\> Vec\<PhysicsLaw\> {  
        // Step 1: Learn causal graph  
        let data \= trajectory.to\_causal\_data();  
        let causal\_graph \= self.dagma.fit(\&data);  
          
        println\!("Learned causal graph:");  
        println\!("{}", causal\_graph);  
          
        // Step 2: For each edge, discover symbolic equation  
        let mut discovered\_laws \= Vec::new();  
          
        for (source, target) in self.extract\_edges(\&causal\_graph) {  
            println\!("\\nDiscovering equation for {} \-\> {}", source, target);  
              
            // Extract source and target data  
            let x\_data \= self.extract\_variable\_data(\&data, source);  
            let y\_data \= self.extract\_variable\_data(\&data, target);  
              
            // Fit symbolic equation  
            let equation \= self.symbolic\_regressor.fit(\&x\_data, \&y\_data);  
              
            println\!("Discovered: {} \= {}", target, equation.to\_string());  
              
            discovered\_laws.push(PhysicsLaw {  
                source,  
                target,  
                equation,  
            });  
        }  
          
        discovered\_laws  
    }  
}

**Success Criteria:**

* \[ \] Rediscover a \= constant (gravity)  
* \[ \] Rediscover v \= v0 \+ at  
* \[ \] Rediscover s \= s0 \+ v0*t \+ 0.5*a\*t^2

**Deliverable:** `examples/physics_discovery/` with working demo

### **Sprint 4.3: Validation & Paper (Weeks 21-24)**

**Engineer Assignment:** Lead (research time)

**Tasks:**

\# Task 4.3.1: Comprehensive evaluation

\#\# Experiments:  
1\. \*\*Falling Objects\*\*  
   \- Ground truth: a \= \-9.8 m/s²  
   \- Measure: Can we discover this?  
     
2\. \*\*Varying Gravity\*\*  
   \- Generate data with gravity \= \[1, 5, 9.8, 15, 20\]  
   \- Measure: Does discovered equation generalize?  
     
3\. \*\*Elastic Collisions\*\*  
   \- Ground truth: conservation of momentum  
   \- Measure: Can we discover p1 \+ p2 \= const?  
     
4\. \*\*Pendulum\*\*  
   \- Ground truth: θ'' \= \-(g/L) sin(θ)  
   \- Measure: Can we discover this nonlinear equation?

\#\# Metrics:  
\- Equation accuracy (MSE vs ground truth)  
\- Generalization (held-out scenarios)  
\- Complexity (number of operations)  
\- Discovery time

\# Task 4.3.2: Write research paper

Title: "Causal Physics Discovery: Learning F=ma from Observations"

Abstract:  
  Current AI systems excel at pattern recognition but struggle to discover  
  fundamental physical laws from observations. We present a method combining  
  causal discovery (DAGMA) with symbolic regression to autonomously discover  
  physics equations from trajectory data. Our system rediscovers Newton's  
  laws of motion, conservation of momentum, and nonlinear dynamics without  
  prior knowledge of these principles.

Sections:  
1\. Introduction  
2\. Related Work (causal discovery, symbolic regression, physics learning)  
3\. Method (DAGMA \+ genetic programming)  
4\. Experiments (4 physics scenarios)  
5\. Results (equations discovered, accuracy, generalization)  
6\. Discussion (limitations, future work)  
7\. Conclusion

Target: NeurIPS workshop, ICLR workshop, or Physical Review Letters

\# Task 4.3.3: Open source release

\- Clean up code  
\- Add tutorials  
\- Create interactive demo (web-based visualization)  
\- Write blog post  
\- Submit to arXiv

**Deliverable:**

* Paper submitted to workshop  
* Code released as `causal-physics` crate  
* Blog post with interactive demo

---

## **Month 6-7: Causal Language Models**

### **Sprint 5.1: Causal Graph Extraction from Text (Weeks 25-28)**

**Engineer Assignment:** Lead \+ NLP Specialist

**Tasks:**

// Task 5.1.1: Parser for causal statements  
pub struct CausalStatementParser {  
    // Patterns for causal language  
    causal\_verbs: Vec\<&'static str\>, // \["causes", "affects", "leads to", ...\]  
    inhibitory\_verbs: Vec\<&'static str\>, // \["prevents", "blocks", "reduces", ...\]  
}

impl CausalStatementParser {  
    pub fn extract\_causal\_relations(\&self, text: \&str) \-\> Vec\<CausalRelation\> {  
        // Pattern matching for causal statements  
        // "Smoking causes cancer" \-\> CausalRelation { cause: "Smoking", effect: "Cancer" }  
        // "Exercise reduces stress" \-\> CausalRelation { cause: "Exercise", effect: "Stress", polarity: Negative }  
    }  
}

// Task 5.1.2: Build causal graph from extracted relations  
pub struct CausalKnowledgeGraph {  
    graph: CausalHypergraph,  
    entity\_index: HashMap\<String, Variable\>,  
}

impl CausalKnowledgeGraph {  
    pub fn add\_from\_text(\&mut self, text: \&str) {  
        let relations \= self.parser.extract\_causal\_relations(text);  
          
        for relation in relations {  
            let cause\_var \= self.get\_or\_create\_variable(\&relation.cause);  
            let effect\_var \= self.get\_or\_create\_variable(\&relation.effect);  
              
            self.graph.add\_interaction(  
                vec\!\[cause\_var\],  
                effect\_var,  
                CausalEdge::from\_confidence(relation.confidence),  
            );  
        }  
    }  
}

// Task 5.1.3: Query answering  
impl CausalKnowledgeGraph {  
    pub fn answer\_intervention\_query(\&self, query: \&str) \-\> Answer {  
        // "What if we increase exercise?"  
        // 1\. Parse query  
        // 2\. Identify intervention target  
        // 3\. Use causal graph to predict effects  
        // 4\. Generate natural language answer  
    }  
}

**Success Criteria:**

* \[ \] Extract causal relations from 1000+ sentences  
* \[ \] 80%+ precision on manual evaluation  
* \[ \] Answer 50+ causal queries correctly

**Deliverable:** `causal-nlp` component

### **Sprint 5.2: Benchmark on Causal Reasoning Tasks (Weeks 29-32)**

**Engineer Assignment:** NLP Specialist

**Tasks:**

// Task 5.2.1: Implement Causal Winograd benchmark  
pub struct CausalWinogradBenchmark {  
    scenarios: Vec\<CausalScenario\>,  
}

struct CausalScenario {  
    text: String,  
    intervention\_question: String,  
    correct\_answer: String,  
    counterfactual\_question: String,  
    counterfactual\_answer: String,  
}

// Example:  
// text: "The trophy doesn't fit in the suitcase because it's too big."  
// intervention: "If we made the trophy smaller, would it fit?"  
// answer: "Yes, making the trophy smaller would allow it to fit."  
// counterfactual: "If the suitcase had been bigger, would it have fit?"  
// answer: "Yes, a bigger suitcase would have accommodated the trophy."

// Task 5.2.2: Evaluate against baselines  
pub fn evaluate\_causal\_reasoning() {  
    let benchmark \= CausalWinogradBenchmark::load();  
      
    // Baseline 1: GPT-4 (zero-shot)  
    let gpt4\_score \= evaluate\_model("gpt-4", \&benchmark);  
      
    // Baseline 2: Claude (zero-shot)  
    let claude\_score \= evaluate\_model("claude", \&benchmark);  
      
    // Our approach: Causal graph \+ reasoning  
    let our\_score \= evaluate\_causal\_graph\_approach(\&benchmark);  
      
    println\!("GPT-4: {:.2}%", gpt4\_score \* 100.0);  
    println\!("Claude: {:.2}%", claude\_score \* 100.0);  
    println\!("Ours: {:.2}%", our\_score \* 100.0);  
}

**Success Criteria:**

* \[ \] Outperform GPT-4 on causal intervention questions  
* \[ \] 70%+ accuracy on counterfactual questions

**Deliverable:** Benchmark results, paper submission

---

## **Month 8-9: Continual Causal Learning**

### **Sprint 6.1: Modular Causal Models (Weeks 33-36)**

**Engineer Assignment:** Lead \+ Systems Engineer

**Tasks:**

// Task 6.1.1: Define causal module interface  
pub trait CausalModule {  
    /// Variables this module operates on  
    fn variables(\&self) \-\> &\[Variable\];  
      
    /// Predict effect of intervention  
    fn predict\_intervention(\&self, state: &\[f32\], intervention: Intervention) \-\> Prediction;  
      
    /// Update module with new observations  
    fn update(\&mut self, observations: &\[Observation\]);  
      
    /// Serialize module for reuse  
    fn save(\&self, path: \&Path) \-\> Result\<()\>;  
    fn load(path: \&Path) \-\> Result\<Self\> where Self: Sized;  
}

// Task 6.1.2: Implement physics module  
pub struct PhysicsModule {  
    discovered\_laws: Vec\<PhysicsLaw\>,  
    causal\_graph: CausalGraph,  
}

impl CausalModule for PhysicsModule {  
    fn predict\_intervention(\&self, state: &\[f32\], intervention: Intervention) \-\> Prediction {  
        // Use discovered laws to predict  
        self.discovered\_laws.iter()  
            .map(|law| law.evaluate(state, intervention))  
            .fold(Prediction::default(), |acc, pred| acc.combine(pred))  
    }  
}

// Task 6.1.3: Context detector for module selection  
pub struct ContextDetector {  
    modules: Vec\<Box\<dyn CausalModule\>\>,  
    module\_activations: Vec\<f32\>,  
}

impl ContextDetector {  
    pub fn select\_module(\&self, observation: \&Observation) \-\> \&dyn CausalModule {  
        // Compute similarity to each module's training distribution  
        let scores: Vec\<f32\> \= self.modules.iter()  
            .map(|module| self.compute\_similarity(observation, module))  
            .collect();  
          
        // Return best match  
        let best\_idx \= scores.iter()  
            .enumerate()  
            .max\_by(|(\_, a), (\_, b)| a.partial\_cmp(b).unwrap())  
            .unwrap().0;  
          
        &\*self.modules\[best\_idx\]  
    }  
}

**Success Criteria:**

* \[ \] Learn 3+ modules (physics, social, cognitive)  
* \[ \] Context detector 85%+ accuracy  
* \[ \] Transfer learning across modules works

**Deliverable:** `modular-causal` framework

---

# **📋 PHASE 3: ECOSYSTEM & ADOPTION (Months 10-24)**

## **Month 10-12: Production Readiness**

### **Sprint 7.1: Rust API Stabilization**

* Semantic versioning (1.0.0 release)  
* Backward compatibility guarantees  
* Deprecation policy  
* LTS releases

### **Sprint 7.2: Python Bindings (PyO3)**

\# pip install agolos-causal  
import agolos\_causal as ac

model \= ac.DAGMA(n\_vars=10)  
graph \= model.fit(data)  
effect \= model.intervene(variable="HeartRate", value=0.6)

### **Sprint 7.3: Performance Optimizations**

* SIMD vectorization for matrix ops  
* Multi-threading for hypergraph evaluation  
* GPU support (via cuBLAS) for large graphs  
* Memory pool allocators

**Target:** 50x speedup on large graphs (d\>1000)

## **Month 13-18: Research Validation**

### **Sprint 8.1: Academic Partnerships**

* MIT, Stanford, CMU collaborations  
* Provide framework for PhD students  
* Co-author papers on applications

### **Sprint 8.2: Industry Validation**

* Healthcare: predict treatment outcomes  
* Robotics: causal world models for manipulation  
* Finance: causal effect of policy changes  
* Digital wellness: personalized interventions

### **Sprint 8.3: Conference Presence**

* NeurIPS: workshop \+ poster  
* ICML: tutorial on causal discovery  
* UAI: full paper on hypergraphs  
* CLeaR: Causal Learning & Reasoning conference

**Goal:** 3+ conference papers, 100+ citations

## **Month 19-24: Community Building**

### **Sprint 9.1: Documentation Hub**

* Interactive tutorials (Jupyter-like)  
* Video courses  
* Case study library  
* API reference

### **Sprint 9.2: Developer Tools**

* VSCode extension for causal graph visualization  
* Web-based playground  
* CLI tools for common tasks  
* Docker containers for reproducibility

### **Sprint 9.3: Governance**

* Open governance model (like Rust Foundation)  
* Contributor guidelines  
* Code of conduct  
* Roadmap transparency

**Goal:** 1000+ GitHub stars, 50+ contributors

---

# **📊 RESOURCE PLANNING**

## **Team Structure**

### **Year 1 (Months 1-12)**

├── Lead Engineer (fen) \- 100%  
│   ├── Core algorithms  
│   ├── Architecture  
│   └── Research direction  
│  
├── Junior Engineer / Collaborator \- 50%  
│   ├── Testing  
│   ├── Documentation  
│   └── Benchmarking  
│  
└── Specialist Consultants \- As needed  
    ├── Physics simulation (Month 4-5)  
    ├── NLP (Month 6-7)  
    └── Systems (Month 8-9)

### **Year 2 (Months 13-24)**

├── Lead Engineer \- 80% (20% research)  
├── Core Engineer \#2 \- 100%  
├── Research Engineer \- 50%  
├── DevRel Engineer \- 50%  
└── Community Manager \- 25%

## **Budget Estimate**

### **Year 1**

* Personnel: $150k-200k (assuming 1.5 FTE avg)  
* Cloud compute: $5k (benchmarks, CI)  
* Conferences: $10k (2-3 conferences)  
* **Total: \~$165k-215k**

### **Year 2**

* Personnel: $300k-400k (3 FTE)  
* Cloud compute: $15k (more benchmarks, GPU)  
* Conferences: $20k (4-5 conferences)  
* Marketing: $10k (website, tools)  
* **Total: \~$345k-445k**

---

# **🎯 SUCCESS METRICS**

## **Technical Metrics**

### **Phase 1 (Foundation)**

* \[ \] Test coverage \>85%  
* \[ \] All benchmarks \<10% regression vs baseline  
* \[ \] Zero critical bugs in core algorithms  
* \[ \] API stability score \>0.9

### **Phase 2 (Research)**

* \[ \] Discover F=ma with \<5% error  
* \[ \] Causal reasoning benchmark \>70% accuracy  
* \[ \] Continual learning without catastrophic forgetting

### **Phase 3 (Adoption)**

* \[ \] 1000+ GitHub stars  
* \[ \] 50+ contributors  
* \[ \] 10+ production users  
* \[ \] 3+ academic citations

## **Research Impact**

* \[ \] 3+ peer-reviewed papers  
* \[ \] 100+ citations within 2 years  
* \[ \] 1+ keynote invitation  
* \[ \] Industry adoption (1+ Fortune 500 company)

## **Community Impact**

* \[ \] 10,000+ downloads  
* \[ \] 50+ community tutorials/blogs  
* \[ \] Active Discord/Slack community (500+ members)  
* \[ \] Framework used in 5+ courses/bootcamps

---

# **⚠️ RISK MITIGATION**

## **Technical Risks**

| Risk | Probability | Impact | Mitigation |
| ----- | ----- | ----- | ----- |
| DAGMA doesn't scale | Medium | High | Benchmark early, have fallback (PC algorithm) |
| Symbolic regression too slow | High | Medium | Use pruning, parallel evaluation |
| Physics discovery fails | Medium | High | Start with simple scenarios, iterate |
| Hypergraph cycles | Low | Medium | Already implementing cycle detection |

## **Resource Risks**

| Risk | Probability | Impact | Mitigation |
| ----- | ----- | ----- | ----- |
| Funding shortage | Medium | High | Seek grants (NSF, industry), sponsors |
| Key person leaves | Low | High | Document everything, pair programming |
| Burnout | Medium | Medium | Sustainable pace, clear milestones |

## **Market Risks**

| Risk | Probability | Impact | Mitigation |
| ----- | ----- | ----- | ----- |
| Competing framework | Medium | Medium | Focus on unique niche (real-time \+ hypergraph) |
| Academic skepticism | Low | Medium | Rigorous validation, peer review |
| Low adoption | Medium | High | Community building, documentation, partnerships |

---

# **🚀 EXECUTION PHILOSOPHY**

## **Principles**

1. **Ship early, iterate fast** \- Release v0.2.0 after Month 3, get feedback  
2. **Research \+ Engineering** \- Balance novel contributions with solid engineering  
3. **Community-first** \- Every feature should serve real users  
4. **Documentation \= Code** \- Treat docs as first-class deliverable  
5. **Benchmark everything** \- No optimization without measurement

## **Weekly Cadence**

Monday:   
  \- Review last week's progress  
  \- Plan this week's tasks  
  \- Prioritize blockers

Tuesday-Thursday:  
  \- Deep work on core tasks  
  \- Pair programming sessions  
  \- Code reviews

Friday:  
  \- Testing & documentation  
  \- Community engagement (answer issues, write blog)  
  \- Research reading

Weekend:  
  \- Optional: exploratory prototyping  
  \- Rest & recharge

## **Monthly Reviews**

* Progress vs plan  
* Metrics review  
* Roadmap adjustments  
* Celebrate wins 🎉

---

# **📌 IMMEDIATE NEXT STEPS (Week 1\)**

\# Day 1: Setup  
\[ \] Create GitHub project board  
\[ \] Set up CI/CD pipeline  
\[ \] Initialize docs/ folder structure

\# Day 2-3: Test Audit  
\[ \] Run coverage report  
\[ \] Document all ignored tests  
\[ \] Prioritize fixes

\# Day 4-5: Benchmark Baseline  
\[ \] Implement basic criterion benchmarks  
\[ \] Record baseline performance  
\[ \] Set up automated regression testing

\# Weekend: Planning  
\[ \] Review this plan with potential collaborators  
\[ \] Identify first external contributor tasks  
\[ \] Write Month 1 detailed sprint plan

---

