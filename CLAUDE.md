# QuEST-FRUGAL Integration

## Important Note
Please also read `dependencies/optimize-cuda-memory-usage-v1/CLAUDE.md` for additional context and information about the FRUGAL memory optimization framework.

## Current Focus: Enable 32+ Qubit Simulations (2025-01-09)

### **Primary Goal**
Enable QuEST to run 32+ qubit simulations that exceed GPU memory capacity (32GB+ requirement for 32 qubits on typical GPUs).

### **Development Plan**

#### Phase 1: UserApplication Prototype (In Progress)
- **File**: Create `dependencies/optimize-cuda-memory-usage-v1/userApplications/tiledCholeskyIncrementalProfile.cu`
- **Base**: Use `tiledCholeskyNaiveGraph.cu` as template
- **Features to implement**:
  1. Graph splitting by memory capacity
  2. Incremental profiling of subgraphs
  3. Cold start with all data in CPU/storage
  4. Profile result merging
- **Test**: Large matrices exceeding GPU memory
- **Timeline**: 2-3 days

#### Phase 2: QuEST Integration
- **File**: Modify `QuEST/src/GPU/QuEST_gpu_memopt.cu`
- **Changes**:
  1. Detect when problem size exceeds GPU memory
  2. Apply incremental profiling automatically
  3. Handle dynamic task/array growth with qubit scaling
- **Test**: 32, 33, 34+ qubit cases
- **Timeline**: 1-2 days

#### Phase 3: API Migration (Optional)
- **Only if needed**: Migrate to execute_v2/memory_manager_v2 if incremental profiling requires newer features
- **Status**: Current API (`profileAndOptimize`, `executeOptimizedGraph`) works but may be limiting

### **Technical Approach**

**Incremental Profiling Strategy**:
```cpp
// Instead of profiling entire graph at once:
auto optimized = profileAndOptimize(graph);

// Split and profile incrementally:
auto subgraphs = splitGraphByMemory(graph, GPU_MEMORY_LIMIT);
for (auto& subgraph : subgraphs) {
    auto partial_optimized = profileAndOptimize(subgraph);
    mergeOptimizationResults(optimized, partial_optimized);
}
```

**Key Insight**: Both task count and array sizes grow with qubit count, unlike current optimize/enlarge which assumes fixed task count.

### **Completed Issues**

#### ✅ **SOLVED: 30-31 Qubit Optimization Failure**
- **Root Cause**: Incorrect MIP constraint treating prefetch/offload as mutually exclusive
- **Solution**: Commented out constraint in `secondStepSolver.cpp` lines 469, 471, 482
- **Status**: 37 qubit cases now solve successfully

## Development Branch
Currently working on `integration/quest` branch in the FRUGAL-base submodule for QuEST-specific modifications.

## Build Environment
When compiling and running, need to use FRUGAL environment in miniconda3:
1. First source miniconda3: `source miniconda3/bin/activate` 
2. For x86 systems: use `miniconda3_x86`
3. Then activate FRUGAL environment: `conda activate frugal` (or appropriate environment name)