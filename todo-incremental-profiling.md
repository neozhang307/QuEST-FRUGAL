# TODO: Incremental Profiling for 32+ Qubit Support

## ✅ SOLVED: Pruning/MIP Constraint Issue
**Problem**: MIP solver failed for 30-31 qubit cases with "No optimal solution found (ResultStatus=2)"  
**Root Cause**: Incorrect constraint `p[i][j] + sum(o[i][j][k]) <= 1` that treated prefetch and offload as mutually exclusive  
**Solution**: Commented out the problematic constraint in `secondStepSolver.cpp` lines 469, 471, 482  
**Status**: RESOLVED - 37 qubit cases now solve successfully

## Current Problem
QuEST cannot run 32+ qubit simulations due to GPU memory limitations (32GB+ requirement exceeds typical GPU capacity). Need incremental profiling to handle applications that exceed GPU memory.

## Development Strategy
Leverage existing stage-based infrastructure in FRUGAL to profile stages individually, using Unified Memory (UM) to handle arrays that exceed GPU capacity.

## Phase 1: Implement UM-based Stage Profiling (2-3 days)

### Core Approach
- Generate subgraphs for each stage separately
- Use Unified Memory for arrays that exceed GPU capacity
- Profile each stage subgraph with `getCudaGraphExecutionTimeline()`
- Merge timelines for optimization

### Detailed Tasks

#### 1.1 Extend MemoryManager for UM Support
- [ ] Add `registerUMAddress()` function to track UM allocations
- [ ] Track UM addresses separately from regular GPU allocations
- [ ] Support both UM and regular memory in same session

#### 1.2 Add UM→Storage Migration
- [ ] Implement `moveUMToStorage()` function
- [ ] Allocate CPU storage for each UM array
- [ ] Copy UM content to storage before main execution
- [ ] Free UM allocations after migration
- [ ] Update address mappings

#### 1.3 Modify QuEST Allocation
- [ ] Add profiling mode flag to QuEST
- [ ] Use `cudaMallocManaged()` when profiling large problems
- [ ] Register UM addresses with MemoryManager
- [ ] Keep same pointers throughout (no remapping needed)

#### 1.4 Implement Stage Subgraph Generation
- [ ] Capture task execution order (not full graph) initially
- [ ] For each stage, replay only that stage's tasks
- [ ] Generate `cudaGraph_t` subgraph per stage
- [ ] Maintain correct task IDs and dependencies

#### 1.5 Add Memory Prefetch Before Profiling
- [ ] Calculate memory requirement per stage
- [ ] Check if stage fits in GPU memory
- [ ] Use `cudaMemPrefetchAsync()` to migrate arrays to GPU
- [ ] **Alternative**: Execute the stage graph once to ensure 100% data migration to GPU
- [ ] Ensure migration complete with `cudaDeviceSynchronize()`
- [ ] Note: Executing graph once guarantees all data is on GPU (more reliable than prefetch)

#### 1.6 Profile Stages Individually
- [ ] For each stage subgraph, call `getCudaGraphExecutionTimeline()`
- [ ] Handle UM page faults gracefully
- [ ] Collect timing information per stage

#### 1.7 Merge Stage Timelines
- [ ] Combine individual stage timelines
- [ ] Adjust timestamps for sequential execution
- [ ] Create unified timeline for optimization

#### 1.8 Test with 32+ Qubit Cases
- [ ] Test with 32 qubit QFT
- [ ] Verify memory stays within GPU limits during profiling
- [ ] Validate optimization results
- [ ] Measure performance overhead

## Phase 2: Optimize Stage Sizes (1 day)

### Goal
Increase stage sizes for better efficiency while staying within GPU memory limits

### Tasks

#### 2.1 Analyze Current Stage Boundaries
- [ ] Profile current stage sizes in QuEST QFT
- [ ] Measure memory usage per stage
- [ ] Identify bottlenecks

#### 2.2 Reduce endStage() Frequency
- [ ] Modify QuEST to use fewer, larger stages
- [ ] Example: Stage per 5-10 qubits instead of per qubit
- [ ] Balance memory usage vs optimization complexity

#### 2.3 Find Optimal Balance
- [ ] Test different stage sizes
- [ ] Measure profiling overhead
- [ ] Measure optimization time
- [ ] Find sweet spot

#### 2.4 Performance Benchmarking
- [ ] Compare execution time: many small stages vs fewer large stages
- [ ] Measure memory transfer overhead
- [ ] Document optimal configuration

## Phase 3: Future - Automatic Graph Splitting (Future TODO)

### Goal
Automatically split graphs based on memory capacity without manual staging

### Future Tasks

#### 3.1 Design Automatic Memory-based Splitting
- [ ] Estimate memory requirements per task
- [ ] Design algorithm to group tasks within memory limit
- [ ] Handle complex dependency patterns

#### 3.2 Implement Dependency-aware Partitioning
- [ ] Respect task dependencies when splitting
- [ ] Minimize cross-partition dependencies
- [ ] Optimize for locality

#### 3.3 Remove Need for Manual endStage()
- [ ] System automatically determines stage boundaries
- [ ] Dynamic adjustment based on available GPU memory
- [ ] Support for heterogeneous GPU configurations

## Implementation Notes

### Key Files
- `dependencies/optimize-cuda-memory-usage-v1/optimization/optimizer.cu` - Stage handling
- `dependencies/optimize-cuda-memory-usage-v1/profiling/memoryManager.cu` - Memory management
- `QuEST/src/GPU/QuEST_gpu_memopt.cu` - QuEST integration

### Success Criteria
- 32+ qubit simulations run successfully
- Memory usage stays within GPU limits during profiling
- Performance overhead acceptable (<2x slowdown)
- Solution reusable for other memory-limited applications

### Timeline
- Phase 1: 2-3 days
- Phase 2: 1 day  
- Phase 3: Future work (document for later implementation)