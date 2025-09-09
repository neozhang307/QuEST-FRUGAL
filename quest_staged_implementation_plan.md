# QuEST Staged Profiling Implementation Plan

## Current Architecture Analysis

### 1. State Vector Sharding
- QuEST splits the quantum state into **16 shards** (NUM_SHARDS = 2^4)
- Each shard contains `numAmpsPerShard` complex amplitudes
- Shards are indexed by 4 global bits, remaining bits are local

### 2. Task Registration
- Tasks are registered with `registerAndExecuteTask()`
- Each task specifies which shards it reads/writes
- Tasks are stored in a global vector for later execution

### 3. Current Optimization Flow
- Capture all QFT operations into a CUDA graph
- Call `profileAndOptimize()` on the entire graph
- Execute optimized graph with address remapping

### 4. Stage Markers
- Already has `memopt::endStage()` calls in the QFT implementation!
- Stages are placed after each qubit's operations in QFT
- Additional stages after swap operations

## Problems with Current Approach

1. **Memory Scaling**: For N qubits, state vector size = 2^N × 16 bytes (complex double)
   - 30 qubits = 16 GB
   - 32 qubits = 64 GB  
   - 34 qubits = 256 GB

2. **Monolithic Profiling**: Tries to profile entire circuit at once

3. **No Data Movement**: Assumes all shards fit in GPU memory

## Proposed Staged Implementation

### Option 1: Use Existing endStage() Markers (Recommended)
- Leverage the existing `memopt::endStage()` calls
- Modify to use `TaskManager_v2` for registration
- Use `profileAndOptimizeStaged()` with automatic stage detection

### Option 2: Create New QuEST_gpu_memopt_staged.cu
- Clean implementation specifically for staged execution
- Better separation of concerns
- Easier to maintain and debug

## Implementation Steps

### Phase 1: Modify Task Registration

```cpp
// Replace current registration with TaskManager_v2
TaskManager_v2 taskManager(true);
std::vector<std::vector<TaskId>> stageTaskIds;
std::vector<TaskId> currentStageTasks;

void registerTaskV2(
  std::vector<StateVecIndex_t> ioShardIndices,
  Task task,
  Qureg qureg
) {
  std::vector<void*> inputs, outputs;
  for (auto i : ioShardIndices) {
    inputs.push_back(qureg.deviceStateVecShards[i].real);
    outputs.push_back(qureg.deviceStateVecShards[i].real);
  }
  
  TaskId id = taskManager.registerTask(task, inputs, outputs);
  currentStageTasks.push_back(id);
}

void endStageV2() {
  if (!currentStageTasks.empty()) {
    stageTaskIds.push_back(currentStageTasks);
    currentStageTasks.clear();
  }
}
```

### Phase 2: Initialize Shards with Offloading

```cpp
void statevec_createQureg_memopt_staged(Qureg* qureg, int numQubits) {
  // ... setup ...
  
  auto& memManager = MemoryManager::getInstance();
  
  for (StateVecIndex_t i = 0; i < qureg->numShards; i++) {
    // Allocate on GPU
    cudaMalloc(&(qureg->deviceStateVecShards[i].real), 
               2 * qureg->numAmpsPerShard * sizeof(qreal));
    
    // Register with memory manager
    memManager.registerManagedMemoryAddress(
      qureg->deviceStateVecShards[i].real, 
      2 * qureg->numAmpsPerShard * sizeof(qreal));
    
    // Immediately offload to CPU storage
    memManager.offloadToStorage(qureg->deviceStateVecShards[i].real);
    
    qureg->deviceStateVecShards[i].imag = 
      qureg->deviceStateVecShards[i].real + qureg->numAmpsPerShard;
  }
}
```

### Phase 3: Use Staged Profiling

```cpp
void applyFullQFTWithStagedMemopt(Qureg* qureg) {
  // Phase 1: Build staged tasks
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Capture operations with stage markers
  for (int q = qureg->numQubitsInStateVec - 1; q >= 0; q--) {
    registerTasksForHadamard(qureg, q);
    if (q > 0) {
      registerTasksForPhaseFunc(qureg, q);
    }
    endStageV2(); // Mark stage boundary
  }
  
  for (int i = 0; i < (qureg->numQubitsInStateVec / 2); i++) {
    registerTasksForSwap(qureg, i, qureg->numQubitsInStateVec - i - 1);
    if ((i + 1) % 2 == 0) {
      endStageV2(); // Mark stage boundary
    }
  }
  
  // Phase 2: Initialize state (on CPU)
  statevec_initZeroStateOnCPU(*qureg);
  
  // Phase 3: Profile and optimize with staging
  auto optimizedGraph = profileAndOptimizeStaged(
    taskManager, stageTaskIds, stream);
  
  // Phase 4: Execute optimized graph
  float runningTime;
  executeOptimizedGraph(
    optimizedGraph,
    [&](int taskId, std::map<void*, void*> addressMapping, cudaStream_t s) {
      taskManager.execute(taskId, s);
    },
    runningTime,
    memManager
  );
}
```

### Phase 4: Handle Large Problem Sizes

- Detect when even a single stage exceeds memory
- Automatically split stages further (e.g., process subset of shards)
- Add configuration for maximum stage size

## Benefits of Staged Approach

1. **Memory Efficiency**:
   - Only one stage's data in GPU at a time
   - Enables 34+ qubit simulations on typical GPUs

2. **Incremental Profiling**:
   - Profile each stage separately
   - Avoid OOM during profiling phase

3. **Better Scalability**:
   - Linear memory requirement per stage
   - Can handle arbitrarily large circuits

4. **Compatibility**:
   - Reuses existing QuEST gate implementations
   - Minimal changes to core QuEST code

## Testing Strategy

1. Start with small circuits (10-20 qubits) to verify correctness
2. Test with 30+ qubits to verify memory savings
3. Compare results with non-optimized version
4. Benchmark performance vs memory tradeoffs

## Next Steps

1. Create `QuEST_gpu_memopt_staged.cu` as a new implementation
2. Implement TaskManager_v2 integration
3. Add staged profiling support
4. Test with increasing qubit counts
5. Add automatic stage splitting for very large stages

## Key Technical Challenges

1. **Shard Dependencies**: Some operations touch multiple shards
2. **Stage Granularity**: Balance between too many small stages (overhead) and too few large stages (memory)
3. **Data Movement Cost**: CPU-GPU transfers may dominate for small stages
4. **Verification**: Ensuring numerical accuracy with staged execution

## Configuration Options

```cpp
struct QuESTStagedConfig {
  size_t maxBytesPerStage;      // Maximum memory per stage
  bool autoSplitLargeStages;    // Auto-split stages that exceed memory
  bool enablePrefetching;        // Prefetch next stage while executing current
  int minTasksPerStage;          // Minimum tasks to form a stage
};
```

## File Structure

```
QuEST/src/GPU/
├── QuEST_gpu_memopt.cu          # Original implementation
├── QuEST_gpu_memopt_staged.cu   # New staged implementation
└── QuEST_gpu_common.h           # Shared definitions
```