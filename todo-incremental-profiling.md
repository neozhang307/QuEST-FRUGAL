# TODO: Incremental Profiling Implementation

## Goal
Enable QuEST to handle 32+ qubit simulations that exceed GPU memory capacity through incremental profiling and staged execution.

## Current Architecture Understanding

### Memory Management
- **FRUGAL manages 16 arrays** (shards), each with arrayId 0-15
- **QuEST wraps these into Qureg** structure for kernels
- **Direct mapping**: arrayId in FRUGAL = shard index in QuEST
- **Address mapping**: Both executor_v1 and executor_v2 support external address mapping via callback

### Key Components in memopt_adapter
1. **allocateShardAndRegister** - Allocates and registers arrays with FRUGAL
2. **tryUpdatingAddress** - Updates pointers when FRUGAL moves arrays (interface with executor)
3. **executeRandomTask** - Updates Qureg structure with new addresses before task execution
4. **moveDataBackToDevice** - Restores data from storage to device (can be replaced with FRUGAL's prefetchAllDataToDevice)

## Implementation Strategy

### Phase 1: TaskManager_v2 with Immediate Execution
**Goal**: Minimal change, keep current execution model but use TaskManager_v2 for task registration

```cpp
cudaGraph_t captureCudaGraphForFullQFT_v2(cudaStream_t stream, Qureg qureg) {
  TaskManager_v2 tmanager(stream);
  std::vector<std::vector<TaskId>> stageTaskIds;
  
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  
  for (int q = qureg.numQubitsInStateVec - 1; q >= 0; q--) {
    std::vector<TaskId> currentStageTasks;
    
    // Register task with TaskManager_v2
    TaskId hadamardTask = tmanager.registerTask<...>(...);
    currentStageTasks.push_back(hadamardTask);
    
    // Execute immediately (captures into graph)
    tmanager.executeTask(hadamardTask, stream);
    
    // Register phase function task
    TaskId phaseTask = tmanager.registerTask<...>(...);
    currentStageTasks.push_back(phaseTask);
    tmanager.executeTask(phaseTask, stream);
    
    stageTaskIds.push_back(currentStageTasks);
    memopt::endStage(stream);  // Mark stage boundary
  }
  
  // Swap operations
  for (int i = 0; i < (qureg.numQubitsInStateVec / 2); i++) {
    // Similar pattern...
  }
  
  cudaGraph_t graph;
  checkCudaErrors(cudaStreamEndCapture(stream, &graph));
  return graph;
}
```

**Benefits**:
- Tasks registered in TaskManager_v2 (provides task structure for optimization)
- Immediate execution maintains current behavior
- Can use both regular and staged optimization
- Minimal code changes

### Phase 2: Switch to executor_v2 and Staged Optimization
**Goal**: Use staged profiling and optimization

```cpp
void applyFullQFTWithMemopt_v2(Qureg* qureg) {
  TaskManager_v2 tmanager(stream);
  std::vector<std::vector<TaskId>> stageTaskIds;
  
  // Phase 1: Register all tasks and build stage structure
  buildQFTTasksAndStages(tmanager, stageTaskIds, *qureg);
  
  // Phase 2: Staged optimization
  auto optimizedGraph = profileAndOptimizeStaged(tmanager, stageTaskIds, stream);
  
  // Phase 3: Execute with executor_v2
  auto& memManager = MemoryManager::getInstance();
  memopt::executeOptimizedGraph(
    optimizedGraph,
    [=](int taskId, std::map<void*, void*> addressUpdate, cudaStream_t stream) {
      memopt_adapter::executeRandomTask(*qureg, taskId, addressUpdate, stream);
    },
    runningTime,
    memManager  // executor_v2 takes MemoryManager reference
  );
  
  // Restore data to device
  memManager.prefetchAllDataToDevice(stream);
  // Update Qureg pointers...
}
```

### Phase 3: Incremental Profiling (Deferred Execution)
**Goal**: Profile stages incrementally without capturing full graph

```cpp
// Register tasks WITHOUT immediate execution
for (int q = qureg.numQubitsInStateVec - 1; q >= 0; q--) {
  // Only register, don't execute
  TaskId hadamardTask = tmanager.registerTask<...>(...);
  TaskId phaseTask = tmanager.registerTask<...>(...);
  
  currentStageTasks.push_back(hadamardTask);
  currentStageTasks.push_back(phaseTask);
  stageTaskIds.push_back(currentStageTasks);
}

// Now profile incrementally - each stage fits in memory
auto optimizedGraph = profileAndOptimizeStaged(tmanager, stageTaskIds, stream);
```

## Key Advantages of This Approach

1. **Incremental Migration**: Each phase builds on the previous one
2. **Keep Current Interface**: The `executeRandomTask` callback works with both executors
3. **Address Mapping Unchanged**: `tryUpdatingAddress` continues to work as the interface
4. **Enable Large Problems**: Staged execution allows 32+ qubit simulations

## Priority Order

1. **Migrate to executor_v2** (enables staged execution)
2. **Implement staged QFT** with TaskManager_v2
3. **Test incremental profiling** for large qubit counts
4. **Optimize MemoryManager integration** (future improvement)

## Testing Milestones

- [ ] Phase 1 works with current qubit sizes (verify no regression)
- [ ] Phase 2 successfully uses staged optimization
- [ ] Phase 3 handles 32+ qubits without OOM
- [ ] Performance comparison vs. non-staged execution

## Notes

- The `addressUpdateMap` interface is intentional and works with both v1 and v2
- FRUGAL's MemoryManager internally tracks array movements
- QuEST's adapter layer correctly bridges between FRUGAL's array view and QuEST's Qureg structure
- Inter-stage offloading (arrayId == -1) is now properly handled in executor.cu