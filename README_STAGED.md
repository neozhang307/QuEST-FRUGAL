# QuEST Staged Profiling Implementation

## Overview
This is a new implementation of QuEST that uses FRUGAL's v2 API with staged profiling to enable simulation of 32+ qubits that exceed GPU memory capacity.

## Key Files

### New Implementation
- `QuEST/src/GPU/QuEST_gpu_memopt_staged.cu` - Main staged implementation using v2 API
- `test_staged_qft.cu` - Test program demonstrating staged QFT execution
- `Makefile.staged` - Build configuration for staged tests

### Original Implementation (for reference)
- `QuEST/src/GPU/QuEST_gpu_memopt.cu` - Original v1 API implementation

## Key Features

### 1. Automatic Memory Detection
```cpp
// Automatically detects when problem size exceeds GPU memory
bool requiresStagedExecution(int numQubits) {
  size_t stateVectorBytes = (1ULL << numQubits) * sizeof(Complex);
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  return (stateVectorBytes > totalMem * 0.8) || (numQubits >= 30);
}
```

### 2. V2 API with TaskManager_v2
```cpp
// Uses FRUGAL v2 API for staged profiling
TaskManager_v2 taskManager(true);
std::vector<std::vector<TaskId>> stageTaskIds;

// Register tasks and mark stage boundaries
registerTaskV2(shardIndices, kernelFunc, qureg, args...);
endStage();  // Mark stage boundary
```

### 3. Cold Start from CPU Storage
```cpp
// Initialize large state vectors on CPU
void statevec_initZeroStateOnCPU(Qureg qureg) {
  // All data starts in CPU storage
  // Managed by FRUGAL's memory manager
}
```

### 4. Staged Profiling and Optimization
```cpp
// Profile and optimize stages incrementally
auto optimizedGraph = profileAndOptimizeStaged(taskManager, stageTaskIds, stream);
```

## Building and Testing

### Prerequisites
1. CUDA toolkit installed
2. FRUGAL library built in `dependencies/optimize-cuda-memory-usage-v1/`
3. Conda environment activated: `conda activate frugal`

### Build
```bash
make -f Makefile.staged
```

### Test Individual Qubit Counts
```bash
make -f Makefile.staged test30  # Test 30 qubits
make -f Makefile.staged test32  # Test 32 qubits (exceeds most GPU memory)
make -f Makefile.staged test34  # Test 34 qubits
```

### Run All Tests
```bash
make -f Makefile.staged test-all
```

## Configuration

### Staged Execution Parameters
```cpp
QuESTStagedConfig config = {
  .maxBytesPerStage = 8GB,        // Maximum memory per stage
  .autoSplitLargeStages = true,   // Auto-split if stage exceeds memory
  .enablePrefetching = true,       // Prefetch next stage data
  .qubitThreshold = 30             // Use staged for 30+ qubits
};
setQuESTStagedConfig(config);
```

## Memory Requirements

| Qubits | State Vector Size | Standard GPU Memory | With Staged |
|--------|------------------|--------------------:|------------:|
| 28     | 4 GB             | 4 GB               | 1 GB/stage  |
| 30     | 16 GB            | 16 GB              | 1 GB/stage  |
| 32     | 64 GB            | 64 GB              | 1 GB/stage  |
| 34     | 256 GB           | 256 GB             | 1 GB/stage  |

## Implementation Status

### ✅ Completed
- GPU memory capacity detection
- V2 API migration with TaskManager_v2
- Staged profiling infrastructure
- Cold start from CPU storage
- Test framework

### 🚧 In Progress
- Kernel implementations from original file
- Gate registration functions
- Full QFT circuit implementation

### 📋 TODO
- Performance benchmarking
- Stage boundary optimization
- Multi-GPU support

## Reverting to Original

If issues arise, the original implementation is preserved in:
- `QuEST/src/GPU/QuEST_gpu_memopt.cu` (v1 API)

Simply use the original file instead of the staged version.

## Technical Notes

1. **Stage Granularity**: Each qubit operation forms a stage in QFT
2. **Memory Movement**: Data moves between CPU storage and GPU as needed
3. **Profiling**: Each stage is profiled independently then merged
4. **Address Remapping**: FRUGAL handles pointer updates automatically

## Troubleshooting

### Out of Memory During Profiling
- Reduce `maxBytesPerStage` in configuration
- Enable `autoSplitLargeStages`

### Slow Execution
- Increase `maxBytesPerStage` to reduce CPU-GPU transfers
- Enable `enablePrefetching` for overlapped data movement

### Verification Failures
- Check that kernels preserve numerical accuracy
- Verify stage boundaries don't break dependencies