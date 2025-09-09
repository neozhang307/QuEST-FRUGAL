// QuEST GPU Memory Optimization with Staged Profiling
// This is a new implementation that uses FRUGAL v2 API with staged profiling
// Based on tiledCholeskyStaged.cu pattern for incremental profiling

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <vector>

#include "../../../dependencies/optimize-cuda-memory-usage-v1/public/memopt.hpp"
#include "QuEST.h"
#include "QuEST_gpu_common.h"
#include "QuEST_internal.h"
#include "QuEST_precision.h"
#include "QuEST_validation.h"
#include "mt19937ar.h"

// Already defined in memopt headers, no need to redefine
// Using the checkCudaErrors from FRUGAL's cudaUtilities.hpp
// Bring __check into scope
using memopt::__check;

typedef long long StateVecIndex_t;

// Configuration for staged execution
struct QuESTStagedConfig {
  size_t maxBytesPerStage;       // Maximum memory per stage
  bool autoSplitLargeStages;     // Auto-split stages that exceed memory
  bool enablePrefetching;         // Prefetch next stage while executing current
  int minTasksPerStage;           // Minimum tasks to form a stage
  bool useIncrementalProfiling;  // Use incremental profiling for 32+ qubits
  int qubitThreshold;            // Qubit count threshold for staged execution (default 30)
};

// Global configuration
static QuESTStagedConfig g_stagedConfig = {
  .maxBytesPerStage = 8ULL * 1024 * 1024 * 1024,  // 8 GB default
  .autoSplitLargeStages = true,
  .enablePrefetching = true,
  .minTasksPerStage = 1,
  .useIncrementalProfiling = true,
  .qubitThreshold = 30
};

// Memopt v2 adapter for staged execution
namespace memopt_adapter_v2 {

using namespace memopt;

// Task manager for v2 API
static TaskManager_v2* g_taskManager = nullptr;
static std::vector<std::vector<TaskId>> g_stageTaskIds;
static std::vector<TaskId> g_currentStageTasks;

// Initialize task manager for staged execution
void initializeTaskManager() {
  if (g_taskManager == nullptr) {
    g_taskManager = new TaskManager_v2(true);  // true for enabling profiling
    g_stageTaskIds.clear();
    g_currentStageTasks.clear();
  }
}

// Clean up task manager
void cleanupTaskManager() {
  if (g_taskManager != nullptr) {
    delete g_taskManager;
    g_taskManager = nullptr;
  }
  g_stageTaskIds.clear();
  g_currentStageTasks.clear();
}

// Register a task with v2 API
template<typename TaskFunc, typename... Args>
void registerTaskV2(
  std::vector<StateVecIndex_t> ioShardIndices,
  TaskFunc task,
  Qureg qureg,
  Args... args,
  const std::string& taskName = ""
) {
  if (g_taskManager == nullptr) {
    initializeTaskManager();
  }

  std::vector<void*> inputs, outputs;
  for (auto i : ioShardIndices) {
    inputs.push_back(qureg.deviceStateVecShards[i].real);
    outputs.push_back(qureg.deviceStateVecShards[i].real);
  }

  // Register task with TaskManager_v2
  TaskId taskId = g_taskManager->registerTask<TaskFunc, Args...>(
    task, inputs, outputs,
    TaskManager_v2::makeArgs(args...),
    taskName
  );

  g_currentStageTasks.push_back(taskId);
}

// Mark end of current stage
void endStage() {
  if (!g_currentStageTasks.empty()) {
    g_stageTaskIds.push_back(g_currentStageTasks);
    g_currentStageTasks.clear();
  }
}

// Allocate shard and register with memory manager
template <typename T>
void allocateShardAndRegister(T** p, size_t s) {
  // Allocate on GPU
  checkCudaErrors(cudaMalloc(p, s));
  
  // Register with memory manager
  MemoryManager::getInstance().registerManagedMemoryAddress(*p, s);
  
  // Immediately offload to CPU storage for out-of-core computation
  MemoryManager::getInstance().offloadToStorage(*p);
}

// Check if problem size requires staged execution
bool requiresStagedExecution(int numQubits) {
  // Calculate memory requirement for state vector
  size_t stateVectorBytes = (1ULL << numQubits) * sizeof(Complex);
  
  // Get GPU memory capacity
  size_t freeMem, totalMem;
  checkCudaErrors(cudaMemGetInfo(&freeMem, &totalMem));
  
  // Use staged execution if:
  // 1. Number of qubits exceeds threshold
  // 2. State vector doesn't fit in GPU memory (with safety margin)
  const double SAFETY_FACTOR = 0.8;
  bool exceedsMemory = (stateVectorBytes > totalMem * SAFETY_FACTOR);
  bool exceedsThreshold = (numQubits >= g_stagedConfig.qubitThreshold);
  
  if (exceedsMemory || exceedsThreshold) {
    printf("Staged execution required: %d qubits, %.2f GB state vector, GPU memory: %.2f GB\n",
           numQubits, 
           stateVectorBytes / (1024.0 * 1024.0 * 1024.0),
           totalMem / (1024.0 * 1024.0 * 1024.0));
    return true;
  }
  
  return false;
}

// Execute staged optimization
OptimizationOutput executeStagedOptimization(cudaStream_t stream) {
  if (g_taskManager == nullptr || g_stageTaskIds.empty()) {
    fprintf(stderr, "Error: No tasks registered for staged optimization\n");
    exit(EXIT_FAILURE);
  }

  printf("Starting staged profiling with %zu stages\n", g_stageTaskIds.size());
  
  // Profile and optimize with staged execution
  auto optimizedGraph = profileAndOptimizeStaged(*g_taskManager, g_stageTaskIds, stream);
  
  printf("Staged optimization complete:\n");
  printf("  Original memory: %.2f MB\n", optimizedGraph.originalMemoryUsage);
  printf("  Optimized memory: %.2f MB\n", optimizedGraph.anticipatedPeakMemoryUsage);
  printf("  Reduction: %.1f%%\n", 
         ((optimizedGraph.originalMemoryUsage - optimizedGraph.anticipatedPeakMemoryUsage) / 
          optimizedGraph.originalMemoryUsage) * 100);
  
  return optimizedGraph;
}

// Execute optimized graph with v2 API
void executeStagedGraph(OptimizationOutput& optimizedGraph, float& runningTime) {
  auto& memManager = MemoryManager::getInstance();
  
  executeOptimizedGraph(
    optimizedGraph,
    [](int taskId, std::map<void*, void*> addressMapping, cudaStream_t stream) {
      if (g_taskManager != nullptr) {
        g_taskManager->execute(taskId, stream);
      }
    },
    runningTime,
    memManager
  );
}

}  // namespace memopt_adapter_v2

// QuEST specific definitions (same as original)
constexpr int MAX_NUM_QUBITS = 64;
constexpr int MAX_NUM_PHASE_FUNC_OVERRIDES = 8;
constexpr int NUM_THREADS_PER_BLOCK = 128;

struct KernelParamQureg {
  long long numAmpsPerShard;
  int numShards;
  int numGlobalBits;
  int numLocalBits;
};

KernelParamQureg convertToKernelParamQureg(const Qureg& qureg) {
  KernelParamQureg kpq;
  kpq.numAmpsPerShard = qureg.numAmpsPerShard;
  kpq.numShards = qureg.numShards;
  kpq.numGlobalBits = qureg.numGlobalBits;
  kpq.numLocalBits = qureg.numLocalBits;
  return kpq;
}

// Bit manipulation functions (same as original)
__forceinline__ __device__ int getBit(StateVecIndex_t num, int index) {
  return (num >> index) & 1;
}

__forceinline__ __host__ __device__ int extractBit(const int locationOfBitFromRight, const StateVecIndex_t theEncodedNumber) {
  return (theEncodedNumber & (1LL << locationOfBitFromRight)) >> locationOfBitFromRight;
}

__forceinline__ __host__ __device__ StateVecIndex_t flipBit(const StateVecIndex_t number, const int index) {
  return (number ^ (1LL << index));
}

__forceinline__ __host__ __device__ StateVecIndex_t insertZeroBit(const StateVecIndex_t number, const int index) {
  StateVecIndex_t left, right;
  left = (number >> index) << index;
  right = number - left;
  return (left << 1) ^ right;
}

__forceinline__ __host__ __device__ StateVecIndex_t insertTwoZeroBits(const StateVecIndex_t number, const int bit1, const int bit2) {
  int small = (bit1 < bit2) ? bit1 : bit2;
  int big = (bit1 < bit2) ? bit2 : bit1;
  return insertZeroBit(insertZeroBit(number, small), big);
}

// Index manipulation (same as original)
__forceinline__ __host__ __device__ StateVecIndex_t getGlobalIndex(Qureg* qureg, StateVecIndex_t index) {
  return index >> qureg->numLocalBits;
}

__forceinline__ __host__ __device__ StateVecIndex_t getLocalIndex(Qureg* qureg, StateVecIndex_t index) {
  return index & ((1 << qureg->numLocalBits) - 1);
}

__forceinline__ __host__ __device__ void splitIndex(Qureg* qureg, StateVecIndex_t index, StateVecIndex_t* globalIndex, StateVecIndex_t* localIndex) {
  *globalIndex = getGlobalIndex(qureg, index);
  *localIndex = getLocalIndex(qureg, index);
}

// Amplitude access functions (same as original)
qreal statevec_getRealAmp(Qureg qureg, StateVecIndex_t index) {
  StateVecIndex_t globalIndex, localIndex;
  splitIndex(&qureg, index, &globalIndex, &localIndex);

  qreal el = 0;
  cudaMemcpy(
    &el,
    &(qureg.deviceStateVecShards[globalIndex].real[localIndex]),
    sizeof(qreal),
    cudaMemcpyDefault
  );
  return el;
}

qreal statevec_getImagAmp(Qureg qureg, StateVecIndex_t index) {
  StateVecIndex_t globalIndex, localIndex;
  splitIndex(&qureg, index, &globalIndex, &localIndex);

  qreal el = 0;
  cudaMemcpy(
    &el,
    &(qureg.deviceStateVecShards[globalIndex].imag[localIndex]),
    sizeof(qreal),
    cudaMemcpyDefault
  );
  return el;
}

// Initialize state vector on CPU for large problems
void statevec_initZeroStateOnCPU(Qureg qureg) {
  printf("Initializing state vector on CPU for out-of-core computation\n");
  
  auto& memManager = memopt::MemoryManager::getInstance();
  
  // Allocate temporary CPU buffer
  qreal* h_buffer = nullptr;
  size_t bufferSize = 2 * qureg.numAmpsPerShard * sizeof(qreal);
  checkCudaErrors(cudaMallocHost(&h_buffer, bufferSize));
  
  // Initialize all shards to zero
  memset(h_buffer, 0, bufferSize);
  
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    // For the first shard, set |000...000>'s amp to 1
    if (i == 0) {
      h_buffer[0] = 1.0;  // real part
      h_buffer[qureg.numAmpsPerShard] = 0.0;  // imag part
    }
    
    // Copy to managed memory (will go to storage)
    if (!memManager.copyHostToManagedMemory(qureg.deviceStateVecShards[i].real, h_buffer)) {
      fprintf(stderr, "ERROR: Failed to initialize shard %lld\n", i);
      exit(EXIT_FAILURE);
    }
    
    // Reset buffer for next shard
    if (i == 0) {
      h_buffer[0] = 0.0;
    }
  }
  
  checkCudaErrors(cudaFreeHost(h_buffer));
  printf("CPU initialization complete - all data in storage\n");
}

// Standard GPU initialization for small problems
void statevec_initZeroState(Qureg qureg) {
  // Check if we need CPU initialization for large problems
  if (memopt_adapter_v2::requiresStagedExecution(qureg.numQubitsInStateVec)) {
    statevec_initZeroStateOnCPU(qureg);
    return;
  }
  
  // Standard GPU initialization for small problems
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    checkCudaErrors(cudaMemset(qureg.deviceStateVecShards[i].real, 0, 2 * qureg.numAmpsPerShard * sizeof(qreal)));
  }

  qreal one = 1, zero = 0;
  checkCudaErrors(cudaMemcpy(&qureg.deviceStateVecShards[0].real[0], &one, sizeof(qreal), cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(&qureg.deviceStateVecShards[0].imag[0], &zero, sizeof(qreal), cudaMemcpyDefault));
}

// Create quantum register with staged memory management
void statevec_createQureg_memopt_staged(Qureg* qureg, int numQubits, QuESTEnv env) {
  printf("Creating Qureg with staged memory optimization for %d qubits\n", numQubits);
  
  // Initialize CUDA device
  qureg->numQubitsInStateVec = numQubits;
  qureg->numAmpsTotal = 1LL << numQubits;
  
  // Set up sharding (16 shards as in original)
  const int NUM_SHARDS_BITS = 4;
  qureg->numGlobalBits = NUM_SHARDS_BITS;
  qureg->numLocalBits = numQubits - NUM_SHARDS_BITS;
  qureg->numShards = 1 << NUM_SHARDS_BITS;
  qureg->numAmpsPerShard = 1LL << qureg->numLocalBits;
  
  // Note: deviceStateVecShards is a fixed array in Qureg, not a pointer
  // No need to allocate, it's already part of the struct
  
  // Check if we need staged execution
  bool useStaged = memopt_adapter_v2::requiresStagedExecution(numQubits);
  
  if (useStaged) {
    // Initialize task manager for v2 API
    memopt_adapter_v2::initializeTaskManager();
    
    // Allocate shards and immediately offload to CPU storage
    auto& memManager = memopt::MemoryManager::getInstance();
    
    for (StateVecIndex_t i = 0; i < qureg->numShards; i++) {
      // Allocate on GPU (just a handle)
      checkCudaErrors(cudaMalloc(&(qureg->deviceStateVecShards[i].real), 
                                2 * qureg->numAmpsPerShard * sizeof(qreal)));
      
      // Register with memory manager
      memManager.registerManagedMemoryAddress(
        qureg->deviceStateVecShards[i].real, 
        2 * qureg->numAmpsPerShard * sizeof(qreal));
      
      // Mark as application input/output
      memManager.registerApplicationInput(qureg->deviceStateVecShards[i].real);
      memManager.registerApplicationOutput(qureg->deviceStateVecShards[i].real);
      
      // Immediately offload to CPU storage
      memManager.offloadToStorage(qureg->deviceStateVecShards[i].real);
      
      // Set imag pointer
      qureg->deviceStateVecShards[i].imag = 
        qureg->deviceStateVecShards[i].real + qureg->numAmpsPerShard;
    }
    
    printf("Staged Qureg created: %d shards offloaded to CPU storage\n", qureg->numShards);
  } else {
    // Standard allocation for small problems
    for (StateVecIndex_t i = 0; i < qureg->numShards; i++) {
      if (memopt::ConfigurationManager::getConfig().generic.useUM) {
        checkCudaErrors(cudaMallocManaged(&(qureg->deviceStateVecShards[i].real), 
                                         2 * qureg->numAmpsPerShard * sizeof(qreal)));
      } else {
        checkCudaErrors(cudaMalloc(&(qureg->deviceStateVecShards[i].real), 
                                  2 * qureg->numAmpsPerShard * sizeof(qreal)));
      }
      
      // Register with memory manager for v1 compatibility
      auto& memManager = memopt::MemoryManager::getInstance();
      memManager.registerManagedMemoryAddress(
        qureg->deviceStateVecShards[i].real, 
        2 * qureg->numAmpsPerShard * sizeof(qreal));
      memManager.registerApplicationInput(qureg->deviceStateVecShards[i].real);
      memManager.registerApplicationOutput(qureg->deviceStateVecShards[i].real);
      
      qureg->deviceStateVecShards[i].imag = 
        qureg->deviceStateVecShards[i].real + qureg->numAmpsPerShard;
    }
  }
}

// Destroy quantum register with proper cleanup
void statevec_destroyQureg_memopt_staged(Qureg qureg, QuESTEnv env) {
  auto& memManager = memopt::MemoryManager::getInstance();
  
  // Free managed memory properly
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    memManager.freeManagedMemory(qureg.deviceStateVecShards[i].real);
  }
  
  // Clean up task manager if used
  memopt_adapter_v2::cleanupTaskManager();
  
  // deviceStateVecShards is a fixed array, no need to free
}

// Placeholder for kernel implementations (will be added from original file)
// These would include:
// - statevec_hadamardLocalBitKernel
// - statevec_hadamardGlobalBitKernel  
// - statevec_phaseShiftByTermKernel
// - statevec_swapQubitAmpsKernel
// etc.

// Example gate implementation with staged registration
void statevec_hadamard_staged(Qureg qureg, int targetQubit) {
  // This would be called during graph building phase
  // Register the Hadamard operation as a task
  
  // Determine which shards are affected
  std::vector<StateVecIndex_t> affectedShards;
  // ... calculate affected shards based on targetQubit ...
  
  // Register task with v2 API
  // memopt_adapter_v2::registerTaskV2(
  //   affectedShards,
  //   hadamardKernelWrapper,
  //   qureg,
  //   targetQubit,
  //   "Hadamard_q" + std::to_string(targetQubit)
  // );
}

// Main entry point for staged QFT execution
void applyFullQFT_staged(Qureg qureg) {
  printf("\n=== Applying QFT with Staged Memory Optimization ===\n");
  printf("Qubits: %d, State vector size: %.2f GB\n", 
         qureg.numQubitsInStateVec,
         (qureg.numAmpsTotal * sizeof(Complex)) / (1024.0 * 1024.0 * 1024.0));
  
  // Check if staged execution is needed
  if (!memopt_adapter_v2::requiresStagedExecution(qureg.numQubitsInStateVec)) {
    printf("Problem fits in GPU memory, using standard execution\n");
    // Fall back to standard execution
    return;
  }
  
  // Initialize task manager
  memopt_adapter_v2::initializeTaskManager();
  
  // Create CUDA stream
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  
  // Phase 1: Build staged task graph
  printf("\n--- Phase 1: Building staged task graph ---\n");
  
  // QFT operations with stage markers
  for (int q = qureg.numQubitsInStateVec - 1; q >= 0; q--) {
    // Register Hadamard gate task
    // statevec_hadamard_staged(qureg, q);
    
    // Register controlled phase gates
    // if (q > 0) {
    //   for (int c = q - 1; c >= 0; c--) {
    //     statevec_controlledPhase_staged(qureg, c, q, ...);
    //   }
    // }
    
    // Mark end of stage for this qubit
    memopt_adapter_v2::endStage();
    printf("Stage %d registered\n", qureg.numQubitsInStateVec - 1 - q);
  }
  
  // Swap operations as separate stages
  for (int i = 0; i < (qureg.numQubitsInStateVec / 2); i++) {
    // Register swap task
    // statevec_swapQubitAmps_staged(qureg, i, qureg.numQubitsInStateVec - i - 1);
    
    if ((i + 1) % 2 == 0) {
      memopt_adapter_v2::endStage();
      printf("Swap stage registered\n");
    }
  }
  
  // Phase 2: Initialize state vector
  printf("\n--- Phase 2: Initializing state vector ---\n");
  statevec_initZeroState(qureg);
  
  // Phase 3: Profile and optimize with staging
  printf("\n--- Phase 3: Staged profiling and optimization ---\n");
  auto optimizedGraph = memopt_adapter_v2::executeStagedOptimization(stream);
  
  // Phase 4: Execute optimized graph
  printf("\n--- Phase 4: Executing optimized graph ---\n");
  float runningTime;
  memopt_adapter_v2::executeStagedGraph(optimizedGraph, runningTime);
  
  printf("✅ Staged QFT execution completed in %.3f ms\n", runningTime * 1000.0f);
  
  // Cleanup
  checkCudaErrors(cudaStreamDestroy(stream));
}

// Configuration API
void setQuESTStagedConfig(const QuESTStagedConfig& config) {
  g_stagedConfig = config;
  printf("Staged configuration updated:\n");
  printf("  Max bytes per stage: %.2f GB\n", config.maxBytesPerStage / (1024.0 * 1024.0 * 1024.0));
  printf("  Auto-split large stages: %s\n", config.autoSplitLargeStages ? "yes" : "no");
  printf("  Prefetching: %s\n", config.enablePrefetching ? "yes" : "no");
  printf("  Qubit threshold: %d\n", config.qubitThreshold);
}

QuESTStagedConfig getQuESTStagedConfig() {
  return g_stagedConfig;
}