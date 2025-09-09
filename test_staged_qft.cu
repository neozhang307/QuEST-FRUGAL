// Test program for staged QFT implementation
// This demonstrates how to use the new staged API for large qubit simulations

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// Include the staged implementation
#include "QuEST/src/GPU/QuEST_gpu_memopt_staged.cu"

// Function to print GPU memory info
void printGPUMemoryInfo(const char* label) {
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("%s - GPU Memory: Total %.2f GB, Free %.2f GB, Used %.2f GB\n", 
         label,
         total_mem / (1024.0 * 1024.0 * 1024.0),
         free_mem / (1024.0 * 1024.0 * 1024.0),
         (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0));
}

// Test function to verify state vector
bool verifyStateVector(Qureg& qureg) {
  printf("Verifying state vector...\n");
  
  // Check that |0> state has amplitude 1
  qreal real0 = statevec_getRealAmp(qureg, 0);
  qreal imag0 = statevec_getImagAmp(qureg, 0);
  
  printf("Amplitude of |0>: %.6f + %.6fi\n", real0, imag0);
  
  // For initial state, should be 1+0i
  bool isValid = (std::abs(real0 - 1.0) < 1e-6) && (std::abs(imag0) < 1e-6);
  
  if (isValid) {
    printf("✅ State vector verification PASSED\n");
  } else {
    printf("❌ State vector verification FAILED\n");
  }
  
  return isValid;
}

// Main test function
int testStagedQFT(int numQubits) {
  printf("\n========================================\n");
  printf("Testing Staged QFT with %d qubits\n", numQubits);
  printf("========================================\n");
  
  // Print initial GPU memory
  printGPUMemoryInfo("Initial");
  
  // Configure staged execution
  QuESTStagedConfig config = getQuESTStagedConfig();
  config.qubitThreshold = 28;  // Use staged for 28+ qubits
  config.maxBytesPerStage = 4ULL * 1024 * 1024 * 1024;  // 4 GB per stage
  config.enablePrefetching = true;
  config.autoSplitLargeStages = true;
  setQuESTStagedConfig(config);
  
  // Create QuEST environment (minimal for testing)
  QuESTEnv env;
  env.numRanks = 1;
  env.rank = 0;
  
  // Create quantum register with staged memory optimization
  Qureg qureg;
  statevec_createQureg_memopt_staged(&qureg, numQubits, env);
  
  printGPUMemoryInfo("After Qureg creation");
  
  // Initialize to |0...0> state
  statevec_initZeroState(qureg);
  
  printGPUMemoryInfo("After initialization");
  
  // Verify initial state
  if (!verifyStateVector(qureg)) {
    printf("Initial state verification failed!\n");
    return -1;
  }
  
  // Apply QFT with staged execution
  applyFullQFT_staged(qureg);
  
  printGPUMemoryInfo("After QFT");
  
  // Clean up
  statevec_destroyQureg_memopt_staged(qureg, env);
  
  printGPUMemoryInfo("After cleanup");
  
  printf("\n✅ Test completed successfully for %d qubits\n", numQubits);
  return 0;
}

// Test harness for different qubit counts
void runIncrementalTests() {
  // Test cases: gradually increase qubit count
  int testCases[] = {10, 20, 28, 30, 31, 32, 33, 34};
  int numTests = sizeof(testCases) / sizeof(testCases[0]);
  
  printf("\n=== Running Incremental Staged QFT Tests ===\n");
  printf("Testing with FRUGAL v2 API and staged profiling\n\n");
  
  for (int i = 0; i < numTests; i++) {
    int numQubits = testCases[i];
    
    // Calculate state vector size
    size_t stateVectorBytes = (1ULL << numQubits) * sizeof(Complex);
    printf("\nTest %d/%d: %d qubits (%.2f GB state vector)\n", 
           i+1, numTests, numQubits,
           stateVectorBytes / (1024.0 * 1024.0 * 1024.0));
    
    // Check if GPU can handle it
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    if (stateVectorBytes > total_mem) {
      printf("⚠️  State vector exceeds GPU memory - staged execution required\n");
    }
    
    // Run test
    int result = testStagedQFT(numQubits);
    
    if (result != 0) {
      printf("❌ Test failed for %d qubits\n", numQubits);
      break;
    }
  }
  
  printf("\n=== All tests completed ===\n");
}

int main(int argc, char** argv) {
  // Initialize CUDA
  cudaSetDevice(0);
  
  // Initialize FRUGAL configuration
  memopt::ConfigurationManager::exportDefaultConfiguration();
  memopt::ConfigurationManager::loadConfiguration("config.json");
  
  if (argc > 1) {
    // Test specific qubit count
    int numQubits = atoi(argv[1]);
    return testStagedQFT(numQubits);
  } else {
    // Run incremental tests
    runIncrementalTests();
  }
  
  return 0;
}