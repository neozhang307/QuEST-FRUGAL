#!/bin/bash
# Minimal build script that excludes problematic profiling libraries

echo "Building QuEST staged test (minimal version)..."
echo "NOTE: Remember to activate FRUGAL environment first!"
echo ""

# Build with only essential libraries
nvcc -std=c++17 -O2 \
    -gencode arch=compute_80,code=sm_80 \
    -I./QuEST/include \
    -I./QuEST/src \
    -I./dependencies/optimize-cuda-memory-usage-v1/include \
    -I./dependencies/optimize-cuda-memory-usage-v1/public \
    test_staged_qft.cu \
    ./dependencies/optimize-cuda-memory-usage-v1/build/profiling/libmemoryManager.a \
    ./dependencies/optimize-cuda-memory-usage-v1/build/profiling/libannotation.a \
    ./dependencies/optimize-cuda-memory-usage-v1/build/optimization/liboptimization.a \
    ./dependencies/optimize-cuda-memory-usage-v1/build/profiling/libmemred.a \
    ./dependencies/optimize-cuda-memory-usage-v1/build/profiling/libpeakMemoryUsageProfiler.a \
    ./dependencies/optimize-cuda-memory-usage-v1/build/optimization/strategies/libstrategies.a \
    ./dependencies/optimize-cuda-memory-usage-v1/build/optimization/strategies/libstrategyUtilities.a \
    -L./dependencies/optimize-cuda-memory-usage-v1/build/vcpkg_installed/arm64-linux/lib -lfmt \
    -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64 \
    -lcudart -lcublas -lcusolver -lcurand -lcupti \
    -Xcompiler -fopenmp -lgomp \
    -o test_staged_minimal

if [ $? -eq 0 ]; then
    echo "Build successful! Executable: test_staged_minimal"
    echo ""
    echo "To test, run:"
    echo "  ./test_staged_minimal 10  # Test with 10 qubits"
    echo "  ./test_staged_minimal 20  # Test with 20 qubits"
    echo "  ./test_staged_minimal 30  # Test with 30 qubits"
else
    echo "Build failed!"
    echo ""
    echo "If you see 'nvcc: command not found', activate FRUGAL environment:"
    echo "  source ~/miniconda3_x86/bin/activate && conda activate frugal"
    echo ""
    echo "If you see CUPTI errors, the profiling library needs CUPTI."
    echo "You may need to exclude cudaGraphExecutionTimelineProfiler."
fi