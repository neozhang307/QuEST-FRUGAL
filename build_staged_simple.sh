#!/bin/bash
# Simple build script for staged implementation

echo "Building QuEST staged test..."

# Find all FRUGAL libraries
FRUGAL_LIBS=$(find dependencies/optimize-cuda-memory-usage-v1/build -name "*.a" | grep -v test | grep -v vcpkg | tr '\n' ' ')

# Build with all libraries linked
nvcc -std=c++17 -O2 \
    -gencode arch=compute_80,code=sm_80 \
    -I./QuEST/include \
    -I./QuEST/src \
    -I./dependencies/optimize-cuda-memory-usage-v1/include \
    -I./dependencies/optimize-cuda-memory-usage-v1/public \
    test_staged_qft.cu \
    $FRUGAL_LIBS \
    -L./dependencies/optimize-cuda-memory-usage-v1/build/vcpkg_installed/arm64-linux/lib -lfmt \
    -lcudart -lcublas -lcusolver -lcurand \
    -Xcompiler -fopenmp -lgomp \
    -o test_staged_qft_simple

if [ $? -eq 0 ]; then
    echo "Build successful! Executable: test_staged_qft_simple"
    echo ""
    echo "To test, run:"
    echo "  ./test_staged_qft_simple 10  # Test with 10 qubits"
    echo "  ./test_staged_qft_simple 20  # Test with 20 qubits"
    echo "  ./test_staged_qft_simple 30  # Test with 30 qubits"
else
    echo "Build failed!"
fi