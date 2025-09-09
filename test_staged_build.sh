#!/bin/bash
# Build and test script for QuEST staged implementation
# This script properly sets up the FRUGAL environment before building

echo "================================================"
echo "QuEST Staged Implementation Build & Test"
echo "================================================"

# Step 1: Activate FRUGAL environment
echo ""
echo "Step 1: Activating FRUGAL conda environment..."
if [ -d ~/miniconda3_x86 ]; then
    source ~/miniconda3_x86/bin/activate
else
    source ~/miniconda3/bin/activate
fi
conda activate frugal

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Step 2: Set environment variables
echo ""
echo "Step 2: Setting environment variables..."
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/dependencies/optimize-cuda-memory-usage-v1/build:$LD_LIBRARY_PATH

# Step 3: Check GPU
echo ""
echo "Step 3: GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Step 4: Build FRUGAL library if needed
echo ""
echo "Step 4: Checking FRUGAL library..."
if [ ! -f dependencies/optimize-cuda-memory-usage-v1/build/profiling/libmemoryManager.a ]; then
    echo "Building FRUGAL library..."
    cd dependencies/optimize-cuda-memory-usage-v1/build
    make -j8
    cd ../../..
else
    echo "FRUGAL library already built"
fi

# Step 5: List available libraries
echo ""
echo "Step 5: Available FRUGAL libraries:"
find dependencies/optimize-cuda-memory-usage-v1/build -name "*.a" -exec basename {} \; | sort -u

# Step 6: Build staged test
echo ""
echo "Step 6: Building staged test executable..."
make -f Makefile.integrated build-staged-test

# Step 7: Run tests if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Step 7: Running tests..."
    echo "================================================"
    
    # Test with small qubit count first
    echo ""
    echo "Testing with 10 qubits (warm-up):"
    ./build_staged/test_staged_qft 10
    
    # Test with larger counts
    for qubits in 20 28 30; do
        echo ""
        echo "Testing with $qubits qubits:"
        ./build_staged/test_staged_qft $qubits
    done
else
    echo ""
    echo "Build failed! Please check the errors above."
    echo "Make sure:"
    echo "  1. FRUGAL conda environment is activated"
    echo "  2. CUDA is properly installed"
    echo "  3. All dependencies are built"
fi

echo ""
echo "================================================"
echo "Script complete!"