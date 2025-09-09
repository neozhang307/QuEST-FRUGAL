# How to Test QuEST Staged Implementation

## IMPORTANT: Always Activate FRUGAL Environment First!

**Before doing ANYTHING, you MUST activate the FRUGAL conda environment:**

```bash
# For x86 systems:
source ~/miniconda3_x86/bin/activate
conda activate frugal

# For ARM systems:
source ~/miniconda3/bin/activate  
conda activate frugal
```

## Testing Methods

### Method 1: Using Integrated Makefile (Recommended)

After activating FRUGAL environment:

```bash
# 1. Build the staged test
make -f Makefile.integrated build-staged-test

# 2. Test with different qubit counts
make -f Makefile.integrated test-staged-10   # 10 qubits
make -f Makefile.integrated test-staged-20   # 20 qubits  
make -f Makefile.integrated test-staged-30   # 30 qubits
make -f Makefile.integrated test-staged-32   # 32 qubits (exceeds most GPUs)

# 3. Or run all tests
make -f Makefile.integrated test-staged-all
```

### Method 2: Direct Compilation

After activating FRUGAL environment:

```bash
# Find all libraries and compile
FRUGAL_LIBS=$(find dependencies/optimize-cuda-memory-usage-v1/build -name "*.a" | grep -v test | tr '\n' ' ')

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
    -o test_staged

# Run tests
./test_staged 10  # Test 10 qubits
./test_staged 20  # Test 20 qubits
./test_staged 30  # Test 30 qubits
```

### Method 3: Compare Original vs Staged

After activating FRUGAL environment:

```bash
# Build both versions
make -f Makefile.integrated build           # Original v1 API
make -f Makefile.integrated build-staged-test  # Staged v2 API

# Compare performance
make -f Makefile.integrated compare
```

## Troubleshooting

### "nvcc: command not found"
**Solution:** You forgot to activate FRUGAL environment!
```bash
source ~/miniconda3_x86/bin/activate && conda activate frugal
```

### "cannot find -lfmt" or other library errors
**Solution:** Build FRUGAL libraries first:
```bash
cd dependencies/optimize-cuda-memory-usage-v1/build
make -j8
cd ../../..
```

### Out of memory errors
- Reduce qubit count for testing
- Check GPU memory: `nvidia-smi`
- Verify staged execution triggers: Should automatically use staging for 30+ qubits

## Expected Output

For successful staged execution with 30+ qubits:
```
Staged execution required: 30 qubits, 16.00 GB state vector, GPU memory: 96.00 GB
Creating Qureg with staged memory optimization for 30 qubits
Staged Qureg created: 16 shards offloaded to CPU storage
...
✅ Staged QFT execution completed in XXX ms
```

## Quick Test Sequence

```bash
# ALWAYS FIRST:
source ~/miniconda3_x86/bin/activate && conda activate frugal

# Then test:
make -f Makefile.integrated check-deps       # Check dependencies
make -f Makefile.integrated gpu-info         # Check GPU memory
make -f Makefile.integrated build-staged-test # Build staged version
make -f Makefile.integrated test-staged-30   # Test 30 qubits
```

## Notes

- The staged implementation automatically detects when to use staging (30+ qubits or when state vector exceeds GPU memory)
- For <30 qubits, it falls back to standard GPU execution
- Build directories are separate: `./build/` (original) vs `./build_staged/` (staged)