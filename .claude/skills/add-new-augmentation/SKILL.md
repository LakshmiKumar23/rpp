---
skill: add-new-augmentation
description: Add a new image/voxel/audio augmentation to RPP with full backend implementation and testing
---

# Add New Augmentation to RPP

This skill guides you through adding a new augmentation primitive to the RPP library, implementing both CPU and HIP backends, and setting up comprehensive testing.

## Prerequisites

Before starting, gather:
1. **Augmentation name** (e.g., "my_new_filter", "color_balance")
2. **Category** (image/voxel/audio/misc)
3. **Input/output requirements** (same dimensions? different? additional parameters?)
4. **Reference implementation** or algorithm description

## Step 1: Define the API

**Location:** `src/include/tensor/rppt_tensor_augmentations.h` (or category-specific header)

Ask the user:
- "What is the augmentation name?"
- "Does it require additional parameters (kernel size, threshold, etc.)?"
- "Does it support all layouts (PKD3/PLN3/PLN1)?"
- "What data types should it support (U8/F32/F16/I8)?"

Then create function signatures:

```cpp
RppStatus rppt_{augmentation_name}_gpu(RppPtr_t srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        RppPtr_t dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        // Add any additional parameters here
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        rppHandle_t rppHandle);

RppStatus rppt_{augmentation_name}_host(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         // Add any additional parameters here
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle);
```

**Action:** Add these signatures to the appropriate header file.

## Step 2: Implement CPU Backend

**Location:** `src/modules/tensor/cpu/kernel/{augmentation_name}.cpp`

Ask the user:
- "Do you have a reference implementation or should I create a basic template?"
- "Which data types should be prioritized (start with U8 and F32)?"

Create the implementation file with:

1. **Data-type specific functions** (one per data type):
   ```cpp
   RppStatus {augmentation_name}_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8u *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    rpp::Handle& handle)
   {
       // Handle layout dispatch (PKD3, PLN3, PLN1)
       // Implement algorithm with SIMD optimizations where possible
   }
   ```

2. **Public wrapper function** that dispatches to data-type implementations:
   ```cpp
   RppStatus rppt_{augmentation_name}_host(RppPtr_t srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            RppPtr_t dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            rppHandle_t rppHandle)
   {
       if (srcDescPtr->dataType == RpptDataType::U8)
           return {augmentation_name}_u8_u8_host_tensor(...);
       else if (srcDescPtr->dataType == RpptDataType::F32)
           return {augmentation_name}_f32_f32_host_tensor(...);
       // ... other data types
       return RPP_ERROR_NOT_IMPLEMENTED;
   }
   ```

**Action:** Create and implement the CPU backend file.

**Checklist:**
- [ ] U8 implementation complete
- [ ] F32 implementation complete
- [ ] F16 implementation (if needed)
- [ ] I8 implementation (if needed)
- [ ] All layouts supported (PKD3/PLN3/PLN1)
- [ ] ROI support implemented
- [ ] SIMD optimizations applied (AVX/SSE where beneficial)

## Step 3: Implement HIP Backend

**Location:** `src/modules/tensor/hip/kernel/{augmentation_name}.hip.cpp`

Ask the user:
- "Should I create optimized GPU kernels or start with a basic implementation?"
- "Are there any special memory access patterns to consider?"

Create the HIP implementation with:

1. **GPU kernel(s)** for different layouts:
   ```cpp
   __global__ void {augmentation_name}_pkd_hip_tensor(...)
   {
       int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
       int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
       
       // Implement GPU algorithm
       // Use shared memory for optimization if needed
   }
   ```

2. **Kernel launcher** that sets up grid/block dimensions:
   ```cpp
   RppStatus hip_exec_{augmentation_name}_tensor(...)
   {
       int globalThreads_x = ...; // Based on image width
       int globalThreads_y = ...; // Based on image height
       int globalThreads_z = ...; // Based on batch size
       
       hipLaunchKernelGGL({augmentation_name}_pkd_hip_tensor,
                          dim3(ceil(globalThreads_x/16.0), ...),
                          dim3(16, ...),
                          0, handle.GetStream(), ...);
       
       return RPP_SUCCESS;
   }
   ```

3. **Public wrapper** that dispatches to appropriate kernel:
   ```cpp
   RppStatus rppt_{augmentation_name}_gpu(...)
   {
       // Dispatch based on layout and data type
       return hip_exec_{augmentation_name}_tensor(...);
   }
   ```

**Action:** Create and implement the HIP backend file.

**Checklist:**
- [ ] GPU kernel(s) implemented
- [ ] Grid/block dimensions optimized
- [ ] Memory access patterns optimized (coalesced reads/writes)
- [ ] Shared memory used where beneficial
- [ ] All data types supported
- [ ] All layouts supported

## Step 4: Add to Test Suite

**Location:** `utilities/test_suite/`

### 4a. Update Test Headers

Edit `utilities/test_suite/rpp_test_suite_image.h` (or appropriate category):

1. Add enum value (use next available number):
   ```cpp
   enum ImageAugmentation {
       // ... existing
       {AUGMENTATION_NAME_UPPER} = {next_number},
   };
   ```

2. Add to `imageAugmentationMap`:
   ```cpp
   const std::map<int, std::vector<std::string>> imageAugmentationMap = {
       // ... existing
       {{number}, {"{augmentation_name}", "HOST", "HIP"}},
   };
   ```

3. **If augmentation has variations** (kernel sizes, interpolation types, etc.):
   - Add to appropriate cases set (e.g., `kernelSizeCases`, `additionalParamCases`)
   ```cpp
   const unordered_set<int> kernelSizeCases = {ERODE, DILATE, ..., {AUGMENTATION_NAME_UPPER}};
   ```

### 4b. Update Test Binaries

Edit `utilities/test_suite/HOST/Tensor_image_host.cpp`:

```cpp
case {AUGMENTATION_NAME_UPPER}:
{
    testCaseName = "{augmentation_name}";
    
    // Set up additional parameters if needed
    // (e.g., kernel size, interpolation type)
    
    startWallTime = omp_get_wtime();
    startCpuTime = clock();
    
    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
        errorCodeCapture = rppt_{augmentation_name}_host(input, srcDescPtr, 
                                                          output, dstDescPtr,
                                                          // additional params
                                                          roiTensorPtrSrc, roiTypeSrc,
                                                          handle);
    else
        missingFuncFlag = 1;
        
    break;
}
```

Repeat for `utilities/test_suite/HIP/Tensor_image_hip.cpp` (change `_host` to `_gpu`).

### 4c. Update Python Test Runner

Edit `utilities/test_suite/HOST/runImageTests.py` and `utilities/test_suite/HIP/runImageTests.py`:

**If augmentation needs parameter variations**, add handling in `run_unit_test()` and `run_performance_test()`:

```python
elif imageAugmentationMap[int(case)][0] == "{augmentation_name}":
    # Add variation loop (kernel sizes, etc.)
    for paramValue in paramRange:
        print(f"./{binName} {srcPath1} {srcPath2} {dstPathTemp} {bitDepth.value} ...")
        result = subprocess.Popen([...], ...)
        log_detected(result, errorLog, ...)
```

**Action:** Update all test files.

**Checklist:**
- [ ] Enum added to header
- [ ] Test case added to HOST binary
- [ ] Test case added to HIP binary
- [ ] Python runner updated (if variations needed)
- [ ] Added to appropriate variation sets (if applicable)

## Step 5: Generate and Validate Golden Outputs

### 5a. Enable Debug Mode

Edit `utilities/test_suite/rpp_test_suite_common.h`:
```cpp
#define DEBUG_MODE 1
```

### 5b. Build and Run Tests

```bash
cd utilities/test_suite/HOST
rm -rf build && mkdir build && cd build
cmake ..
make -j8
cd ..
python3 runImageTests.py --case_list {case_number} --test_type 0 --batch_size 3
```

This generates `.bin` files in the current directory.

### 5c. Organize Golden Outputs

```bash
cd utilities/test_suite
python3 scripts/organize_reference_outputs.py --source HOST --dest REFERENCE_OUTPUT
```

### 5d. Verify Outputs

Ask the user:
- "Please visually inspect the generated outputs to ensure correctness"
- "Are the outputs as expected?"

If outputs are correct, proceed. If not, debug the implementation.

### 5e. Run QA Tests

Disable debug mode (`DEBUG_MODE 0`), then run QA tests:

```bash
cd utilities/test_suite/HOST
python3 runImageTests.py --case_list {case_number} --test_type 0 --qa_mode 1 --batch_size 3
```

Verify all tests pass.

**Action:** Generate, organize, and validate golden outputs.

**Checklist:**
- [ ] Debug mode enabled
- [ ] Tests run successfully
- [ ] `.bin` files generated
- [ ] Outputs organized into REFERENCE_OUTPUT/
- [ ] Visual verification completed
- [ ] QA tests pass with golden outputs
- [ ] Debug mode disabled

## Final Steps

1. **Build both backends** to ensure compilation succeeds:
   ```bash
   # HIP backend
   mkdir build-hip && cd build-hip
   cmake ../rpp
   make -j8
   
   # CPU backend
   mkdir build-cpu && cd build-cpu
   cmake ../rpp -DBACKEND=CPU
   make -j8
   ```

2. **Run comprehensive tests**:
   ```bash
   cd utilities/test_suite/HOST
   python3 runImageTests.py --case_list {case_number} --test_type 0 --qa_mode 1 --batch_size 3
   python3 runImageTests.py --case_list {case_number} --test_type 1 --num_runs 100
   
   cd ../HIP
   python3 runImageTests.py --case_list {case_number} --test_type 0 --qa_mode 1 --batch_size 3
   python3 runImageTests.py --case_list {case_number} --test_type 1 --num_runs 100
   ```

3. **Commit changes**:
   - Source files (`src/include/`, `src/modules/`)
   - Test files (`utilities/test_suite/`)
   - Golden outputs (`utilities/test_suite/REFERENCE_OUTPUT/{augmentation_name}/`)

## Summary

Report to the user:
- Files created/modified
- Test case number assigned
- QA test results
- Performance metrics (if available)
- Next steps (documentation, additional optimizations, etc.)

**Final Checklist:**
- [ ] API defined in headers
- [ ] CPU backend implemented and tested
- [ ] HIP backend implemented and tested
- [ ] Test suite updated
- [ ] Golden outputs generated and validated
- [ ] QA tests passing
- [ ] Performance tests running
- [ ] Both backends compile successfully
- [ ] Ready for commit
