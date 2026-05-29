/*
 * Template for test case in HIP test binary
 *
 * Location: utilities/test_suite/HIP/Tensor_image_hip.cpp
 *
 * Instructions:
 * 1. Add this case block in the switch statement
 * 2. Replace {{AUGMENTATION_NAME}} and {{AUGMENTATION_NAME_UPPER}}
 * 3. Set up any additional parameters (same as HOST version)
 * 4. Call the appropriate _gpu function
 * 5. Structure is nearly identical to HOST version, just uses _gpu instead of _host
 */

case {{AUGMENTATION_NAME_UPPER}}:
{
    testCaseName = "{{augmentation_name}}";

    // ===== Set up additional parameters (if needed) =====
    // Example for kernel size:
    // Rpp32u kernelSize = additionalParam;

    // Example for array parameters (allocate on GPU):
    // Rpp32f myParamTensor[batchSize];
    // for(int i = 0; i < batchSize; i++)
    //     myParamTensor[i] = 1.5f;
    // Rpp32f *myParamTensorGpu = nullptr;
    // CHECK_RETURN_STATUS(hipMalloc(&myParamTensorGpu, batchSize * sizeof(Rpp32f)));
    // CHECK_RETURN_STATUS(hipMemcpy(myParamTensorGpu, myParamTensor,
    //                                batchSize * sizeof(Rpp32f), hipMemcpyHostToDevice));

    // ===== Start timing =====
    startWallTime = omp_get_wtime();
    startCpuTime = clock();

    // ===== Call the augmentation function =====
    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
    {
        errorCodeCapture = rppt_{{augmentation_name}}_gpu(d_input, srcDescPtr,
                                                           d_output, dstDescPtr,
                                                           // PASS ADDITIONAL PARAMETERS HERE
                                                           // Example: kernelSize,
                                                           // Example: myParamTensorGpu,
                                                           d_roiTensorPtrSrc, roiTypeSrc,
                                                           handle);
    }
    else
    {
        missingFuncFlag = 1;
    }

    // ===== Clean up GPU memory (if allocated) =====
    // if (myParamTensorGpu != nullptr)
    //     CHECK_RETURN_STATUS(hipFree(myParamTensorGpu));

    break;
}

// ===== KEY DIFFERENCES FROM HOST VERSION =====
/*
 * 1. Input/output pointers use d_input and d_output (device memory)
 * 2. ROI pointer uses d_roiTensorPtrSrc (device memory)
 * 3. Call _gpu function instead of _host
 * 4. Additional parameter arrays must be allocated on GPU with hipMalloc
 * 5. Must copy parameter arrays to GPU with hipMemcpy
 * 6. Must free GPU parameter arrays with hipFree
 */

// ===== VARIATION EXAMPLES =====

/*
 * Example 1: Simple augmentation with scalar parameter
 */
// case BRIGHTNESS:
// {
//     testCaseName = "brightness";
//
//     Rpp32f alphaTensor[batchSize];
//     Rpp32f betaTensor[batchSize];
//     for(int i = 0; i < batchSize; i++)
//     {
//         alphaTensor[i] = 1.0f;
//         betaTensor[i] = 50.0f;
//     }
//
//     Rpp32f *alphaTensorGpu, *betaTensorGpu;
//     CHECK_RETURN_STATUS(hipMalloc(&alphaTensorGpu, batchSize * sizeof(Rpp32f)));
//     CHECK_RETURN_STATUS(hipMalloc(&betaTensorGpu, batchSize * sizeof(Rpp32f)));
//     CHECK_RETURN_STATUS(hipMemcpy(alphaTensorGpu, alphaTensor,
//                                    batchSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
//     CHECK_RETURN_STATUS(hipMemcpy(betaTensorGpu, betaTensor,
//                                    batchSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
//
//     startWallTime = omp_get_wtime();
//     startCpuTime = clock();
//     if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
//         errorCodeCapture = rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr,
//                                                 alphaTensorGpu, betaTensorGpu,
//                                                 d_roiTensorPtrSrc, roiTypeSrc, handle);
//     else
//         missingFuncFlag = 1;
//
//     CHECK_RETURN_STATUS(hipFree(alphaTensorGpu));
//     CHECK_RETURN_STATUS(hipFree(betaTensorGpu));
//     break;
// }

/*
 * Example 2: Filter with kernel size (no GPU memory needed for scalar)
 */
// case EMBOSS:
// {
//     testCaseName = "emboss";
//     Rpp32u kernelSize = additionalParam;  // Scalar - pass directly
//
//     startWallTime = omp_get_wtime();
//     startCpuTime = clock();
//     if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
//         errorCodeCapture = rppt_emboss_gpu(d_input, srcDescPtr, d_output, dstDescPtr,
//                                             kernelSize, d_roiTensorPtrSrc, roiTypeSrc, handle);
//     else
//         missingFuncFlag = 1;
//     break;
// }
