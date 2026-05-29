/*
 * Template for test case in HOST test binary
 *
 * Location: utilities/test_suite/HOST/Tensor_image_host.cpp
 *
 * Instructions:
 * 1. Add this case block in the switch statement
 * 2. Replace {{AUGMENTATION_NAME}} and {{AUGMENTATION_NAME_UPPER}}
 * 3. Set up any additional parameters your augmentation needs
 * 4. Call the appropriate _host function
 */

case {{AUGMENTATION_NAME_UPPER}}:
{
    testCaseName = "{{augmentation_name}}";

    // ===== Set up additional parameters (if needed) =====
    // Example for kernel size:
    // Rpp32u kernelSize = additionalParam;

    // Example for array parameters:
    // Rpp32f myParamTensor[batchSize];
    // for(int i = 0; i < batchSize; i++)
    //     myParamTensor[i] = 1.5f;

    // ===== Start timing =====
    startWallTime = omp_get_wtime();
    startCpuTime = clock();

    // ===== Call the augmentation function =====
    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
    {
        errorCodeCapture = rppt_{{augmentation_name}}_host(input, srcDescPtr,
                                                            output, dstDescPtr,
                                                            // PASS ADDITIONAL PARAMETERS HERE
                                                            // Example: kernelSize,
                                                            // Example: myParamTensor,
                                                            roiTensorPtrSrc, roiTypeSrc,
                                                            handle);
    }
    else
    {
        missingFuncFlag = 1;
    }

    break;
}

// ===== VARIATION EXAMPLES =====

/*
 * Example 1: Simple augmentation (brightness, flip, etc.)
 * No additional parameters needed - just call the function directly
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
//     startWallTime = omp_get_wtime();
//     startCpuTime = clock();
//     if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
//         errorCodeCapture = rppt_brightness_host(input, srcDescPtr, output, dstDescPtr,
//                                                  alphaTensor, betaTensor,
//                                                  roiTensorPtrSrc, roiTypeSrc, handle);
//     else
//         missingFuncFlag = 1;
//     break;
// }

/*
 * Example 2: Filter with kernel size (emboss, gaussian, etc.)
 * Uses additionalParam for kernel size
 */
// case EMBOSS:
// {
//     testCaseName = "emboss";
//     Rpp32u kernelSize = additionalParam;
//
//     startWallTime = omp_get_wtime();
//     startCpuTime = clock();
//     if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
//         errorCodeCapture = rppt_emboss_host(input, srcDescPtr, output, dstDescPtr,
//                                              kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
//     else
//         missingFuncFlag = 1;
//     break;
// }

/*
 * Example 3: Geometric transform with interpolation type
 * Uses additionalParam for interpolation type
 */
// case RESIZE:
// {
//     testCaseName = "resize";
//     RpptInterpolationType interpolationType = static_cast<RpptInterpolationType>(additionalParam);
//
//     startWallTime = omp_get_wtime();
//     startCpuTime = clock();
//     if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
//         errorCodeCapture = rppt_resize_host(input, srcDescPtr, output, dstDescPtr,
//                                              dstImgSizes, interpolationType,
//                                              roiTensorPtrSrc, roiTypeSrc, handle);
//     else
//         missingFuncFlag = 1;
//     break;
// }

/*
 * Example 4: Dual-input augmentation (blend, add, etc.)
 * Uses input_second for the second input
 */
// case BLEND:
// {
//     testCaseName = "blend";
//
//     Rpp32f alphaTensor[batchSize];
//     for(int i = 0; i < batchSize; i++)
//         alphaTensor[i] = 0.5f;
//
//     startWallTime = omp_get_wtime();
//     startCpuTime = clock();
//     if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32)
//         errorCodeCapture = rppt_blend_host(input, input_second, srcDescPtr,
//                                             output, dstDescPtr, alphaTensor,
//                                             roiTensorPtrSrc, roiTypeSrc, handle);
//     else
//         missingFuncFlag = 1;
//     break;
// }
