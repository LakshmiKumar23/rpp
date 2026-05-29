/*
 * Template for HIP backend implementation
 *
 * Location: src/modules/tensor/hip/kernel/{{augmentation_name}}.hip.cpp
 *
 * Instructions:
 * 1. Replace {{AUGMENTATION_NAME}} with your augmentation name
 * 2. Implement GPU kernels for each layout (PKD3, PLN3, PLN1)
 * 3. Optimize for coalesced memory access
 * 4. Use shared memory where beneficial
 * 5. Consider using texture memory for read-only data
 * 6. Test with different block/grid dimensions for optimal performance
 */

#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// Helper device functions (if needed)
__device__ __forceinline__ uchar process_pixel_u8(uchar pixel)
{
    // IMPLEMENT PIXEL-LEVEL ALGORITHM HERE
    return pixel;  // PLACEHOLDER
}

__device__ __forceinline__ float process_pixel_f32(float pixel)
{
    // IMPLEMENT PIXEL-LEVEL ALGORITHM HERE
    return pixel;  // PLACEHOLDER
}

// PKD3 (NHWC) kernel for U8
__global__ void {{augmentation_name}}_pkd3_u8_hip_tensor(uchar *srcPtr,
                                                         uint2 srcStridesNH,
                                                         uchar *dstPtr,
                                                         uint2 dstStridesNH,
                                                         // ADD ADDITIONAL PARAMETERS HERE
                                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;  // Process 8 pixels at a time
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                  ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    // Process 8 pixels (24 bytes) in vectorized manner
    d_float8 src_f8;
    rpp_hip_load24_pkd3_and_unpack_to_float8_pln3(srcPtr + srcIdx, &src_f8);

    // IMPLEMENT ALGORITHM HERE
    // Example: Apply transformation to each channel
    // src_f8.f4[0] = process(src_f8.f4[0]);  // R channel
    // src_f8.f4[1] = process(src_f8.f4[1]);  // G channel
    // src_f8.f4[2] = process(src_f8.f4[2]);  // B channel

    rpp_hip_pack_float8_pln3_and_store24_pkd3(dstPtr + dstIdx, &src_f8);
}

// PLN3 (NCHW) kernel for U8
__global__ void {{augmentation_name}}_pln3_u8_hip_tensor(uchar *srcPtr,
                                                         uint3 srcStridesNCH,
                                                         uchar *dstPtr,
                                                         uint3 dstStridesNCH,
                                                         // ADD ADDITIONAL PARAMETERS HERE
                                                         int channelsDst,
                                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;  // Process 8 pixels
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);

    // IMPLEMENT ALGORITHM HERE
    // Process all channels
    for(int c = 0; c < channelsDst; c++)
    {
        // src_f8 = process_channel(src_f8, c);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx + c * dstStridesNCH.y, &src_f8);
    }
}

// PLN1 (grayscale) kernel for U8
__global__ void {{augmentation_name}}_pln1_u8_hip_tensor(uchar *srcPtr,
                                                         uint2 srcStridesNH,
                                                         uchar *dstPtr,
                                                         uint2 dstStridesNH,
                                                         // ADD ADDITIONAL PARAMETERS HERE
                                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);

    // IMPLEMENT ALGORITHM HERE for grayscale
    // src_f8 = process(src_f8);

    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
}

// F32 kernels (similar structure to U8)
__global__ void {{augmentation_name}}_pkd3_f32_hip_tensor(float *srcPtr,
                                                          uint2 srcStridesNH,
                                                          float *dstPtr,
                                                          uint2 dstStridesNH,
                                                          // ADD ADDITIONAL PARAMETERS HERE
                                                          RpptROIPtr roiTensorPtrSrc)
{
    // IMPLEMENT F32 PKD3 KERNEL
}

// Kernel launcher - dispatches to appropriate kernel
RppStatus hip_exec_{{augmentation_name}}_tensor(Rpp8u *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 // ADD ADDITIONAL PARAMETERS HERE
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh_host(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;  // Divide by 8 and round up
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    // Choose optimal block dimensions
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        // PKD3 to PKD3
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hipLaunchKernelGGL({{augmentation_name}}_pkd3_u8_hip_tensor,
                              dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                              dim3(localThreads_x, localThreads_y, localThreads_z),
                              0,
                              handle.GetStream(),
                              srcPtr,
                              make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                              dstPtr,
                              make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                              // PASS ADDITIONAL PARAMETERS HERE
                              roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            // Launch F32 kernel
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        // PLN3 to PLN3 or PLN1 to PLN1
        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            if (srcDescPtr->c == 1)
            {
                // PLN1
                hipLaunchKernelGGL({{augmentation_name}}_pln1_u8_hip_tensor,
                                  dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                  dim3(localThreads_x, localThreads_y, localThreads_z),
                                  0,
                                  handle.GetStream(),
                                  srcPtr,
                                  make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                  dstPtr,
                                  make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                  // PASS ADDITIONAL PARAMETERS HERE
                                  roiTensorPtrSrc);
            }
            else
            {
                // PLN3
                hipLaunchKernelGGL({{augmentation_name}}_pln3_u8_hip_tensor,
                                  dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                  dim3(localThreads_x, localThreads_y, localThreads_z),
                                  0,
                                  handle.GetStream(),
                                  srcPtr,
                                  make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                  dstPtr,
                                  make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                  // PASS ADDITIONAL PARAMETERS HERE
                                  dstDescPtr->c,
                                  roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}

// Public API wrapper
RppStatus rppt_{{augmentation_name}}_gpu(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         // ADD ADDITIONAL PARAMETERS HERE
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle)
{
    return hip_exec_{{augmentation_name}}_tensor(static_cast<Rpp8u*>(srcPtr), srcDescPtr,
                                                  static_cast<Rpp8u*>(dstPtr), dstDescPtr,
                                                  // PASS ADDITIONAL PARAMETERS HERE
                                                  roiTensorPtrSrc, roiType,
                                                  *static_cast<rpp::Handle*>(rppHandle));
}
