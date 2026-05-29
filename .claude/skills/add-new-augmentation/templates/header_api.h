/*
 * Template for adding new augmentation API to header file
 *
 * Location: src/include/tensor/rppt_tensor_augmentations.h
 *
 * Instructions:
 * 1. Replace {{AUGMENTATION_NAME}} with your augmentation name (e.g., my_new_filter)
 * 2. Add any additional parameters between dstDescPtr and roiTensorPtrSrc
 * 3. Add documentation comments describing the function
 * 4. Insert these function declarations in the appropriate section of the header
 */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief {{AUGMENTATION_NAME}} augmentation on HOST backend
 * \ingroup group_tensor_augmentations
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as srcDescPtr)
 *
 * ADD ADDITIONAL PARAMETER DOCUMENTATION HERE IF NEEDED
 * For example:
 * \param [in] kernelSize kernel size for the operation (odd values: 3, 5, 7, 9)
 * \param [in] stdDevTensor standard deviation values per batch
 *
 * \param [in] roiTensorPtrSrc ROI data in HOST memory
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_{{AUGMENTATION_NAME}}_host(RppPtr_t srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          RppPtr_t dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          // ADD ADDITIONAL PARAMETERS HERE
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief {{AUGMENTATION_NAME}} augmentation on HIP backend
 * \ingroup group_tensor_augmentations
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as srcDescPtr)
 *
 * ADD ADDITIONAL PARAMETER DOCUMENTATION HERE IF NEEDED
 *
 * \param [in] roiTensorPtrSrc ROI data in HIP memory
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_{{AUGMENTATION_NAME}}_gpu(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         // ADD ADDITIONAL PARAMETERS HERE
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
