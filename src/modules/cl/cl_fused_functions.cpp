#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"
#include<iostream>

RppStatus
color_twist_cl( cl_mem srcPtr,RppiSize srcSize,
                cl_mem dstPtr, const float alpha/*contrast*/, const float beta /*brightness*/, const float hue_factor /*hue factor*/, const float sat /* saturation_factor*/,
                RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{((srcSize.width + 15)/16) * 16, ((srcSize.height + 15)/16) * 16, 1};

    if (chnFormat == RPPI_CHN_PLANAR)
    {
       handle.AddKernel("", "", "colortwist.cl", "colortwist_pln", vld, vgd, "")(srcPtr, dstPtr, alpha, beta, hue_factor, sat, srcSize.height, srcSize.width);
    }
    else
    {
        handle.AddKernel("", "", "colortwist.cl", "colortwist_pkd", vld, vgd, "")(srcPtr, dstPtr, alpha, beta, hue_factor, sat, srcSize.height, srcSize.width);
    }
   
    return RPP_SUCCESS;
}

RppStatus
color_twist_cl_batch ( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle,  RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;

    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{ max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "colortwist.cl", "colortwist_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                                        plnpkdind
                                                                                        );
    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_cl( cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiSize dstSize, Rpp32u crop_pox_x,
                                        Rpp32u crop_pos_y, Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, 
                                        unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{                                                                                                                                                                                   
     int plnpkdind;
     int batch_size = handle.GetBatchSize();
     
   if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
   else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
       
    Rpp32u max_src_height, max_src_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_src_height, &max_src_width);

    // Threads should be launched w.r.t Destinalton Sizes
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height , handle.GetBatchSize()};
    handle.AddKernel("", "", "crop_mirror_normalize.cl", "crop_mirror_normalize_batch", vld, vgd, "")(srcPtr, dstPtr,                                                                                                                                                                                                                                                                               
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,       
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                        // handle.GetBatchSize(),
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind);       
    return RPP_SUCCESS;
} 

RppStatus
crop_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{                                                                                                                                                                                   
     int plnpkdind;
     int batch_size = handle.GetBatchSize();
     
   if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
   else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

    // Threads should be launched w.r.t Destinalton Sizes
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "crop.cl", "crop_batch", vld, vgd, "")(srcPtr, dstPtr,                                                                                                                                                                                                                                                                               
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,       
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                        // handle.GetBatchSize(),
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind);       
    return RPP_SUCCESS;
} 

RppStatus
resize_crop_mirror_cl_batch( cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{                                                                                                                                                                                   
     int plnpkdind;
     int batch_size = handle.GetBatchSize();
     
   if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
   else
        plnpkdind = 3;

    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
       
    Rpp32u max_src_height, max_src_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_src_height, &max_src_width);

    // Threads should be launched w.r.t Destinalton Sizes
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    handle.AddKernel("", "", "resize.cl", "resize_crop_mirror_batch", vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height, 
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,                                                                                                                                                                                                                                                                              
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,       
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        channel,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                        // handle.GetBatchSize(),
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        plnpkdind);       
    return RPP_SUCCESS;
} 

