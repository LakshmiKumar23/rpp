"""
Template for Python test runner variations

Location: utilities/test_suite/HOST/runImageTests.py and utilities/test_suite/HIP/runImageTests.py

Instructions:
1. Add variation handling if your augmentation needs parameter variations
2. Most augmentations don't need any changes - the default 'else' case handles them
3. Only add custom handling for kernel sizes, interpolation types, noise types, etc.
"""

# ========== VARIATION TYPE 1: Kernel Size ==========
# For filters that vary by kernel size (box_filter, gaussian, emboss, etc.)
# Add to the kernel size conditional:

def run_unit_test(...):
    # ... existing code ...

    if imageAugmentationMap[int(case)][0] in {"erode", "dilate", "box_filter",
                                                "median_filter", "gaussian_filter",
                                                "emboss", "{{augmentation_name}}"}:  # Add here
        for kernelSize in range(3, 10, 2):  # Tests kernel sizes 3, 5, 7, 9
            print(f"./{binName} {srcPath1} {srcPath2} {dstPathTemp} {bitDepth.value} {outputFormatToggle.value} {case} {kernelSize}")
            result = subprocess.Popen([buildFolderPath + "/build/" + binName,
                                      srcPath1, srcPath2, dstPathTemp,
                                      str(bitDepth.value), str(outputFormatToggle.value),
                                      str(case), str(kernelSize),
                                      str(numRuns), str(testType), str(layout), "0",
                                      str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log_detected(result, errorLog, imageAugmentationMap[int(case)][0],
                        get_bit_depth(bitDepth.value),
                        get_image_layout_type(layout, outputFormatToggle.value, "HIP"))

# ========== VARIATION TYPE 2: Interpolation Type ==========
# For geometric transforms (resize, rotate, warp, etc.)
# Add to the interpolation conditional:

    elif imageAugmentationMap[int(case)][0] in {"resize", "rotate", "warp_affine",
                                                  "remap", "warp_perspective",
                                                  "{{augmentation_name}}"}:  # Add here if needed
        if singleImageFlag:
            interpolationRange = 1  # nearestneighbor only for single image
        elif imageAugmentationMap[int(case)][0] in {"warp_perspective", "remap"}:
            interpolationRange = 2  # Only 2 types for these
        else:
            interpolationRange = 6  # All 6 types: BICUBIC, BILINEAR, GAUSSIAN, NEAREST, LANCZOS, TRIANGULAR

        for interpolationType in range(interpolationRange):
            print(f"./{binName} {srcPath1} {srcPath2} {dstPathTemp} {bitDepth.value} {outputFormatToggle.value} {case} {interpolationType}")
            result = subprocess.Popen([buildFolderPath + "/build/" + binName, ...])
            log_detected(...)

# ========== VARIATION TYPE 3: Custom Parameter Range ==========
# For augmentations with custom parameter variations
# Add a new elif block:

    elif imageAugmentationMap[int(case)][0] == "{{augmentation_name}}":
        # Define your custom parameter range
        for paramValue in range(min_value, max_value, step):  # Customize this
            print(f"./{binName} {srcPath1} {srcPath2} {dstPathTemp} {bitDepth.value} {outputFormatToggle.value} {case} {paramValue}")
            result = subprocess.Popen([buildFolderPath + "/build/" + binName,
                                      srcPath1, srcPath2, dstPathTemp,
                                      str(bitDepth.value), str(outputFormatToggle.value),
                                      str(case), str(paramValue),
                                      str(numRuns), str(testType), str(layout), "0",
                                      str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log_detected(result, errorLog, imageAugmentationMap[int(case)][0],
                        get_bit_depth(bitDepth.value),
                        get_image_layout_type(layout, outputFormatToggle.value, "HIP"))

# ========== NO VARIATIONS ==========
# For simple augmentations without variations, no changes needed!
# The default 'else' case handles them:

    else:
        print(f"./{binName} {srcPath1} {srcPath2} {dstPathTemp} {bitDepth.value} {outputFormatToggle.value} {case} 0 {numRuns} {testType} {layout}")
        result = subprocess.Popen([buildFolderPath + "/build/" + binName,
                                  srcPath1, srcPath2, dstPathTemp,
                                  str(bitDepth.value), str(outputFormatToggle.value),
                                  str(case), "0",  # additionalParam = 0 for no variations
                                  str(numRuns), str(testType), str(layout), "0",
                                  str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_detected(...)

# ========== REPEAT FOR PERFORMANCE TESTS ==========
# The same pattern applies to run_performance_test() function
# Just search for the similar if/elif blocks and add your augmentation

# ========== PROFILING TESTS (HIP only) ==========
# Also add to run_performance_test_with_profiler() for profiling support
# Same pattern as above, but also handles the output file naming

# ========== PROFILING RESULTS CONSOLIDATION ==========
# If you added variations, also update the profiling results section
# around line 507 in runImageTests.py:

def consolidate_profiling_results():
    # ... existing code ...

    if imageAugmentationMap[int(CASE_NUM)][0] in {"erode", "dilate", "box_filter",
                                                    "median_filter", "gaussian_filter",
                                                    "emboss", "{{augmentation_name}}"}:
        KSIZE_LIST = [3, 5, 7, 9]
        for KSIZE in KSIZE_LIST:
            CASE_FILE_PATH = CASE_RESULTS_DIR + f"/output_case{CASE_NUM}_bitDepth{BIT_DEPTH}_oft{OFT}_kernelSize{KSIZE}.stats.csv"
            fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter)
            # ...
