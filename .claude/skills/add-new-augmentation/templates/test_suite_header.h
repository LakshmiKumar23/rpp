/*
 * Template for test suite header additions
 *
 * Location: utilities/test_suite/rpp_test_suite_image.h (or voxel/audio/misc)
 *
 * Instructions:
 * 1. Add enum value (use next available number after checking existing enums)
 * 2. Add to imageAugmentationMap with supported backends
 * 3. If augmentation has parameter variations, add to appropriate case sets
 */

// ========== STEP 1: Add to enum ==========
// Location: enum ImageAugmentation section

enum ImageAugmentation
{
    // ... existing augmentations (0-105)
    {{AUGMENTATION_NAME_UPPER}} = {{NEXT_NUMBER}},  // e.g., MY_NEW_FILTER = 106
};

// ========== STEP 2: Add to imageAugmentationMap ==========
// Location: imageAugmentationMap definition

const std::map<int, std::vector<std::string>> imageAugmentationMap =
{
    // ... existing mappings
    {{{NEXT_NUMBER}}, {"{{augmentation_name}}", "HOST", "HIP"}},  // Add both backends, or just "HOST" or just "HIP"
};

// ========== STEP 3: Add to variation case sets (if applicable) ==========
// Only add to these if your augmentation needs parameter variations

// For kernel-based filters that vary by kernel size
const unordered_set<int> kernelSizeCases = {
    ERODE, DILATE, BOX_FILTER, MEDIAN_FILTER, GAUSSIAN_FILTER, EMBOSS,
    {{AUGMENTATION_NAME_UPPER}}  // Add if your augmentation uses kernel sizes
};

// For augmentations with additional parameters
const unordered_set<int> additionalParamCases = {
    NOISE, RESIZE, ROTATE, WARP_AFFINE, WARP_PERSPECTIVE, ERODE, DILATE,
    BOX_FILTER, SOBEL_FILTER, MEDIAN_FILTER, GAUSSIAN_FILTER, REMAP,
    CHANNEL_PERMUTE, EMBOSS,
    {{AUGMENTATION_NAME_UPPER}}  // Add if your augmentation has additional parameters
};

// For augmentations with dual inputs
const unordered_set<int> dualInputCases = {
    BLEND, NON_LINEAR_BLEND, CROP_AND_PATCH, MAGNITUDE, PHASE,
    BITWISE_AND, BITWISE_XOR, BITWISE_OR,
    // {{AUGMENTATION_NAME_UPPER}}  // Uncomment and add if needs two inputs
};

// For augmentations with interpolation type variations
const unordered_set<int> interpolationTypeCases = {
    RESIZE, ROTATE, WARP_AFFINE, WARP_PERSPECTIVE, REMAP,
    // {{AUGMENTATION_NAME_UPPER}}  // Uncomment and add if needs interpolation
};

// For augmentations that output PLN1 (grayscale)
const unordered_set<int> pln1OutTypeCases = {
    COLOR_TO_GREYSCALE, SOBEL_FILTER,
    // {{AUGMENTATION_NAME_UPPER}}  // Uncomment and add if outputs grayscale
};

// ========== EXAMPLE VARIATIONS ==========

/*
 * Example 1: Simple augmentation with no variations
 * Just add to enum and imageAugmentationMap, nothing else needed.
 */

/*
 * Example 2: Filter with kernel size variations
 * Add to: kernelSizeCases, additionalParamCases
 * Python runner will automatically test with kernel sizes 3, 5, 7, 9
 */

/*
 * Example 3: Geometric transform with interpolation
 * Add to: interpolationTypeCases, additionalParamCases
 * Python runner will test with all interpolation types
 */
