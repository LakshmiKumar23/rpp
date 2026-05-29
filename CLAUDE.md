# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMD ROCm Performance Primitives (RPP) is a comprehensive, high-performance computer vision library for AMD processors with HIP (GPU) or CPU backends. It provides 2D image, 3D voxel, audio, and miscellaneous augmentation primitives.

**Key characteristics:**
- Dual backend architecture: HIP (GPU) and CPU (HOST)
- Test suites validate functionality and performance across backends
- Uses `.rgb` files with embedded 24-byte headers (not JPEG/PNG directly)
- Supports multiple data types: U8, F16, F32, I8
- Multiple layouts: PLN1 (1-channel planar), PLN3 (3-channel planar NCHW), PKD3 (3-channel packed NHWC)

## Build System

### Quick Build (HIP Backend - Default)
```bash
mkdir build-hip && cd build-hip
cmake ../rpp
make -j8
sudo make install
```

### CPU Backend Build
```bash
mkdir build-cpu && cd build-cpu
cmake ../rpp -DBACKEND=CPU
make -j8
sudo make install
```

### Build Options
- `BACKEND`: `HIP` (default) or `CPU`
- `RPP_AUDIO_SUPPORT`: `ON` (default) or `OFF`
- Installation paths: `${ROCM_PATH}/lib` (libraries), `${ROCM_PATH}/include/rpp` (headers), `${ROCM_PATH}/share/rpp` (samples/tests)

## Architecture

### Source Code Structure
```
src/
├── include/
│   ├── common/          # Common headers (rppdefs.h, rpp_version.h)
│   └── tensor/          # Tensor API headers
└── modules/
    └── tensor/
        ├── cpu/kernel/  # CPU backend implementations (C/AVX/SSE)
        └── hip/kernel/  # HIP backend implementations (GPU kernels)
```

### Backend Implementation Pattern
Each augmentation has parallel implementations:
- **CPU**: `src/modules/tensor/cpu/kernel/` - uses C, SSE, AVX optimizations
- **HIP**: `src/modules/tensor/hip/kernel/` - uses ROCm/HIP GPU kernels

The build system selects the appropriate backend based on the `BACKEND` CMake variable. Both backends implement the same API defined in `src/include/tensor/`.

### Test Suite Architecture

Test suites are under `utilities/test_suite/` with four categories:
- **Image** (`Tensor_image_host.cpp`, `Tensor_image_hip.cpp`): 2D image augmentations
- **Voxel** (`Tensor_voxel_host.cpp`, `Tensor_voxel_hip.cpp`): 3D voxel augmentations
- **Audio** (`Tensor_audio_host.cpp`, `Tensor_audio_hip.cpp`): Audio augmentations
- **Misc** (`Tensor_misc_host.cpp`, `Tensor_misc_hip.cpp`): Miscellaneous operations

Each test binary is controlled by Python runners (`runImageTests.py`, etc.) that:
1. Build the C++ test binary via CMake
2. Execute tests with various parameter combinations
3. Compare outputs against golden references (QA mode)
4. Report performance metrics

## Running Tests

### Image Test Suite

**Location:** `utilities/test_suite/HOST/` or `utilities/test_suite/HIP/`

**Unit tests (run once, save outputs):**
```bash
cd utilities/test_suite/HOST  # or HIP
python3 runImageTests.py --case_start 0 --case_end 102 --test_type 0
```

**Performance tests (100 runs, timing statistics):**
```bash
python3 runImageTests.py --case_start 0 --case_end 102 --test_type 1 --num_runs 100
```

**QA mode (validate against golden outputs):**
```bash
python3 runImageTests.py --case_start 0 --case_end 102 --test_type 0 --qa_mode 1 --batch_size 3
```

**Run specific test cases:**
```bash
python3 runImageTests.py --case_list 0 2 4 21 --test_type 0
```

**HIP profiling (GPU kernel analysis):**
```bash
python3 runImageTests.py --test_type 1 --profiling YES
```

### Single Test Execution

To run a specific test case manually (without Python wrapper):
```bash
cd utilities/test_suite/HOST/build
# Arguments: src1 src2 dst bitdepth outputFormat caseNum additionalParam numRuns testType layout verbosity qaMode decoderType batchSize roi scriptPath
./Tensor_image_host <src1_path> <src2_path> <output_path> 0 0 21 0 1 0 0 0 0 0 3 0 0 0 0 <script_path>
```

### Test Parameters
- `test_type`: `0` = unit tests (1 run), `1` = performance tests (100 runs default)
- `qa_mode`: `0` = no validation, `1` = compare against golden outputs
- `decoder_type`: `0` = packed `.rgb` files (default), `1` = OpenCV image loading
- `batch_size`: Number of images to process together (QA mode requires batch_size=3)
- `case_start/case_end`: Test case range (0-102 for images)

## Input Format: .rgb Files

**Critical:** Test binaries do NOT decode JPEG/PNG. They require pre-decoded `.rgb` files with embedded headers.

### Converting JPEGs to .rgb
```bash
cd utilities/test_suite/scripts
python3 jpeg_to_rgb_conversion.py /path/to/jpegs [--out-dir /path/to/output] [--recursive]
```

This creates `.rgb` files with:
- 24-byte header: magic (0x52474242 "RGBB"), version, width, height, channels (1/3/4), reserved
- Followed by raw pixel data (row-major, tight packing)

### YUV Input Format
YUV files use `.yuv + .info` sidecar approach:
- `.yuv`: NV12 YUV data (Y plane + interleaved UV)
- `.info`: Text file with `width=`, `height=`, `color_range=`, `col_standard=` key-value pairs

## Generating Golden Outputs

Golden outputs are reference `.bin` files used for QA validation.

**Step 1:** Enable debug mode in `utilities/test_suite/rpp_test_suite_common.h`:
```cpp
#define DEBUG_MODE 1
```

**Step 2:** Run test suite (generates `.bin` files in current directory):
```bash
cd utilities/test_suite/HOST
python3 runImageTests.py --case_start 0 --case_end 102 --test_type 0 --batch_size 3
```

**Step 3:** Organize outputs into `REFERENCE_OUTPUT/` structure:
```bash
cd utilities/test_suite
python3 scripts/organize_reference_outputs.py --source HOST --dest REFERENCE_OUTPUT
```

**Step 4:** Disable debug mode (set `DEBUG_MODE 0`) for normal testing.

## Code Conventions

### Data Type Suffixes
Files and functions use suffixes to indicate data type:
- `_u8`: 8-bit unsigned integer
- `_f32`: 32-bit float
- `_f16`: 16-bit float
- `_i8`: 8-bit signed integer

### Layout Conventions
- **PKD3**: Packed/interleaved 3-channel (NHWC) - `RGB RGB RGB ...`
- **PLN3**: Planar 3-channel (NCHW) - `RRR... GGG... BBB...`
- **PLN1**: Planar 1-channel (grayscale)

### Test Case Naming
Test outputs follow the pattern: `{function}_{datatype}[_{variation}].bin`
- Example: `brightness_u8.bin`, `resize_f32_interpolationTypeBILINEAR.bin`, `emboss_u8_kernelSize3.bin`

## Common Issues

### QA Test Failures
- Regenerate goldens if RGB→gray conversion formula changes (BT.601 luma weights)

### Build Failures
- Ensure ROCm is installed: `sudo amdgpu-install --usecase=rocm`
- Minimum ROCm version: 7.0.0
- GPU requirement for HIP backend: gfx908 or higher
- Check `CMAKE_PREFIX_PATH` includes `${ROCM_PATH}/lib/cmake`

## File Header Validation
The `.rgb` file parser validates:
- Magic number: 0x52474242 ("RGBB")
- Version: 1
- Dimensions: width > 0, height > 0 (no arbitrary upper limit)
- Channels: 1 (grayscale), 3 (RGB), or 4 (CMYK)

Invalid headers cause test failures with clear error messages indicating the file path and issue.

## Performance Testing
Performance tests measure:
- Wall time (always)
- GPU kernel time (HIP backend with `--profiling YES`)
- Min/max/average across multiple runs
- Comparison to baseline (QA performance mode)

Results are logged to `OUTPUT_PERFORMANCE_LOGS_*/` directories with CSV summaries.
