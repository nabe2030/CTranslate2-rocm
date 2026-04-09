# CTranslate2 ROCm 7.13 Support for gfx1151 (Strix Halo)

## Overview

This branch adds ROCm 7.13 compatibility to the [pigeek/CTranslate2-rocm](https://github.com/pigeek/CTranslate2-rocm) fork, enabling faster-whisper GPU inference on AMD Radeon 8060S (gfx1151 / Strix Halo) with TheRock nightly ROCm packages.

### Why this fork instead of upstream CTranslate2?

The official [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) does not include any HIP/ROCm code. The pigeek fork integrates ROCm support (`WITH_HIP` build option, CUDA→HIP API translation, MIOpen support) directly into the v4.1.0 source tree, eliminating the version mismatch issues with faster-whisper's `ctranslate2>=4.0` requirement.

However, the pigeek fork targets ROCm 6.x. ROCm 7.x introduced several API incompatibilities that this patch resolves.

## Test Environment

| Component | Version |
|---|---|
| Hardware | GMKtec EVO-X2, Ryzen AI Max+ 395, Radeon 8060S (gfx1151) |
| OS | Ubuntu 25.10 (kernel 6.18.20) |
| ROCm | TheRock nightly 7.13.0a20260402 (`/opt/rocm-nightly`) |
| Compiler | AMD clang 23.0.0 (from TheRock) |
| Python | 3.13.7 |
| faster-whisper | 1.2.1 (GPU inference confirmed with large-v3) |

> **Note on Ubuntu 26.04 LTS:** Ubuntu 26.04 LTS is expected to release on April 23, 2026. The ROCm 7 patches and build instructions in this document should be directly applicable to 26.04 as well, potentially with simplified ROCm installation via official apt packages.

## ROCm 7.x Incompatibilities

### 1. Device math functions no longer implicitly included

ROCm 7 removed the automatic inclusion of `__clang_hip_math.h`. Device-side calls to `sinf`, `cosf`, `expf`, `tanhf`, `erff`, `powf`, `rsqrtf`, `fabsf`, `fmaxf`, `nearbyintf`, and `__fdividef` fail with "no matching function" errors.

**Fix:** New header `src/cuda/hip_math_compat.h` provides `__device__` wrappers that call OCML intrinsics directly (e.g., `__ocml_sin_f32`). Also provides `sincospi` (needed by rocrand), double-precision wrappers, and device-side placement `operator new` (needed by thrust).

### 2. CMake target name changes

`hiprand` changed to `hip::hiprand` as a CMake imported target.

**Fix:** Updated `CMakeLists.txt` link target.

### 3. Thrust headers no longer implicitly included

`thrust::counting_iterator`, `thrust::reduce`, `thrust::max_element`, etc. require explicit includes in ROCm 7 (`<thrust/iterator/counting_iterator.h>`, `<thrust/reduce.h>`, `<thrust/extrema.h>`, etc.).

**Fix:** Added explicit thrust includes in `src/cuda/utils.h`.

## Changed Files

### New file
- **`src/cuda/hip_math_compat.h`** — ROCm 7 compatibility header with OCML-based `__device__` math function wrappers

### Modified files
| File | Change |
|---|---|
| `CMakeLists.txt` | `hiprand` → `hip::hiprand` link target |
| `src/cuda/utils.h` | Include `hip_math_compat.h`; add explicit thrust header includes |
| `src/cuda/random.h` | Include `hip/hip_runtime.h` and `hip_math_compat.h` before hiprand |
| `src/cuda/primitives.cu` | Replace `min()` with ternary operator for device compatibility |
| `src/ops/softmax_gpu.cu` | Replace `::max()` with ternary operator |
| `src/ops/dequantize_gpu.cu` | Replace `__fdividef` with plain division under `#ifdef CT2_USE_HIP` |
| `src/ops/layer_norm_gpu.cu` | Include `hip_math_compat.h` |
| `src/ops/rms_norm_gpu.cu` | Include `hip_math_compat.h` |

All changes are guarded by `#ifdef CT2_USE_HIP` and do not affect CUDA builds.

## Build Instructions

### Prerequisites

- TheRock nightly tarball extracted to `/opt/rocm-nightly`:
  ```bash
  curl -L -o /tmp/therock.tar.gz \
    https://rocm.nightlies.amd.com/tarball/therock-dist-linux-gfx1151-7.13.0a20260402.tar.gz
  sudo mkdir -p /opt/rocm-nightly
  sudo tar xzf /tmp/therock.tar.gz -C /opt/rocm-nightly
  sudo ln -sf /opt/rocm-nightly /opt/rocm
  ```

- TheRock pip wheels for rocBLAS Tensile kernels (tarball does not include gfx1151 Tensile libraries):
  ```bash
  python3 -m venv ~/rocm-therock
  source ~/rocm-therock/bin/activate
  pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries]"
  deactivate
  ```

### CMake + Build

```bash
cd ~/CTranslate2
mkdir -p build && cd build

CLANG=/opt/rocm-nightly/lib/llvm/bin/clang++

cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_HIP=ON \
      -DWITH_CUDNN=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
      -DCMAKE_C_COMPILER=/opt/rocm-nightly/lib/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=$CLANG \
      -DCMAKE_HIP_COMPILER=$CLANG \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-nightly \
      -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm-nightly -I/opt/rocm-nightly/include -include hip/hip_runtime.h" \
      -DCMAKE_PREFIX_PATH=/opt/rocm-nightly \
      -DCMAKE_INSTALL_PREFIX=$HOME/ctranslate2-install \
      -DWITH_MKL=OFF -DWITH_DNNL=OFF -DBUILD_CLI=OFF \
      -DOPENMP_RUNTIME=COMP \
      ..

cmake --build . -j$(nproc)
cmake --install .
```

### Python bindings + faster-whisper

```bash
python3 -m venv ~/rocm713-whisper
source ~/rocm713-whisper/bin/activate
pip install --upgrade pip setuptools wheel pybind11

cd ~/CTranslate2/python
CXXFLAGS="-I$HOME/ctranslate2-install/include" \
LDFLAGS="-L$HOME/ctranslate2-install/lib -L/opt/rocm-nightly/lib" \
CT2_INSTALL_PREFIX=$HOME/ctranslate2-install \
pip install --no-build-isolation .

pip install faster-whisper soundfile tqdm
```

**Important:** Do NOT install PyTorch in this venv. PyTorch's bundled ROCm SDK conflicts with `/opt/rocm-nightly` (LLVM symbol mismatch). faster-whisper inference does not require PyTorch.

### Runtime environment

```bash
export LD_LIBRARY_PATH=/opt/rocm-nightly/lib:/opt/rocm-nightly/lib/llvm/lib:$HOME/ctranslate2-install/lib:$LD_LIBRARY_PATH
export ROCBLAS_TENSILE_LIBPATH=$HOME/rocm-therock/lib/python3.13/site-packages/_rocm_sdk_libraries_gfx1151/lib/rocblas/library
export ROCM_PATH=/opt/rocm-nightly
```

### Verify

```python
import ctranslate2
print(ctranslate2.__version__)              # 4.1.0
print(ctranslate2.get_cuda_device_count())  # 1

from faster_whisper import WhisperModel
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
# Model loaded on GPU
```

## Known Issues

- **MIOpen warning about missing `gfx1151_20.HIP.fdb.txt`**: Harmless. MIOpen will compile kernels online on first use.
- **`xnack 'Off' was requested for a processor that does not support it`**: Harmless warning.
- **Tarball missing rocBLAS Tensile kernels**: Must set `ROCBLAS_TENSILE_LIBPATH` to TheRock pip wheels location.

## Credits

- [pigeek/CTranslate2-rocm](https://github.com/pigeek/CTranslate2-rocm) — Original ROCm 6.x port
- [davidguttman/whisper-rocm](https://github.com/davidguttman/whisper-rocm) — Initial ROCm patch for CTranslate2 v3.23.0
- ROCm 7.13 patches generated with assistance from [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) (Anthropic)
