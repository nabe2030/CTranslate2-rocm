# CTranslate2 ROCm 7.13 Fix Guide — Claude Code Instructions

This document was passed to [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) (Anthropic's AI coding agent) to generate the ROCm 7.13 compatibility patches for CTranslate2.

## Initial Prompt to Claude Code

> This project (pigeek/CTranslate2-rocm) needs to be fixed to build with ROCm 7.13 + gfx1151.
>
> 1. Read this fix guide to understand all errors
> 2. Check current changes with `git diff`
> 3. Run cmake + build, check errors
> 4. Fix remaining errors and repeat until build succeeds
> 5. Run `cmake --install .` when done

## Build Environment

- OS: Ubuntu 25.10, Python 3.13.7
- ROCm: TheRock nightly 7.13.0a20260402 (`/opt/rocm-nightly`)
- Compiler: `/opt/rocm-nightly/lib/llvm/bin/clang++`
- Target: gfx1151 (Radeon 8060S / Strix Halo)

## CMake Command

```bash
cd ~/CTranslate2/build && rm -rf *
CLANG=/opt/rocm-nightly/lib/llvm/bin/clang++
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_HIP=ON -DWITH_CUDNN=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
      -DCMAKE_C_COMPILER=/opt/rocm-nightly/lib/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=$CLANG \
      -DCMAKE_HIP_COMPILER=$CLANG \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-nightly \
      -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm-nightly -I/opt/rocm-nightly/include -include hip/hip_runtime.h" \
      -DCMAKE_PREFIX_PATH=/opt/rocm-nightly \
      -DCMAKE_INSTALL_PREFIX=$HOME/ctranslate2-install \
      -DWITH_MKL=OFF -DWITH_DNNL=OFF -DBUILD_CLI=OFF \
      -DOPENMP_RUNTIME=COMP ..
cmake --build . -j$(nproc)
```

## Error List (All Build Errors)

The following errors were collected from `cmake --build . -j$(nproc) 2>&1 | grep "error:" | sort -u`:

### Category 1: thrust API changes

In ROCm 7, thrust headers are no longer implicitly included. `thrust::counting_iterator`, `thrust::reduce`, `thrust::max_element` etc. require explicit `#include` directives.

```
src/cuda/helpers.h:107:21: error: no template named 'counting_iterator' in namespace 'thrust'
src/cuda/helpers.h:98:55: error: no template named 'counting_iterator' in namespace 'thrust'
src/cuda/primitives.cu:62:42: error: no template named 'counting_iterator' in namespace 'thrust'
src/cuda/primitives.cu:121:26: error: no member named 'reduce' in namespace 'thrust'
src/cuda/primitives.cu:131:35: error: no member named 'max_element' in namespace 'thrust'
src/cuda/primitives.cu:141:26: error: no member named 'reduce' in namespace 'thrust'
src/cuda/primitives.cu:707:39: error: no member named 'reduce' in namespace 'thrust'
src/ops/gumbel_max_gpu.cu:32:19: error: no template named 'counting_iterator' in namespace 'thrust'
```

**Affected files:**
- `src/cuda/helpers.h` (lines 98, 107)
- `src/cuda/primitives.cu` (lines 62, 121, 131, 141, 707)
- `src/ops/gumbel_max_gpu.cu` (line 32)

### Category 2: Device math function signature changes

In ROCm 7, `__clang_hip_math.h` is no longer automatically included. Float device math functions like `expf`, `tanhf`, `erff` etc. are undefined in device code. Need to provide `__device__` overloads or use OCML intrinsics directly.

```
src/cuda/helpers.h:288:32: error: no matching function for call to 'erff'
src/cuda/helpers.h:296:34: error: no matching function for call to 'tanhf'
src/cuda/helpers.h:296:79: error: no matching function for call to 'powf'
src/cuda/helpers.h:304:27: error: no matching function for call to 'expf'
src/cuda/helpers.h:312:27: error: no matching function for call to 'expf'
src/cuda/helpers.h:320:16: error: no matching function for call to 'tanhf'
src/cuda/primitives.cu:694:14: error: no matching function for call to 'expf'
src/ops/gumbel_max_gpu.cu:19:26: error: no matching function for call to 'logf'
src/ops/layer_norm_gpu.cu:182:16: error: no matching function for call to 'fmaxf'
src/ops/layer_norm_gpu.cu:184:22: error: no matching function for call to 'rsqrtf'
src/ops/quantize_gpu.cu:11:22: error: no matching function for call to 'fabsf'
src/ops/quantize_gpu.cu:11:32: error: no matching function for call to 'fabsf'
src/ops/quantize_gpu.cu:51:16: error: no matching function for call to 'nearbyintf'
src/ops/rms_norm_gpu.cu:39:21: error: no matching function for call to 'rsqrtf'
```

**Affected functions:** `erff`, `tanhf`, `powf`, `expf`, `logf`, `fmaxf`, `rsqrtf`, `fabsf`, `nearbyintf`

### Category 3: CUDA-specific intrinsics

```
src/ops/dequantize_gpu.cu:12:16: error: use of undeclared identifier '__fdividef'
```

`__fdividef` does not exist in HIP. Replace with `(a) / (b)` under `#ifdef CT2_USE_HIP`.

### Category 4: Namespace issues

```
src/ops/softmax_gpu.cu:180:16: error: no member named 'max' in the global namespace
src/cuda/primitives.cu:326:20: error: use of undeclared identifier 'min'
```

Replace `::max()` and `min()` with ternary operators or `std::max` / `std::min`.

### Category 5: rocrand header errors

```
/opt/rocm-nightly/include/rocrand/rocrand_normal.h:90:9: error: use of undeclared identifier 'sincospi'
/opt/rocm-nightly/include/rocrand/rocrand_normal.h:166:9: error: use of undeclared identifier 'sincospi'
/opt/rocm-nightly/include/rocrand/rocrand_log_normal.h:620:12: error: no matching function for call to 'expf'
/opt/rocm-nightly/include/rocrand/rocrand_log_normal.h:644:19: error: no matching function for call to 'expf'
/opt/rocm-nightly/include/rocrand/rocrand_log_normal.h:644:48: error: no matching function for call to 'expf'
```

These are errors inside rocrand's own headers. They need `sincospi`, `expf` etc. to be defined as `__device__` functions before `hiprand_kernel.h` is included.

## Fix Strategy

1. Only modify code within `#ifdef CT2_USE_HIP` guards
2. For thrust: add explicit `#include` directives for the new header paths
3. For math functions: provide `__device__` wrappers using OCML intrinsics, or use `static_cast<float>()` wrappers
4. For CUDA intrinsics (`__fdividef`): replace with HIP-compatible alternatives under `#ifdef CT2_USE_HIP`
5. After changes, run `cmake --build . -j$(nproc)` to verify
