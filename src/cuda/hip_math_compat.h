#ifndef CT2_HIP_MATH_COMPAT_H
#define CT2_HIP_MATH_COMPAT_H

// ROCm 7 device math compatibility
// ROCm 7 nightly no longer auto-includes __clang_hip_math.h, so standard math
// functions (sinf, cosf, etc.) are only available as __host__ from glibc.
// This header provides __device__ overloads using OCML intrinsics.

#ifdef CT2_USE_HIP

#include <new>
#include <cstdio>

// Device-side placement new (normally provided by __clang_hip_runtime_wrapper.h)
#if !defined(__CLANG_CUDA_WRAPPERS_NEW)
#define __CLANG_CUDA_WRAPPERS_NEW
__device__ inline void* operator new(size_t, void* __p) noexcept { return __p; }
__device__ inline void* operator new[](size_t, void* __p) noexcept { return __p; }
#endif

// Float C wrappers
__device__ inline float sinf(float x) { return __ocml_sin_f32(x); }
__device__ inline float cosf(float x) { return __ocml_cos_f32(x); }
__device__ inline float expf(float x) { return __ocml_exp_f32(x); }
__device__ inline float logf(float x) { return __ocml_log_f32(x); }
__device__ inline float tanhf(float x) { return __ocml_tanh_f32(x); }
__device__ inline float erff(float x) { return __ocml_erf_f32(x); }
__device__ inline float powf(float x, float y) { return __ocml_pow_f32(x, y); }
__device__ inline float rsqrtf(float x) { return __ocml_rsqrt_f32(x); }
__device__ inline float fabsf(float x) { return __ocml_fabs_f32(x); }
__device__ inline float fmaxf(float x, float y) { return __ocml_fmax_f32(x, y); }
__device__ inline float nearbyintf(float x) { return __ocml_nearbyint_f32(x); }
__device__ inline float __fdividef(float x, float y) { return x / y; }

// C++ float overloads (used by thrust/rocrand via std::exp, std::log etc.)
__device__ inline float exp(float x) { return __ocml_exp_f32(x); }
__device__ inline float log(float x) { return __ocml_log_f32(x); }

// Double wrappers (needed by rocrand internals)
__device__ inline double log(double x) { return __ocml_log_f64(x); }
__device__ inline double sqrt(double x) { return __ocml_sqrt_f64(x); }
__device__ inline double exp(double x) { return __ocml_exp_f64(x); }

// sincospi (used by rocrand for normal distribution)
__device__ inline void sincospi(double x, double* s, double* c) {
    *s = __ocml_sinpi_f64(x);
    *c = __ocml_cospi_f64(x);
}
__device__ inline void sincospi(float x, float* s, float* c) {
    *s = __ocml_sinpi_f32(x);
    *c = __ocml_cospi_f32(x);
}

#endif // CT2_USE_HIP

#endif // CT2_HIP_MATH_COMPAT_H
