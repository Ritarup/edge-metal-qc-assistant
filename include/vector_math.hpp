#pragma once

#include <cstddef>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace qc::vision {

// PatchCore compares hundreds of floats per bank entry. Independent NEON
// accumulators let Cortex-A76 process these without a scalar dependency chain.
inline float descriptor_dot(const float* a, const float* b, std::size_t count) {
    std::size_t i = 0;
    float result = 0.0f;
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= count; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    result = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#endif
    for (; i < count; ++i) result += a[i] * b[i];
    return result;
}

} // namespace qc::vision
