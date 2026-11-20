// BitNet I2_S GEMV/GEMM declarations
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_gemv_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s_scaled(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t weight_row_stride,
    const void * GGML_RESTRICT vy, size_t act_row_stride, int nr, int nc,
    enum ggml_type type, const float * act_scales, const int32_t * act_sums);

// Interleaved (8-row) variants — operate on repacked weight data
void ggml_gemv_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc);

#ifdef __cplusplus
}
#endif
