#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-cpu-i2s.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static void check_close(float actual, float expected, float tolerance, const char * message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::fprintf(stderr, "FAILED: %s: actual %.9g expected %.9g\n", message, actual, expected);
        std::exit(1);
    }
}

static void run_case(ggml_type type, int k, int rows) {
    const bool grouped = type == GGML_TYPE_I2_S_G32;
    const int groups = grouped ? k/32 : 1;
    const size_t row_bytes = ggml_row_size(type, k);
    std::vector<float> master(rows*k);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < k; ++i) {
            const int v = (r*41 + i*37 + i/32*13) % 29 - 14;
            master[r*k + i] = (i%11 == 0 || r == 1) ? 0.0f : v*(0.013f + 0.002f*(i/32));
        }
    }

    std::vector<uint8_t> packed(rows*row_bytes, 0xcd);
    const size_t written = grouped
        ? quantize_i2_s_g32(master.data(), packed.data(), rows, k, nullptr)
        : quantize_i2_s_row(master.data(), packed.data(), rows, k, nullptr);
    require(written == packed.size(), "physical tensor byte count");

    std::vector<float> dequant(rows*k, -99.0f);
    std::vector<float> expected(rows*k, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const uint8_t * qrow = packed.data() + r*row_bytes;
        const float * scales = reinterpret_cast<const float *>(qrow + k/4);
        for (int g = 0; g < groups; ++g) {
            const int begin = grouped ? g*32 : 0;
            const int end = grouped ? begin + 32 : k;
            float abs_sum = 0.0f;
            for (int i = begin; i < end; ++i) abs_sum += std::fabs(master[r*k + i]);
            const float scale = std::max(abs_sum/(end - begin), 1e-8f);
            require(scales[g] == scale, "native absmean scale and scale ordering");
            for (int i = begin; i < end; ++i) {
                float q = std::nearbyint(master[r*k + i]/scale);
                q = std::max(-1.0f, std::min(1.0f, q));
                const int code = (qrow[i/4] >> (6 - 2*(i%4))) & 3;
                require(code == int(q) + 1, "2-bit code and byte ordering");
                expected[r*k + i] = q*scale;
            }
        }
        if (grouped) dequantize_row_i2_s_g32(qrow, dequant.data() + r*k, k);
        else dequantize_row_i2_s_row(qrow, dequant.data() + r*k, k);
        for (int i = 0; i < k; ++i) {
            require(dequant[r*k + i] == expected[r*k + i], "exact dequantized ternary weight");
        }
    }

    constexpr int cols = 5;
    std::vector<float> activations(cols*k);
    std::vector<int8_t> q8(cols*k);
    std::vector<float> act_scales(cols);
    std::vector<int32_t> act_sums(cols);
    for (int c = 0; c < cols; ++c) {
        for (int i = 0; i < k; ++i) {
            activations[c*k + i] = c == 4 ? 0.0f : std::sin(0.17f*i + 0.31f*c)*0.7f;
        }
        quantize_row_i8_s(activations.data() + c*k, q8.data() + c*k, k, &act_scales[c], &act_sums[c]);
    }

    std::vector<float> gemm(rows*cols, -123.0f);
    ggml_gemm_i2_i8_s_scaled(k, gemm.data(), rows, packed.data(), row_bytes,
        q8.data(), k, cols, rows, type, act_scales.data(), act_sums.data());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double reference = 0.0;
            if (act_scales[c] != 0.0f) {
                for (int i = 0; i < k; ++i) reference += expected[r*k + i]*q8[c*k + i]/act_scales[c];
            }
            const float tolerance = c == 4 ? 0.0f : 2e-5f*std::max(1.0, std::fabs(reference));
            check_close(gemm[c*rows + r], reference, tolerance, "W1.58A8 GEMV/GEMM known answer");
        }
    }
    std::printf("ok: %s K=%d rows=%d cols=%d\n", ggml_type_name(type), k, rows, cols);
}

static void check_legacy() {
    float input[32] = {};
    input[0] = -2.0f;
    input[1] = 0.0f;
    input[2] = 3.0f;
    uint8_t packed[8 + 32] = {};
    require(quantize_i2_s(input, packed, 1, 32, nullptr) == sizeof(packed), "legacy I2_S byte count");
    require((packed[0] >> 6) == 0 && ((packed[0] >> 4) & 3) == 1 && ((packed[0] >> 2) & 3) == 2,
        "legacy I2_S codes unchanged");
    require(*reinterpret_cast<float *>(packed + 8) == 2.0f, "legacy I2_S scale unchanged");

    uint8_t dequant_input[32] = {};
    dequant_input[0] = (0 << 6) | (1 << 4) | (2 << 2);
    float output[128];
    dequantize_row_i2_s(dequant_input, output, 128, 2.0f);
    require(output[0] == -2.0f && output[32] == 0.0f && output[64] == 2.0f,
        "legacy I2_S dequantization unchanged");
    std::printf("ok: legacy I2_S regression\n");
}

static void check_ggml_mul_mat(ggml_type type) {
    constexpr int k = 128;
    constexpr int rows = 5;
    constexpr int cols = 5;
    const int test_cols = type == GGML_TYPE_I2_S ? cols - 1 : cols;
    ggml_init_params params = {16*1024*1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "GGML matmul context");
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, k, rows);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, test_cols);

    std::vector<float> expected(rows*k, 0.0f);
    if (type == GGML_TYPE_I2_S) {
        constexpr float scale = 0.125f;
        auto * packed = static_cast<uint8_t *>(weight->data);
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < 32; ++j) {
                uint8_t byte = 0;
                for (int part = 0; part < 4; ++part) {
                    const int i = part*32 + j;
                    const int code = (r*7 + i*11)%3;
                    byte |= code << (6 - 2*part);
                    expected[r*k + i] = (code - 1)*scale;
                }
                packed[r*(k/4) + j] = byte;
            }
        }
        *reinterpret_cast<float *>(packed + rows*(k/4)) = scale;
    } else {
        std::vector<float> master(rows*k);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < k; ++i) {
                master[r*k + i] = ((r*41 + i*37 + i/32*13)%29 - 14)*(0.013f + 0.002f*(i/32));
            }
        }
        const size_t written = type == GGML_TYPE_I2_S_G32
            ? quantize_i2_s_g32(master.data(), weight->data, rows, k, nullptr)
            : quantize_i2_s_row(master.data(), weight->data, rows, k, nullptr);
        require(written == ggml_nbytes(weight), "GGML quantized tensor byte count");
        for (int r = 0; r < rows; ++r) {
            const auto * qrow = static_cast<const uint8_t *>(weight->data) + r*weight->nb[1];
            if (type == GGML_TYPE_I2_S_G32) dequantize_row_i2_s_g32(qrow, expected.data() + r*k, k);
            else dequantize_row_i2_s_row(qrow, expected.data() + r*k, k);
        }
    }

    std::vector<float> activations(test_cols*k);
    std::vector<int8_t> q8(test_cols*k);
    std::vector<float> act_scales(test_cols);
    std::vector<int32_t> act_sums(test_cols);
    for (int c = 0; c < test_cols; ++c) {
        for (int i = 0; i < k; ++i) activations[c*k + i] = std::sin(0.17f*i + 0.31f*c)*0.7f;
        quantize_row_i8_s(activations.data() + c*k, q8.data() + c*k, k, &act_scales[c], &act_sums[c]);
    }
    std::memcpy(activation->data, activations.data(), activations.size()*sizeof(float));

    ggml_tensor * result = ggml_mul_mat(ctx, weight, activation);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    require(ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS, "GGML CPU graph execution");
    const auto * actual = static_cast<const float *>(result->data);
    for (int c = 0; c < test_cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            double reference = 0.0;
            for (int i = 0; i < k; ++i) reference += expected[r*k + i]*q8[c*k + i]/act_scales[c];
            check_close(actual[c*rows + r], reference, 2e-5f*std::max(1.0, std::fabs(reference)),
                "GGML CPU mul_mat known answer");
        }
    }
    std::printf("ok: GGML CPU mul_mat %s K=%d rows=%d cols=%d threads=3\n",
        ggml_type_name(type), k, rows, test_cols);
    ggml_free(ctx);
}

int main() {
    check_legacy();
    check_ggml_mul_mat(GGML_TYPE_I2_S);
    for (ggml_type type : {GGML_TYPE_I2_S_ROW, GGML_TYPE_I2_S_G32}) {
        check_ggml_mul_mat(type);
        for (int k : {32, 64, 128, 256}) {
            for (int rows : {1, 3, 5}) run_case(type, k, rows);
        }
        ggml_init_params params = {16*1024*1024, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        require(ctx != nullptr, "GGML context");
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, 128, 5);
        require(tensor->nb[1] == ggml_row_size(type, 128), "GGML physical row stride");
        require(ggml_nbytes(tensor) == 5*ggml_row_size(type, 128), "GGML tensor allocation size");
        ggml_free(ctx);
    }
    std::printf("all scaled I2_S known-answer tests passed\n");
    return 0;
}
