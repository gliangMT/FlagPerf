#include <mublas.h>
#include <musa_runtime.h>
#include <chrono>
#include <iostream>
#include <musa_bf16.h>

constexpr int M = 8192;
constexpr int N = 8192;
constexpr int K = 8192;

struct PrecisionConfig {
    musaDataType_t musaType;
    mublasComputeType_t mublasType;
    int bytesPerElement;
    const char* name;
    int NUM_ITERATIONS ;
    int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
    __mt_bfloat16  *d_A, *d_B, *d_C;

    musaMallocManaged(&d_A, M * K * config.bytesPerElement);
    musaMallocManaged(&d_B, K * N * config.bytesPerElement);
    if (config.musaType == MUSA_R_8I) {
        musaMallocManaged(&d_C, M * N * sizeof(float));
    } else {
        musaMallocManaged(&d_C, M * N * config.bytesPerElement);
    }

    mublasHandle_t handle;
    mublasCreate(&handle);

    __mt_bfloat16 alpha = __float2bfloat16(1.0f);
    __mt_bfloat16 beta = __float2bfloat16(0.0f);

    for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
        if (config.musaType == MUSA_R_8I) {
            mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.musaType, M,
                         d_B, config.musaType, K,
                         &beta,
                         d_C, MUSA_R_32I, M,
                         config.mublasType, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.musaType, M,
                         d_B, config.musaType, K,
                         &beta,
                         d_C, config.musaType, M,
                         config.mublasType, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    musaError_t syncError = musaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    if (syncError != musaSuccess) {
        std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
    }

    for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
        if (config.musaType == MUSA_R_8I) {
            mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.musaType, M,
                         d_B, config.musaType, K,
                         &beta,
                         d_C, MUSA_R_32I, M,
                         config.mublasType, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                         M, N, K, &alpha,
                         d_A, config.musaType, M,
                         d_B, config.musaType, K,
                         &beta,
                         d_C, config.musaType, M,
                         config.mublasType, MUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    syncError = musaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (syncError != musaSuccess) {
        std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Average " << config.name << " Single Op Duration: " << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

    double time_second = duration.count() / 1.0e6;
    double flops = 2.0 * M * N * K * config.NUM_ITERATIONS;
    double FLOPS = flops / time_second;
    double TFLOPS = FLOPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-BF16=" << TFLOPS << "TFLOPS" << std::endl;

    musaFree(d_A);
    musaFree(d_B);
    musaFree(d_C);

    mublasDestroy(handle);
}

int main() {
    PrecisionConfig bf16 = {
        MUSA_R_16BF,
        MUBLAS_COMPUTE_32F,
        2,
        "BF16",
        50000,
        10
    };

    test(bf16);

    return 0;
}

