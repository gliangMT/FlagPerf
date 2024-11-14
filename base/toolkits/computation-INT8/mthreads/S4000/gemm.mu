// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <mublas.h>
#include <musa_runtime.h>

#include <chrono>
#include <iostream>

constexpr int M = 8192;
constexpr int N = 8192;
constexpr int K = 8192;

struct PrecisionConfig {
  musaDataType_t musaType;
  mublasComputeType_t mublasType;
  int bytesPerElement;
  const char* name;
  int NUM_ITERATIONS;
  int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig& config) {
  int8_t* d_A, * d_B;
  int32_t* d_C;

  musaMalloc(&d_A, M * K * config.bytesPerElement);
  musaMalloc(&d_B, K * N * config.bytesPerElement);
  if (config.musaType == MUSA_R_8I) {
    musaMalloc(&d_C, M * N * sizeof(float));
  }
  else {
    musaMalloc(&d_C, M * N * config.bytesPerElement);
  }

  mublasHandle_t handle;
  mublasCreate(&handle);

  int alpha = 1;
  int beta = 0;

  for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
    if (config.musaType == MUSA_R_8I) {
      mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N, M, N, K, &alpha, d_A,
        config.musaType, M, d_B, config.musaType, K, &beta, d_C,
        MUSA_R_32I, M, config.mublasType,
        MUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    else {
      mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N, M, N, K, &alpha, d_A,
        config.musaType, M, d_B, config.musaType, K, &beta, d_C,
        config.musaType, M, config.mublasType,
        MUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
  }

  musaError_t syncError = musaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  if (syncError != musaSuccess) {
    std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
  }

  for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
    if (config.musaType == MUSA_R_8I) {
      mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N, M, N, K, &alpha, d_A,
        config.musaType, M, d_B, config.musaType, K, &beta, d_C,
        MUSA_R_32I, M, config.mublasType,
        MUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    else {
      mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N, M, N, K, &alpha, d_A,
        config.musaType, M, d_B, config.musaType, K, &beta, d_C,
        config.musaType, M, config.mublasType,
        MUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
  }
  syncError = musaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  if (syncError != musaSuccess) {
    std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
  }
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Average " << config.name << " Single Op Duration: "
    << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

  double time_second = duration.count() / 1.0e6;
  double ops = 2.0 * M * N * K * config.NUM_ITERATIONS;
  double OPS = ops / time_second;
  double TOPS = OPS / 1.0e12;

  std::cout << "[FlagPerf Result]" << "computation-INT8=" << TOPS << "TOPS"
    << std::endl;

  musaFree(d_A);
  musaFree(d_B);
  musaFree(d_C);

  mublasDestroy(handle);
}

int main() {
  PrecisionConfig int8 = { MUSA_R_8I, MUBLAS_COMPUTE_32I, 1, "INT8", 50000, 10 };

  test(int8);

  return 0;
}