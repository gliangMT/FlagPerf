// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <stdio.h>
#include <musa_runtime.h>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (16ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 10000

void checkCudaError(musaError_t err, const char *msg) {
    if (err != musaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, musaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *d_src, *d_dst;
    musaEvent_t start, end;
    float elapsed_time;

    checkCudaError(musaMalloc(&d_src, SIZE), "musaMalloc");
    checkCudaError(musaMalloc(&d_dst, SIZE), "musaMalloc");

    checkCudaError(musaEventCreate(&start), "musaEventCreate");
    checkCudaError(musaEventCreate(&end), "musaEventCreate");

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkCudaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyDeviceToDevice), "musaMemcpy");
    }

    checkCudaError(musaEventRecord(start), "musaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkCudaError(musaMemcpy(d_dst, d_src, SIZE, musaMemcpyDeviceToDevice), "musaMemcpy");
    }

    checkCudaError(musaEventRecord(end), "musaEventRecord");
    checkCudaError(musaEventSynchronize(end), "musaEventSynchronize");

    checkCudaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");

    double bandwidth = 2.0 * SIZE * ITERATIONS / (elapsed_time / 1000.0);

    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGiB/s\n", bandwidth / (1024.0 * 1024.0 * 1024.0));
    printf("[FlagPerf Result]main_memory-bandwidth=%.2fGB/s\n", bandwidth / (1000.0 * 1000.0 * 1000.0));

    checkCudaError(musaFree(d_src), "musaFree");
    checkCudaError(musaFree(d_dst), "musaFree");
    checkCudaError(musaEventDestroy(start), "musaEventDestroy");
    checkCudaError(musaEventDestroy(end), "musaEventDestroy");

    return 0;
}
