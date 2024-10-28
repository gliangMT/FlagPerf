// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
#include <stdio.h>
#include <musa_runtime.h>
#include <nccl.h>
#include <vector>
#include <iostream>
#include <iomanip>

#define GB (1024ULL * 1024ULL * 1024ULL)
#define SIZE (4ULL * GB)
#define WARMUP_ITERATIONS 100
#define ITERATIONS 10000

void checkCudaError(musaError_t err, const char *msg) {
    if (err != musaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, musaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkNcclError(ncclResult_t result, const char *msg) {
    if (result != ncclSuccess) {
        fprintf(stderr, "NCCL Error: %s: %s\n", msg, ncclGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int num_gpus = 8;
    int devs[num_gpus] = {0, 1, 2, 3, 4, 5, 6, 7};

    musaEvent_t start, end;
    float elapsed_time;
    std::vector<float*> d_src(num_gpus);
    std::vector<float*> d_dst(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<musaStream_t> streams(num_gpus);

    for (int i = 0; i < num_gpus; ++i) {
        checkCudaError(musaSetDevice(devs[i]), "musaSetDevice");
        checkCudaError(musaMalloc(&d_src[i], SIZE), "musaMalloc");
        checkCudaError(musaMalloc(&d_dst[i], SIZE), "musaMalloc");
        checkCudaError(musaMemset(d_src[i], 1.0f, SIZE), "musaMemset");
        checkCudaError(musaStreamCreate(&streams[i]), "musaStreamCreate");
    }
    checkNcclError(ncclCommInitAll(comms.data(), num_gpus, devs), "ncclCommInitAll");

    checkCudaError(musaEventCreate(&start), "musaEventCreate");
    checkCudaError(musaEventCreate(&end), "musaEventCreate");
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        checkNcclError(ncclGroupStart(), "ncclGroupStart");
        for (int j = 0; j < num_gpus; ++j) {
            checkNcclError(ncclAllReduce((const void*)d_src[j], (void*)d_dst[j], SIZE / sizeof(float), ncclFloat, ncclSum, comms[j], streams[j]), "ncclAllReduce");
        }
        checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
        for (int j = 0; j < num_gpus; ++j){
            checkCudaError(musaStreamSynchronize(streams[j]), "musaStreamSynchronize");
        } 
    }

    checkCudaError(musaEventRecord(start), "musaEventRecord");

    for (int i = 0; i < ITERATIONS; ++i) {
        checkNcclError(ncclGroupStart(), "ncclGroupStart");
        for (int j = 0; j < num_gpus; ++j) {
            checkNcclError(ncclAllReduce((const void*)d_src[j], (void*)d_dst[j], SIZE / sizeof(float), ncclFloat, ncclSum, comms[j], streams[j]), "ncclAllReduce");
        }
        checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
        for (int j = 0; j < num_gpus; ++j){
            checkCudaError(musaStreamSynchronize(streams[j]), "musaStreamSynchronize");
        }
    }
    checkCudaError(musaEventRecord(end), "musaEventRecord"); 
    checkCudaError(musaEventSynchronize(end), "musaEventSynchronize");
    checkCudaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");
    /*
        algbw = S/t
    Considering that each rank has a bandwidth to the outside world of B, the time to perform an allReduce operation of S elements is at best :
        t = (S*2*(n-1)) / (n*B)
    Indeed, we have S elements, 2*(n-1) operations per element, and n links of bandwidth B to perform them. Reordering the equation, we find that
        t = (S/B) * (2*(n-1)/n)
    Therefore, to get an AllReduce bandwidth measurement which we can compare to the hardware peak bandwidth, we compute :
        B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)
    More details can be found in https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    NVIDIA specifies the 600GBps for intra-server connect as a bidirectional bandwidth, 
    meaning each node can simultaneously upload and download at 300GBps. 
    To better reflect the ratio of the tested value to the specified value and 
    to align with common understanding of NVIDIA's product capabilities, 
    we have multiplied the bandwidth result here by two.
    */
    double algbw = SIZE * ITERATIONS / (elapsed_time / 1000.0);
    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-algbw=" 
              << std::fixed << std::setprecision(2) << algbw / (1024.0 * 1024.0 * 1024.0) 
              << "GiB/s" << std::endl;

    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-algbw=" 
              << std::fixed << std::setprecision(2) << algbw / (1000.0 * 1000.0 * 1000.0) 
              << "GB/s" << std::endl;
    double bandwidth = algbw * (2.0 * (num_gpus-1) / num_gpus);
    bandwidth = bandwidth + bandwidth;
    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=" 
              << std::fixed << std::setprecision(2) << bandwidth / (1024.0 * 1024.0 * 1024.0) 
              << "GiB/s" << std::endl;

    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=" 
              << std::fixed << std::setprecision(2) << bandwidth / (1000.0 * 1000.0 * 1000.0) 
              << "GB/s" << std::endl;
    for (int i = 0; i < num_gpus; ++i) {
        checkCudaError(musaFree(d_src[i]), "musaFree");
        checkCudaError(musaFree(d_dst[i]), "musaFree");
        checkNcclError(ncclCommDestroy(comms[i]), "ncclCommDestroy");
    }
    checkCudaError(musaEventDestroy(start), "musaEventDestroy");
    checkCudaError(musaEventDestroy(end), "musaEventDestroy");
    return 0;
}
