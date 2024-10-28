// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
#include <stdio.h>
#include <musa_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>

#define SIZE (1024ULL * 1024ULL * 1024ULL * sizeof(float))
#define WARMUP_ITERATIONS 1000
#define ITERATIONS 5000

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

void checkMPIError(int result, const char *msg) {
    if (result != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(result, error_string, &length);
        fprintf(stderr, "MPI Error: %s: %s\n", msg, error_string);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    float *d_tensor;
    musaEvent_t start, end;
    float elapsed_time;

    checkMPIError(MPI_Init(&argc, &argv), "MPI_Init");
    int rank, nranks;
    checkMPIError(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMPIError(MPI_Comm_size(MPI_COMM_WORLD, &nranks), "MPI_Comm_size");
    checkCudaError(musaSetDevice(0), "musaSetDevice");

    ncclComm_t comm;
    musaStream_t stream;

    ncclUniqueId id;
    if (rank == 0) {
        checkNcclError(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    checkNcclError(ncclCommInitRank(&comm, nranks, id, rank), "ncclCommInitRank");
    checkCudaError(musaStreamCreate(&stream), "musaStreamCreate");
    
    checkCudaError(musaMalloc(&d_tensor, SIZE), "musaMalloc");

    checkCudaError(musaEventCreate(&start), "musaEventCreate");
    checkCudaError(musaEventCreate(&end), "musaEventCreate");

    checkNcclError(ncclGroupStart(), "ncclGroupStart");
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        if (rank == 0) {
            checkNcclError(ncclSend(d_tensor, SIZE / sizeof(float), ncclFloat, 1, comm, stream), "ncclSend");
        }
        else if (rank == 1){
            checkNcclError(ncclRecv(d_tensor, SIZE / sizeof(float), ncclFloat, 0, comm, stream), "ncclRecv");
        }
    }
    checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
    checkCudaError(musaStreamSynchronize(stream), "musaStreamSynchronize");
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");

    checkCudaError(musaEventRecord(start), "musaEventRecord");
    checkNcclError(ncclGroupStart(), "ncclGroupStart");
    for (int i = 0; i < ITERATIONS; ++i) {
        if (rank == 0) {
            checkNcclError(ncclSend(d_tensor, SIZE / sizeof(float), ncclFloat, 1, comm, stream), "ncclSend");
        }
        else if (rank == 1){
            checkNcclError(ncclRecv(d_tensor, SIZE / sizeof(float), ncclFloat, 0, comm, stream), "ncclRecv");
        }
    }
    checkNcclError(ncclGroupEnd(), "ncclGroupEnd");
    checkCudaError(musaStreamSynchronize(stream), "musaStreamSynchronize");
    checkMPIError(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    checkCudaError(musaEventRecord(end), "musaEventRecord");
    checkCudaError(musaEventSynchronize(end), "musaEventSynchronize");
    checkCudaError(musaEventElapsedTime(&elapsed_time, start, end), "musaEventElapsedTime");

    double bandwidth = SIZE * ITERATIONS / (elapsed_time / 1000.0) + SIZE * ITERATIONS / (elapsed_time / 1000.0);
    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=" 
              << std::fixed << std::setprecision(2) << bandwidth / (1024.0 * 1024.0 * 1024.0) 
              << "GiB/s" << std::endl;

    std::cout << "[FlagPerf Result]interconnect-MPI_intraserver-bandwidth=" 
              << std::fixed << std::setprecision(2) << bandwidth / (1000.0 * 1000.0 * 1000.0) 
              << "GB/s" << std::endl;
    checkCudaError(musaEventDestroy(start), "musaEventDestroy");
    checkCudaError(musaEventDestroy(end), "musaEventDestroy");
    checkCudaError(musaFree(d_tensor), "musaFree");
    checkNcclError(ncclCommDestroy(comm), "ncclCommDestroy");
    checkCudaError(musaStreamDestroy(stream), "musaStreamDestroy");
    checkMPIError(MPI_Finalize(), "MPI_Finalize");
    return 0;
}
