#include <iostream>
#include <musa_fp16.h>
#include <mma.h>
#include <chrono>
#include <cmath>

using namespace mtmusa;

// 矩阵块大小
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 全局矩阵大小（需是 WMMA 块大小的倍数）
constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 2048;

// 检查 MUSA 错误
#define MUSA_CHECK(status)                                                           \
    {                                                                                \
        musaError_t error = status;                                                 \
        if (error != musaSuccess) {                                                 \
            std::cerr << "MUSA Error: " << musaGetErrorString(error) << std::endl;  \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

// 使用 WMMA 实现的 BF16 矩阵乘法内核
__global__ void wmma_bf16_gemm_kernel(const __mt_bfloat16* A, const __mt_bfloat16* B, float* C, int m, int n, int k) {
    // 每个线程块负责计算一个 WMMA 矩阵块
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);


    if (warpM * WMMA_M >= m || warpN * WMMA_N >= n) return;

    // 声明 WMMA 片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __mt_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __mt_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累积器片段为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 遍历 K 维度分块
    for (int i = 0; i < k; i += WMMA_K) {
        // 从全局内存加载 A 和 B 的片段
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * k + i, k);
        wmma::load_matrix_sync(b_frag, B + i * n + warpN * WMMA_N, n);

        // 执行矩阵乘法并累加结果
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 将累积结果存回全局内存
    wmma::store_matrix_sync(C + warpM * WMMA_M * n + warpN * WMMA_N, c_frag, n, wmma::mem_row_major);
}

// #define UNROLL_NUM 16
// #define ITERS 256
// using namespace mtmusa;

// #define MMA_4(a, b, c)        \
//   wmma::mma_sync(c, a, b, c); \
//   wmma::mma_sync(c, a, b, c); \
//   wmma::mma_sync(c, a, b, c); \
//   wmma::mma_sync(c, a, b, c);

// #define MMA_16(a, b, c) \
//   MMA_4(a, b, c);       \
//   MMA_4(a, b, c);       \
//   MMA_4(a, b, c);       \
//   MMA_4(a, b, c);



// __device__ void compute_mma_bf16_impl(const __mt_bfloat16* A, const __mt_bfloat16* B, float* C, int M, int N, int K) {
//   wmma::fragment<wmma::matrix_a, M, N, K, __mt_bfloat16, wmma::row_major>
//       a_frag;
//   wmma::fragment<wmma::matrix_b, M, N, K, __mt_bfloat16, wmma::row_major>
//       b_frag;
//   wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
//   // #pragma unroll
//   for (int i = 0; i < ITERS; i++) {
//     MMA_16(a_frag, b_frag, c_frag);
//   }
//   wmma::store_matrix_sync((float*)ptr, c_frag, M, wmma::mem_col_major);
// }


// 初始化 BF16 矩阵数据
void initialize_matrices(__mt_bfloat16* A, __mt_bfloat16* B, int sizeA, int sizeB) {
    for (int i = 0; i < sizeA; i++) A[i] = __mt_bfloat16(float(rand() % 5));
    for (int i = 0; i < sizeB; i++) B[i] = __mt_bfloat16(float(rand() % 5));
}

// 主函数
int main() {
    // 主机和设备指针
    __mt_bfloat16* h_A, * h_B;
    float* h_C;
    __mt_bfloat16* d_A, * d_B;
    float* d_C;

    int sizeA = M * K;
    int sizeB = K * N;
    int sizeC = M * N;

    // 分配主机内存
    h_A = new __mt_bfloat16[sizeA];
    h_B = new __mt_bfloat16[sizeB];
    h_C = new float[sizeC];

    // 初始化主机矩阵
    initialize_matrices(h_A, h_B, sizeA, sizeB);

    // 分配设备内存
    MUSA_CHECK(musaMalloc(&d_A, sizeA * sizeof(__mt_bfloat16)));
    MUSA_CHECK(musaMalloc(&d_B, sizeB * sizeof(__mt_bfloat16)));
    MUSA_CHECK(musaMalloc(&d_C, sizeC * sizeof(float)));

    // 将数据从主机拷贝到设备
    MUSA_CHECK(musaMemcpy(d_A, h_A, sizeA * sizeof(__mt_bfloat16), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, sizeB * sizeof(__mt_bfloat16), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemset(d_C, 0, sizeC * sizeof(float)));

    // 配置 MUSA 网格和线程块
    dim3 blockDim(32, 32);  // 每个线程块 32x32 个线程
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);  // 根据矩阵尺寸计算网格大小

    // 性能测量
    constexpr int numIterations = 100;
    for (int i = 0; i < 10; i++) { // 预热
        wmma_bf16_gemm_kernel<<<gridDim, blockDim >>> (d_A, d_B, d_C, M, N, K);
    }
    musaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; i++) {
        wmma_bf16_gemm_kernel<<<gridDim, blockDim >>> (d_A, d_B, d_C, M, N, K);
    }
    MUSA_CHECK(musaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // 从设备拷贝结果（可选，验证正确性）
    MUSA_CHECK(musaMemcpy(h_C, d_C, sizeC * sizeof(float), musaMemcpyDeviceToHost));

    // 计算性能
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double total_ops = 2.0 * M * N * K * numIterations;
    double tflops = total_ops / (elapsed_seconds * 1e12);

    std::cout << "Elapsed Time: " << elapsed_seconds * 1e3 / numIterations << " ms\n";
    std::cout << "Performance: " << tflops << " TFLOPS\n";

    // 释放资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    MUSA_CHECK(musaFree(d_A));
    MUSA_CHECK(musaFree(d_B));
    MUSA_CHECK(musaFree(d_C));

    return 0;
}
