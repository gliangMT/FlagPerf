#include <iostream>
#include <musa_runtime.h>

#include "mute/tensor.hpp"

#include "mutlass/mutlass.h"

#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"

#include "mutlass/util/command_line.h"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/packed_stride.hpp"
#include "mutlass/util/host_tensor.h"
#include "mutlass/util/reference/device/tensor_fill.h"
#include "mutlass/util/reference/device/tensor_compare.h"
#include "mutlass/util/reference/device/gett.hpp"
#include <chrono>

#define MUSA_CHECK(status)                                              \
  {                                                                     \
    musaError_t error = status;                                         \
    if (error != musaSuccess) {                                         \
      std::cerr << "Got bad musa status: " << musaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define MUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    mutlass::Status error = status;                                                              \
    if (error != mutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got mutlass error: " << mutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

constexpr int NUM_ITERATIONS = 1000;
constexpr int WARMUP_ITERATIONS = 10;

constexpr int M_INPUT = 8192;
constexpr int N_INPUT = 8192;
constexpr int K_INPUT = 8192;
constexpr int L_INPUT = 1;

using namespace mute;

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
    mutlass::DeviceAllocation<Element>& block,
    uint64_t seed = 2023) {

    Element scope_max, scope_min;
    int bits_input = mutlass::sizeof_bits<Element>::value;

    if constexpr (std::is_same_v<Element, mutlass::bfloat16_t>) {
        // 对于 bfloat16 类型，使用构造函数进行初始化
        scope_max = mutlass::bfloat16_t(2);   // 使用构造函数进行初始化
        scope_min = mutlass::bfloat16_t(-2);  // 使用构造函数进行初始化
    }
    else if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
    }
    else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
    }
    else {
        scope_max = 8;
        scope_min = -8;
    }

    mutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, scope_max, scope_min, 0);

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int test() {
    musaDeviceProp props;

    MUSA_CHECK(musaGetDeviceProperties(&props, 0));

    if (props.major * 10 + props.minor != 22) {
        std::cout
            << "This example requires a GPU of MooreThreads's Quyuan Architecture.\n";
        return 0;
    }

    //
    // Build Gemm Kernel
    //
    using LayoutA = mutlass::layout::ColumnMajor;
    using LayoutB = mutlass::layout::ColumnMajor;
    using LayoutC = mutlass::layout::ColumnMajor;
    using LayoutD = mutlass::layout::ColumnMajor;

    // using ElementA = mutlass::half_t;
    // using ElementB = mutlass::half_t;
    // using ElementC = mutlass::half_t;
    // using ElementD = mutlass::half_t;

    using ElementA = mutlass::bfloat16_t;
    using ElementB = mutlass::bfloat16_t;
    using ElementC = mutlass::bfloat16_t;
    using ElementD = mutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;

    using ArchTag = mutlass::arch::Mp22;
    using OpClass = mutlass::arch::OpClassTensorOp;

    // 16Byte Alignment
    static constexpr int AlignmentA = 16 / sizeof(ElementA);
    static constexpr int AlignmentB = 16 / sizeof(ElementB);
    static constexpr int AlignmentC = 16 / sizeof(ElementC);
    static constexpr int AlignmentD = 16 / sizeof(ElementD);

    using CollectiveMainloop = typename mutlass::gemm::collective::CollectiveBuilder<    // 矩阵计算主循环
        ArchTag, OpClass,
        ElementA, LayoutA, AlignmentA,                        // Operand A
        ElementB, LayoutB, AlignmentB,                        // Operand B
        ElementAccumulator,
        Shape<_256, _256, _32>,                               // TileShape
        Shape<_1, _1, _1>,                                    // ClusterShape
        Layout<Shape<_2, _2, _1>>,                              // AtomLayout
        mutlass::gemm::collective::PermuteLayoutAuto,         // PermuteLayoutType
        mutlass::gemm::collective::StageCountAuto,            // StageCountType
        mutlass::gemm::collective::KernelScheduleAuto         // KernelScheduleType
    >::CollectiveOp;

    using CollectiveEpilogue = typename mutlass::epilogue::collective::CollectiveBuilder<   // 尾部计算
        ArchTag, OpClass,
        Shape<_256, _256, _32>,                               // TileShape
        Shape<_1, _1, _1>,                                    // ClusterShape
        mutlass::epilogue::collective::EpilogueTileAuto,      // EpilogueTileType
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,                        // Operand C
        ElementD, LayoutD, AlignmentD,                        // Output  D
        mutlass::epilogue::collective::EpilogueScheduleAuto   // EpilogueScheduleType
    >::CollectiveOp;

    using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    //
    // Initialize operands
    //
    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    ProblemShapeType problem_size = ProblemShapeType{ M_INPUT,N_INPUT,K_INPUT,L_INPUT };

    auto [M, N, K, L] = problem_size;
    uint64_t seed = 0;

    StrideA stride_A = mutlass::make_mute_packed_stride(StrideA{}, mute::make_shape(M, K, L));
    StrideB stride_B = mutlass::make_mute_packed_stride(StrideB{}, mute::make_shape(N, K, L));
    StrideC stride_C = mutlass::make_mute_packed_stride(StrideC{}, mute::make_shape(M, N, L));
    StrideD stride_D = mutlass::make_mute_packed_stride(StrideD{}, mute::make_shape(M, N, L));

    mutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
    mutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
    mutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
    mutlass::DeviceAllocation<typename Gemm::ElementD> block_D;
    mutlass::DeviceAllocation<typename Gemm::ElementD> block_ref_D;


    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    float alpha(1.0);
    float beta(0.0);


    typename Gemm::Arguments arguments{
      mutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{alpha, beta},
       block_C.get(), stride_C, block_D.get(), stride_D},
      mutlass::KernelHardwareInfo{}
    };

    // Instantiate MUTLASS kernel depending on templates
    Gemm gemm;

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    mutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    MUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize MUTLASS kernel with arguments and workspace pointer
    MUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Warm up
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        MUTLASS_CHECK(gemm.run());
    }
    auto musa_err = musaDeviceSynchronize();
    if (musaSuccess != musa_err) {
        std::cerr << "ERROR: GEMM operator execution failed. with error :";
        std::cerr << musaGetErrorString(musa_err) << "\n";
        return 1;
    }

    musaEvent_t e_start, e_end;
    float elapsed_time;

    // Run Gemm Kernel
    auto start = std::chrono::high_resolution_clock::now();
    MUSA_CHECK(musaEventCreate(&e_start));
    MUSA_CHECK(musaEventCreate(&e_end));

    MUSA_CHECK(musaEventRecord(e_start));
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        MUTLASS_CHECK(gemm.run());
    }

    musa_err = musaDeviceSynchronize();
    if (musaSuccess != musa_err) {
        std::cerr << "ERROR: GEMM operator execution failed. with error :";
        std::cerr << musaGetErrorString(musa_err) << "\n";
        return 1;
    }

    MUSA_CHECK(musaEventRecord(e_end));
    MUSA_CHECK(musaEventSynchronize(e_end));
    MUSA_CHECK(musaEventElapsedTime(&elapsed_time, e_start, e_end));


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Average " << "BF16" << " Single Op Duration: " << duration.count() / NUM_ITERATIONS << " us" << std::endl;

    std::cout << "[MUSA EVENTS]Average " << "BF16" << " Single Op Duration: " << elapsed_time * 1000 / NUM_ITERATIONS << " us" << std::endl;

    double time_second = duration.count() / 1.0e6;
    double flops = 2.0 * M * N * K * NUM_ITERATIONS;
    double FLOPS = flops / time_second;
    double TFLOPS = FLOPS / 1.0e12;

    std::cout << "[FlagPerf Result]" << "computation-BF16=" << TFLOPS << "TFLOPS" << std::endl;

    //
    // Verify
    //
    mutlass::reference::device::gett(
        problem_size,
        block_A.get(), stride_A,
        block_B.get(), stride_B,
        ElementAccumulator{},
        block_C.get(), stride_C,
        block_ref_D.get(), stride_D,
        alpha, beta);

    musa_err = musaDeviceSynchronize();
    if (musaSuccess != musa_err) {
        std::cerr << "ERROR: GEMM reference execution failed. with error :";
        std::cerr << musaGetErrorString(musa_err) << "\n";
        return 1;
    }

    // Compare
    bool passed = mutlass::reference::device::BlockCompareEqual(
        block_D.get(),
        block_ref_D.get(),
        block_D.size());

    if (passed) {
        std::cout << "MUTLASS GEMM verification passed.\n";
        return 0;
    }
    else {
        std::cerr << "ERROR: MUTLASS GEMM verification failed.\n";
        return 1;
    }
}


int main() {
    test();
    return 0;
}
