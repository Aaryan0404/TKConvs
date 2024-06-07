#define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#ifdef TORCH_COMPILE
#include "src/kittens.cuh"
#else
#include "../../../src/kittens.cuh"
#endif

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

using namespace kittens;

__global__ void gemm(int m, int n, int k, __half const* A, __half const* B, __half* C) {
    const int row = blockIdx.y * MMA_M;
    const int col = blockIdx.x * MMA_N;

    if (row >= m && col >= n) {
        return;
    }

    rt_hf_1x1<> accum;
    zero(accum);

    rt_hf_1x1<> a_mat;
    rt<__half2, 1, 1, kittens::ducks::rt_layout::col> b_mat;

    #pragma unroll
    for (auto i = 0; i < (k + MMA_K - 1) / MMA_K; i++) {

        load(a_mat, A + row * k + i * MMA_K, k);
        load(b_mat, B + i * MMA_K * n + col, n);

        mma_AB(accum, a_mat, b_mat, accum);
    }
    store(C + row * n + col, accum, n);
}

////////////////
// KERNEL END //
////////////////

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void gemm(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(c);

    TORCH_CHECK(a.scalar_type() == c10::ScalarType::Half, "A must be of type torch.float16");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Half, "B must be of type torch.float16");
    TORCH_CHECK(c.scalar_type() == c10::ScalarType::Half, "C must be of type torch.float16");

    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(1);

    c10::Half *a_ptr = a.data_ptr<c10::Half>();
    c10::Half *b_ptr = b.data_ptr<c10::Half>();
    c10::Half *c_ptr = c.data_ptr<c10::Half>();

    half* A = reinterpret_cast<half*>(a_ptr);
    half* B = reinterpret_cast<half*>(b_ptr);
    half* C = reinterpret_cast<half*>(c_ptr);

    const dim3 block_dim{32u};
    const dim3 grid_dim{(unsigned int)(n + MMA_N - 1) / MMA_N,
                        (unsigned int)(m + MMA_M - 1) / MMA_M};

    gemm<<<grid_dim, block_dim>>>(m, n, k, A, B, C);
}
#else
#include "harness.impl"
#endif