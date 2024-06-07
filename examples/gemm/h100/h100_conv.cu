#include "../../../src/kittens.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>

//////////////////
// KERNEL START //
//////////////////

#define MMA_N 128
#define MMA_K 128

#define NUM_WORKERS    (16)
#define NUM_WARPGROUPS (NUM_WORKERS/4)

#define MMA_M (64*NUM_WARPGROUPS)

using namespace kittens;

#define a_smem_tile kittens::st_bf<4, 8, wgmma_swizzle_l>
#define b_smem_tile kittens::st_bf<8, 8, wgmma_swizzle_l>
#define c_smem_tile kittens::st_bf<4, 8, wgmma_swizzle_l>

__global__ __launch_bounds__(NUM_WARPGROUPS*kittens::WARPGROUP_THREADS,1)
void simple_gemm(int m, int n, int k, CUtensorMap* A, CUtensorMap* B, CUtensorMap* C) {
    
    const int block_row = blockIdx.y * NUM_WARPGROUPS;
    const int block_col = blockIdx.x;

    const int K_tiles   = k / MMA_K;
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    int tic = 0, toc = 1;

    a_smem_tile (&a_smem)[2][NUM_WARPGROUPS] = al.allocate<a_smem_tile, 2, NUM_WARPGROUPS>();
    b_smem_tile (&b_smem)[2]                 = al.allocate<b_smem_tile, 2>();
    
    rt_fl_1x8<> ab_reg;
    zero(ab_reg);

    __shared__ uint64_t smem_barrier; 
    const int wg = warpid() / 4;

    if (warpid() == 0) {
        tma::init_barrier(smem_barrier);
        tma::set_bytes(smem_barrier, size_bytes<typeof(a_smem[0][0]), NUM_WARPGROUPS> + size_bytes<typeof(b_smem[0])>);

        for(int wg = 0; wg < NUM_WARPGROUPS; wg++) {
            tma::load_async(a_smem[tic][wg], A, smem_barrier, block_row+wg, 0);
        }
        tma::load_async(b_smem[tic], B, smem_barrier, block_col, 0);
    }
    __syncthreads();

    for (int i = 0; i < K_tiles; i++, tic^=1, toc^=1) {
        tma::arrive_and_wait(smem_barrier, tic);

        if(warpid() == 0) {
            if (i + 1 < K_tiles) {
                tma::set_bytes(smem_barrier, size_bytes<typeof(a_smem[0][0]), NUM_WARPGROUPS> + size_bytes<typeof(b_smem[0])>);
                
                for(int wg = 0; wg < NUM_WARPGROUPS; wg++) {
                    tma::load_async(a_smem[toc][wg], A, smem_barrier, block_row+wg, i+1);
                }

                tma::load_async(b_smem[toc], B, smem_barrier, block_col, i+1);
            }
        }
        __syncwarp(); 

        warpgroup::mma_fence(ab_reg);
        warpgroup::mma_ABt(ab_reg, a_smem[tic][wg], b_smem[tic]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();
    }
    __syncthreads();
    warpgroup::store(a_smem[0][wg], ab_reg);
    __syncthreads();

    if (warpid() % 4 == 0) {
        tma::store_async(C, a_smem[0][wg], block_row + wg, block_col);
        tma::store_commit_group();
    }
    tma::store_async_wait();
}
////////////////
// KERNEL END //
////////////////

#include "harness_conv.impl"
