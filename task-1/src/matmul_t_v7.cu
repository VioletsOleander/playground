#include "parameters.h"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <mma.h>

#define FLOAT4(POINTER) (reinterpret_cast<float4 *>(POINTER))
#define UINT32(POINTER) (reinterpret_cast<uint32_t *>(POINTER))

using namespace nvcuda;
namespace cg = cooperative_groups;

__global__ void MatMulFP16(const int m, const int n, const int k,
                           half *a, half *b, half *c) {

    // size of matrix a: m * k (elements)
    // size of matrix b: k * n (elements)
    // size of matrix c: m * n (elements)

    half *gm_a = a;
    half *gm_b = b;
    half *gm_c = c;

    extern __shared__ half sm[];

    half *sm_tileA = sm;                                     // tileA shape (BLOCK_TILE_Y, BLOCK_TILE_K), size of tileA: BLOCK_TILE_Y * BLOCK_TILE_K (elements)
    half *sm_tileB = sm_tileA + BLOCK_TILE_K * BLOCK_TILE_Y; // tileB shape (BLOCK_TILE_K, BLOCK_TILE_X), size of tileB: BLOCK_TILE_K * BLOCK_TILE_X (elements)
    int bufferSize = BLOCK_TILE_K * (BLOCK_TILE_X + BLOCK_TILE_Y);

    // one block computes BLOCK_TILE_X elements in x dim
    int globalIdxBlk_x = BLOCK_TILE_X * blockIdx.x;
    // one block computes BLOCK_TILE_Y elements in y dim
    int globalIdxBlk_y = BLOCK_TILE_Y * blockIdx.y;

    // threadIdx in block from sequential view
    int seqThreadIdx_blk = threadIdx.x;
    // threadIdx in warp from sequential view
    int seqThreadIdx_warp = seqThreadIdx_blk % 32;
    // warp index in block, each block is assumed to have blockdim.x * blockdim.y / 32 = 256 / 32 = 8 warps
    int warpIdx = seqThreadIdx_blk / 32;

    int mappedThreadIdx_warp;
    if (seqThreadIdx_warp % 2 == 0) {
        mappedThreadIdx_warp = seqThreadIdx_warp / 2;
    } else {
        mappedThreadIdx_warp = seqThreadIdx_warp / 2 + 16;
    }

    // block tileC is divided into a grid of warp tiles,
    // the shape of gird is (BLOCK_TILE_X / WARP_TILE_X, BLOCK_TILE_Y / WARP_TILEY) = (2, 4)
    int warpTileYIdx_c = warpIdx / (BLOCK_TILE_X / WARP_TILE_X); // each row takes BLOCK_TILE_X / WARP_TILE_X warp tiles
    int warpTileXIdx_c = warpIdx % (BLOCK_TILE_X / WARP_TILE_X);
    int warpTileYOffset_c = warpTileYIdx_c * WARP_TILE_Y;
    int warpTileXOffset_c = warpTileXIdx_c * WARP_TILE_X;

    auto pipeline = cuda::make_pipeline();

    uint32_t c_array[WARP_TILE_Y / 16][WARP_TILE_X / 8][2] = {0}; // each mma.m16n8k16 needs a thread to store 4 half in 2 registers for multiplicand C

    for (int computeIdx = 0, fetchIdx = 0; computeIdx < k / BLOCK_TILE_K; computeIdx++) { // exact division assumed
        for (; fetchIdx < k / BLOCK_TILE_K && fetchIdx < computeIdx + N_PIPELINE_STAGE; fetchIdx++) {

            pipeline.producer_acquire();

            int bufferOffset = (fetchIdx % N_PIPELINE_STAGE) * bufferSize;

            // 16 x 16 fragment is used as data fetching unit for matrix A
            // each warp fetch (256 / 16 / 8) = 2 units, each thread in a warp fetch 8 half elements
            // for warp to fetch a unit:
            // gm_a[globalIdxBlk_y + warpIdx * 32 + unitIdx * 16 (+16)][fetchIdx * BLOCK_TILE_K (+16)] to
            // sm_tileA[warpIdx * 8 + unitIdx * 4 (+4)][0 (+64)]
            // for a thread in a warp:
            // gm_a[globalIdxBlk_y + warpIdx * 32 + unitIdx * 16 + seqThreadIdx_warp / 2][fetchIdx * BLOCK_TILE_K + seqThreadIdx_warp % 2 * 8 (+8)] to
            // sm_tileA[warpIdx * 8 + unitIdx * 4 + mappedThreadIdx_warp / 8][mappedThreadIdx_warp % 8 * 8 (+8)]
            // if seqThreadIdx_warp % 2 == 0
            //   mappedThreadIdx_warp = seqThreadIdx_warp / 2
            // else
            //   mappedThreadIdx_warp = seqThreadIdx_warp / 2 + 16
            for (int unitIdx = 0; unitIdx < BLOCK_TILE_Y / 16 / N_WARP_PER_BLOCK; unitIdx++) {
                // cuda::memcpy_async(FLOAT4(sm_tileA + bufferOffset + (warpIdx * 8 + unitIdx * 4 + mappedThreadIdx_warp / 8) * 64 + mappedThreadIdx_warp % 8 * 8),
                //    FLOAT4(gm_a + (globalIdxBlk_y + warpIdx * 32 + unitIdx * 16 + seqThreadIdx_warp / 2) * n + fetchIdx * BLOCK_TILE_K + seqThreadIdx_warp % 2 * 8),
                //    4 * sizeof(float), pipeline);
                FLOAT4(sm_tileA + bufferOffset + (warpIdx * 8 + unitIdx * 4 + mappedThreadIdx_warp / 8) * 64 + mappedThreadIdx_warp % 8 * 8)
                [0] = FLOAT4(gm_a + (globalIdxBlk_y + warpIdx * 32 + unitIdx * 16 + seqThreadIdx_warp / 2) * n + fetchIdx * BLOCK_TILE_K + seqThreadIdx_warp % 2 * 8)[0];
            }

            // 16 x 16 fragment is used as data fetching unit of matrix B
            // each warp fetch (128 / 16 / 8) = 1 units, each thread in a warp fetch 8 half elements
            // for warp:
            // gm_b[fetchIdx * BLOCK_TILE_K (+16)][globalIdxBlk_x + warpIdx * 16 (+16)] to
            // sm_tileB[warpIdx * 4 (+4)][0 (+64)]
            // for a thread in a warp:
            // gm_b[fetchIdx * BLOCK_TILE_K + seqThreadIdx_warp % 2 * 8 (+8)][globalIdxBlk_x + warpIdx * 16 + seqThreadIdx_warp / 2] to
            // sm_tileB[mappedThreadIdx_warp % 8 * 8 (+8)][warpIdx * 4 + mappedThreadIdx_warp / 8]
            {
                // cuda::memcpy_async(FLOAT4(sm_tileB + bufferOffset + (warpIdx * 4 + mappedThreadIdx_warp / 8) * 64 + mappedThreadIdx_warp % 8 * 8),
                //    FLOAT4(gm_b + (globalIdxBlk_x + warpIdx * 16 + seqThreadIdx_warp / 2) * k + fetchIdx * BLOCK_TILE_K + seqThreadIdx_warp % 2 * 8),
                //    4 * sizeof(float), pipeline);
                FLOAT4(sm_tileB + bufferOffset + (warpIdx * 4 + mappedThreadIdx_warp / 8) * 64 + mappedThreadIdx_warp % 8 * 8)
                [0] = FLOAT4(gm_b + (globalIdxBlk_x + warpIdx * 16 + seqThreadIdx_warp / 2) * k + fetchIdx * BLOCK_TILE_K + seqThreadIdx_warp % 2 * 8)[0];
            }

            pipeline.producer_commit();
        }

        pipeline.consumer_wait();

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        int bufferOffset = (computeIdx % N_PIPELINE_STAGE) * bufferSize;

        // a warp is responsible for a 64 x 64 tile,
        // which is divided into a (4 x 8) grid of (16 x 8) fragments
        // each fragment is calculated by a m16n8k16 mma operation,
        // which needs 8 elements in 4 regiseters for multiplicand A and 4 elements in 2 registers for multiplicand B
        uint32_t a_array[WARP_TILE_Y / 16][4];
        uint32_t b_array[WARP_TILE_X / 8][2];

        for (int i_frag = 0; i_frag < WARP_TILE_Y / 16; i_frag++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 "
                         "{%0, %1, %2, %3}, "
                         "[%4];\n\t"
                         : "=r"(a_array[i_frag][0]), "=r"(a_array[i_frag][1]), "=r"(a_array[i_frag][2]), "=r"(a_array[i_frag][3])
                         : "l"(sm_tileA + bufferOffset + (seqThreadIdx_warp / 8) * 64 + seqThreadIdx_warp % 8 * 8));
        }

        for (int j_frag = 0; j_frag < WARP_TILE_X / 8; j_frag += 2) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 "
                         "{%0, %1, %2, %3}, "
                         "[%4];\n\t"
                         : "=r"(b_array[j_frag][0]), "=r"(b_array[j_frag + 1][0]), "=r"(b_array[j_frag][1]), "=r"(b_array[j_frag + 1][1])
                         : "l"(sm_tileB + bufferOffset + (seqThreadIdx_warp / 8) * 64 + seqThreadIdx_warp % 8 * 8));
        }

        for (int i_frag = 0; i_frag < WARP_TILE_Y / 16; i_frag++) {
            for (int j_frag = 0; j_frag < WARP_TILE_X / 8; j_frag++) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                             "{%0, %1}, "
                             "{%2, %3, %4, %5}, "
                             "{%6, %7}, "
                             "{%8, %9};\n\t"
                             : "=r"(c_array[i_frag][j_frag][0]), "=r"(c_array[i_frag][j_frag][1])
                             : "r"(a_array[i_frag][0]), "r"(a_array[i_frag][1]), "r"(a_array[i_frag][2]), "r"(a_array[i_frag][3]),
                               "r"(b_array[j_frag][0]), "r"(b_array[j_frag][1]),
                               "r"(c_array[i_frag][j_frag][0]), "r"(c_array[i_frag][j_frag][1]));
            }
        }

        pipeline.consumer_release();
    }

    for (int i_frag = 0; i_frag < WARP_TILE_Y / 16; i_frag++) {
        for (int j_frag = 0; j_frag < WARP_TILE_X / 8; j_frag++) {
            // for warp:
            // frag_c to
            // gm_c[globalIdxBlk_y + warpTileYIdx_c + i_frag * 16 (+16)][globalIdxBlk_x + warpXOffset_c + j_frag * 8 (+8)]
            // for thread:
            // c_array[i_frag][j_frag][0] to
            // gm_c[globalIdxBlk_y + warpTileYIdx_c + i_frag * 16 + seqThreadIdx_warp / 4][globalIdxBlk_x + warpXOffset_c + j_frag * 8 + seqThreadIdx_warp % 4 * 2 (+2)]
            // c_array[i_frag][j_frag][1] to
            // gm_c[globalIdxBlk_y + warpTileYIdx_c + i_frag * 16 + seqThreadIdx_warp / 4 + 8][globalIdxBlk_x + warpXOffset_c + j_frag * 8 + seqThreadIdx_warp % 4 * 2 (+2)]
            UINT32(gm_c + (globalIdxBlk_y + warpTileYOffset_c + i_frag * 16 + seqThreadIdx_warp / 4) * n + globalIdxBlk_x + warpTileXOffset_c + j_frag * 8 + seqThreadIdx_warp % 4 * 2)
            [0] = c_array[i_frag][j_frag][0];
            UINT32(gm_c + (globalIdxBlk_y + warpTileYOffset_c + i_frag * 16 + seqThreadIdx_warp / 4 + 8) * n + globalIdxBlk_x + warpTileXOffset_c + j_frag * 8 + seqThreadIdx_warp % 4 * 2)
            [0] = c_array[i_frag][j_frag][1];
        }
    }
}