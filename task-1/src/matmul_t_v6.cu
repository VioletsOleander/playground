#include "parameters.h"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <mma.h>

#define FLOAT4(POINTER) (reinterpret_cast<float4 *>(POINTER))
#define UINT32(POINTER) (reinterpret_cast<uint32_t *>(POINTER))

using namespace nvcuda;

__global__ void MatMulFP16(const int m, const int n, const int k,
                           half *a, half *b, half *c) {

    // size of matrix a: m * k (elements)
    // size of matrix b: k * n (elements)
    // size of matrix c: m * n (elements)

    half *gm_a = a;
    half *gm_b = b;
    half *gm_c = c;

    extern __shared__ half sm[];

    half *sm_tileA = sm;                               // tileA shape (BLOCK_TILE_Y, BLOCK_TILE_K), size of tileA: BLOCK_TILE_Y * BLOCK_TILE_K (elements)
    half *sm_tileB = sm + BLOCK_TILE_K * BLOCK_TILE_Y; // tileB shape (BLOCK_TILE_K, BLOCK_TILE_X), size of tileB: BLOCK_TILE_K * BLOCK_TILE_X (elements)
    int bufferSize = BLOCK_TILE_K * (BLOCK_TILE_X + BLOCK_TILE_Y);

    int globalIdxBlk_x = BLOCK_TILE_X * blockIdx.x; // one block computes BLOCK_TILE_X elements in x dim
    int globalIdxBlk_y = BLOCK_TILE_Y * blockIdx.y; // one block computes BLOCK_TILE_Y elements in y dim

    int seqThreadIdx_warp = threadIdx.x % 32; // threadIdx in warp from sequential view
    int warpIdx = threadIdx.x / 32;           // warp index in block, each block is assumed to have blockdim.x * blockdim.y / 32 = 256 / 32 = 8 warps

    // block tileC is divided into a grid of warp tiles,
    // the shape of gird is (BLOCK_TILE_Y / WARP_TILE_Y, BLOCK_TILE_X / WARP_TILE_X) = (4, 2)
    int warpTileYIdx_c = warpIdx / (BLOCK_TILE_X / WARP_TILE_X); // each row takes BLOCK_TILE_X / WARP_TILE_X = 2 warp tiles
    int warpTileXIdx_c = warpIdx % (BLOCK_TILE_X / WARP_TILE_X);
    int warpTileYOffset_c = warpTileYIdx_c * WARP_TILE_Y;
    int warpTileXOffset_c = warpTileXIdx_c * WARP_TILE_X;

    int rowBase = seqThreadIdx_warp / 8;
    int colBase = seqThreadIdx_warp % 8;

    auto pipeline = cuda::make_pipeline();

    uint32_t c_array[WARP_TILE_Y / FRAG_M][WARP_TILE_X / FRAG_N][2] = {0}; // each mma.m16n8k16 needs a thread to store 4 half in 2 registers for multiplicand C

    for (int computeIdx = 0, fetchIdx = 0; computeIdx < k / BLOCK_TILE_K; computeIdx++) { // exact division assumed
        for (; fetchIdx < k / BLOCK_TILE_K && fetchIdx < computeIdx + N_PIPELINE_STAGE; fetchIdx++) {

            pipeline.producer_acquire();

            int bufferOffset = (fetchIdx % N_PIPELINE_STAGE) * bufferSize;

            // the gm_a is cloumn major, whose shape is (m, k)
            // each warp fetch a (32, 8) chunk of half, which can also be viewd as (4, 8) chunk of half sector
            // the 4 x 8 half sector chunk layout is further viewd as 8 x 4 by merging consectuive two cols into one col
            // the sm_tileA is row major, whose shape is set to (BLOCK_TILE_Y * BLOCK_TILE_K / 64 , 64)
            // before the 8 x 4 chunk of half sector is stored into sm_tileA, in each of its row ( row 0 except ),
            // the half sectors will recalculate its col index with col_idx_new = row_idx ^ col_idx,
            // the whole chunk of half sector the warp fetches will take up a (4, 64) place in sm_tileA
            // the warp will fetch BLOCK_TILE_K / 8 = 2 such chunks to take up a (BLOCK_TILE_K / 2 = 8, 64) place in total
            // above is the data fetching process for a warp for tileA

            // the code following assumed N_WARP_PER_BLOCK = 8 and BLOCK_TILE_Y = 256, therefore BLOCK_TILE_Y / N_WARP_PER_BLOCK = 32
            // for warp, it fetches a chunk in a batch:
            // the warp fetches gm_a[globalIdxBlk_y + warpIdx * 32 (+32)][fetchIdx * BLOCK_TILE_K + i_batch * 8 (+8)]
            //               to sm_tileA[warpIdx * 8 + i_batch * 4 (+4)][0...64]
            // for thread, it fetches a half sector in a batch:
            // the thread fetches gm_a[globalIdxBlk_y + warpIdx * 32 + seqThreadIdx_warp % 4 * 8 (+8)][fetchIdx * BLOCK_TILE_K + i_batch * 8 + seqThreadIdx_warp / 4]
            // let row = warpIdx * 8 + i_batch * 4 + rowBase, col = colBase
            //               originally, to sm_tileA[row][col * 8 (+8)]
            //               after swizzle, to sm_tileA[row][(colBase ^ rowBase) * 8 (+8)]
            for (int i_batch = 0; i_batch < BLOCK_TILE_K / 8; i_batch++) {
                int col = colBase ^ rowBase;
                int row = rowBase + warpIdx * 8 + i_batch * 4;
                cuda::memcpy_async(FLOAT4(sm_tileA + bufferOffset + row * 64 + col * 8),
                                   FLOAT4(gm_a + (fetchIdx * BLOCK_TILE_K + i_batch * 8 + seqThreadIdx_warp / 4) * k + globalIdxBlk_y + warpIdx * 32 + seqThreadIdx_warp % 4 * 8),
                                   4 * sizeof(float), pipeline);
            }

            // the gm_b is row major, whose shape is (k, n)
            // each thread fetch a half sector, whose size is 128bit = 8 half
            // each warp fetch a (8, 32) chunk of half, which can also be view as (8, 4) chunk of half sector
            // the 8 x 4 half sector chunk layout is further viewed as 4 x 8 by merging consectuive two rows into one row
            // the sm_tileB is row major, whose shape is set to (BLOCK_TILE_X * BLOCK_TILE_K / 64, 64)
            // before the 4 x 8 chunk of half sector is stored into sm_tileB, in each of its row (row 0 except),
            // the half sectors will recalculate its col index with col_idx_new = row_idx ^ col_idx,
            // the whole chunk of half sector the warp fetches will take up a (4, 64) place in sm_tileB
            // above is the data fetching process for a warp for tileB

            // the code following assumed N_WARP_PER_BLOCK = 8 and BLOCK_TILE_X = 128 and BLOCK_TILE_K = 16, therefore N_WARP_PER_BLOCK * 256 = BLOCK_TILE_X * BLOCK_TILE_K
            // for warp, it fetches a chunk:
            // the warp fetches gm_b[fetchIdx * BLOCK_TILE_K + warpIdx / 4 * 8 (+8)][globalIdxBlk_x + warpIdx % 4 * 32 (+32)]
            //               to sm_tileB[(warpIdx % 4) * 8 + warpIdx / 4 * 4 (+4)][0...64]
            // for thread, it fetches a half sector:
            // the thread fetches gm_b[fetchIdx * BLOCK_TILE_K + warpIdx / 4 * 8 + seqThreadIdx_warp / 4][globalIdxBlk_x + warpIdx % 4 * 32 + seqThreadIdx_warp % 4 * 8 (+8)]
            // let row = warpIdx * 4 + rowBase, col = colBase
            //             originally, to sm_tileB[row][col * 8 (+8)]
            //             after swizzle, to sm_tileB[row][(colBase ^ rowBase) * 8 (+8)]
            {
                int col = colBase ^ rowBase;
                int row = rowBase + (warpIdx % 4) * 8 + warpIdx / 4 * 4;
                cuda::memcpy_async(FLOAT4(sm_tileB + bufferOffset + row * 64 + col * 8),
                                   FLOAT4(gm_b + (fetchIdx * BLOCK_TILE_K + warpIdx / 4 * 8 + seqThreadIdx_warp / 4) * n + globalIdxBlk_x + warpIdx % 4 * 32 + seqThreadIdx_warp % 4 * 8),
                                   4 * sizeof(float), pipeline);
            }

            pipeline.producer_commit();
        }

        pipeline.consumer_wait();

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        int bufferOffset = (computeIdx % N_PIPELINE_STAGE) * bufferSize;

        // a warp is responsible a (64 x 64) tile, which is divided into 16 (16 x 16) sub tile,
        // which is calculated from 2 (16, 16) @ (16, 8) matrix multiplication ( mma.m16n8k16 )
        // therefore, a warp needs to issue 2 * 16 = 32 mma.m16n8k16

        // each mma.m16n8k16 needs each thread stores 8 half in 4 registers for multiplicand A
        // a_array is responsible for storing WARPT_TILE_Y / 16 = 4 16 x 16 fragment's data
        uint32_t a_array[4][4];
        // each mma.m16n8k16 needs each thread stores 4 half in 2 registers for multiplicand B
        // b_array is responsible fro storing WARP_TILE_X / 8 = 2 16 x 8 fragments's data
        uint32_t b_array[8][2];

        // fetch a 16x16 fragment from sm_tileA to registers
        // note that two 16 x 16 fragments are stored in the same 8 x 64 chunk in shared memory,
        // and the location of their half sectors are interleaved
        for (int i_frag = 0; i_frag < WARP_TILE_Y / 16; i_frag++) {
            // calculate in-chunk index
            int row = seqThreadIdx_warp % 16 / 2;                                 // every two thread takes over a row
            int col = seqThreadIdx_warp % 16 % 8 / 2 + seqThreadIdx_warp % 2 * 4; // every two thread takes over a col

            // for threads whose in warp idx is in 16-31
            if (seqThreadIdx_warp > 15) {
                if (row % 2 == 0) {
                    col += 1;
                } else {
                    col -= 1;
                }
            }

            // for the second fragment in the chunk
            if (i_frag % 2 != 0) {
                if (seqThreadIdx_warp % 16 / 4 % 2 == 0) {
                    col += 2;
                } else {
                    col -= 2;
                }
            }

            int chunkYOffset_a = warpTileYIdx_c * 16 + i_frag / 2 * 8;

            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 "
                         "{%0, %1, %2, %3}, "
                         "[%4];\n\t"
                         : "=r"(a_array[i_frag][0]), "=r"(a_array[i_frag][1]), "=r"(a_array[i_frag][2]), "=r"(a_array[i_frag][3])
                         : "l"(sm_tileA + bufferOffset + (chunkYOffset_a + row) * 64 + col * 8));

            // for (int _i = 0; _i < 4; _i++) {
            //     asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(a_array[i_frag][_i]) : "r"(a_array[i_frag][_i]));
            // }
        }

        // fetch 2 16x8 matrix (1 16x16 matrix) from sm_tileB to registers
        // note that 2 16 x 16 (4 16x8 fragments) fragments are stored in the same 8 x 64 chunk in shared memory,
        // and the location of their half sectors are interleaved
        for (int j_frag = 0; j_frag < WARP_TILE_X / 8; j_frag += 2) {
            // calculate in-chunk index
            int row = seqThreadIdx_warp % 16 / 2;                                 // every two thread takes over a row
            int col = seqThreadIdx_warp % 16 % 8 / 2 + seqThreadIdx_warp % 2 * 4; // every two thread takes over a col

            // for threads whose in warp idx is in 16-31
            if (seqThreadIdx_warp > 15) {
                if (row % 2 == 0) {
                    col += 1;
                } else {
                    col -= 1;
                }
            }

            if (j_frag == 2 || j_frag == 6) {
                if (seqThreadIdx_warp % 16 / 4 % 2 == 0) {
                    col += 2;
                } else {
                    col -= 2;
                }
            }

            int chunkYOffset_b = warpTileXIdx_c * 16 + j_frag / 4 * 8;

            asm volatile("ldmatrix.sync.aligned.m8n8.trans.x4.b16 "
                         "{%0, %1, %2, %3}, "
                         "[%4];\n\t"
                         : "=r"(b_array[j_frag][0]), "=r"(b_array[j_frag][1]), "=r"(b_array[j_frag + 1][0]), "=r"(b_array[j_frag + 1][1])
                         : "l"(sm_tileB + bufferOffset + (chunkYOffset_b + row) * 64 + col * 8));
            for (int _i = 0; _i < 2; _i++) {
                asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(b_array[j_frag][_i]) : "r"(b_array[j_frag][_i]));
                asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(b_array[j_frag + 1][_i]) : "r"(b_array[j_frag + 1][_i]));
            }
        }

        pipeline.consumer_release();

        for (int i_frag = 0; i_frag < WARP_TILE_Y / 16; i_frag++) {
            for (int j_frag = 0; j_frag < WARP_TILE_X / 8; j_frag++) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                             "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n\t"
                             : "=r"(c_array[i_frag][j_frag][0]), "=r"(c_array[i_frag][j_frag][1])
                             : "r"(a_array[i_frag][0]), "r"(a_array[i_frag][1]), "r"(a_array[i_frag][2]), "r"(a_array[i_frag][3]),
                               "r"(b_array[j_frag][0]), "r"(b_array[j_frag][1]), "r"(c_array[i_frag][j_frag][0]), "r"(c_array[i_frag][j_frag][1]));
            }
        }
    }

    for (int i_frag = 0; i_frag < WARP_TILE_Y / 16; i_frag++) {
        for (int j_frag = 0; j_frag < WARP_TILE_X / 8; j_frag += 2) {
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