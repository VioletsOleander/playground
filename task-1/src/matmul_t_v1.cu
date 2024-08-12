#include "parameters.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

#define FRAG_M 16
#define FRAG_N 16
#define FRAG_K 16
#define FLOAT(POINTER) (reinterpret_cast<float *>(POINTER))

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

    int smYStride_a = BLOCK_TILE_Y + FRAG_M;
    int smXStride_b = BLOCK_TILE_X + FRAG_N;

    half *sm_tileA = sm;                              // tileA shape(smYStride_a, BLOCK_TILE_K), size of tileA: (smYStride_a) * BLOCK_TILE_K (elements)
    half *sm_tileB = sm + smYStride_a * BLOCK_TILE_K; // tileB shape(BLOCK_TILE_K, smXStride_b), size of tileB: BLOCK_TILE_K * (smXStride_b) (elements)

    // one block computes BLOCK_TILE_X elements in x dim
    int globalIdxBlk_x = BLOCK_TILE_X * blockIdx.x;
    // one block computes BLOCK_TILE_Y elements in y dim
    int globalIdxBlk_y = BLOCK_TILE_Y * blockIdx.y;

    // threadIdx in block from sequential view
    int seqThreadIdx_blk = threadIdx.y * blockDim.y + threadIdx.x;
    int seqThreadIdx_warp = seqThreadIdx_blk % 32;
    int warpIdx = seqThreadIdx_blk / 32;

    int warpIdx_y = warpIdx / (BLOCK_TILE_X / WARP_TILE_X);
    int warpIdx_x = warpIdx % (BLOCK_TILE_X / WARP_TILE_X);

    int warpOffset_y = warpIdx_y * WARP_TILE_Y;
    int warpOffset_x = warpIdx_x * WARP_TILE_X;

    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, half> frag_c[WARP_TILE_Y / FRAG_M][WARP_TILE_X / FRAG_N];
#pragma unroll
    for (int i_frag = 0; i_frag < WARP_TILE_Y / FRAG_M; i_frag++) {
#pragma unroll
        for (int j_frag = 0; j_frag < WARP_TILE_X / FRAG_N; j_frag++) {
            wmma::fill_fragment(frag_c[i_frag][j_frag], 0);
        }
    }

    for (int i = 0; i < k / BLOCK_TILE_K; i++) { // exact division assumed

        // each thread fetch 2 continuous elements, each warp fetch 64 continuous elements
        // from gm_a[globalIdxBlk_y + (seqThreadIdx_warp + WARP_SIZE * i_warp) * 2][i * BLOCK_TILE_K + warpIdx + i_batch * N_WARP_PER_BLOCK]
        // to sm_tileA[(seqThreadIdx_warp  + WARP_SIZE * i_warp) * 2][warpIdx + i_batch * N_WARP_PER_BLOCK],
        // i_warp in [0, BLOCK_TILE_Y / WARP_SIZE / 2]
#pragma unroll
        for (int i_batch = 0; i_batch < BLOCK_TILE_K / N_WARP_PER_BLOCK; i_batch++) {
#pragma unroll
            for (int i_warp = 0; i_warp < BLOCK_TILE_Y / WARP_SIZE / 2; i_warp++) {
                FLOAT(sm_tileA + (warpIdx + i_batch * N_WARP_PER_BLOCK) * smYStride_a + (seqThreadIdx_warp + WARP_SIZE * i_warp) * 2)
                [0] = FLOAT(gm_a + (i * BLOCK_TILE_K + warpIdx + i_batch * N_WARP_PER_BLOCK) * k + globalIdxBlk_y + (seqThreadIdx_warp + WARP_SIZE * i_warp) * 2)[0];
            }
        }

        // fetch gm_b[i * BLOCK_TILE_K + warpIdx][globalIdxBlk_x + (seqThreadIdx_warp + WARP_SIZE * i_warp) * 2]
        // to sm_tileB[warpIdx + i_batch * N_WARP_PER_BLOCK][(seqThreadIdx + WARP_SIZE * i_warp) * 2],
        // i_warp in [0, BLOCK_TILE_X / WARP_SIZE / 2]
#pragma unroll
        for (int i_batch = 0; i_batch < BLOCK_TILE_K / N_WARP_PER_BLOCK; i_batch++) {
#pragma unroll
            for (int i_warp = 0; i_warp < BLOCK_TILE_X / WARP_SIZE / 2; i_warp++) {
                FLOAT(sm_tileB + (warpIdx + i_batch * N_WARP_PER_BLOCK) * smXStride_b + (seqThreadIdx_warp + WARP_SIZE * i_warp) * 2)
                [0] = FLOAT(gm_b + (i * BLOCK_TILE_K + warpIdx + i_batch * N_WARP_PER_BLOCK) * n + globalIdxBlk_x + (seqThreadIdx_warp + WARP_SIZE * i_warp) * 2)[0];
            }
        }

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::col_major> frag_a[WARP_TILE_Y / FRAG_M];
        wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> frag_b[WARP_TILE_X / FRAG_N];

#pragma unroll
        for (int k_frag = 0; k_frag < BLOCK_TILE_K / FRAG_K; k_frag++) {
#pragma unroll
            for (int i_frag = 0; i_frag < WARP_TILE_Y / FRAG_M; i_frag++) {
                // sm_tileA[warpOffset_y + i_frag * FRAG_M (+FRAG_M)][k_frag * FRAG_K (+FRAG_N)]
                // to frag_a[i_frag]
                wmma::load_matrix_sync(frag_a[i_frag], sm_tileA + (k_frag * FRAG_K) * smYStride_a + warpOffset_y + i_frag * FRAG_M,
                                       smYStride_a);
            }
#pragma unroll
            for (int j_frag = 0; j_frag < WARP_TILE_X / FRAG_N; j_frag++) {
                // sm_tileB[k_frag * FRAG_K (+FRAG_M)][warpOffset_x + j_frag * FRAG_N (+FRAG_N)]
                wmma::load_matrix_sync(frag_b[j_frag], sm_tileB + (k_frag * FRAG_K) * smXStride_b + warpOffset_x + j_frag * FRAG_N,
                                       smXStride_b);
            }

#pragma unroll
            for (int i_frag = 0; i_frag < WARP_TILE_Y / FRAG_M; i_frag++) {
#pragma unroll
                for (int j_frag = 0; j_frag < WARP_TILE_X / FRAG_N; j_frag++) {
                    wmma::mma_sync(frag_c[i_frag][j_frag], frag_a[i_frag], frag_b[j_frag], frag_c[i_frag][j_frag]);
                }
            }
        }

        __syncthreads(); // sync to make sure every thread finished computation so sm_tileA and sm_tileB are allowed to be overwirtten
    }

#pragma unroll
    for (int i_frag = 0; i_frag < WARP_TILE_Y / FRAG_M; i_frag++) {
#pragma unroll
        for (int j_frag = 0; j_frag < WARP_TILE_X / FRAG_N; j_frag++) {
            // frag_c[i_frag][j_frag] to
            // gm_c[globalIdxBlk_y + warpOffset_y + i_frag * FRAG_M (+FRAG_M)][globalIdxBlk_x + warpOffset_x + j_frag * FRAG_N (+FRAG_N)]
            wmma::store_matrix_sync(gm_c + (globalIdxBlk_y + warpOffset_y + i_frag * FRAG_M) * n + globalIdxBlk_x + warpOffset_x + j_frag * FRAG_N,
                                    frag_c[i_frag][j_frag], n, wmma::mem_row_major);
        }
    }
}