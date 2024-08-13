#include "parameters.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

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

    half *sm_tileA = sm;                               // tileA shape(BLOCK_TILE_Y BLOCK_TILE_K), size of tileA: BLOTK_TILE_Y * BLOCK_TILE_K (elements)
    half *sm_tileB = sm + BLOCK_TILE_Y * BLOCK_TILE_K; // tileB shape(BLOCK_TILE_K, BLOCK_TILE_X), size of tileB: BLOCK_TILE_K * BLOCK_TILE_X (elements)

    // one block computes BLOCK_TILE_X elements in x dim
    int globalIdxBlk_x = BLOCK_TILE_X * blockIdx.x;
    // one block computes BLOCK_TILE_Y elements in y dim
    int globalIdxBlk_y = BLOCK_TILE_Y * blockIdx.y;

    // threadIdx in block from sequential view
    int seqThreadIdx_blk = threadIdx.y * blockDim.y + threadIdx.x;
    int seqThreadIdx_warp = seqThreadIdx_blk % 32;
    int warpIdx = seqThreadIdx_blk / 32;

    int warpYIdx_c = warpIdx / (BLOCK_TILE_X / WARP_TILE_X);
    int warpXIdx_c = warpIdx % (BLOCK_TILE_X / WARP_TILE_X);

    int warpYOffset_c = warpYIdx_c * WARP_TILE_Y;
    int warpXOffset_c = warpXIdx_c * WARP_TILE_X;

    int warpYOffset_a = (warpIdx * BLOCK_TILE_Y / N_WARP_PER_BLOCK); // each warp is responsible for fetching BLOCK_TILE_Y / N_WARP_PER_BLOCK rows of data
    int warpXOffset_b = (warpIdx * BLOCK_TILE_X / N_WARP_PER_BLOCK); // each warp is responsible for fetching BLOCK_TILE_X / N_WARP_PER_BLOCK columns of data

    // each thread in a warp fetch 2 half data in a batch,
    // thus each row in sm_tileA takes BLOCK_TILE_K / 2 threads,
    int threadYOffset_a = seqThreadIdx_warp / (BLOCK_TILE_K / 2);
    int threadXOffset_a = seqThreadIdx_warp % (BLOCK_TILE_K / 2) * 2;
    // and each column in sm_tileB takes BLOCK_TILE_K / 2 threads
    int threadXOffset_b = seqThreadIdx_warp / (BLOCK_TILE_K / 2);
    int threadYOffset_b = seqThreadIdx_warp % (BLOCK_TILE_K / 2) * 2;

    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, half> frag_c[WARP_TILE_Y / FRAG_M][WARP_TILE_X / FRAG_N];
#pragma unroll
    for (int i_frag = 0; i_frag < WARP_TILE_Y / FRAG_M; i_frag++) {
#pragma unroll
        for (int j_frag = 0; j_frag < WARP_TILE_X / FRAG_N; j_frag++) {
            wmma::fill_fragment(frag_c[i_frag][j_frag], 0);
        }
    }

    for (int i = 0; i < k / BLOCK_TILE_K; i++) { // exact division assumed

        // for warp:
        // from gm_a[globalIdxBlk_y + warpYOffset_a (+ BLOCK_TILE_Y/N_WARP_PER_BLOCK)][i * BLOCK_TILE_K...i * BLOCK_TILE_K + BLOCK_TILE_K]
        // to sm_tileA[warpYOffset_a (+ BLOCK_TILE_Y/N_WARP_PER_BLOCK)][0...BLOCK_TILE_K]
        // for thread:
        // from gm_a[globalIdxBlk_y + warpYOffset_a + batchYOffset_a + threadYOffset_a][i * BLOCK_TILE_K + threadXOffset_a (+1)]
        // to sm_tileA[warpYOffset_a + batchYOffset_a + threadYOffset_a][threadXOffset_a (+1)]
#pragma unroll
        for (int i_batch = 0; i_batch < BLOCK_TILE_Y / N_WARP_PER_BLOCK * BLOCK_TILE_K / 64; i_batch++) { // each batch fetch 64 half data, which takes 64 / BLOCK_TILE_K rows in sm_tileA
            int batchYOffset_a = i_batch * (64 / BLOCK_TILE_K);
            FLOAT(sm_tileA + (warpYOffset_a + batchYOffset_a + threadYOffset_a) * BLOCK_TILE_K + threadXOffset_a)
            [0] = FLOAT(gm_a + (globalIdxBlk_y + warpYOffset_a + batchYOffset_a + threadYOffset_a) * k + i * BLOCK_TILE_K + threadXOffset_a)[0];
        }

        // for warp:
        // from gm_b[i * BLOCK_TILE_K...i * BLOCK_TILE_K + BLOCK_TILE_K][globalIdxBlk_x + warpXOffset_b (+ BLOCK_TILE_X/N_WARP_PER_BLOCK)]
        // to sm_tileB[0...BLOCK_TILE_K][warpXOffset_b (+ BLOCK_TILE_X/N_WARP_PER_BLOCK)]
        // for thread:
        // from gm_b[i * BLOCK_TILE_K + thraedYOffset_b (+1)][globalIdxBlk_x + warpXOffset_b + batchXOffset_b + threadXOffset_b]
        // to sm_tileB[threadYOffset_b (+1)][warpXOffset_b + batchXOffset_b + threadXOffset_b]
#pragma unroll
        for (int i_batch = 0; i_batch < BLOCK_TILE_X / N_WARP_PER_BLOCK * BLOCK_TILE_K / 64; i_batch++) { // each batch fetch 64 half data, which takes 64 / BLOCK_TILE_K cols in sm_tileB
            int batchXOffset_b = i_batch * (64 / BLOCK_TILE_K);
            FLOAT(sm_tileB + (warpXOffset_b + batchXOffset_b + threadXOffset_b) * BLOCK_TILE_K + threadYOffset_b)
            [0] = FLOAT(gm_b + (globalIdxBlk_x + warpXOffset_b + batchXOffset_b + threadXOffset_b) * k + i * BLOCK_TILE_K + threadYOffset_b)[0];
        }

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> frag_a[WARP_TILE_Y / FRAG_M];
        wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, half, wmma::col_major> frag_b[WARP_TILE_X / FRAG_N];

#pragma unroll
        for (int k_frag = 0; k_frag < BLOCK_TILE_K / FRAG_K; k_frag++) {
#pragma unroll
            for (int i_frag = 0; i_frag < WARP_TILE_Y / FRAG_M; i_frag++) {
                // from sm_tileA[warpYOffset_c + i_frag * FRAG_M (+FRAG_M)][k_frag * FRAG_K (+FRAG_K)]
                // to frag_a[i_frag]
                wmma::load_matrix_sync(frag_a[i_frag], sm_tileA + (warpYOffset_c + i_frag * FRAG_M) * BLOCK_TILE_K + k_frag * FRAG_K,
                                       BLOCK_TILE_K);
            }
#pragma unroll
            for (int j_frag = 0; j_frag < WARP_TILE_X / FRAG_N; j_frag++) {
                // from sm_tileB[k_frag * FRAG_K (+FRAG_K)][warpXOffset_c + j_frag * FRAG_N (+FRAG_N)]
                // to frag_b[j_frag]
                wmma::load_matrix_sync(frag_b[j_frag], sm_tileB + (warpXOffset_c + j_frag * FRAG_N) * BLOCK_TILE_K + k_frag * FRAG_K,
                                       BLOCK_TILE_K);
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
            wmma::store_matrix_sync(gm_c + (globalIdxBlk_y + warpYOffset_c + i_frag * FRAG_M) * n + globalIdxBlk_x + warpXOffset_c + j_frag * FRAG_N,
                                    frag_c[i_frag][j_frag], n, wmma::mem_row_major);
        }
    }
}