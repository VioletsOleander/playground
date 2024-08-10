#include "parameters.h"
#define FLOAT4(POINTER) (reinterpret_cast<float4 *>(POINTER))

__global__ void MatMulFP32(const int m, const int n, const int k,
                           float *__restrict__ a, float *__restrict__ b, float *__restrict__ c) {

    // size of matrix a: m * k (elements)
    // size of matrix b: k * n (elements)
    // size of matrix c: m * n (elements)

    float *gm_a = a;
    float *gm_b = b;
    float *gm_c = c;

    extern __shared__ float sm[];

    float *sm_tileA = sm;                                             // size of tileA: blockDim.y * THREAD_TILE_Y * BLOCK_TILE_K (elements)
    float *sm_tileB = sm + blockDim.y * THREAD_TILE_Y * BLOCK_TILE_K; // size of tileB: BLOCK_TILE_K * THRAD_TILE_X * blockDim.x (elements)

    // one block computes blockDim.x * THREAD_TILE_X elements in x dim
    int globalIdxBlk_x = blockDim.x * THREAD_TILE_X * blockIdx.x;
    // one block computes blockDim.y * THREAD_TILE_Y elements in y dim
    int globalIdxBlk_y = blockDim.y * THREAD_TILE_Y * blockIdx.y;

    // threadIdx in block from sequential view
    int seqThreadIdx_blk = threadIdx.y * blockDim.y + threadIdx.x;
    int seqThreadIdx_warp = seqThreadIdx_blk % 32;
    int warpIdx = seqThreadIdx_blk / 32;

    float vals[THREAD_TILE_X][THREAD_TILE_X] = {};
    for (int i = 0; i < k / BLOCK_TILE_K; i++) { // exact division assumed

        // fetch gm_a[globalIdxBlk_y + seqThreadIdx_warp + WARP_SIZE * i_warp][i * BLOCK_TILE_K + warpIdx]
        // to sm_tileA[seqThreadIdx_warp + WARP_SIZE * i_warp][warpIdx],  i_warp in [0, BLOCK_TILE_X / WARP_SIZE]
#pragma unroll
        for (int i_batch = 0; i_batch < 2; i_batch++) {
#pragma unroll
            for (int i_warp = 0; i_warp < 4; i_warp++) {
                sm_tileA[(warpIdx + 8 * i_batch) * BLOCK_TILE_Y + seqThreadIdx_warp + WARP_SIZE * i_warp] =
                    gm_a[(i * BLOCK_TILE_K + warpIdx + 8 * i_batch) * k + globalIdxBlk_y + seqThreadIdx_warp + WARP_SIZE * i_warp];
            }
        }

        // fetch gm_b[i * BLOCK_TILE_K + warpIdx][globalIdxBlk_x + seqThreadIdx_warp + WARP_SIZE * i_warp]
        // to sm_tileB[warpIdx][seqThreadIdx + WARP_SIZE * i_warp], i_warp in [0, BLOCK_TILE_X / WARP_SIZE]
#pragma unroll
        for (int i_batch = 0; i_batch < 2; i_batch++) {
#pragma unroll
            for (int i_warp = 0; i_warp < 4; i_warp++) {
                sm_tileB[(warpIdx + 8 * i_batch) * BLOCK_TILE_X + seqThreadIdx_warp + WARP_SIZE * i_warp] =
                    gm_b[(i * BLOCK_TILE_K + (warpIdx + 8 * i_batch)) * n + globalIdxBlk_x + seqThreadIdx_warp + WARP_SIZE * i_warp];
            }
        }

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        float bufferYDim_a[THREAD_TILE_Y];
        float bufferXDim_b[THREAD_TILE_X];
#pragma unroll
        for (int _k = 0; _k < BLOCK_TILE_K; _k++) {

#pragma unroll
            for (int _i = 0, _j = 0; _i < THREAD_TILE_Y && _j < THREAD_TILE_X; _i++, _j++) {
                // read sm_tileA[threadIdx.y * THREAD_TILE_Y + _i][_k]
                bufferYDim_a[_i] = sm_tileA[_k * BLOCK_TILE_Y + threadIdx.y * THREAD_TILE_Y + _i];
                // read sm_tileB[_k][threadIdx.x * THREAD_TILE_X + _j]
                bufferXDim_b[_j] = sm_tileB[_k * BLOCK_TILE_X + threadIdx.x * THREAD_TILE_X + _j];
            }

#pragma unroll
            for (int _i = 0; _i < THREAD_TILE_Y; _i++) {
#pragma unroll
                for (int _j = 0; _j < THREAD_TILE_X; _j++) {
                    vals[_i][_j] += bufferYDim_a[_i] * bufferXDim_b[_j];
                }
            }
        } // the loop takes (BLOCK_TILE_K * (THREAD_TILE_X + THREAD_TILE_Y)) shared memory read in total

        __syncthreads(); // sync to make sure every thread finished computation so sm_tileA and sm_tileB are allowed to be overwirtten
    } // the loop takes (2 * k/BLOCK_TILE_K) global memory read in total,
    // and takes (k * (THREAD_TILE_X + THREAD_TILE_Y)) shared memory read in total

    // be responsible for computing the (globalIdx_y (+8), globalIdx_x (+8)) element in matrix c
    // one thread comuptes THREAD_TILE_X elements in x dim
    int globalIdx_x = globalIdxBlk_x + THREAD_TILE_X * threadIdx.x;
    // one thread comuptes THREAD_TILE_Y elements in y dim
    int globalIdx_y = globalIdxBlk_y + THREAD_TILE_Y * threadIdx.y;

    // wirte global memory THREAD_TILE_X * THRAD_TILE_Y / 4 = 16 times
#pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++) {
        // write vals[i][0 (+THREAD_TILE_X)] to gm_c[globalIdx_y + i][globalIdx_x (+THREAD_TILE_X)]
        FLOAT4(gm_c + (globalIdx_y + i) * n + globalIdx_x)
        [0] = FLOAT4(&vals[i][0])[0];
        FLOAT4(gm_c + (globalIdx_y + i) * n + globalIdx_x + 4)
        [0] = FLOAT4(&vals[i][4])[0];
    }
    // computation for THREAD_TILE_X * THREAD_TILE_Y output element takes
    // ( 2 * k/BLOCK_TILE_K + THREAD_TILE_X * THREAD_TILE_Y / 4 ) global memory access,
    // and takes ( k * (THREAD_TILE_X + THREAD_TILE_Y) + 2 * k/BLOCK_TILE_K ) shared memory access
    // on average, computation for 1 output element takes
    // ( 2 * k/BLOCK_TILE_K + THREAD_TILE_X * THREAD_TILE_Y / 4 ) / (THREAD_TILE_X * THREAD_TILE_Y) global memory access,
    // and takes ( k * (THREAD_TILE_X + THREAD_TILE_Y) + 2 * k/BLOCK_TILE_K ) / (THREAD_TILE_X * THREAD_TILE_Y) shared memory access
}