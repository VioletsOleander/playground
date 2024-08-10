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
    int seqThreadIdx = threadIdx.x * blockDim.x + threadIdx.y;
    int smYIdx_a = seqThreadIdx / 2; // 8 / 4 = 2 threads each row
    int smXIdx_a = (seqThreadIdx % 2) * 4;
    int smYIdx_b = seqThreadIdx / 32; // 128 / 4 = 32 thread each row
    int smXIdx_b = (seqThreadIdx % 32) * 4;

    float vals[THREAD_TILE_X][THREAD_TILE_X] = {};
    for (int i = 0; i < k / BLOCK_TILE_K; i++) { // exact division assumed
        // be responsible for fetching four element each for tileA and tileB from gm to sm,
        // read global memory 2 times, write shared memory 2 times

        // fetch gm_a[(globalIdxBlk_y + smYIdx_a) * n][i * BLOCK_TILE_K + smXIdx_a (+4)] to sm_tileA[smYIdx_a][smXIdx_a (+4)]
        FLOAT4(sm_tileA + smYIdx_a * BLOCK_TILE_K + smXIdx_a)
        [0] = FLOAT4(gm_a + (globalIdxBlk_y + smYIdx_a) * k + i * BLOCK_TILE_K + smXIdx_a)[0];

        // fetch gm_b[i * BLOCK_TILE_K + smYIdx_b][globalIdxBlk_x + smXIdx_b (+4)] to sm_tileB[smYIdx_b][smXIdx_b (+4)]
        FLOAT4(sm_tileB + smYIdx_b * BLOCK_TILE_X + smXIdx_b)
        [0] = FLOAT4(gm_b + (i * BLOCK_TILE_K + smYIdx_b) * n + globalIdxBlk_x + smXIdx_b)[0];

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        float bufferYDim_a[THREAD_TILE_Y];
        float bufferXDim_b[THREAD_TILE_X];
#pragma unroll
        for (int _k = 0; _k < BLOCK_TILE_K; _k++) {

#pragma unroll
            for (int _i = 0, _j = 0; _i < THREAD_TILE_Y && _j < THREAD_TILE_X; _i++, _j++) {
                // read sm_tileA[threadIdx.y * THREAD_TILE_Y + _i][_k]
                bufferYDim_a[_i] = sm_tileA[(threadIdx.y * THREAD_TILE_Y + _i) * BLOCK_TILE_K + _k];
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