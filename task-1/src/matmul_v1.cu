#include "parameters.h"
#include <cuda_runtime.h>

__global__ void MatMulFP32(const int m, const int n, const int k,
                           const float *a, const int lda,
                           const float *b, const int ldb,
                           float *c, const int ldc) {

    // size of matrix a: m * k (elements)
    // size of matrix b: k * n (elements)
    // size of matrix c: m * n (elements)

    const float *gm_a = a;
    const float *gm_b = b;
    float *gm_c = c;

    // blockDim.x == blockDim.y == tileK is assumed, which means size(tileA) == size(tileB) == size(block)
    const int tileK = WIDTH_BLOCK_TILE;

    extern __shared__ float sm[];

    float *sm_tileA = sm;                      // size of tileA: blockDim.y * tileK (elements)
    float *sm_tileB = sm + blockDim.y * tileK; // size of tileB: tileK * blockDim.x (elements)

    // be responsible for computing the (globalIdx_y, globalIdx_x) element in matrix c
    int globalIdx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int globalIdx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (globalIdx_y < m && globalIdx_x < n) {
        float val = 0.0;
        for (int i = 0; i < k / tileK; i++) { // exact division assumed
            // be responsible for fetching one element each for tileA and tileB from gm to sm, read global memory twice, write shared memory twice
            // fetch gm_a[globalIdx_y, i * tileK + threadIdx.x] to sm_tileA[threadIdx.y, threadIdx.x]
            sm_tileA[threadIdx.y * tileK + threadIdx.x] = gm_a[globalIdx_y * lda + i * tileK + threadIdx.x];
            // fetch gm_b[i * tileK + threadIdx.y, globalIdx_x] to sm_tileB[threadIdx.y, threadIdx.x]
            sm_tileB[threadIdx.y * blockDim.x + threadIdx.x] = gm_b[(i * tileK + threadIdx.y) * ldb + globalIdx_x];

            __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

            // be responsible for computing a partial sum for sm_tileC[threadIdx.y, threadIdx.x]
            for (int j = 0; j < tileK; j++) {
                float val_a = sm_tileA[threadIdx.y * tileK + j];      // sm_tileA[threadIdx.y, j]
                float val_b = sm_tileB[j * blockDim.x + threadIdx.x]; // sm_tileB[j, threadIdx.x]
                val += val_a * val_b;
            } // the loop takes ( 2 * tileK ) shared memory read in total

            __syncthreads(); // sync to make sure every thread finished computation so sm_tileA and sm_tileB are allowed to be overwirtten
        } // the loop takes ( 2 * k/tileK ) global memory read in total, and takes ( 2 * k ) shared memory read in total
        // write global memory once
        gm_c[globalIdx_y * ldc + globalIdx_x] = val;
    }
    // computation for one output element takes ( 2 * k/tileK + 1 ) global memory access, and takes ( 2 * k + 2 * k/tileK ) shared memory access
}