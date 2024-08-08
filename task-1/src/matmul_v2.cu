#include "parameters.h"
#include <cuda_runtime.h>

__global__ void MatMulFP32(const int m, const int n, const int k,
                           const float *a, const float *b, float *c,
                           const int offset_x, const int offset_y) {

    // size of matrix a: m * k (elements)
    // size of matrix b: k * n (elements)
    // size of matrix c: m * n (elements)

    const float *gm_a = a;
    const float *gm_b = b;
    float *gm_c = c;

    // blockDim.x == blockDim.y == tileK is assumed,
    // which means size(tileA partition) == size(tileB partition) == size(block)
    const int tileK = WIDTH_BLOCK_TILE;

    // size(shared memory) == corsenFactor * ( size(tileA partition) + size(tileB partition) )
    extern __shared__ float sm[];

    float *sm_tileA = sm;                             // size of tileA partition: blockDim.y * tileK (elements)
    float *sm_tileB = sm + blockDim.y * tileK;        // size of tileB partition: tileK * blockDim.x (elements)
    int partSize = tileK * (blockDim.y + blockDim.x); // partition size = size(tileA) + size(tileB) (elements)

    // be responsible for computing the (globalIdx_y, globalIdx_x), (globalIdx_y + offset_y, globalIdx_x),
    // (globalIdx_y + offset_y, globalIdx_x + offset_x), (globalIdx_y, globalIdx_x + offset_x) element in matrix c
    int globalIdx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int globalIdx_y = blockDim.y * blockIdx.y + threadIdx.y;

    // remove boundary check for simplicity
    float val_1, val_2, val_3, val_4;
    val_1 = val_2 = val_3 = val_4 = 0.0;
#pragma unroll
    for (int i = 0; i < k / tileK; i++) { // exact division assumed
        // be responsible for doing the following divider times:
        // fetching one element each for tileA partition and tileB partition from gm to sm,
        // read global memory 2 * divider = 4 times, write shared memory 2 * divider = 4 times

        // gm_a[globalIdx_y][i * tileK + threadIdx.x] to sm_tileA[threadIdx.y][threadIdx.x]
        sm_tileA[threadIdx.y * tileK + threadIdx.x] = gm_a[globalIdx_y * k + i * tileK + threadIdx.x];
        // gm_b[i * tileK + threadIdx.y][globalIdx_x] to sm_tileB[threadIdx.y][threadIdx.x]
        sm_tileB[threadIdx.y * blockDim.x + threadIdx.x] = gm_b[(i * tileK + threadIdx.y) * n + globalIdx_x];

        // gm_a[globalIdx_y + offset_y][i * tileK + threadIdx.x] to (sm_tileA + partSize)[threadIdx.y][threadIdx.x]
        (sm_tileA + partSize)[threadIdx.y * tileK + threadIdx.x] = gm_a[(globalIdx_y + offset_y) * k + i * tileK + threadIdx.x];
        // gm_b[i * tileK + threadIdx.y][globalIdx_x + offset_x] to (sm_tileB + partSize)[threadIdx.y][threadIdx.x]
        (sm_tileB + partSize)[threadIdx.y * blockDim.x + threadIdx.x] = gm_b[(i * tileK + threadIdx.y) * n + offset_x + globalIdx_x];

        __syncthreads(); // sync to make sure every element in sm_tileA and sm_tileB gets ready

        // be responsible for computing a partial sum for val_1, val_2, val_3, val_4
#pragma unroll
        for (int j = 0; j < tileK; j++) {
            float val_a1 = sm_tileA[threadIdx.y * tileK + j];                   // sm_tileA[threadIdx.y][j]
            float val_a2 = (sm_tileA + partSize)[threadIdx.y * tileK + j];      // (sm_tileA + partSize)[threadIdx.y][j]
            float val_b1 = sm_tileB[j * blockDim.x + threadIdx.x];              // sm_tileB[j][threadIdx.x]
            float val_b2 = (sm_tileB + partSize)[j * blockDim.x + threadIdx.x]; // (sm_tileB + partSize)[j][threadIdx.x]
            val_1 += val_a1 * val_b1;
            val_2 += val_a1 * val_b2;
            val_3 += val_a2 * val_b1;
            val_4 += val_a2 * val_b2;
        } // the loop takes ( 4 * tileK ) shared memory read in total

        __syncthreads(); // sync to make sure every thread finished computation so sm_tileA and sm_tileB are allowed to be overwirtten
    } // the loop takes ( 4 * k/tileK ) global memory read in total, and takes ( 4 * k ) shared memory read in total
    // write global memory 1 * corsenFactor times
    gm_c[globalIdx_y * n + globalIdx_x] = val_1;                         // gm_c[globalIdx_y][globalIdx_x]
    gm_c[globalIdx_y * n + globalIdx_x + offset_x] = val_2;              // gm_c[globalIdx_y][globalIdx_x + offset_x]
    gm_c[(globalIdx_y + offset_y) * n + globalIdx_x] = val_3;            // gm_c[globalIdx_y + offset_y][globalIdx_x]
    gm_c[(globalIdx_y + offset_y) * n + globalIdx_x + offset_x] = val_4; // gm_c[globalIdx_y + offset_y][globalIdx_x + offset_x]
    // computation for 4 output element takes ( 4 * k/tileK + 4 ) global memory access, and takes ( 4 * k + 4 * k/tileK ) shared memory access
    // computation for 1 output element takes ( k/tileK + 1 ) global memory access, and takes ( k + k/tileK ) shared memory access
}