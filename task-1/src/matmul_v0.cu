#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parameters.h"

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

    // be responsible for computing the (globalIdx_y, globalIdx_x) element in matrix c
    int globalIdx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int globalIdx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (globalIdx_y < m && globalIdx_x < n) {
        float val = 0.0;
        for (int i = 0; i < k; i++) {
            // read global memory twice
            float val_a = gm_a[globalIdx_y * lda + i]; // (globalIdx_y, i) in matrix a
            float val_b = gm_b[i * ldb + globalIdx_x]; // (i, globalIdx_x) in matrix b
            val += val_a * val_b;
        }
        // write global memory once
        gm_c[globalIdx_y * ldc + globalIdx_x] = val;
    }
    // computation for one output element takes (2 * k + 1) global memory access
}