#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

void (cublasHandle_t handle, int m, int n, int k, float *d_A, int lda, float *d_B, int ldb, float *d_R, int ldr){
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_R, m);
}