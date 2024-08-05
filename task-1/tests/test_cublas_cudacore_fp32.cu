#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void GenMatFP32(int, int, float*);

int main(){
    int m=M, n=N, k=K;
    float run_time = 0.0, sum_run_time = 0.0;
    float *a, *b, *r;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //allocate memory for matrices
    const size_t a_mem_size = m * k * sizeof(float);
    const size_t b_mem_size = k * n * sizeof(float);
    const size_t r_mem_size = m * n * sizeof(float);
    a = (float*)malloc(a_mem_size);
    b = (float*)malloc(b_mem_size);
    r = (float*)malloc(r_mem_size);
    //generate random matrices
    GenMatFP32(m, k, a);
    GenMatFP32(k, n, b);

    //create handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    //allocate memory in device
    float *d_A, *d_B, *d_R;
    cudaMalloc((void**)&d_A, a_mem_size);
    cudaMalloc((void**)&d_B, b_mem_size);
    cudaMalloc((void**)&d_R, r_mem_size);
    
    cudaMemcpy(d_A, a, a_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, b_mem_size, cudaMemcpyHostToDevice);

     //run for (N_REP+N_WARMUP) times
    for(int i=0; i<(N_REP+N_WARMUP); i++){
        if (i<N_WARMUP){
            //warm up
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_R, n);
            continue;
        }
        //running and timing
        cudaEventRecord(start, NULL);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_R, n);

        cudaEventRecord(stop, NULL); 
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&run_time, start, stop);
        sum_run_time += run_time;


    }

     //calculate tflops and average error
    float msecPerMatrixMul = sum_run_time / N_REP;
    double flopsPerMatrixMul = 2.0 * m * k * n;
    double tflops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerMatrixMul / 1000.0f);

    printf("TFLOPS is: %lf\n", tflops);
        cudaMemcpy(r, d_R, r_mem_size, cudaMemcpyDeviceToHost);

        //free memories in device
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_R);
    //free memories in host
    free(a);
    free(b);
    free(r);

    return 0;
}