#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "parameters.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

void MMult_ref(int, int, int, float *, int , float *, int, float *, int);
void MMult_fp16(int, int, int, __half *, int, __half *, int, __half *, int);
void gen_mat_fp32(int, int, float*);
void mat_32_16(int, int, float *, __half *);
void mat_16_32(int, int, __half *, float *);
float comp_mat(int, int, float *, float *);


int main(){
    int m=M, n=N, k=K;
    int lda = k, ldb = n, ldr = n;
    float run_time = 0.0, sum_run_time = 0.0;
    float err = 0.0;
    __half *a, *b, *r;
    float *a_32, *b_32, *r_32, *r_ref;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //allocate memory for matrices
    const size_t a_mem_size = m * k * sizeof(__half);
    const size_t b_mem_size = k * n * sizeof(__half);
    const size_t r_mem_size = m * n * sizeof(__half);
    const size_t a_32_mem_size = m * k * sizeof(float); 
    const size_t b_32_mem_size = k * n * sizeof(float);
    const size_t r_32_mem_size = m * n * sizeof(float);
    const size_t r_ref_mem_size = m * n * sizeof(float);

    a = (__half*)malloc(a_mem_size);
    b = (__half*)malloc(b_mem_size);
    r = (__half*)malloc(r_mem_size);
    a_32 = (float*)malloc(a_32_mem_size);
    b_32 = (float*)malloc(b_32_mem_size);
    r_32 = (float*)malloc(r_32_mem_size);
    r_ref = (float*)malloc(r_ref_mem_size);


    //generate random matrices
    gen_mat_fp32(m, k, a_32);
    gen_mat_fp32(k, n, b_32);
    mat_32_16(m, k, a_32, a);
    mat_32_16(k, n, b_32, b);

    //get benchmark
    MMult_ref(m, n, k, a_32, lda, b_32, ldb, r_ref, ldr); 


    //create handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    __half alpha = __float2half(1.0);
    __half beta = __float2half(0.0); 

    // float alpha = 1.0;
    // float beta = 0.0;    //this is wrong, err is 1.0


    //allocate memory in device
    __half  *d_A, *d_B, *d_R;
    cudaMalloc((void**)&d_A, a_mem_size);
    cudaMalloc((void**)&d_B, b_mem_size);
    cudaMalloc((void**)&d_R, r_mem_size);
    

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    //generate random matrices
    gen_mat_fp32(m, k, a_32);
    gen_mat_fp32(k, n, b_32);
    mat_32_16(m, k, a_32, a);
    mat_32_16(k, n, b_32, b);

    cudaMemcpy(d_A, a, a_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, b_mem_size, cudaMemcpyHostToDevice);
    

    //run for (N_REP+N_WARMUP) times
    for(int i=0; i<(N_REP+N_WARMUP); i++){
        // warm up N_WARMUP times
        if (i < N_WARMUP){
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                      n, m, k, 
                                      &alpha, 
                                      d_B, CUDA_R_16F, n, 
                                      d_A, CUDA_R_16F, k, 
                                      &beta, 
                                      d_R, CUDA_R_16F, n, 
                                      CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            continue;
        }
        //running and timing  N_REP times
        cudaEventRecord(start, NULL);

        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                  n, m, k, 
                                  &alpha, 
                                  d_B, CUDA_R_16F, n, 
                                  d_A, CUDA_R_16F, k, 
                                  &beta, 
                                  d_R, CUDA_R_16F, n, 
                                  CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        cudaEventRecord(stop, NULL); 
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&run_time, start, stop);
        sum_run_time += run_time;
    }
    cudaMemcpy(r, d_R, r_mem_size, cudaMemcpyDeviceToHost);
    //compare result and benchmark
    mat_16_32(m, n, r, r_32);
    err = comp_mat(m, n, r_ref, r_32);
     //calculate tflops
    float msecPerMatrixMul = sum_run_time / N_REP;
    double flopsPerMatrixMul = 2.0 * m * k * n;
    double tflops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerMatrixMul / 1000.0f);

    printf("TFLOPS is: %lf\naverage error is: %f\n", tflops, err);
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