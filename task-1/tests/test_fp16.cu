#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"
//#include "MMult_v0.cu"
//#include "CompareMat.cpp"
//#include "gen_mat.cu"
//#include "MatMulRef.cpp"
//#include "tran_mat.cu"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void MatMulRef(int, int, int, float *, int , float *, int, float *, int);
void MatMulFP16(int, int, int, __half *, int, __half *, int, __half *, int);
void GenMatFP32(int, int, float*);
void mat_32_16(int, int, float *, __half *);
void mat_16_32(int, int, __half *, float *);
float CompareMat(int, int, float *, float *);

int main(){
    int m=M, n=N, k=K;
    int lda = k, ldb = n, ldr = n;
    float *a_32, *b_32, *r_32, *r_ref;
    float run_time = 0.0, sum_run_time = 0.0;
    float err = 0.0;
    __half *a, *b, *r;

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
    GenMatFP32(m, k, a_32);
    GenMatFP32(k, n, b_32);
    mat_32_16(m, k, a_32, a);
    mat_32_16(k, n, b_32, b);

    //get benchmark
    MatMulRef(m, n, k, a_32, lda, b_32, ldb, r_ref, ldr); 

    //allocate memory in device
    __half *d_A, *d_B, *d_R;
    cudaMalloc((void**)&d_A, a_mem_size);
    cudaMalloc((void**)&d_B, b_mem_size);
    cudaMalloc((void**)&d_R, r_mem_size);
    
    cudaMemcpy(d_A, a, a_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, b_mem_size, cudaMemcpyHostToDevice);

    //run for (N_REP+N_WARMUP) times
    for(int i=0; i<(N_REP+N_WARMUP); i++){
        //warm up
        if (i<N_WARMUP){
            MatMulFP16(m, n, k, d_A, lda, d_B, ldb, d_R, ldr);
            continue;
        }
        //running and timing  N_REP times
        cudaEventRecord(start, NULL);

        MatMulFP16(m, n, k, d_A, lda, d_B, ldb, d_R, ldr);

        cudaEventRecord(stop, NULL); 
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&run_time, start, stop);
        sum_run_time += run_time;
    }

    cudaMemcpy(r, d_R, r_mem_size, cudaMemcpyDeviceToHost);

    //compare result and benchmark
    mat_16_32(m, n, r, r_32);
    err = CompareMat(m, n, r_ref, r_32);

    //calculate tflops and average error
    float msecPerMatrixMul = sum_run_time / N_REP;
    double flopsPerMatrixMul = 2.0 * m * k * n;
    double tflops = (flopsPerMatrixMul * 1.0e-12f) / (msecPerMatrixMul / 1000.0f);

    printf("TFLOPS is: %lf\naverage error is: %f\n", tflops, err);

    //free memories in device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);

    //free memories in host
    free(a);
    free(b);
    free(r);
    free(a_32);
    free(b_32);
    free(r_ref);

    return 0;
}