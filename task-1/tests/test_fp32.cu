#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"

#include <cuda_runtime.h>

void MatMulRef(int, int, int, float *, int , float *, int, float *, int);
__global__ void MatMulFP32(int, int, int, float *, int, float *, int, float *, int);
void GenMatFP32(int, int, float*);
float CompareMat(int, int, float *, float *);

int main(){
    int m=M, n=N, k=K;
    int lda = k, ldb = n, ldr = n;
    float run_time = 0.0, sum_run_time = 0.0;
    float err = 0.0;
    float *a, *b, *r, *r_ref; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //allocate memory for matrices
    const size_t a_mem_size = m * k * sizeof(float);
    const size_t b_mem_size = k * n * sizeof(float);
    const size_t r_mem_size = m * n * sizeof(float);
    const size_t r_ref_mem_size = m * n * sizeof(float);
    a = (float*)malloc(a_mem_size);
    b = (float*)malloc(b_mem_size);
    r = (float*)malloc(r_mem_size);
    r_ref = (float*)malloc(r_ref_mem_size);
    //generate random matrices
    GenMatFP32(m, k, a);
    GenMatFP32(k, n, b);

    //get benchmark
    MatMulRef(m, n, k, a, lda, b, ldb, r_ref, ldr); 

    //allocate memory in device
    float *d_A, *d_B, *d_R;
    cudaMalloc((void**)&d_A, a_mem_size);
    cudaMalloc((void**)&d_B, b_mem_size);
    cudaMalloc((void**)&d_R, r_mem_size);
    
    cudaMemcpy(d_A, a, a_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, b_mem_size, cudaMemcpyHostToDevice);

    //run for (N_REP+N_WARMUP) times
    for(int i=0; i<(N_REP+N_WARMUP); i++){
        //warm up
        if (i<N_WARMUP){
            MatMulFP32<<<1024, 64>>>(m, n, k, d_A, lda, d_B, ldb, d_R, ldr);
            continue;
        }
        //running and timing  N_REP times
        cudaEventRecord(start, NULL);

        MatMulFP32<<<1024, 64>>>(m, n, k, d_A, lda, d_B, ldb, d_R, ldr);

        cudaEventRecord(stop, NULL); 
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&run_time, start, stop);
        sum_run_time += run_time;
    }

    //compare result and benchmark
    cudaMemcpy(r, d_R, r_mem_size, cudaMemcpyDeviceToHost);
    err = CompareMat(m, n, r_ref, r);

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
    free(r_ref);

    return 0;
}