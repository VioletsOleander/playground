#include "parameters.h"
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void MatMulRef(const int, const int, const int, float *, int, float *, int, float *, int);
__global__ void MatMulFP32(const int, const int, const int, float *, float *, float *);
__global__ void MatMulFP16(const int, const int, const int, half *, half *, half *);
void GenMatFP32(int, int, float *);
void ConvertMat_32_16(int, int, float *, half *);
void ConvertMat_16_32(int, int, half *, float *);
float CompareMat(int, int, float *, float *);
void TransposeMatFP16(int, int, half *);

int main() {
    const int m = _M, n = _N, k = _K;
    const int lda = _K, ldb = _N, ldc = _N, ldr = _N;
    // printf("Matrix A: (%d * %d), Matrix B: (%d * %d), Matrix C: (%d * %d)\n",
    //    m, k, k, n, m, n);

    // allocate memory for matrices
    const size_t memSize_a = m * lda * sizeof(float);
    const size_t memSize_b = k * ldb * sizeof(float);
    const size_t memSize_c = m * ldc * sizeof(float);
    const size_t memSize_r = m * ldc * sizeof(float);
    half *h_a = (half *)malloc(memSize_a);
    half *h_b = (half *)malloc(memSize_b);
    half *h_c = (half *)malloc(memSize_c);
    float *h_a32 = (float *)malloc(memSize_a);
    float *h_b32 = (float *)malloc(memSize_b);
    float *h_c32 = (float *)malloc(memSize_c);
    float *h_r32 = (float *)malloc(memSize_r);

    // generate random matrices
    GenMatFP32(m, k, h_a32);
    GenMatFP32(k, n, h_b32);
    ConvertMat_32_16(m, k, h_a32, h_a);
    ConvertMat_32_16(k, n, h_b32, h_b);

    // get reference result
    MatMulRef(m, n, k, h_a32, lda, h_b32, ldb, h_r32, ldr);

    TransposeMatFP16(k, n, h_b);

    // allocate memory in device
    half *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, memSize_a);
    cudaMalloc((void **)&d_b, memSize_b);
    cudaMalloc((void **)&d_c, memSize_c);

    cudaMemcpy(d_a, h_a, memSize_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, memSize_b, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float runTime = 0.0, runTimeSum = 0.0;

    // configure kernel launch
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    assert(BLOCK_DIM_X * BLOCK_DIM_Y / WARP_SIZE == N_WARP_PER_BLOCK);

    const int numElementXDim_blk = BLOCK_TILE_X;
    const int numElementYDim_blk = BLOCK_TILE_Y;

    const int numElementXDim_grid = m;
    const int numElementYDim_grid = n;

    const int numBlockXDim_grid = (numElementXDim_grid + numElementXDim_blk - 1) / numElementXDim_blk;
    const int numBlockYDim_grid = (numElementYDim_grid + numElementYDim_blk - 1) / numElementYDim_blk;
    dim3 dimGrid(numBlockXDim_grid, numBlockYDim_grid);

    // const int offset_x = n / DIVIDER;
    // const int offset_y = m / DIVIDER;

    const int tileDim_y = numElementYDim_blk;
    const int tileDim_x = numElementXDim_blk;
    const int tileDim_k = BLOCK_TILE_K;

    // shared memory usage by each block tile
    size_t sMemPerBlk = (tileDim_y + tileDim_x) * tileDim_k * sizeof(half) * N_PIPELINE_STAGE;
    assert(sMemPerBlk <= SM_PER_BLOCK);

    // run (N_REP+N_WARMUP) times
    for (int i = 0; i < (N_REP + N_WARMUP); i++) {
        // warm up
        if (i < N_WARMUP) {
            MatMulFP16<<<dimGrid, dimBlock, sMemPerBlk>>>(m, n, k, d_a, d_b, d_c);
            continue;
        }
        // run and timing N_REP times
        cudaEventRecord(start, NULL);

        MatMulFP16<<<dimGrid, dimBlock, sMemPerBlk>>>(m, n, k, d_a, d_b, d_c);

        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        runTimeSum += runTime;
    }

    cudaMemcpy(h_c, d_c, memSize_r, cudaMemcpyDeviceToHost);

    ConvertMat_16_32(m, n, h_c, h_c32);

    // compare result against reference
    float error = 0.0;
    error = CompareMat(m, n, h_r32, h_c32);

    // calculate tflops and average error
    float msecPerMatMul = runTimeSum / N_REP;
    double flopsPerMatMul = 2.0 * m * k * n;
    double tflops = (flopsPerMatMul * 1.0e-12f) / (msecPerMatMul / 1000.0f);

    printf("TFLOPS is: %lf\naverage error is: %f\n", tflops, error);

    // free device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // free host memories
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_a32);
    free(h_b32);
    free(h_c32);
    free(h_r32);

    return 0;
}