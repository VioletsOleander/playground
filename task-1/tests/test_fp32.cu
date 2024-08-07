#include "parameters.h"
#include <cassert>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void MatMulRef(const int, const int, const int, float *, int, float *, int, float *, int);
__global__ void MatMulFP32(const int, const int, const int, const float *, const int, const float *, const int, float *, const int);
void GenMatFP32(int, int, float *);
float CompareMat(int, int, float *, float *);

int main() {
    const int m = M, n = N, k = K;
    const int lda = K, ldb = N, ldc = N, ldr = N;
    // printf("Matrix A: (%d * %d), Matrix B: (%d * %d), Matrix C: (%d * %d)\n",
    //    m, k, k, n, m, n);

    // allocate memory for matrices
    const size_t memSize_a = m * lda * sizeof(float);
    const size_t memSize_b = k * ldb * sizeof(float);
    const size_t memSize_c = m * ldc * sizeof(float);
    const size_t memSize_r = m * ldc * sizeof(float);
    float *h_a = (float *)malloc(memSize_a);
    float *h_b = (float *)malloc(memSize_b);
    float *h_c = (float *)malloc(memSize_c);
    float *h_r = (float *)malloc(memSize_r);

    // generate random matrices
    GenMatFP32(m, k, h_a);
    GenMatFP32(k, n, h_b);

    // get reference result
    MatMulRef(m, n, k, h_a, lda, h_b, ldb, h_r, ldr);

    // allocate memory in device
    float *d_a, *d_b, *d_r;

    cudaMalloc((void **)&d_a, memSize_a);
    cudaMalloc((void **)&d_b, memSize_b);
    cudaMalloc((void **)&d_r, memSize_r);

    cudaMemcpy(d_a, h_a, memSize_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, memSize_b, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float runTime = 0.0, runTimeSum = 0.0;

    // configure kernel launch
    const int numWarp_blk = 8;
    const int numThread_blk = numWarp_blk * N_THR_PER_WARP;
    const int numThreadXDim_blk = sqrt((double)numThread_blk);
    const int numThreadYDim_blk = numThread_blk / numThreadXDim_blk; // exact division assumed
    dim3 dimBlock(numThreadXDim_blk, numThreadYDim_blk);

    const int numThreadXDim_grid = m; // each thread responsible for one output
    const int numThreadYDim_grid = n;
    const int numBlockXDim_grid = (numThreadXDim_grid + numThreadXDim_blk - 1) / numThreadXDim_blk;
    const int numBlockYDim_grid = (numThreadYDim_grid + numThreadYDim_blk - 1) / numThreadYDim_blk;
    dim3 dimGrid(numBlockXDim_grid, numBlockYDim_grid);
    // printf("Grid dim: (%d, %d), Block dim: (%d, %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    const int tileDim_m = numThreadYDim_blk;
    const int tileDim_n = numThreadXDim_blk;
    assert(tileDim_m == tileDim_n); // square tile block assumed
    const int tileDim_k = WIDTH_BLOCK_TILE;
    // printf("Blocktile A dim: (%d, %d), BlockTile B dim: (%d, %d), Blocktile C dim: (%d, %d)\n",
    //    tileDim_m, tileDim_k, tileDim_k, tileDim_n, tileDim_m, tileDim_n);
    // shared memory usage by each block tile
    size_t sMemPerBlk = (tileDim_m * tileDim_k + tileDim_n * tileDim_k) * sizeof(float);
    // printf("Shared memory usage: %d bytes per block\n", sMemPerBlk);
    assert(sMemPerBlk < SM_PER_BLOCK);

    // run (N_REP+N_WARMUP) times
    for (int i = 0; i < (N_REP + N_WARMUP); i++) {
        // warm up
        if (i < N_WARMUP) {
            MatMulFP32<<<dimGrid, dimBlock, sMemPerBlk>>>(m, n, k, d_a, lda, d_b, ldb, d_r, ldr);
            continue;
        }
        // run and timing N_REP times
        cudaEventRecord(start, NULL);

        MatMulFP32<<<dimGrid, dimBlock, sMemPerBlk>>>(m, n, k, d_a, lda, d_b, ldb, d_r, ldr);

        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runTime, start, stop);
        runTimeSum += runTime;
    }

    cudaMemcpy(h_c, d_r, memSize_r, cudaMemcpyDeviceToHost);

    // compare result against reference
    float error = 0.0;
    error = CompareMat(m, n, h_r, h_c);

    // calculate tflops and average error
    float msecPerMatMul = runTimeSum / N_REP;
    double flopsPerMatMul = 2.0 * m * k * n;
    double tflops = (flopsPerMatMul * 1.0e-12f) / (msecPerMatMul / 1000.0f);

    printf("TFLOPS is: %lf\naverage error is: %f\n", tflops, error);

    // free device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r);

    // free host memories
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_r);

    return 0;
}