#include <stdio.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;



void MatMulFP16(int m, int n, int k, half *d_A, int lda, half *d_B, int ldb, half *d_R, int ldr){
    /*
	Add your code here
	*/
}