#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "parameters.h"



__global__ void MMult_fp32(int m, int n, int k, float *d_A, int lda, float *d_B, int ldb, float *d_R, int ldr){
	

}