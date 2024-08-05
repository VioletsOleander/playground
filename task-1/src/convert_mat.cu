#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void mat_32_16(int m, int n, float *a, __half *b){
    for(int i=0; i<m*n; i++){
        b[i] = __float2half(a[i]);
    }
}

void mat_16_32(int m, int n, __half *a, float *b){
    for(int i=0; i<m*n; i++){
        b[i] = __half2float(a[i]);
    }
}