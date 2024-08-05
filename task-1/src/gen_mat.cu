#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>


void GenMatFP16(int m, int n, __half* mat){
    srand(time(NULL));
    int cnt;
    float tmp;
    for(cnt=0; cnt<m*n; cnt++){
        tmp = (float)rand() / RAND_MAX;
        mat[cnt] = __half2float(tmp);
    }
}

void GenMatFP32(int m, int n, float *mat){
    srand(time(NULL));
    int cnt;
    for(cnt=0; cnt<m*n; cnt++){
        mat[cnt] = (float)rand() / RAND_MAX;
    }
}
