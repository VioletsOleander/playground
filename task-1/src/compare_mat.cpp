#include <stdio.h>
#include <stdlib.h>

float CompareMat(int m, int n, float *a, float *b){
    float gap = 0.0;
    float err_sum = 0.0;
    float a_err_sum = 0.0; //average sum of errors

    for(int i=0; i<m*n; i++){
        gap = abs(a[i] - b[i]);
        err_sum += gap/a[i];
    }

    a_err_sum = err_sum/(m*n);

    return a_err_sum;
}