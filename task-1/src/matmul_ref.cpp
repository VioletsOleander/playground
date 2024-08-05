#include "cblas.h"

void MatMulRef(int m, int n, int k, float *a, int lda, float *b, int ldb, float *r, int ldr){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, r, ldr);
}