#include <cuda_fp16.h>
void TransposeMatFP32(int m, int n, float *mat) {
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < n; j++) {
            float tmp = mat[i * n + j];
            mat[i * n + j] = mat[j * n + i];
            mat[j * n + i] = tmp;
        }
    }
}
void TransposeMatFP16(int m, int n, half *mat) {
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < n; j++) {
            float tmp = mat[i * n + j];
            mat[i * n + j] = mat[j * n + i];
            mat[j * n + i] = tmp;
        }
    }
}