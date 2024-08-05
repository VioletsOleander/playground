//size of matrices
//mat(M,K)@mat(K,N)=mat(M,N)
#define M 4096
#define N 4096
#define K 4096

//repeated times
#define N_REP 100

#define N_WARMUP 10

//range of elements in matrix
#define ELE_MIN 0
#define ELE_MAX 1

//GPU A100 properties
#define SM_PER_BLOCK 49152 //shared memory per block (bytes)
#define MAX_THR_PER_BLOCK 1024 //max number of threads per block
