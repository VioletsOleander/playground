// size of matrices
// mat(M,K)@mat(K,N)=mat(M,N)
#define _M 4096
#define _N 4096
#define _K 4096

// repeat times
#define N_REP 100

#define N_WARMUP 10

// A100 GPU properties
#define N_MULTIPROCESSOR 108 // number of stream multiprocessors per device

#define N_PIPELINE_STAGE 1 // pipeline stages, usage of pipeline starts from v3

#define SM_PER_MULTIPROCESSOR 167936    // size of shared memory per stream multiprocessor (bytes)
#define N_REG_PER_MULTIPROCESSOR 65536  // number of registers per stream multiprocessor
#define MAX_BLK_PER_MULTIPROCESSOR 32   // max number of blocks per stream multiprocessor
#define MAX_THR_PER_MULTIPROCESSOR 2048 // max number of threads per stream multiprocessor

#define SM_PER_BLOCK 49152       // size of shared memory per block (bytes)
#define N_REG_PER_BLOCK 65536    // number of registers per block
#define MAX_THR_PER_BLOCK 1024   // max number of threads per block
#define MAX_THR_PER_BLOCK_X 1024 // max number of threads in block x dim
#define MAX_THR_PER_BLOCK_Y 1024 // max number of threads in block y dim
#define MAX_THR_PER_BLOCK_Z 24   // max number of threads in block z dim

#define N_THR_PER_WARP 32   // number of threads per warp
#define WIDTH_BLOCK_TILE 32 // width of block tile

#define CORSEN_FACTOR 4 // corsen factor
#define DIVIDER 2       // sqrt(CORSEN_FACTOR)
