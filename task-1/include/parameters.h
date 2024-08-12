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

#define WARP_SIZE 32 // number of threads per warp

#define CORSEN_FACTOR 16 // corsen factor
#define DIVIDER 2        // sqrt(CORSEN_FACTOR)

#define BLOCK_TILE_K 16 // width of block tile
#define STRIDE_IN_TILE 2
#define BLOCK_DIM_X 16 // thread num in block dim x
#define BLOCK_DIM_Y 16 // thread num in block dim y

#define BLOCK_TILE_X 128 // element num in block tile x dim
#define BLOCK_TILE_Y 256 // element num in block tile y dim
#define WARP_TILE_X 64   // element num in warp tile x dim
#define WARP_TILE_Y 64   // element num in warp tile y dim
#define THREAD_TILE_X 8  // element num in thread tile x dim
#define THREAD_TILE_Y 8  // element num in thread tile y dim

#define N_WARP_PER_BLOCK 8