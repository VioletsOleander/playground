#include "parameters.h"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

__global__ void MatMulFP32(const int m, const int n, const int k,
                           const float *a, const float *b, float *c,
                           const int offset_x, const int offset_y) {

    // size of matrix a: m * k (elements)
    // size of matrix b: k * n (elements)
    // size of matrix c: m * n (elements)

    const float *gm_a = a;
    const float *gm_b = b;
    float *gm_c = c;

    // blockDim.x == blockDim.y == tileK is assumed,
    // which means size(tileA partition) == size(tileB partition) == size(block)
    const int tileK = WIDTH_BLOCK_TILE;
    const int halfTileK = tileK / 2;
    const int tileNum = k / tileK; // exact divition assumed

    // size(shared memory) == corsenFactor * ( size(tileA partition) + size(tileB partition) ) * N_PIPELINE_STAGE
    extern __shared__ float sm[];

    float *sm_tileA = sm;                             // size of tileA partition: blockDim.y * tileK (elements)
    float *sm_tileB = sm + blockDim.y * tileK;        // size of tileB partition: tileK * blockDim.x (elements)
    int partSize = tileK * (blockDim.y + blockDim.x); // partition size = size(tileA) + size(tileB) (elements)
    int bufferSize = 2 * partSize;                    // buffer size = 2 * partSize = 2 * ( size(tileA) + size(tileB) ) (elements)

    // be responsible for computing the (globalIdx_y, globalIdx_x), (globalIdx_y + offset_y, globalIdx_x),
    // (globalIdx_y + offset_y, globalIdx_x + offset_x), (globalIdx_y, globalIdx_x + offset_x) element in matrix c
    int globalIdx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int globalIdx_y = blockDim.y * blockIdx.y + threadIdx.y;

    // pipeline instance to manage stages
    cuda::pipeline<cuda::thread_scope::thread_scope_thread> pipeline = cuda::make_pipeline();

    float val_1, val_2, val_3, val_4;
    val_1 = val_2 = val_3 = val_4 = 0.0;
    // boundary check is removed for simplicity
    for (int computeTileIdx = 0, fetchTileIdx = 0; computeTileIdx < tileNum; computeTileIdx++) {
        // before the first compute stage, issue N_PIPELINE_STAGE asynchronous memcpy opertion to fill the shared memory
        // after the first compute stage, the fetchTileIdx should be ( N_PIPELINE_STAGE - 1 ) ahead of computeTileIdx
        // after the first compute stage, the fetchTileIdx will increase with the same pace of computeTileIdx
        for (; fetchTileIdx < tileNum && fetchTileIdx < computeTileIdx + N_PIPELINE_STAGE; fetchTileIdx++) {
            // In each fetch stage, the thread is responsible for doing the following DIVIDER times:
            // fetching one element each for tileA partition and tileB partition from gm to sm
            // the thread read global memory ( 2 * DIVIDER = 4 ) times, write shared memory ( 2 * DIVIDER = 4 ) times in total

            // the shared memory is N_PIPELINE_STAGE-way buffered
            int bufferIdx = fetchTileIdx % N_PIPELINE_STAGE;
            int bufferOffset = bufferIdx * bufferSize;

            // acquire a stage from the head of the pipeline queue
            pipeline.producer_acquire();

            // gm_a[globalIdx_y][i * tileK + threadIdx.x] to sm_tileA[threadIdx.y][threadIdx.x]
            cuda::memcpy_async((sm_tileA + bufferOffset + threadIdx.y * tileK + threadIdx.x),
                               (gm_a + globalIdx_y * k + fetchTileIdx * tileK + threadIdx.x), sizeof(float), pipeline);
            // gm_a[globalIdx_y][i * tileK + halfTileK + threadIdx.x] to sm_tileA[threadIdx.y][threadIdx.x + halfTileK]
            cuda::memcpy_async((sm_tileA + bufferOffset + threadIdx.y * tileK + threadIdx.x + halfTileK),
                               (gm_a + globalIdx_y * k + fetchTileIdx * tileK + halfTileK + threadIdx.x), sizeof(float), pipeline);
            // gm_b[i * tileK + threadIdx.y][globalIdx_x] to sm_tileB[threadIdx.y][threadIdx.x]
            cuda::memcpy_async((sm_tileB + bufferOffset + threadIdx.y * blockDim.x + threadIdx.x),
                               (gm_b + (fetchTileIdx * tileK + threadIdx.y) * n + globalIdx_x), sizeof(float), pipeline);
            // gm_b[i * tileK + threadIdx.y + halfTileK][globalIdx_x] to sm_tileB[threadIdx.y + halfTileK][threadIdx.x]
            cuda::memcpy_async((sm_tileB + bufferOffset + (threadIdx.y + halfTileK) * blockDim.x + threadIdx.x),
                               (gm_b + (fetchTileIdx * tileK + threadIdx.y + halfTileK) * n + globalIdx_x), sizeof(float), pipeline);
            // gm_a[globalIdx_y + offset_y][i * tileK + threadIdx.x] to (sm_tileA + partSize)[threadIdx.y][threadIdx.x]
            cuda::memcpy_async((sm_tileA + partSize + bufferOffset + threadIdx.y * tileK + threadIdx.x),
                               (gm_a + (globalIdx_y + offset_y) * k + fetchTileIdx * tileK + threadIdx.x), sizeof(float), pipeline);
            // gm_a[globalIdx_y + offset_y][i * tileK + halfTileK + threadIdx.x] to (sm_tileA + partSize)[threadIdx.y][threadIdx.x + halfTilek]
            cuda::memcpy_async((sm_tileA + partSize + bufferOffset + threadIdx.y * tileK + threadIdx.x + halfTileK),
                               (gm_a + (globalIdx_y + offset_y) * k + fetchTileIdx * tileK + halfTileK + threadIdx.x), sizeof(float), pipeline);
            // gm_b[i * tileK + threadIdx.y][globalIdx_x + offset_x] to (sm_tileB + partSize)[threadIdx.y][threadIdx.x]
            cuda::memcpy_async((sm_tileB + partSize + bufferOffset + threadIdx.y * blockDim.x + threadIdx.x),
                               (gm_b + (fetchTileIdx * tileK + threadIdx.y) * n + offset_x + globalIdx_x), sizeof(float), pipeline);
            // gm_b[i * tileK + threadIdx.y + halfTileK][globalIdx_x + offset_x] to (sm_tileB + partSize)[threadIdx.y + halfTileK][threadIdx.x]
            cuda::memcpy_async((sm_tileB + partSize + bufferOffset + (threadIdx.y + halfTileK) * blockDim.x + threadIdx.x),
                               (gm_b + (fetchTileIdx * tileK + threadIdx.y + halfTileK) * n + offset_x + globalIdx_x), sizeof(float), pipeline);

            // commit issued asynchronous memcpy operations to the acquire stage
            pipeline.producer_commit();
        }
        // wait for the stage in the pipeline tail to complete
        pipeline.consumer_wait();

        // sync to make sure every thread in block complete their waiting,
        // therefore every element in sm_tileA, sm_tileB, sm_tileA + partSize, sm_tileB + partSize is ready
        __syncthreads();

        int bufferIdx = computeTileIdx % N_PIPELINE_STAGE;
        int bufferOffset = bufferIdx * bufferSize;

        // be responsible for computing a partial sum for val_1, val_2, val_3, val_4
        for (int j = 0; j < tileK; j++) {
            float val_a1 = (sm_tileA + bufferOffset)[threadIdx.y * tileK + j];                 // sm_tileA[threadIdx.y][j]
            float val_a2 = (sm_tileA + partSize + bufferOffset)[threadIdx.y * tileK + j];      // (sm_tileA + partSize)[threadIdx.y][j]
            float val_b1 = (sm_tileB + bufferOffset)[j * blockDim.x + threadIdx.x];            // sm_tileB[j][threadIdx.x]
            float val_b2 = (sm_tileB + partSize + bufferOffset)[j * blockDim.x + threadIdx.x]; // (sm_tileB + partSize)[j][threadIdx.x]
            val_1 += val_a1 * val_b1;
            val_2 += val_a1 * val_b2;
            val_3 += val_a2 * val_b1;
            val_4 += val_a2 * val_b2;
        } // the loop takes ( 4 * tileK ) shared memory read in total

        __syncthreads();
        // release the required stage for reuse
        pipeline.consumer_release();
    }
    // the loop takes ( 4 * k/tileK ) global memory read in total, and takes ( 4 * k ) shared memory read in total
    // write global memory 1 * corsenFactor times
    gm_c[globalIdx_y * n + globalIdx_x] = val_1;                         // gm_c[globalIdx_y][globalIdx_x]
    gm_c[globalIdx_y * n + globalIdx_x + offset_x] = val_2;              // gm_c[globalIdx_y][globalIdx_x + offset_x]
    gm_c[(globalIdx_y + offset_y) * n + globalIdx_x] = val_3;            // gm_c[globalIdx_y + offset_y][globalIdx_x]
    gm_c[(globalIdx_y + offset_y) * n + globalIdx_x + offset_x] = val_4; // gm_c[globalIdx_y + offset_y][globalIdx_x + offset_x]
    // computation for 4 output element takes ( 4 * k/tileK + 4 ) global memory access, and takes ( 4 * k + 4 * k/tileK ) shared memory access
    // computation for 1 output element takes ( k/tileK + 1 ) global memory access, and takes ( k + k/tileK ) shared memory access
}