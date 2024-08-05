#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(){
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    if(!deviceCount){
        printf("No devices found suppoting CUDA.\n");
    }
    else{
        printf("Detected %d CUDA capable device(s).\n", deviceCount);
    }

    FILE *fp;
    fp = fopen("../docs/lab_device_properties.txt","w");
    if(fp == NULL){
        printf("Error opening file \"../docs/lab_device_properties.txt\".\n");
    }

    for(int i=0; i<deviceCount; i++){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        fprintf(fp, "Device %d: %s \n", i, deviceProp.name);

        fprintf(fp, " Device Level Properties:\n");
        fprintf(fp, "  Compute Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
        fprintf(fp, "  Total amount of global memory on device: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        fprintf(fp, "  Total amount of constant memory on device : %zu bytes\n", deviceProp.totalConstMem);
        fprintf(fp, "  Number of available multiprocessors on device: %d\n", deviceProp.multiProcessorCount);

        fprintf(fp, " Grid Level Properties:\n");
        fprintf(fp, "  Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

        fprintf(fp, " Stream Multiprocessor Level Properties:\n");
        fprintf(fp, "  Total amount of shared memory per multiprocessor: %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
        fprintf(fp, "  Number of registers available per multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
        fprintf(fp, "  Maximum number of blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
        fprintf(fp, "  Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

        fprintf(fp, " Block Level Properties:\n");
        fprintf(fp, "  Total amount of shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        fprintf(fp, "  Total amount of reserved shared memory per block: %zu bytes\n", deviceProp.reservedSharedMemPerBlock);
        fprintf(fp, "  Number of registers available per block: %d\n", deviceProp.regsPerBlock);
        fprintf(fp, "  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(fp, "  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

        fprintf(fp, " Other Properties:\n");
        fprintf(fp, "  Warp size: %d\n", deviceProp.warpSize);
        fprintf(fp, "  Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        fprintf(fp, "  Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
        if (i != deviceCount - 1) fprintf(fp, "\n");
    }

    fclose(fp);

    return 0;
}