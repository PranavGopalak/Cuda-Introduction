#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA.\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Total Global Memory:                       %.2f MB\n",
               (float)deviceProp.totalGlobalMem / (1 << 20));
        printf("  Shared Memory Per Block:                   %.2f KB\n",
               (float)deviceProp.sharedMemPerBlock / 1024);
        printf("  Registers Per Block:                       %d\n", deviceProp.regsPerBlock);
        printf("  Warp Size:                                 %d\n", deviceProp.warpSize);
        printf("  Maximum Threads Per Block:                 %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum Threads Dimensions:                [%d, %d, %d]\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Maximum Grid Size:                         [%d, %d, %d]\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Clock Rate:                                %.2f MHz\n", deviceProp.clockRate * 1e-3f);
        printf("  Total Constant Memory:                     %.2f KB\n", (float)deviceProp.totalConstMem / 1024);
        printf("  Texture Alignment:                         %zu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent Kernels:                        %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
        printf("  Device Overlap:                            %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
        printf("  Compute Mode:                              %d\n", deviceProp.computeMode);
        printf("Number of Streaming Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);
        long long totalMaxBlocks = (long long)deviceProp.maxGridSize[0];
        printf("Total maximum number of blocks: %lld\n", totalMaxBlocks);
    }

    return EXIT_SUCCESS;
}
