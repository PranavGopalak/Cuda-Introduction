/* File:     cuda_hello1.cu
 * Purpose:  Implement ``hello, world'' on a gpu using CUDA.
 *           Each thread started by the call to the kernel,
 *           prints a message.  This version can start multiple
 *           thread blocks.
 *
 * Compile:  nvcc -o hello hello.cu
 * Run:      ./hello <number of thread blocks> <number of threads>
 *
 * Input:    None
 * Output:   A message from each thread.
 *
 * Note:     Requires an Nvidia GPU with compute capability >= 2.0
 *
 * IPP2:     6.6 (pp. 299 and ff.)
 *
 * Sam Siewert - modified to have more friendly command line processing
 *
 */

#include <stdio.h>
#include <cuda.h> /* Header file for CUDA */

/* Device code:  runs on GPU */
__global__ void Hello(void)
{

    printf("Hello from thread (%d,%d,%d) in block (%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
} /* Hello */

/* Host code:  Runs on CPU */
int main(int argc, char *argv[])
{
    int blk_ct = 108 * 33 * 2;     /* Number of thread blocks */
    int th_per_blk = 1024; /* Number of threads in each block */
    
    if (argc == 1)
    {
        printf("will use default 1 block, with 1 thread\n");
    }
    else if (argc == 2)
    {
        blk_ct = strtol(argv[1], NULL, 10); /* Get number of blocks from command line */
    }
    else if (argc == 3)
    {
        blk_ct = strtol(argv[1], NULL, 10);     /* Get number of blocks from command line */
        th_per_blk = strtol(argv[2], NULL, 10); /* Get number of threads per block from command line */
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100 * 1024 * 1024);  // Set to 10 MB
    size_t newSize = 800 * 1024 * 1024; // 8 MB
    cudaError_t err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, newSize);
    if (err != cudaSuccess) {
    fprintf(stderr, "Failed to set printf buffer size: %s\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
    }

    // Optionally, verify the buffer size
    size_t currentSize;
    err = cudaDeviceGetLimit(&currentSize, cudaLimitPrintfFifoSize);
    if (err == cudaSuccess) {
    printf("Printf buffer size set to: %zu bytes\n", currentSize);
    } else {
    fprintf(stderr, "Failed to get printf buffer size: %s\n", cudaGetErrorString(err));
    }
    printf("Maximum Threads Per Block:                 %d\n", deviceProp.maxThreadsPerBlock);
    long long totalMaxBlocks = (long long)deviceProp.maxGridSize[0];
    printf("Total maximum number of blocks: %lld\n", totalMaxBlocks);
    printf("Total Possible hello messages = %lld x %d\n", totalMaxBlocks , deviceProp.maxThreadsPerBlock);
    Hello<<<blk_ct, th_per_blk>>>();
    /* Start blk_ct*th_per_blk threads on GPU, */

    cudaDeviceSynchronize(); /* Wait for GPU to finish */

    return 0;
} /* main */
