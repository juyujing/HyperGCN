#include <stdio.h>
#include <cuda_runtime_api.h>

void release_cuda_memory(void* device_ptr) {
    cudaError_t status = cudaFree(device_ptr);
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed to release CUDA memory: %s\n", cudaGetErrorString(status));
    }
}
