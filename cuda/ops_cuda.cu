#include <cuda_runtime.h>

__global__ void smoke_kernel() {
}

void cuda_smoke_test() {
    smoke_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}