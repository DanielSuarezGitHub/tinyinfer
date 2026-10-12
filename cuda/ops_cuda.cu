#include <cuda_runtime.h>
#include "../include/tinyinfer/tensor.hpp"

namespace tinyinfer
{

    __global__ void reluKernel(const float *input, float *output, std::size_t n)
    {
        std::size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        for (std::size_t i = idx; i < n; i += gridDim.x * blockDim.x)
        {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }

    __global__ void addKernel(const float *a, const float *b, float *ouput, std::size_t n)
    {
        std::size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        for (std::size_t i = idx; i < n; i += gridDim.x * blockDim.x)
        {
            ouput[i] = a[i] + b[i];
        }
    }

    __global__ void matmulKernel(
        const float *a,
        const float *b,
        float *out,
        int numRowsA,
        int sharedDim,
        int numColsB)
    {
        std::size_t row = threadIdx.y + blockDim.y * blockIdx.y;
        std::size_t col = threadIdx.x + blockDim.x * blockIdx.x;

        // A: numRowsA x sharedDim
        // B: sharedDim x numColsB
        // C: numRowsA x numColsB

        if (row < numRowsA && col < numColsB) {
            float f = 0.0;
            for (std::size_t i = 0; i < sharedDim; i++) {
                f += a[row * sharedDim + i] * b[col + i * numColsB];
            }
            out[row * numColsB + col] = f;
        }
    }

    Tensor reluCUDA(const Tensor &input)
    {
        Tensor T{input.shape(), Device::CUDA};
        const int blockSize = 256;
        const int numBlocks = static_cast<int>(std::min<std::size_t>((input.numel() + blockSize - 1) / blockSize, 16 * 48));
        reluKernel<<<numBlocks, blockSize>>>(input.data(), T.data(), input.numel());
        return T;
    }

    Tensor addCUDA(const Tensor &a, const Tensor &b)
    {
        Tensor T{a.shape(), Device::CUDA};
        const int blockSize = 256;
        const int numBlocks = static_cast<int>(std::min<std::size_t>((a.numel() + blockSize - 1) / blockSize, 16 * 48));
        addKernel<<<numBlocks, blockSize>>>(a.data(), b.data(), T.data(), a.numel());
        return T;
    }

    Tensor mamtulCUDA(const Tensor &a, const Tensor &b)
    {
        const int numRowsA = static_cast<int>(a.shape()[0]);
        const int sharedDim = static_cast<int>(a.shape()[1]);
        const int numColsB = static_cast<int>(b.shape()[1]);

        Tensor result{{a.shape()[0], b.shape()[1]}, Device::CUDA};

        const dim3 blockSize{16, 16};
        const dim3 numBlocks{
            (static_cast<unsigned int>(numColsB) + blockSize.x - 1) / blockSize.x,
            (static_cast<unsigned int>(numRowsA) + blockSize.y - 1) / blockSize.y};

        matmulKernel<<<numBlocks, blockSize>>>(
            a.data(),
            b.data(),
            result.data(),
            numRowsA,
            sharedDim,
            numColsB);

        return result;
    }

}
