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

    Tensor reluCUDA(const Tensor &input)
    {
        Tensor T{input.shape(), Device::CUDA};
        const int blockSize = 256;
        const int numBlocks = static_cast<int>(std::min<std::size_t>((input.numel() + blockSize - 1) / blockSize, 16 * 48));
        reluKernel<<<numBlocks, blockSize>>>(input.data(), T.data(), input.numel());
        return T;
    }

}