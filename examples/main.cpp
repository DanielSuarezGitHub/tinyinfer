#include <cuda_runtime.h>
#include "tinyinfer/tensor.hpp"
#include "tinyinfer/ops.hpp"

using namespace tinyinfer;

int main()
{
    constexpr std::size_t N = 1024;

    Tensor a{{N, N}, Device::CPU};
    Tensor b{{N, N}, Device::CPU};

    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        a.data()[i] = 1.0f;
        b.data()[i] = 2.0f;
    }

    Tensor aGPU = a.to(Device::CUDA);
    Tensor bGPU = b.to(Device::CUDA);

    Tensor c = mamtul(aGPU, bGPU);

    cudaDeviceSynchronize();
}