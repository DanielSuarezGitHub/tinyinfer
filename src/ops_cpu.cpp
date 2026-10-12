#include "tinyinfer/ops.hpp"

namespace tinyinfer
{

    Tensor reluCUDA(const Tensor &input);
    Tensor reluCPU(const Tensor &input);
    Tensor addCUDA(const Tensor &a, const Tensor &b);
    Tensor addCPU(const Tensor &a, const Tensor &b);
    Tensor mamtulCUDA(const Tensor &a, const Tensor &b);
    Tensor mamtulCPU(const Tensor &a, const Tensor &b);



    Tensor mamtul(const Tensor &a, const Tensor &b)
    {
        if (a.device() != b.device())
        {
            throw std::runtime_error("incompatible device");
        }
        else if (a.shape().size() != 2 || b.shape().size() != 2)
        {
            throw std::runtime_error("Dimension mismatch");
        }
        else if (a.shape()[1] != b.shape()[0])
        {
            throw std::runtime_error("incompatible shapes");
        }

        if (a.device() == Device::CPU) {
            return mamtulCPU(a,b);
        } else {
            return mamtulCUDA(a,b);
        }
    };

    Tensor relu(const Tensor &input)
    {
        if (input.device() == Device::CPU)
        {
            return reluCPU(input);
        }
        else
        {
            return reluCUDA(input);
        }
    }

    Tensor add(const Tensor &a, const Tensor &b)
    {
        if (a.device() != b.device())
        {
            throw std::runtime_error("incompatible device");
        }
        else if (a.shape() != b.shape())
        {
            throw std::runtime_error("incompatible shapes");
        }

        if (a.device() == Device::CPU)
        {
            return addCPU(a, b);
        }
        else
        {
            return addCUDA(a, b);
        }
    }

    Tensor mamtulCPU(const Tensor& a, const Tensor& b)
{
    const std::size_t M = a.shape()[0];
    const std::size_t K = a.shape()[1];
    const std::size_t N = b.shape()[1];

    Tensor result{{M, N}, Device::CPU};

    const float* A = a.data();
    const float* B = b.data();
    float* C = result.data();

    for (std::size_t i = 0; i < M; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;

            for (std::size_t k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }

    return result;
}

    Tensor addCPU(const Tensor &a, const Tensor &b)
    {
        Tensor T{a.shape(), Device::CPU};
        const float *Adata = a.data();
        const float *Bdata = b.data();
        float *Tdata = T.data();
        for (std::size_t i = 0; i < a.numel(); i++)
        {
            Tdata[i] = Bdata[i] + Adata[i];
        }
        return T;
    }

    Tensor reluCPU(const Tensor &input)
    {
        Tensor T{input.shape(), Device::CPU};
        const float *dataI = input.data();
        float *dataT = T.data();
        for (std::size_t i = 0; i < input.numel(); i++)
        {
            dataT[i] = dataI[i] > 0 ? dataI[i] : 0;
        }

        return T;
    }

}
