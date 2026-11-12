#include "tinyinfer/ops.hpp"
#include <stdexcept>

namespace tinyinfer
{

    Tensor reluCUDA(const Tensor &input);
    Tensor reluCPU(const Tensor &input);
    Tensor addCUDA(const Tensor &a, const Tensor &b);
    Tensor addCPU(const Tensor &a, const Tensor &b);
    void matmulCUDAInto(const Tensor &a, const Tensor &b, Tensor &output);
    void matmulCPUInto(const Tensor &a, const Tensor &b, Tensor &output);
    Tensor addBiasCUDA(const Tensor &a, const Tensor &b);
    Tensor addBiasCPU(const Tensor &a, const Tensor &b);

    Tensor addBias(const Tensor &a, const Tensor &b)
    {
        if (a.shape().size() != 2 || b.shape().size() != 1)
        {
            throw std::runtime_error("addBias needs a matrix and a bias vector");
        }
        if (a.device() != b.device())
        {
            throw std::runtime_error("incompatible device");
        }

        if (a.shape()[1] != b.shape()[0])
        {
            throw std::runtime_error("incompatible shapes");
        }

        if (a.device() == Device::CPU)
        {
            return addBiasCPU(a, b);
        }
        else
        {
            return addBiasCUDA(a, b);
        }
    }

    Tensor addBiasCPU(const Tensor &a, const Tensor &b)
    {
        const std::size_t M = a.shape()[0];
        const std::size_t K = a.shape()[1];
        const float *A = a.data();
        const float *B = b.data();

        Tensor result{{M, K}, Device::CPU};
        float *C = result.data();
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
              C[K * i + j]  = A[K * i + j] + B[j];
            }
        }
        return result;
    }

    Tensor matmul(const Tensor &a, const Tensor &b)
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

        Tensor output({a.shape()[0], b.shape()[1]}, a.device());
        matmulInto(a, b, output);
        return output;
    };

    void matmulInto(const Tensor &a, const Tensor &b, Tensor &output)
    {
        if (a.shape().size() != 2 || b.shape().size() != 2 ||
            a.shape()[1] != b.shape()[0] || a.device() != b.device())
        {
            throw std::runtime_error("incompatible matmul inputs");
        }
        if (output.shape() != std::vector<std::size_t>{a.shape()[0], b.shape()[1]} ||
            output.device() != a.device() || output.data() == a.data() || output.data() == b.data())
        {
            throw std::runtime_error("incompatible matmul output");
        }
        if (a.device() == Device::CPU)
            matmulCPUInto(a, b, output);
        else
            matmulCUDAInto(a, b, output);
    }

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

    void matmulCPUInto(const Tensor &a, const Tensor &b, Tensor &result)
    {
        const std::size_t M = a.shape()[0];
        const std::size_t K = a.shape()[1];
        const std::size_t N = b.shape()[1];

        const float *A = a.data();
        const float *B = b.data();
        float *C = result.data();

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
